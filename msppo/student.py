"""The student policy: Dex4D's `ActorPointNetTransformer`, re-dimensioned.

Ported from `Dex4D-Simulation/dex4d_policy/dex4d/algorithms/rl/dagger/module.py:152`
and `algo/pn_utils/transformer.py`. Two things about it are easy to get wrong
from the paper alone and are worth stating plainly:

  * the transformer attends over MODALITY tokens, not over points -- each point
    set is collapsed to a single token by a PointNet first, so the sequence is 6
    long, not 200;
  * there is NO temporal axis. History enters only through the previous-action
    token and the auxiliary next-state prediction.

Token layout (theirs -> ours):

    robot_qpos      22 -> 9
    robot_qvel      22 -> 9
    action          22 -> 8      <- the action mean is READ OUT of this token
    obj+goal kp   256x6 -> 64x6
    (none)                7      tcp_pose, which their xArm6 obs has no analogue of
    (none)             64x3      scene_kp: optional fixture points (not used by
                                 the released student)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """BERT-style, matching `algo/pn_utils/transformer.py:37` including its
    pre-norm-then-residual ordering (`x = ln(x); x = x + attn(x)`), which is not
    the usual pre-norm form but is what the reference does."""

    def __init__(self, d, heads, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln_2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d), nn.Dropout(dropout))

    def forward(self, x):
        x = self.ln_1(x)
        x = x + self.attn(x, x, x, need_weights=False)[0]
        x = self.ln_2(x)
        return x + self.mlp(x)


class PointNetToken(nn.Module):
    """`Simple6DPointNetBackbone` (`ppo/module.py:42`): centre each xyz block,
    append the block mean, shared Conv1d MLP, then a max/mean split aggregation.

    The centre-and-append trick matters here more than upstream: masked points
    arrive as exact zeros, so a plain mean pool would drag the descriptor toward
    the origin as occlusion varies. Splitting the aggregation so half the channels
    are max-pooled keeps a path that zeros cannot dominate.
    """

    def __init__(self, pc_dim: int, feature_dim: int = 128):
        super().__init__()
        assert pc_dim % 3 == 0
        self.blocks = pc_dim // 3
        self.conv = nn.Sequential(
            nn.Conv1d(pc_dim * 2, 128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 1, bias=False), nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(256, 256), nn.ReLU(),
                                  nn.Linear(256, feature_dim))

    def forward(self, pc):                       # [B,N,pc_dim] -> [B,F]
        n = pc.shape[1]
        feats = []
        for b in range(self.blocks):
            xyz = pc[:, :, 3 * b: 3 * b + 3]
            m = xyz.mean(dim=1, keepdim=True)
            feats += [m.expand(-1, n, -1), xyz - m]
        f = self.conv(torch.cat(feats, dim=-1).transpose(1, 2)).transpose(1, 2)
        sep = f.shape[-1] // 2
        g = torch.cat([f[..., :sep].amax(dim=1), f[..., sep:].mean(dim=1)], dim=-1)
        return self.head(g)


class StudentTransformer(nn.Module):
    def __init__(self, num_kp=64, act_dim=8, d=128, layers=4, heads=4,
                 psi=False, psi_token=False,
                 dropout=0.0, scene_kp=True, prev_action=True,
                 init_logstd=-0.5, predict_future=True, with_critic=False,
                 pose_obs=False, tcp=True, qdim=9, lang_dim=0,
                 rel_token=False, goal_sig=False, goal_form="mean", goal_k=4):
        super().__init__()
        self.num_kp, self.act_dim = num_kp, act_dim
        # Joint count. 9 on every Panda task (7 arm + 2 finger) and 7 on PushT,
        # whose panda_stick has no gripper. This was hardcoded to 9 in the slice
        # map AND in the two tokenizer Linears; the two would have disagreed
        # silently on PushT, because the slice map would have over-read qpos into
        # qvel while the Linear still accepted the wrong-width vector.
        self.qdim = qdim
        # Task-identity channel for a MIXED-TASK student, appended to the END of
        # the observation so every existing layout stays a prefix and every
        # existing checkpoint loads unchanged (default 0 = absent).
        #   0    none      -- purely goal-conditioned, the hypothesis under test
        #   K    one-hot   -- the correct control for the language arm
        #   512  t5-small  -- the same frozen encoder TraceGen conditions on
        #                     (TraceGen/models/model_flow.py:81-83), looked up
        #                     from assets/instr_t5.npz; no T5 runs at control rate
        #                     (the asset file is not shipped; lang=none in the
        #                     released student)
        #
        # ⚠️ With a fixed instruction set (15 strings, 5 tasks) the language
        # channel carries at most log2(n_tasks) bits -- it is NOT more INFORMATIVE
        # than the one-hot, only more extensible. Any claim that language helps
        # must be made against the one-hot arm, never against no-channel.
        self.lang_dim = lang_dim
        self.scene_kp, self.prev_action = scene_kp, prev_action
        # Dex4D's student takes joint angle, joint velocity, last action and the
        # masked paired points -- and nothing else (Fig.2(b), Table VIII). tcp is
        # not privileged (it is qpos through forward kinematics) but it is not in
        # their list, so `tcp=False` is what a literal port looks like.
        self.tcp = tcp
        self.predict_future = predict_future
        # `pose_obs` consumes the SE(3) the env already solved from the (occluded)
        # keypoints instead of the raw points. Same trunk, same readout token,
        # same auxiliary heads -- only the point encoders are replaced, so a
        # comparison against the raw-point student isolates the encoding.
        self.pose_obs = pose_obs

        # Flat slice map. Mirrors `kp_env.KeypointStackCube._augment`'s student
        # branch exactly; a mismatch here is silent, so `msppo/checks.py` asserts
        # the env's dim against this layout.
        p = 0
        self.sl = {}
        for k, n in ([("qpos", qdim), ("qvel", qdim)] + ([("tcp", 7)] if tcp else [])):
            self.sl[k] = slice(p, p + n); p += n
        if prev_action:
            self.sl["action"] = slice(p, p + act_dim); p += act_dim
        if pose_obs:
            # Mirrors kp_env's `student_recon` branch exactly. A mismatch here is
            # silent, so msppo/checks.py asserts the env dim against this layout.
            for k in (["obj_pose", "goal_pose"] + (["scene_pose"] if scene_kp else [])):
                self.sl[k] = slice(p, p + 7); p += 7
            nrel = 4 if scene_kp else 3
            self.nrel = nrel
            self.sl["rel"] = slice(p, p + 3 * nrel); p += 3 * nrel
            # RELIABILITY CHANNEL: [rmse (m), sigma2/sigma1] of the solve that
            # produced this goal. Placed after `rel` and before `lang` so every
            # older layout stays a prefix and older checkpoints load unchanged.
            #
            # It is NOT in goal_slices(). The pair says how far to trust the
            # goal, not where the goal is: with the goal block zeroed it cannot
            # be inverted to recover a target, so the no-goal ablation stays
            # honest without zeroing it. (If a later arm makes the signal
            # goal-dependent, that argument must be re-made, not assumed.)
            self.goal_sig = bool(goal_sig)
            if self.goal_sig:
                self.sl["goal_sig"] = slice(p, p + 2); p += 2
            # K-SAMPLE GOAL FORM. goal_pose ALWAYS holds the
            # K-mean; these slots carry what the arm adds on top:
            #   sd   diag sd of the K sample positions (world axes), 3
            #   k    the K sample poses themselves, K x 7, order-free tokens
            # Both can be inverted to a goal, so goal_extra_slices() lists them
            # for --no-trace zeroing (unlike goal_sig, which only says how far
            # to trust the goal). Placed after goal_sig so every older layout
            # stays a prefix and older checkpoints load unchanged.
            self.goal_form, self.goal_k = goal_form, int(goal_k)
            if goal_form == "sd":
                self.sl["goal_sd"] = slice(p, p + 3); p += 3
            elif goal_form == "k":
                self.sl["goal_k"] = slice(p, p + 7 * self.goal_k); p += 7 * self.goal_k
            elif goal_form == "ell":
                # DIRECTION ONLY. The principal axis u of the K
                # sample positions, as the upper triangle of u u^T (a line, not
                # a vector: no sign ambiguity). Magnitude is deliberately absent
                # -- sample spread does not track error but its axis does.
                # All-zero when the samples coincide (identity / a=0).
                self.sl["goal_ell"] = slice(p, p + 6); p += 6
            elif goal_form != "mean":
                raise ValueError(f"goal_form {goal_form!r} not in mean/sd/k/ell")
            # psi (k1, h) rides in the objgoal token: Linear(14 -> d) becomes Linear(18 -> d). Same token, no new
            # attention slot: the command modulates the object/goal representation the action reads from.
            self.psi = bool(psi)
            if self.psi:
                self.sl["psi"] = slice(p, p + 4); p += 4
        else:
            self.psi = False
            self.goal_form, self.goal_k = "mean", int(goal_k)
            self.sl["obj"] = slice(p, p + num_kp * 3); p += num_kp * 3
            self.sl["goal"] = slice(p, p + num_kp * 3); p += num_kp * 3
            if scene_kp:
                self.sl["scene"] = slice(p, p + num_kp * 3); p += num_kp * 3
        # psi_token: psi as its OWN token (Linear(4 -> d)) instead of riding in the objgoal token, so the
        # student can attend to the command without also attending to the goal pose.
        self.psi_token = bool(psi_token) and self.psi
        if lang_dim:
            self.sl["lang"] = slice(p, p + lang_dim); p += lang_dim
        self.obs_dim = p

        self.tok = nn.ModuleDict({"qpos": nn.Linear(qdim, d),
                                  "qvel": nn.Linear(qdim, d)})
        self.order = ["qpos", "qvel"]
        if tcp:
            self.tok["tcp"] = nn.Linear(7, d)
            self.order.append("tcp")
        if prev_action:
            self.tok["action"] = nn.Linear(act_dim, d)
            self.order.append("action")
        # GIVE `rel` A TOKEN OF ITS OWN.
        #
        # `rel` (obj-tcp, goal-tcp, goal-obj, [scene-obj], all 3-D world-axis
        # TRANSLATIONS) was only ever consumed as the tail of the scene token,
        # `Linear(7+12 -> d)`. With `scene_kp=False` that token does not exist,
        # so under --no-scene the block was packed into the observation and read
        # by NOTHING: d(action)/d(rel) was exactly 0.000e+00 and
        # a +100 m perturbation moved the action by exactly 0.
        #
        # Setting rel_token gives it its own token and shrinks the scene token to
        # the fixture pose alone, so the two encodings differ in one place only.
        # It adds NO information -- the observation is byte-identical -- which is
        # what makes it the necessary control.
        self.rel_token = bool(rel_token) and pose_obs
        if pose_obs:
            # One token for the object+goal pair (14) and one for the scene pose
            # plus the relative vectors, so the token COUNT matches the raw-point
            # student and only the encoder type differs.
            self.tok["objgoal"] = nn.Linear(14 + (4 if (self.psi and not self.psi_token) else 0), d)
            self.order.append("objgoal")
            if self.psi_token:
                self.tok["psi"] = nn.Linear(4, d)
                self.order.append("psi")
            if self.rel_token:
                self.tok["rel"] = nn.Linear(3 * self.nrel, d)
                self.order.append("rel")
                if scene_kp:
                    self.tok["scene"] = nn.Linear(7, d)
                    self.order.append("scene")
            elif scene_kp:
                self.tok["scene"] = nn.Linear(7 + 12, d)
                self.order.append("scene")
            if getattr(self, "goal_sig", False):
                self.tok["goal_sig"] = nn.Linear(2, d)
                self.order.append("goal_sig")
            if self.goal_form == "sd":
                self.tok["goal_sd"] = nn.Linear(3, d)
                self.order.append("goal_sd")
            elif self.goal_form == "ell":
                self.tok["goal_ell"] = nn.Linear(6, d)
                self.order.append("goal_ell")
            elif self.goal_form == "k":
                # ONE Linear shared by the K samples -> K tokens. No positional
                # encoding anywhere in this trunk, so the K tokens are order-free
                # by construction (checked in tests/test_student_form.py).
                self.tok["goal_k"] = nn.Linear(7, d)
                self.order.append("goal_k")
        else:
            self.tok["objgoal"] = PointNetToken(6, d)
            self.order.append("objgoal")
            if scene_kp:
                self.tok["scene"] = PointNetToken(3, d)
                self.order.append("scene")
        if lang_dim:
            self.tok["lang"] = nn.Linear(lang_dim, d)
            self.order.append("lang")
        # Read the action out of the previous-action token, as upstream does
        # (`dagger/module.py:194`). Without prev_action there is no such token, so
        # fall back to the paired-keypoint one -- the only token that always
        # carries goal information.
        self.out_idx = self.order.index("action" if prev_action else "objgoal")

        self.blocks = nn.Sequential(*[TransformerBlock(d, heads, dropout)
                                      for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, act_dim))
        self.out[1].weight.data *= 0.01          # `dagger/module.py:173`
        self.log_std = nn.Parameter(torch.ones(act_dim) * init_logstd)

        if predict_future:
            self.future = nn.ModuleDict({"qpos": nn.Linear(d, qdim),
                                         "qvel": nn.Linear(d, qdim)})
            for m in self.future.values():
                m.weight.data *= 0.01

        # Only the PPO-from-scratch baseline needs this: PPO on the same
        # observation and the same trunk, so that the comparison isolates
        # DISTILLATION rather than the architecture. DAgger never queries a
        # value, so the head is absent by default and the distilled checkpoints
        # stay free of unused parameters.
        self.with_critic = with_critic
        if with_critic:
            self.critic_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def tokens(self, obs):
        t = []
        for k in self.order:
            if self.pose_obs and k == "objgoal":
                t.append(self.tok["objgoal"](
                    torch.cat([obs[:, self.sl["obj_pose"]],
                               obs[:, self.sl["goal_pose"]]]
                              + ([obs[:, self.sl["psi"]]] if (self.psi and not self.psi_token) else []), dim=-1)))
            elif self.pose_obs and k == "scene":
                x = obs[:, self.sl["scene_pose"]]
                if not self.rel_token:
                    x = torch.cat([x, obs[:, self.sl["rel"]]], dim=-1)
                t.append(self.tok["scene"](x))
            elif k == "rel":
                t.append(self.tok["rel"](obs[:, self.sl["rel"]]))
            elif k == "goal_k":
                g = obs[:, self.sl["goal_k"]].reshape(-1, self.goal_k, 7)
                t.extend(self.tok["goal_k"](g).unbind(dim=1))      # K tokens

            elif k == "objgoal":
                o = obs[:, self.sl["obj"]].reshape(-1, self.num_kp, 3)
                g = obs[:, self.sl["goal"]].reshape(-1, self.num_kp, 3)
                t.append(self.tok["objgoal"](torch.cat([o, g], dim=-1)))
            elif k == "scene":
                t.append(self.tok["scene"](obs[:, self.sl["scene"]].reshape(-1, self.num_kp, 3)))
            else:
                t.append(self.tok[k](obs[:, self.sl[k]]))
        return torch.stack(t, dim=1)                       # [B, T, d]

    def forward(self, obs):
        z = self.blocks(self.tokens(obs))
        mean = self.out(z[:, self.out_idx])
        fut = None
        if self.predict_future:
            fut = {k: self.future[k](z[:, self.order.index(k)]) for k in ("qpos", "qvel")}
        return mean, fut

    # ---- the interface `eval_rl.py` and `arm_t.py` already expect of an agent --
    def encode(self, obs):
        return obs

    def actor_mean(self, obs):
        return self.forward(obs)[0]

    def get_value(self, obs):
        z = self.blocks(self.tokens(obs))
        return self.critic_head(z[:, self.out_idx]).squeeze(-1)

    def get_action_and_value(self, obs, action=None):
        z = self.blocks(self.tokens(obs))
        mean = self.out(z[:, self.out_idx])
        dist = torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))
        if action is None:
            action = dist.sample()
        # Without a critic (the distilled arms) return zeros: DAgger never asks
        # for a value, and the 4-tuple contract keeps the existing evaluators
        # working unchanged.
        v = (self.critic_head(z[:, self.out_idx]).squeeze(-1) if self.with_critic
             else torch.zeros(mean.shape[0], device=mean.device))
        return (action, dist.log_prob(action).sum(1), dist.entropy().sum(1), v)


def student_from_cfg(cfg, act_dim, env=None, device="cuda"):
    """Build the StudentTransformer a run.json describes -- ONE place, so the
    four evaluators cannot drift apart.

    Two fields decide the architecture and both are silent when wrong:

      point_obs   the PAIRED-POINT encoding (609-D) vs the pose encoding (66-D).
                  Different observation AND different tokenizer; a mismatch is a
                  load_state_dict failure at best and a wrong number at worst.
      n_pts       NOT always num_kp. `--query grid` feeds all 400 TraceGen query
                  points (~4.9 on the peg, rest masked to zero), so a student
                  built with num_kp=64 mis-slices a 3633-D observation.

    `env` is consulted only as a fallback for runs written before `n_pts` was
    recorded; run.json wins when it has the field.
    """
    point_obs = bool(cfg.get("point_obs", False))
    n_pts = cfg.get("n_pts")
    if n_pts is None:
        n_pts = (getattr(env, "n_pts", None) if point_obs else None) or cfg.get("num_kp", 64)
    return StudentTransformer(num_kp=n_pts, act_dim=act_dim,
                              pose_obs=not point_obs,
                              goal_sig=cfg.get("goal_sig", False),
                              # K-sample goal form; absent in older run.json files
                              goal_form=cfg.get("goal_form", "mean"),
                              goal_k=cfg.get("goal_k", 4),
                              # absent in older run.json files, so those
                              # checkpoints rebuild unchanged
                              rel_token=cfg.get("rel_token", False),
                              scene_kp=cfg.get("scene_kp", True),
                              tcp=cfg.get("tcp_obs", True),
                              # 9 for every run written before PushT existed, so
                              # the default keeps those checkpoints loadable.
                              qdim=cfg.get("qdim", 9),
                              psi=bool(cfg.get("psi", False)), psi_token=bool(cfg.get("psi_token", False)),
                              prev_action=True).to(device)


def goal_slices(student):
    """(goal_sl, goal_rel_sl) for either encoding -- ONE place.

    `--no-trace` zeroes the goal block; `--strict-no-trace` additionally zeroes
    the two `rel` vectors that carry it (tcp->goal, obj->goal), because with
    obj_pose given the goal POSITION is otherwise exactly recoverable -- measured
    at 3.07e-05 mm. The slice names must never be hardcoded:

      pose encoding   sl["goal_pose"] + rel[3:9]
      point encoding  sl["goal"]      + (nothing: there is no rel block, so
                                         zeroing `goal` already removes every
                                         goal-carrying dimension)
    """
    if "goal_pose" in student.sl:                       # pose encoding
        r = student.sl["rel"]
        return student.sl["goal_pose"], slice(r.start + 3, r.start + 9)
    return student.sl["goal"], slice(0, 0)              # point encoding


def goal_extra_slices(student):
    """Goal-carrying slots the K-sample arms add (`goal_sd`, `goal_k`), to be
    zeroed alongside goal_slices() under --no-trace. Empty for every earlier
    layout, so the existing goal_slices() call sites keep their 2-tuple;
    an evaluator that zeroes the goal must ALSO zero these."""
    return [student.sl[k] for k in ("goal_sd", "goal_k", "goal_ell") if k in student.sl]
