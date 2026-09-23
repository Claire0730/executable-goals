"""Keypoint wrapper for StackCube-v1, under the SHARED task contract.

WHY A SECOND STACKCUBE FILE. `kp_env.py` already wraps this task, but it carries
723 lines of goal-error-study machinery (`goal_mode`, `goal_error_m`,
`GOAL_LEAK_DIMS`, four observation modes) and, decisively, **no camera** --
`msppo/kp_env.py` builds with `obs_mode="state"` and nothing in the StackCube
line (`train_rl` / `distill` / `arm_t` / `eval_rl`) contains a single reference to
`sensor_data` or a perception layer. So the earlier `kp_env.py` line is
state-only; the released StackCube teacher and student use this file's
perception path.

This file exists to put StackCube on the same footing as the other three tasks:
the shared `kp_teacher` / `task_student_env` / `task_distill` stack, with the
RGB-D perception path and the `--goal-delta` interface. It does NOT touch
`kp_env.py`, so no existing StackCube result can move.

FOUR THINGS READ FROM THE ENV, NOT ASSUMED
──────────────────────────────────────────
1.  Layout is 48-D, the offsets verified in `kp_env.py:32-34` by dumping
    obs_mode="state_dict":
        qpos(9) qvel(9) tcp_pose(7) cubeA_pose(7) cubeB_pose(7)
        tcp_to_cubeA(3) tcp_to_cubeB(3) cubeA_to_cubeB(3)

2.  SUCCESS IS POSITION-ONLY, AND IT NEEDS THE GRIPPER OPEN.
    `stack_cube.py:112-131` requires `on_cubeB AND static AND NOT grasped`. The
    ungrasp requirement is what an earlier custom reward missed here, stalling in
    a release valley, so the official reward is used untouched (`--w-kp 0`).
    Tolerances: xy is ||[.02,.02]||+.005 = 33 mm but **z is only 5 mm**. Depth is
    the binding axis, and depth is exactly where a monocular trace is weakest.

3.  THE GOAL ORIENTATION IS NOT CONSTRAINED. `evaluate()` checks position only,
    so demanding a specific yaw would invent a requirement the task does not
    have. `kp_env._set_oracle_goal` keeps cubeA's own yaw for exactly this
    reason, and this file does the same -- but COMMITS it at episode start rather
    than tracking the live yaw, because a planner emits one goal per episode.

4.  The fixture is cubeB, a real second body -- unlike LiftPegUpright, which has
    none.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from msppo.kp import CUBE_HALF
from msppo.peg_kp_env import to_world, unit_box_keypoints

# Verified in kp_env.py:32-34 against obs_mode="state_dict".
PROPRIO = slice(0, 25)          # qpos 9 + qvel 9 + tcp_pose 7
TCP_P = slice(18, 21)
OBJ_P, OBJ_Q = slice(25, 28), slice(28, 32)      # cubeA
SCN_P, SCN_Q = slice(32, 35), slice(35, 39)      # cubeB


class KeypointStack:
    """Composition, not gym.Wrapper: ManiSkillVectorEnv is a VectorEnv and fails
    gym.Wrapper's isinstance check."""

    has_scene = True
    RAW_DIM = 48

    def __init__(self, env, num_kp=64, reward="stock", w_kp=0.0, seed=0,
                 device="cuda", ap2ap_fields=True, last_action=False,
                 sig_obs=False, w_bump=0.0, bump_mode="cum", contact_obs=False, priv_obs=False,
                 shaping_true_goal=False):
        assert reward == "stock", (
            "StackCube's official reward is used untouched: its staged form "
            "already contains the ungrasp term a custom reward here missed")
        self.env, self.base, self.device = env, env.unwrapped, device
        self.num_kp, self.reward_kind, self.w_kp = num_kp, reward, w_kp
        self.num_envs = self.base.num_envs

        # CUBE-B DISPLACEMENT PENALTY. The stock reward has no term
        # for knocking cubeB aside, and the goal is bound to cubeB's LIVE pose
        # (`_goal_from_obs` reads base_obs[:, SCN_P]), so after a shove the goal
        # simply follows and the policy still scores. Success uses the live
        # `pos_A - pos_B` too. Nothing anywhere says "do not hit it".
        self.w_bump = float(w_bump)
        # bump_mode "cum" penalises cubeB's displacement from its episode start.
        # That version DIVERGED (grasp 20x worse at equal iteration, KL 0.035 ->
        # 0.937): once cubeB has moved the penalty is stuck on for the rest of
        # the episode no matter what the policy does, so it is unrecoverable and
        # simply adds return variance on top of the reaching gradient.
        # "delta" penalises only this step's INCREASE in displacement -- zero
        # while cubeB is untouched, recoverable, and it prices the act of
        # bumping rather than the history of having bumped. The released
        # StackCube teacher (sc_v9_nz03b_s0) uses bump_mode 'delta'.
        self.bump_mode = bump_mode
        self._p_b0 = torch.zeros((self.num_envs, 3), device=device)
        self._p_bprev = torch.zeros((self.num_envs, 3), device=device)

        # A cube never changes shape, so the canonical set is a fixed [K,3].
        # Resampling it per reconfiguration is what silently sent an earlier
        # StackCube policy out of distribution (88% -> 7%), per `kp.py`.
        self._canon = (unit_box_keypoints(num_kp, seed).to(device) * CUBE_HALF)

        # Committed at reset, refreshed per env on done -- docstring item 3.
        self._goal_q = torch.zeros((self.num_envs, 4), device=device)
        self._goal_q[:, 0] = 1.0
        self._last_action = torch.zeros(
            (self.num_envs, env.single_action_space.shape[0]), device=device)

        self.ap2ap_fields = ap2ap_fields
        sp = self.env.single_observation_space
        base_dim = sp.shape[-1] if sp.shape is not None else self.RAW_DIM
        assert base_dim == self.RAW_DIM, (
            f"stock observation is {base_dim}, expected {self.RAW_DIM}")
        if ap2ap_fields:
            base_dim += 20      # is_grasped 1 + tcp_to_obj 3 + obj_vel 6
                                # + goal_pose 7 + obj_to_goal 3
        self.last_action_obs = last_action
        if last_action:
            base_dim += env.single_action_space.shape[0]

        # See `peg_kp_env` for why these 4 dims sit here and not elsewhere.
        self.sig_obs = bool(sig_obs)
        if self.sig_obs:
            base_dim += 4
        # CONTACT CHANNEL. Every StackCube teacher variant sits on the same
        # clean-vs-noised frontier, so no reweighting moves it; the policy is
        # short of INFORMATION, not of tuning. The release decision
        # needs "cubeA is resting on cubeB", which under goal noise cannot be
        # read off position. Pairwise contact force says it directly. 4 dims:
        # cubeA-cubeB force vector (3) and its magnitude, log1p-compressed
        # because contact forces span decades. A real arm has an F/T sensor, so
        # this is a channel the student can also have at deployment.
        self.contact_obs = bool(contact_obs)
        if self.contact_obs:
            base_dim += 4
        # PRIVILEGED BLOCK for an asymmetric critic: true obj_p + true goal_p
        # from the CLEAN observation. Sits after sig, before the keypoints.
        self.priv_obs = bool(priv_obs)
        self.shaping_true_goal = bool(shaping_true_goal)
        self.priv_slice = slice(base_dim, base_dim + 6) if self.priv_obs else None
        if self.priv_obs:
            base_dim += 6

        self.kp_slice = slice(base_dim, base_dim + num_kp * 6)
        total = base_dim + num_kp * 6
        self.single_observation_space = gym.spaces.Box(
            -np.inf, np.inf, (total,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (self.num_envs, total), dtype=np.float32)
        self.single_action_space = self.env.single_action_space
        self.action_space = self.env.action_space

    # ------------------------------------------------------------ keypoints --
    def _canon_b(self, base_obs):
        return self._canon[None].expand(base_obs.shape[0], -1, -1)

    def _goal_from_obs(self, base_obs):
        """(p, q): cubeA resting on top of cubeB, at the yaw committed at reset.

        z is cubeB's centre plus one full cube edge (2 * half), which is where a
        resting cubeA's CENTRE sits -- not plus one half.
        """
        p = base_obs[:, SCN_P].clone()
        p[:, 2] = base_obs[:, SCN_P][:, 2] + CUBE_HALF * 2
        return p, self._goal_q

    def keypoints(self, base_obs):
        """(obj_kp, goal_kp, scene_kp), [B,K,3] world, paired by index.

        Read from the OBSERVATION vector, never from `self.base.cubeA.pose`: the
        vector env auto-resets before returning, so the live property already
        holds the NEXT episode for any env that finished (`peg_kp_env.py:206`).
        """
        canon = self._canon_b(base_obs)
        obj = to_world(canon, base_obs[:, OBJ_P], base_obs[:, OBJ_Q])
        gp, gq = self._goal_from_obs(base_obs)
        goal = to_world(canon, gp, gq)
        scene = to_world(canon, base_obs[:, SCN_P], base_obs[:, SCN_Q])
        return obj, goal, scene

    def _extra_fields(self, base_obs):
        gp, gq = self._goal_from_obs(base_obs)
        return torch.cat([
            self.base.agent.is_grasping(self.base.cubeA).float()[:, None],  # 1
            base_obs[:, OBJ_P] - base_obs[:, TCP_P],                        # 3
            self.base.cubeA.linear_velocity,
            self.base.cubeA.angular_velocity,                               # 6
            gp, gq,                                                          # 7
            gp - base_obs[:, OBJ_P],                                         # 3
        ], dim=-1)

    # TEACHER-SIDE OBSERVATION NOISE. Rewriting the pose slices here makes the
    # stock block, the AP2AP block and the keypoints all consistent, because all
    # three are derived from `obs` below. Default None = exactly the old path.
    def _priv_row(self, clean):
        gp, _ = self._goal_from_obs(clean)
        return torch.cat([clean[:, OBJ_P], gp], dim=-1)

    def _contact_row(self, b):
        """[b,4] cubeA-cubeB contact force, log1p-compressed, plus its magnitude."""
        f = self.base.scene.get_pairwise_contact_forces(self.base.cubeA, self.base.cubeB)
        m = f.norm(dim=-1, keepdim=True)
        return torch.cat([torch.log1p(f.abs()) * torch.sign(f), torch.log1p(m)], dim=-1)

    def _sig_row(self, b):
        """[b,4] noise-scale channel; zeros when no injector is attached, so a
        sig_obs run with no noise is a clean control rather than an error."""
        # DISTILLATION OVERRIDE. Under `--teacher-perc` there is no injector;
        # the reliability the teacher should condition on is the STUDENT's own
        # estimate, which `teacher_perc` writes here before calling _augment.
        ov = getattr(self, "_sig_override", None)
        if ov is not None:
            return ov
        n = getattr(self, "obs_noise", None)
        if n is None or not hasattr(n, "sig_row"):
            return torch.zeros((b, 4), device=self.device)
        return n.sig_row()

    def _noisy(self, obs):
        n = getattr(self, "obs_noise", None)
        if n is None:
            return obs
        # per-env episode boundary: `step` stores the done mask just before
        # calling `_augment`; reset leaves it None (global fallback)
        n.tick(getattr(self, "_done_mask", None))
        self._done_mask = None
        return n.apply(obs)

    def _augment(self, obs):
        return self._augment_core(self._noisy(obs), clean=obs)

    def _augment_core(self, obs, clean=None):
        """Layout only -- no noise tick. Split out so the TERMINAL observation of
        a finishing episode can be laid out with that episode's goal without
        advancing the noise injector a second time in the same step."""
        obj, goal, _ = self.keypoints(obs)
        b = obs.shape[0]
        parts = [obs]
        if self.ap2ap_fields:
            parts.append(self._extra_fields(obs))
        if self.last_action_obs:
            parts.append(self._last_action)
        if self.sig_obs:
            parts.append(self._sig_row(b))
        if self.contact_obs:
            parts.append(self._contact_row(b))
        if self.priv_obs:
            parts.append(self._priv_row(clean if clean is not None else obs))
        parts += [obj.reshape(b, -1), goal.reshape(b, -1)]
        return torch.cat(parts, dim=-1), obj, goal

    def raw_from_sim(self):
        """[B,48] the stock state vector, rebuilt from the LIVE sim, for
        PERCEPTION mode where `obs_mode="rgb+depth+segmentation"` withholds the
        object poses. Safe at the one point DAgger calls it -- see
        `liftpeg_kp_env.raw_from_sim`."""
        a = self.base.agent
        tcp = a.tcp.pose
        A, B = self.base.cubeA.pose, self.base.cubeB.pose
        return torch.cat([
            a.robot.get_qpos()[:, :9], a.robot.get_qvel()[:, :9],
            tcp.raw_pose, A.raw_pose, B.raw_pose,
            A.p - tcp.p, B.p - tcp.p, B.p - A.p], dim=-1).float()

    def teacher_observation(self, raw=None):
        return self._augment(raw if raw is not None else self.raw_from_sim())[0]

    # ------------------------------------------------------------------ gym --
    def _commit_goal(self, raw, idx=None):
        q = raw[:, OBJ_Q]
        b = raw[:, SCN_P]
        if idx is None:
            self._goal_q = q.clone()
            self._p_b0 = b.clone(); self._p_bprev = b.clone()
        else:
            self._goal_q[idx] = q[idx]
            self._p_b0[idx] = b[idx]; self._p_bprev[idx] = b[idx]

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_action = torch.zeros_like(self._last_action)
        raw = obs if torch.is_tensor(obs) else self.raw_from_sim()
        self._commit_goal(raw)
        aug, _, _ = self._augment(raw)
        return (aug if torch.is_tensor(obs) else obs), info

    def step(self, action):
        self._last_action = action.detach()
        obs, rew, term, trunc, info = self.env.step(action)
        raw = obs if torch.is_tensor(obs) else self.raw_from_sim()
        done = (term | trunc).bool()
        # TERMINAL VALUE BOOTSTRAP. `ppo.py:100-107` bootstraps
        # V(next) from the observation `step` returns, which under auto_reset is
        # the NEW episode's reset state -- unless the env supplies
        # `final_observation_kp`, the old episode's terminal state laid out with
        # the old episode's goal. `kp_env.py` supplies it; this env supplies it
        # too (below). With gamma 0.8 and a normalised reward ~0.85/step while
        # holding the cube on cubeB, holding is worth ~0.85/(1-0.8) = 4.2
        # whereas releasing, bootstrapped from a fresh reset state worth ~1-2,
        # is worth ~1 + 0.8*1.5 = 2.2. Without this block the agent is trained
        # NOT to let go.
        # Built BEFORE `_commit_goal`, so `self._goal_q` is still the old goal.
        fo_obj = fo_goal = None
        # Only in STATE mode: under perception the vector env stores a dict of
        # camera tensors there, and PPO's bootstrap (the only consumer) never
        # runs on the perception path -- distillation is supervised.
        if bool(done.any()) and torch.is_tensor(info.get("final_observation")):
            fo = info["final_observation"]
            n = getattr(self, "obs_noise", None)
            fo_aug, fo_obj, fo_goal = self._augment_core(n.apply(fo) if n is not None else fo, clean=fo)
            if self.shaping_true_goal:
                fo_obj, fo_goal, _ = self.keypoints(fo)
            info["final_observation_kp"] = fo_aug[done]
        if bool(done.any()):
            self._commit_goal(raw, done)
        self._done_mask = done          # consumed by _noisy -> tick(done)
        aug, obj, goal = self._augment(raw)
        if self.shaping_true_goal:
            # shaping toward the TRUE goal (kp_env's `shaping_true_goal`): the
            # dense term then carries information the observation does not,
            # which is the point -- reward is privileged, observation is not
            obj, goal, _ = self.keypoints(raw)
        gd = (obj - goal).norm(dim=-1).mean(-1)
        if fo_obj is not None:
            # the shaping term scores the transition that just happened, which
            # for a finished env is its terminal state, not the reset state
            gd = torch.where(done, (fo_obj - fo_goal).norm(dim=-1).mean(-1), gd)
        info["kp_dist"] = gd
        if self.w_kp:
            rew = rew + self.w_kp * (1 - torch.tanh(5.0 * gd))
        if self.w_bump:
            # TRUE cubeB pose, never the observation: the observation carries the
            # injected noise, and a penalty computed from it would make the
            # reward dishonest -- the one property the noise experiment rests on.
            pb = self.base.cubeB.pose.p
            d_b = (pb - self._p_b0).norm(dim=-1)
            info["cubeB_disp"] = d_b
            if self.bump_mode == "delta":
                d_b = (pb - self._p_bprev).norm(dim=-1) * 25.0   # per-step, scaled: 2 mm -> tanh(2.5)
            self._p_bprev = pb.clone()
            # Penalise DISPLACEMENT, not contact: cubeA must end up resting ON
            # cubeB, so contact is part of the task and being pushed away is the
            # harm. tanh saturates so one big shove is not unboundedly worse than
            # a moderate one. Not gated on `is_cubeA_on_cubeB` -- a gate would be
            # an exploitable discontinuity; the ~1-2 mm of legitimate settling
            # after stacking costs only tanh(0.075) ~ 0.07.
            rew = rew - self.w_bump * torch.tanh(d_b / CUBE_HALF)
        return (aug if torch.is_tensor(obs) else obs), rew, term, trunc, info

    def close(self):
        self.env.close()


def make_stack_kp_env(num_envs=256, num_kp=64, reward="stock", w_kp=0.0, seed=0,
                      control="pd_joint_delta_pos", max_episode_steps=50,
                      reconfig_freq=0, reward_mode="normalized_dense",
                      ap2ap_fields=True, ignore_terminations=False,
                      robot="panda_wristcam", last_action=False,
                      perception=False,
                      tracegen_camera=False, sig_obs=False, w_bump=0.0,
                      priv_obs=False, shaping_true_goal=False, bump_mode="cum",
                      contact_obs=False):
    # MSGEN_NO_RGB: the distillation path renders RGB every step and reads it
    # NOWHERE (perception uses depth per step and segmentation once at t=0;
    # only msppo.multi_video touches rgb). Dropping the shading pass is a
    # candidate free speedup -- OFF by default so recording keeps working.
    import os as _os
    _chan = "depth+segmentation" if _os.environ.get("MSGEN_NO_RGB") not in (None, "", "0") \
        else "rgb+depth+segmentation"
    mode = _chan if perception else "state"
    extra = {}
    if tracegen_camera:
        from mani_skill.utils import sapien_utils
        from msgen.tasks import IMAGE_SIZE, get_task
        c = get_task("stack")["camera"]
        extra["sensor_configs"] = {"base_camera": dict(
            pose=sapien_utils.look_at(eye=c["eye"], target=c["target"]),
            width=IMAGE_SIZE, height=IMAGE_SIZE, fov=c["fov"])}
    env = gym.make("StackCube-v1", num_envs=num_envs, obs_mode=mode,
                   control_mode=control, sim_backend="physx_cuda",
                   max_episode_steps=max_episode_steps, reward_mode=reward_mode,
                   reconfiguration_freq=reconfig_freq, robot_uids=robot, **extra)
    env = ManiSkillVectorEnv(env, auto_reset=True,
                             ignore_terminations=ignore_terminations)
    return KeypointStack(env, num_kp=num_kp, reward=reward, w_kp=w_kp,
                         seed=seed, ap2ap_fields=ap2ap_fields,
                         last_action=last_action, sig_obs=sig_obs,
                         w_bump=w_bump, bump_mode=bump_mode, contact_obs=contact_obs, priv_obs=priv_obs,
                         shaping_true_goal=shaping_true_goal)
