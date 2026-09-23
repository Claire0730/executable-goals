"""Executor environment for every task in `kp_teacher.TASKS`: occluded points
in, solved SE(3) or raw paired points out.

Generalises `peg_student_env.py`. Two things are deliberately SIMPLER here and
both are limitations, not improvements:

  * THE TEACHER IS OUR OWN. peg distils from a frozen teacher checkpoint, so
    `peg_student_env` has to rebuild that teacher's 436-D observation field by
    field. Here the teacher is trained by `kp_teacher.py` on
    this very env, so the teacher view is just the keypoint env's own augmented
    observation -- the same tensor, no adapter, nothing to desync.

  * THERE ARE TWO PERCEPTION PATHS AND THEY ARE NOT COMPARABLE.

    `perception=False` (the state path) takes the object and fixture poses from
    the SIMULATOR and corrupts them with the synthetic occlusion + noise model
    below. `perception=True` renders RGB-D through the task's FITTED TraceGen
    camera (`msgen/tasks.py`) and derives every pose from that frame, reusing
    `peg_perception.PegPerception` -- which needed only an actor name to
    generalise, the peg defaults being kept so the peg numbers that rest on it
    cannot move.

    ⚠️ The state path is optimistic; only `perception=True` is comparable to
    the paper's rows.

    ⚠️ The two are also not interchangeable at the CHECKPOINT level. Perception
    poses live in the PCA frame of the t=0 point set, not a CAD frame, so a
    student distilled on one path cannot be evaluated on the other.

THE TWO ERROR COMPONENTS, and the split is the whole point (ported verbatim in
spirit from `peg_student_env._occlude`):

    kp_noise_m   PER-POINT iid. Kabsch averages it over the surviving points, so
                 it shrinks by about sqrt(n). 5 mm of this lands as ~2.9 mm of
                 solved-pose error -- measured on peg.
    kp_bias_m    PER-OBJECT, PER-EPISODE, applied to every point of that object
                 identically. Kabsch CANNOT average it out: it passes straight
                 through into the solved translation.

Real 3D tracking error has both. `kp_bias_m` defaults to 0 to match the existing
peg runs, and that default is the project's largest known deployability gap.

OBSERVATION LAYOUTS. Both mirror `msppo.student.StudentTransformer`'s slice map
exactly, and both are asserted against the declared space on every pack, because
a mismatch is otherwise silent (it once cost a full training run):

    pose    proprio(2*qdim [+7 tcp]) | prev_action | obj_pose 7 | goal_pose 7
            [| scene_pose 7] | rel 3*(3 or 4)
    point   proprio(2*qdim [+7 tcp]) | prev_action | obj K*3 | goal K*3
            [| scene K*3]
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from msppo.kabsch import kabsch
from msppo.peg_student_env import _R_to_quat


class TaskStudentEnv:
    """Wraps the task's keypoint environment. Composition again: the thing
    underneath is a VectorEnv and fails gym.Wrapper's isinstance check."""

    # Which env attribute is the object and which is the fixture. This is the
    # only task-specific thing the perception layer needs, and it is an ENV
    # ATTRIBUTE name, not the SAPIEN actor name: PushT's actor is called "Tee"
    # but the attribute is `tee` (push_t.py:233), and the goal marker's attribute
    # is `goal_tee`. LiftPegUpright has no fixture at all -- None.
    ACTORS = {"liftpeg": ("peg", None), "pusht": ("tee", "goal_tee"),
              "pushcube": ("cube", None), "placesphere": ("sphere", None),
              "stack": ("cubeA", "cubeB"),
              # goal_site is hidden (pick_cube.py:104) -- no fixture, no scene
              # channel; the goal reaches the student only through the planner
              "pickcube": ("cube", None),
              # the rectangular hole rim, whose four corners are non-collinear
              # (sigma2/sigma0 = 0.897) unlike the peg's own traced points
              "peginsert": ("peg", "box")}

    def __init__(self, kpenv, qdim, task=None, seed=0, device="cuda",
                 kp_mask_ratio=2, kp_noise_m=0.005, kp_mask_height=True,
                 kp_bias_m=0.0, kp_bias_rand=True, prev_action=True,
                 scene_kp=True, tcp_obs=True, point_obs=False,
                 perception=False, track_px=0.0, depth_noise_m=0.0,
                 query="free", goal_delta=None, anchor=0.0, psi=False, psi_bank=None,
                 goal_error_rel=None, goal_err_scale=None, goal_rel_se3=False,
                 goal_sig=False, goal_sig_bank=None, goal_form="mean", goal_k=4):
        self.kp = kpenv
        # INTERACTION-SITE ANCHOR. 0.0 = the object's centroid, which is the only
        # behaviour this project has ever used. Non-zero re-anchors BOTH obj_pose
        # and goal_pose to a point along the object's principal axis, as a
        # fraction of that axis's half-extent: -1 and +1 are the two ends.
        #
        # This is the student's FRAME OF REFERENCE, not a command. The teacher is
        # untouched and DAgger still imitates its actions; what changes is whether
        # the policy reasons about "the object at its centre" or "the object at
        # the point where the interaction happens".
        #
        # Applied to the canonical set BEFORE Kabsch, so obj and goal shift
        # together and their difference stays meaningful.
        self.anchor = float(anchor)
        self.task, self.perception = task, perception
        # K planner samples (a list of (dR, dt)) or one (a tuple); kept as a
        # list so the composition below is written once for both.
        self.goal_delta = ([goal_delta] if isinstance(goal_delta, tuple)
                           else goal_delta)
        self._delta_g7 = None
        self._delta_gs = None                     # [B,K,7] composed samples
        self.env, self.base, self.device = kpenv.env, kpenv.base, device
        self.num_envs = kpenv.num_envs
        self.num_kp = kpenv.num_kp
        self.qdim = qdim
        # Dict space in perception mode -- fall back to the keypoint env's own
        # verified stock width, which is what `raw_from_sim()` produces.
        _sp = self.env.single_observation_space
        self.raw_dim = _sp.shape[-1] if _sp.shape is not None else kpenv.RAW_DIM
        self.act_dim = kpenv.single_action_space.shape[0]

        # scene_kp is TRI-STATE, and the third value exists for mixed-task
        # training:
        #   True    a real fixture is perceived (peg's hole rim, StackCube's cubeB)
        #   False   no scene block in the layout at all
        #   "zero"  the block IS in the layout and is always zeros
        #
        # "zero" is what lets a fixture task and a fixture-less task share ONE
        # student: the widths match and the absent fixture is a constant. The
        # zeroing must happen at training time (see `arm_t.rollout`) -- masking
        # it only at eval would give the student a channel in distillation that
        # does not exist at deployment.
        if scene_kp is True and not kpenv.has_scene:
            raise SystemExit(
                "this task has no fixture to perceive. Pass scene_kp=False (drop "
                "the block) or scene_kp='zero' (keep the block, fill with zeros -- "
                "needed when mixing with a task that DOES have a fixture).")

        self.prev_action, self.scene_kp = prev_action, scene_kp
        self.tcp_obs, self.point_obs = tcp_obs, point_obs
        self.kp_mask_ratio, self.kp_noise_m = kp_mask_ratio, kp_noise_m
        self.kp_mask_height, self.kp_bias_m = kp_mask_height, kp_bias_m
        self.kp_bias_rand = kp_bias_rand
        self._bias_obj = self._bias_scene = None
        self._gen = torch.Generator(device=device).manual_seed(seed)
        self._prev_act = torch.zeros((self.num_envs, self.act_dim), device=device)
        self._keep_obj = torch.ones((self.num_envs, self.num_kp),
                                    dtype=torch.bool, device=device)
        self._keep_scene = torch.ones_like(self._keep_obj)

        pro = 2 * qdim + (7 if tcp_obs else 0)
        n_sets = 3 if scene_kp else 2
        self._n_sets = n_sets
        if point_obs:
            dim = pro + self.act_dim + self.num_kp * 3 * n_sets
        else:
            dim = pro + self.act_dim + 7 * n_sets + 3 * (4 if scene_kp else 3)
            if goal_sig:
                dim += 2
            # K-SAMPLE GOAL FORM. The goal_pose slot ALWAYS
            # holds the K-mean; these slots add what the arm sees on top:
            #   sd   diag sd of the K sample positions, world axes  (+3)
            #   k    the K sample poses themselves, order-free      (+7K)
            # Same offsets as StudentTransformer (after goal_sig, before lang).
            if goal_form == "sd":
                dim += 3
            elif goal_form == "k":
                dim += 7 * int(goal_k)
            elif goal_form == "ell":
                dim += 6
            elif goal_form != "mean":
                raise SystemExit(f"goal_form {goal_form!r} not in mean/sd/k/ell")
            # PSI BLOCK: 4 dims = k1 approach unit vector (3) + h carry height (1),
            # appended LAST. Training: the frame-patched kp env's own per-episode command (`_frame_k1`, `_frame_h`),
            # the same values the teacher acted on. Deployment: `psi_bank` [N,4], extracted from the planner trace
            # (msgen.trace_seg) -- no simulator state.
            if psi:
                dim += 4
        if psi and point_obs:
            raise SystemExit("psi is defined for the POSE layout only.")
        self.psi, self.psi_bank = bool(psi), psi_bank
        self.goal_form, self.goal_k = goal_form, int(goal_k)
        if self.goal_form != "mean" and point_obs:
            raise SystemExit("goal_form sd/k is defined for the POSE layout only.")
        self._goal_samples = None                 # [B,K,7] behind the last obs
        self.single_observation_space = gym.spaces.Box(
            -np.inf, np.inf, (dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (self.num_envs, dim), dtype=np.float32)
        self.single_action_space = kpenv.single_action_space
        self.action_space = kpenv.action_space
        self.n_pts = self.num_kp

        # TRAINING-TIME GOAL ERROR, scene-relative. The planner's measured error
        # decomposed in each scene's own object->goal frame (fraction of the
        # required travel + two lateral offsets + a relative rotation), so a
        # scene needing 100 mm and one needing 350 mm get proportionate errors
        # rather than the same world-frame offset. Built by
        # `msppo.peg_relbank export`. Drawn ONCE PER EPISODE, never per step:
        # resampling every control step averages the error away and trains
        # against a problem that is easier than the real one.
        self.goal_error_rel = goal_error_rel
        self.goal_err_scale = goal_err_scale
        # THE GOAL REPRESENTATION SWAP. The 7 goal-carrying slots keep
        # their offsets, their token (`Linear(14->d)` over obj||goal) and their
        # entry in `goal_slices()`; only their CONTENTS change, from the goal's
        # absolute world pose to the object-frame relative transform
        #     dt = R_obj^T (t_goal - t_obj)      (3)
        #     dq = q_obj^-1 (x) q_goal           (4, canonicalised to w >= 0)
        # Holding the layout and the parameter count fixed is the point: the arm
        # then isolates the REPRESENTATION and nothing else. Both quantities are
        # BILINEAR in (obj_pose, goal_pose), so the affine tokenizer cannot form
        # them from the absolute encoding -- this is information the current
        # student provably cannot compute, not a re-parameterisation of what it
        # already sees.
        self.goal_rel_se3 = bool(goal_rel_se3)
        # RELIABILITY CHANNEL. Two dims appended after `rel`:
        #     [0] rmse  metres, the residual of the solve that produced this goal
        #     [1] s21   sigma2/sigma1 of that solve's cross-covariance
        # Both are computed from the PLANNER's own output and the segmentation,
        # never from simulator truth, so they exist on a real robot too -- that
        # is what separates this from an oracle channel.
        #
        # Where the values come from, in priority order:
        #   goal_sig_bank   [N,2] per scene, for the row-4 eval, where the goal
        #                   comes from a rendered bank rather than an injection
        #   the relbank     during injection training, indexed by the same
        #                   per-episode draw as the error itself
        #   (0.0, 1.0)      the true goal: a perfect fit, perfectly conditioned
        self.goal_sig = bool(goal_sig)
        self.goal_sig_bank = goal_sig_bank
        if self.goal_sig and point_obs:
            raise SystemExit("goal_sig is defined for the POSE layout only.")
        if self.goal_rel_se3 and point_obs:
            raise SystemExit(
                "goal_rel_se3 is defined for the POSE layout only; the point "
                "layout has no goal pose to re-express.")
        self._rel_idx = self._rel_a = None
        # GOAL-ERROR CURRICULUM. The teacher learned this the hard
        # way (obs_noise2.PoseNoiseV2._a_max): anneal the UPPER BOUND of the
        # severity draw, never the value. Annealing the value ends training with
        # every episode at full error and removes the near-clean episodes the
        # optimiser needs to climb. `warmup` is a leading stretch at a_max=0 so
        # the student first learns the task at all, then meets the error.
        self.goal_err_warmup = 0.0
        self.goal_err_curriculum = 0.0
        self._noise_prog = 1.0
        # GOAL COVARIANCE, reduced to (translation, rotation) magnitudes. Written
        # to BOTH the student's own channel and `_goal_sig_row`, which is what
        # `teacher_perc` hands the teacher -- the two sides must never disagree
        # about how blurred the goal is.
        self._cloud_sig = None
        # PIPELINE-IDENTITY TRAINING. `goal_error_rel` applies an
        # error to the TRUE goal pose and re-solves Kabsch on clean local points;
        # deployment (`goal_delta`) composes the planner transform with the
        # PERCEIVED points, occlusion mask and few-point fallback included. Two
        # different code paths. This pool feeds the DEPLOYMENT path during
        # training, with a random bank row per episode (scene identity does not
        # need to match -- the transform is relative, and the relbank route has
        # always sampled a random scene index too).
        self.goal_delta_pool = None      # [(dR[N,3,3], dt[N,3]), ...]
        self._gd_idx = None
        self._obj_attr = self.ACTORS.get(task, (None, None))[0]
        if goal_error_rel is not None and not perception:
            raise SystemExit(
                "goal_error_rel is wired into the PERCEPTION path only. The "
                "state path builds its goal points directly and would silently "
                "ignore the injection, which is a wrong number rather than an "
                "error.")

        self.perc = None
        if perception:
            if task not in self.ACTORS:
                raise SystemExit(f"no actor map for task {task!r}")
            obj_name, scene_name = self.ACTORS[task]
            if scene_kp is True and scene_name is None:
                raise SystemExit(
                    f"{task} has no fixture actor, so perception cannot supply a "
                    f"scene channel. Pass scene_kp=False or scene_kp='zero'.")
            from msppo.peg_perception import PegPerception
            self.perc = PegPerception(
                kpenv.env, num_kp=self.num_kp, seed=seed, device=device,
                track_px=track_px, depth_noise_m=depth_noise_m, query=query,
                obj_name=obj_name,
                # In "zero" mode nothing reads the scene pose, so pointing the
                # perception layer at a fixture that does not exist would only
                # invent work and a failure mode.
                scene_name=(scene_name if scene_kp is True else None))
            if query == "grid":
                from msgen.tasks import NUM_KPS
                self.n_pts = NUM_KPS
                if point_obs:
                    # grid mode hands back all 400 TraceGen query points, not
                    # num_kp -- sizing the encoder with num_kp mis-slices the
                    # observation, silently. Re-declare the space.
                    pro = 2 * qdim + (7 if tcp_obs else 0)
                    dim = (2 * qdim + (7 if tcp_obs else 0) + self.act_dim
                           + self.n_pts * 3 * self._n_sets)
                    self.single_observation_space = gym.spaces.Box(
                        -np.inf, np.inf, (dim,), dtype=np.float32)
                    self.observation_space = gym.spaces.Box(
                        -np.inf, np.inf, (self.num_envs, dim), dtype=np.float32)
        elif goal_delta is not None:
            # Same refusal `peg_student_env` makes, for the same reason: composing
            # the delta needs a PERCEIVED object pose. On the state path the obj
            # channel is solved against a CAD template from the simulator's pose,
            # so composing there would put simulator state back into the very
            # channel the delta representation exists to keep it out of.
            raise SystemExit("goal_delta requires perception=True; the state "
                             "path has no perceived object pose to compose with")

    # ----------------------------------------------------------- perception --
    def reset_masks(self, idx=None):
        """One occlusion side per EPISODE. Resampling every control step would
        average the occlusion away and make the task easier than the real one --
        the same argument `peg_student_env` makes for the goal error.

        The side is drawn in the object's CANONICAL frame, as
        `peg_student_env.reset_masks` does: an occluding side belongs to the
        object, so it must not spin as the object does.

        `idx` refreshes only the envs that just reset. peg's version refreshes
        every env whenever ANY env finishes, which quietly gives mid-trajectory
        envs a new occlusion side; this keeps that option (idx=None) but does not
        force it.
        """
        from msppo.masking import mask_oneside
        canon = self.kp._canon[None].expand(self.num_envs, -1, -1)
        new_o = mask_oneside(canon, self.kp_mask_ratio, self._gen)
        new_s = mask_oneside(canon, self.kp_mask_ratio, self._gen)
        if idx is None:
            self._keep_obj, self._keep_scene = new_o, new_s
            self._prev_act.zero_()
        else:
            self._keep_obj[idx], self._keep_scene[idx] = new_o[idx], new_s[idx]
            self._prev_act[idx] = 0.0
        self._resample_bias(idx)
        self._resample_goal_error(idx)
        self.resample_goal_delta(idx)

    def _resample_goal_error(self, idx=None):
        """One planner-error draw per episode, per env.

        The severity `a` is also needed by PIPELINE-IDENTITY training, which has
        a delta pool instead of a relbank -- there `a` interpolates the composed
        planner goal toward the true one, so the same warmup/ramp applies.
        """
        if self.goal_error_rel is None and self.goal_delta_pool is None:
            return
        if self.goal_error_rel is None:
            new_a = torch.rand((self.num_envs,), generator=self._gen,
                               device=self.device) * self._a_max()
            if self._rel_a is None or idx is None:
                self._rel_a = new_a
            else:
                self._rel_a[idx] = new_a[idx]
            return
        n = self.goal_error_rel["frac"].shape[0]
        new_i = torch.randint(0, n, (self.num_envs,), generator=self._gen,
                              device=self.device)
        if self.goal_err_scale == "uniform":
            # curriculum, NOT distribution-matched: a~U(0,1) halves the mean
            # error the student ever sees (E[a]=0.5) relative to deployment.
            new_a = torch.rand((self.num_envs,), generator=self._gen,
                               device=self.device)
        elif isinstance(self.goal_err_scale, str) and \
                self.goal_err_scale.startswith("bern"):
            # DISTRIBUTION-MATCHED MIXTURE. a in {0, 1} with
            # P(a=1)=p: every noisy episode carries the FULL measured planner
            # error, so the conditional error distribution is EXACTLY the
            # deployed one, while the 1-p clean episodes preserve precise
            # execution (the property that removed the uniform ramp's true-goal
            # tax). "bern0.7" -> p=0.7; bare "bern" -> p=0.5.
            tail = self.goal_err_scale[4:].lstrip(":")
            p_noisy = float(tail) if tail else 0.5
            new_a = (torch.rand((self.num_envs,), generator=self._gen,
                                device=self.device) < p_noisy).float()
        elif self.goal_err_scale is None:
            new_a = torch.ones((self.num_envs,), device=self.device)
        else:
            new_a = torch.full((self.num_envs,), float(self.goal_err_scale),
                               device=self.device)
        new_a = new_a * self._a_max()
        if self._rel_idx is None or idx is None:
            self._rel_idx, self._rel_a = new_i, new_a
        else:
            self._rel_idx[idx], self._rel_a[idx] = new_i[idx], new_a[idx]

    def set_noise_progress(self, f):
        """Fraction of training elapsed, 0..1. Called once per distillation
        iteration; drives the goal-error curriculum below."""
        self._noise_prog = float(f)

    def _a_max(self):
        """Upper bound of the per-episode severity at this point in training."""
        w, c = self.goal_err_warmup, self.goal_err_curriculum
        f = self._noise_prog
        if f < w:
            return 0.0
        if c <= 0:
            return 1.0
        return min(1.0, (f - w) / c)

    def _set_cloud_sig(self, gs):
        """[B,2] goal reliability from the K-sample cloud's own spread.

        Deployable by construction: a planner emits K samples and this needs
        nothing but those samples -- no simulator state. It is the sampling-cloud
        covariance, reduced to two magnitudes. The cloud's DIRECTION was
        measured separately and gave no gain, so only the magnitudes are
        carried here.
        """
        import math as _m
        z = torch.zeros((self.num_envs, 2), device=self.device)
        if gs is None or gs.shape[1] < 2:
            self._cloud_sig = self._goal_sig_row = z
            return
        t = gs[..., :3].std(dim=1, unbiased=False).norm(dim=-1, keepdim=True)
        q = gs[..., 3:]
        qm = q.mean(dim=1, keepdim=True)
        qm = qm / qm.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        # |dot| : q and -q are the same rotation, so the sign must not count
        dot = (q * qm).sum(-1, keepdim=True).abs().clamp(0.0, 1.0)
        r = (2.0 * torch.arccos(dot)).mean(dim=1)
        sig = torch.cat([(t / 0.15).clamp(0, 1),
                         (r / _m.radians(60.0)).clamp(0, 1)], dim=-1)
        self._cloud_sig = self._goal_sig_row = sig

    def resample_goal_delta(self, idx=None):
        """New random bank row per episode, and mark the composed goal stale so
        `student_observation_perc` rebuilds it for those envs."""
        if self.goal_delta_pool is None:
            return
        N = self.goal_delta_pool[0][0].shape[0]
        new_i = torch.randint(0, N, (self.num_envs,), generator=self._gen,
                              device=self.device)
        if self._gd_idx is None or idx is None:
            self._gd_idx = new_i
        else:
            self._gd_idx[idx] = new_i[idx]
        self.goal_delta = [tuple(x[self._gd_idx] for x in row)
                           for row in self.goal_delta_pool]
        # the composed goal is built from the OLD rows; force a rebuild
        if idx is None:
            self._delta_g7 = self._delta_gs = None
            self._delta_stale = None
        else:
            m = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            m[idx] = True
            self._delta_stale = m if getattr(self, "_delta_stale", None) is None \
                else (self._delta_stale | m)

    @staticmethod
    def _interp_pose(g0, g1, a):
        """Poses [...,7] interpolated g0 -> g1 by a in [0,1]. a=0 gives EXACTLY
        g0, so the warmup really does hand the student the true goal."""
        from msppo.peg_relbank import slerp_from_identity
        a0 = a.reshape(a.shape[0])                       # [B], before padding
        a = a.reshape(*a.shape, *([1] * (g0.dim() - a.dim())))
        p = g0[..., :3] + a * (g1[..., :3] - g0[..., :3])
        q0, q1 = g0[..., 3:], g1[..., 3:]
        q0c = torch.cat([q0[..., :1], -q0[..., 1:]], dim=-1)
        dq = TaskStudentEnv._qmul(q0c.reshape(-1, 4), q1.reshape(-1, 4))
        dq = torch.where(dq[:, :1] < 0, -dq, dq)          # q and -q are the same
        # a is per-ENV but dq is flattened over (env, K): broadcast a across the
        # K samples first, then flatten. `expand` on the raw [B] shape silently
        # demands B == B*K and only matches when K == 1.
        aq = a0.reshape(-1, *([1] * (g0.dim() - 2))).expand(g0.shape[:-1]).reshape(-1)
        step = slerp_from_identity(dq, aq)
        q = TaskStudentEnv._qmul(q0.reshape(-1, 4), step).reshape(q0.shape)
        return torch.cat([p, q], dim=-1)

    def _apply_rel_error(self, gp, gq):
        """Replace the true goal with what the planner would have predicted.

        Returns [B,K,7], K = the relbank's column count (1 for a legacy bank).
        One scene index and ONE severity `a` per episode; the K columns of that
        scene are expanded together, so the samples stay a cloud around one
        draw rather than K unrelated draws."""
        from msppo.peg_relbank import expand, scale_error
        # PushCube names its cube `obj` while ACTORS says "cube"; fall back to the scene actor
        _a = getattr(self.kp.base, self._obj_attr, None)
        if _a is None:
            _a = self.kp.base.scene.actors[self._obj_attr]
        obj_p = _a.pose.p
        frac, lv, lw, dq = scale_error(self.goal_error_rel, self._rel_idx,
                                       self._rel_a)
        if frac.dim() == 1:
            frac, lv, lw, dq = frac[:, None], lv[:, None], lw[:, None], dq[:, None]
        B, K = frac.shape
        rep = lambda x: x[:, None].expand(B, K, x.shape[-1]).reshape(B * K, -1)  # noqa: E731
        p = expand(rep(obj_p), rep(gp), frac.reshape(-1), lv.reshape(-1), lw.reshape(-1))
        q = self._qmul(dq.reshape(B * K, 4), rep(gq))
        return torch.cat([p, q], dim=-1).reshape(B, K, 7)

    @staticmethod
    def _ell(gs, eps=1e-10):
        """[B,K,7] -> [B,6]: upper triangle of u u^T, u = principal axis (unit,
        world frame) of the K sample POSITIONS about their mean. Coincident
        samples (covariance ~ 0, i.e. a=0 or one goal repeated K times) give
        the zero vector, so the identity case carries no direction at all."""
        c = gs[..., :3] - gs[..., :3].mean(dim=1, keepdim=True)         # [B,K,3]
        cov = torch.einsum("bki,bkj->bij", c, c) / gs.shape[1]          # [B,3,3]
        w, v = torch.linalg.eigh(cov.double())                          # ascending
        u = v[..., -1].to(gs.dtype)                                     # [B,3]
        live = (w[:, -1] > eps).to(gs.dtype)[:, None]
        u = u * live
        return torch.stack([u[:, 0] * u[:, 0], u[:, 0] * u[:, 1], u[:, 0] * u[:, 2],
                            u[:, 1] * u[:, 1], u[:, 1] * u[:, 2], u[:, 2] * u[:, 2]], dim=-1)

    @staticmethod
    def _summarise(gs):
        """[B,K,7] -> (mean pose [B,7], position sd [B,3]). Quaternions are
        sign-aligned to sample 0 before averaging (q and -q are one rotation);
        K is small and the cloud tight (~30 mm), so the normalised
        mean is a valid rotation. Population sd (unbiased=False) so K=1 -> 0."""
        p = gs[..., :3].mean(dim=1)
        q = gs[..., 3:7]
        q = torch.where((q * q[:, :1]).sum(-1, keepdim=True) < 0, -q, q)
        q = q.mean(dim=1)
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return torch.cat([p, q], dim=-1), gs[..., :3].std(dim=1, unbiased=False)

    @staticmethod
    def _qmul(a, b):
        """Hamilton product, SAPIEN (w,x,y,z)."""
        aw, ax, ay, az = a.unbind(-1)
        bw, bx, by, bz = b.unbind(-1)
        return torch.stack([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw], dim=-1)

    def _resample_bias(self, idx=None):
        if not self.kp_bias_m:
            self._bias_obj = self._bias_scene = None
            return
        sh = (self.num_envs, 3)
        sig = self.kp_bias_m
        if self.kp_bias_rand:
            sig = torch.rand((self.num_envs, 1), generator=self._gen,
                             device=self.device) * self.kp_bias_m
        b_o = torch.randn(sh, generator=self._gen, device=self.device) * sig
        b_s = torch.randn(sh, generator=self._gen, device=self.device) * sig
        if self._bias_obj is None or idx is None:
            self._bias_obj, self._bias_scene = b_o, b_s
        else:
            self._bias_obj[idx], self._bias_scene[idx] = b_o[idx], b_s[idx]

    def _occlude(self, pts, keep, bias=None):
        """Height mask + noise, then ZERO the dropped points -- upstream zeroes
        rather than removes and the encoders expect a fixed-length set."""
        from msppo.masking import mask_height
        if self.kp_mask_height:
            keep = mask_height(pts, keep, self._gen)
        out = pts
        if self.kp_noise_m > 0:
            out = out + torch.randn(out.shape, generator=self._gen,
                                    device=out.device) * self.kp_noise_m
        if bias is not None:
            out = out + bias[:, None, :]
        return out * keep[..., None], keep

    def _solve_pose(self, kp, canon, keep=None):
        """Weighted Kabsch. The weights are not cosmetic: masked points are
        ZEROED rather than removed, and an unweighted fit treats those zeros as
        real observations at the origin -- 94.4 mm error unweighted against
        2.7 mm weighted, measured on the cube."""
        w = None if keep is None else keep.to(kp.dtype)
        R, t, _ = kabsch(canon, kp, weights=w)
        return torch.cat([t, _R_to_quat(R)], dim=-1)

    # ------------------------------------------------------------ the views --
    def _proprio(self, raw):
        return raw[:, :2 * self.qdim + (7 if self.tcp_obs else 0)]

    def _tcp(self, raw):
        # tcp_pose sits immediately after qvel in both tasks' stock layouts.
        return raw[:, 2 * self.qdim: 2 * self.qdim + 3]

    def student_observation(self, raw):
        obj, goal, scene = self.kp.keypoints(raw)
        canon = self.kp._canon_b(raw)
        obj_o, ko = self._occlude(obj, self._keep_obj, self._bias_obj)
        proprio = self._proprio(raw)
        if self.point_obs:
            return self._pack_points(proprio, obj_o, goal, scene, ko)
        op = self._solve_pose(obj_o, canon, ko)
        # goal_kp is a PREDICTION, not a perception: it carries its own error and
        # is never occluded, so it is solved unweighted.
        gp = self._solve_pose(goal, canon)
        sp = None
        if self.scene_kp is True:
            scene_o, ks = self._occlude(scene, self._keep_scene, self._bias_scene)
            sp = self._solve_pose(scene_o, canon, ks)
        return self._pack_pose(proprio, op, gp, sp, self._tcp(raw))

    @staticmethod
    def _qconj(q):
        """Conjugate, SAPIEN (w,x,y,z). For a unit quaternion this is the
        inverse rotation."""
        return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device,
                                dtype=q.dtype)

    def _goal_block(self, op, gp):
        """The 7 dims that carry the goal: absolute pose, or -- under
        `goal_rel_se3` -- the object-frame relative transform in the SAME slots.

        `dq` is canonicalised to w >= 0 because q and -q are the same rotation:
        without it the student sees a sign flip as a discontinuity, the same
        trap `peg_relbank.slerp_from_identity` documents on its own input."""
        if not self.goal_rel_se3:
            return gp
        from msppo.peg_kp_env import quat_to_R
        R = quat_to_R(op[:, 3:7])                                  # [B,3,3]
        dt = torch.einsum("bji,bj->bi", R, gp[:, :3] - op[:, :3])  # R^T v
        dq = self._qmul(self._qconj(op[:, 3:7]), gp[:, 3:7])
        dq = torch.where(dq[:, :1] < 0, -dq, dq)
        return torch.cat([dt, dq], dim=-1)

    def _pack_pose(self, proprio, op, gp, sp, tcp, gs=None):
        """gp: what goes in the goal_pose slot (the K-mean when gs is given).
        gs: [B,K,7] samples behind it, for the sd/k slots; None means K=1 == gp."""
        if self.scene_kp == "zero":
            sp = torch.zeros_like(op)
        parts = [proprio]
        if self.prev_action:
            parts.append(self._prev_act)
        # `rel` below stays in ABSOLUTE world coordinates even under
        # goal_rel_se3, so the two encodings differ in exactly one block.
        parts += [op, self._goal_block(op, gp)]
        if self.scene_kp:
            parts.append(sp)
        parts += [op[:, :3] - tcp, gp[:, :3] - tcp, gp[:, :3] - op[:, :3]]
        if self.scene_kp:
            parts.append(sp[:, :3] - op[:, :3])
        if self.goal_sig:
            parts.append(self._sig_row())
        if gs is None:
            gs = gp[:, None]
        if self.goal_form != "mean" and gs.shape[1] != self.goal_k:
            # a single goal on a K-form student: the same pose K times (sd = 0)
            if gs.shape[1] != 1:
                raise SystemExit(f"goal_form {self.goal_form} expects K={self.goal_k} "
                                 f"samples, got {gs.shape[1]}")
            gs = gs.expand(-1, self.goal_k, -1)
        self._goal_samples = gs
        if self.goal_form == "sd":
            parts.append(gs[..., :3].std(dim=1, unbiased=False))
        elif self.goal_form == "k":
            parts.append(gs.reshape(gs.shape[0], -1))
        elif self.goal_form == "ell":
            parts.append(self._ell(gs))
        if self.psi:
            parts.append(self._psi_row(parts[0].shape[0]))
        out = torch.cat(parts, dim=-1)
        assert out.shape[1] == self.single_observation_space.shape[0], (
            f"pose layout {out.shape[1]} != declared "
            f"{self.single_observation_space.shape[0]}")
        return out

    def _psi_row(self, b):
        """[B,4] the path command psi = (k1, h). Bank row per env at deployment, else the frame-patched kp env's command."""
        if self.psi_bank is not None:
            return self.psi_bank[:b].to(self.device)
        k1, h = getattr(self.kp, "_frame_k1", None), getattr(self.kp, "_frame_h", None)
        if k1 is None or h is None:
            raise SystemExit("psi=True but the kp env carries no frame command: apply msppo.frame_patches.apply(tasks) "
                             "before building the env, or pass psi_bank")
        return torch.cat([k1[:b], h[:b, None]], dim=-1)

    def _sig_row(self):
        """[B,2] reliability for the goal each env is currently being given."""
        if self._cloud_sig is not None:
            return self._cloud_sig
        if self.goal_sig_bank is not None:
            return self.goal_sig_bank
        if self.goal_error_rel is not None and self.goal_error_rel.get("sig") is not None:
            if self._rel_idx is None:
                self._resample_goal_error()
            return self.goal_error_rel["sig"][self._rel_idx]
        # the true goal: zero residual, perfectly conditioned
        return torch.tensor([0.0, 1.0], device=self.device).expand(self.num_envs, 2)

    def _pack_points(self, proprio, obj, goal, scene, ko=None,
                     already_occluded=False):
        if self.scene_kp == "zero":
            scene = torch.zeros_like(obj)
            already_occluded = True
        """Order must mirror `StudentTransformer(pose_obs=False).sl` exactly:
        obj, then goal, then scene. `student.tokens` stacks obj and goal into one
        6-channel block, which is the PAIRED encoding -- channel i is
        [current_xyz, target_xyz] of THE SAME query point. A silent permutation
        here would still train and would be a DECOUPLED encoding wearing a paired
        label; Dex4D measures that variant at 0.203 against 0.600."""
        b = proprio.shape[0]
        parts = [proprio]
        if self.prev_action:
            parts.append(self._prev_act)
        parts += [obj.reshape(b, -1), goal.reshape(b, -1)]
        if self.scene_kp:
            # In the perception path the points arrive ALREADY occluded by real
            # geometry (a depth test), so applying the synthetic one-side mask on
            # top would occlude twice and make the perception arm strictly harder
            # than the state arm for a reason that is not perception.
            scene_o = scene if already_occluded else self._occlude(
                scene, self._keep_scene, self._bias_scene)[0]
            parts.append(scene_o.reshape(b, -1))
        out = torch.cat(parts, dim=-1)
        assert out.shape[1] == self.single_observation_space.shape[0], (
            f"point layout {out.shape[1]} != declared "
            f"{self.single_observation_space.shape[0]}")
        return out

    # ------------------------------------------------------ perception path --
    def _apply_anchor(self, idx=None):
        """Shift the canonical set so its origin sits at the interaction site.

        `idx` IS NOT OPTIONAL IN PRACTICE. This shift is applied to the canonical
        set IN PLACE, and `perc.reset(idx=...)` restores every env NOT in `idx`
        from a snapshot that is ALREADY shifted. Without the mask, those envs are
        shifted a second time on every partial reset -- and episodes terminate at
        different times, so partial resets are the steady state. Without the
        mask, an env that never resets drifts along axis 0 linearly and without
        bound. `anchor=0.0` returns early, so an unanchored run does not
        exercise this path.

        The perception canonical frame is the PCA frame of the t=0 points
        (`peg_perception`: centroid at the origin, axes from PCA with the signs
        pinned), so axis 0 IS the principal axis and the shift is one
        subtraction. The half-extent comes from the point set itself, not from
        CAD, because the whole perception path is CAD-free.
        """
        if not self.anchor or self.perc is None:
            return
        c = self.perc.canon_obj
        keep = self.perc.keep0_obj.to(c.dtype)[..., None]
        a0 = c[..., 0:1] * keep
        half = (a0.amax(1, keepdim=True) - a0.amin(1, keepdim=True)) * 0.5
        off = torch.zeros_like(c[:, :1, :])
        off[..., 0] = self.anchor * half.squeeze(-1)
        if idx is not None:
            m = idx if idx.dtype == torch.bool else torch.zeros(
                c.shape[0], dtype=torch.bool, device=c.device).scatter_(0, idx, True)
            off = off * m[:, None, None].to(off.dtype)
        self.perc.canon_obj = c - off

    def perc_reset(self, obs_dict, idx=None):
        """Pick query points and fix the canonical frames. `idx` re-picks ONLY the
        envs that just reset: the others are mid-trajectory and their canonical
        frame defines the pose convention, so moving it under a running policy
        would make the observation discontinuous."""
        self.perc.reset(obs_dict, idx=idx)
        # Re-anchoring must follow EVERY reset, including the partial ones: the
        # canonical frame is rebuilt from scratch for the envs in `idx`, so their
        # anchor offset is gone unless it is reapplied here -- but ONLY for those
        # envs. See _apply_anchor for what the unmasked version cost.
        self._apply_anchor(idx)
        # The delta cache is PER-EPISODE state. Nuking it globally here was the
        # receding-carrot bug reborn: the task terminates on success, so from the
        # first termination onward EVERY env's goal was recomposed against its
        # CURRENT pose each step -- the goal walks away from the policy chasing
        # it. PickCube identity read 0.039 against a 0.980 student; peg is
        # unaffected (its eval resets perception once, before the loop).
        if idx is None:
            self._delta_g7 = None
            self._delta_gs = None
            self._delta_stale = None
        elif self._delta_g7 is not None:
            m = idx if idx.dtype == torch.bool else torch.zeros(
                self.num_envs, dtype=torch.bool,
                device=self.device).scatter_(0, idx, True)
            if getattr(self, "_delta_stale", None) is None:
                self._delta_stale = m.clone()
            else:
                self._delta_stale |= m

    def perc_sig_row(self):
        """[B,4] DEPLOYABLE reliability estimate for the teacher's sig channel.

        Object side: the fraction of query points the tracker LOST this step.
        Points drop out through occlusion and depth disagreement, and a solve on
        fewer points is a worse solve -- this is the object-side proxy measured
        earlier (survivor count / Kabsch conditioning), and it needs no privileged
        state. Goal side: 0 when the goal is the true one (row 3); under
        `--goal-delta` the planner's own reliability belongs here (row 4).
        """
        ko = getattr(self, "_perc_keep", None)
        z = torch.zeros((self.num_envs, 1), device=self.device)
        if ko is None:
            o = z
        else:
            o = (1.0 - ko.float().mean(dim=-1, keepdim=True)).clamp(0, 1)
        g = getattr(self, "_goal_sig_row", None)
        if g is None:
            gt = gr = z
        elif g.shape[-1] >= 2:          # (translation, rotation) spread
            gt, gr = g[:, 0:1], g[:, 1:2]
        else:
            gt = gr = g
        return torch.cat([o, o, gt, gr], dim=-1)

    def student_observation_perc(self, obs_dict):
        """The student view with obj/scene/goal ALL derived from a rendered RGB-D
        frame. Poses live in the PCA frame of the t=0 point set, NOT a CAD frame,
        so a policy distilled on `student_observation` cannot be evaluated here --
        the two use different conventions for the same geometry.
        """
        from msppo.kabsch import kabsch
        from msppo.peg_kp_env import quat_to_R, to_world
        from msppo.peg_student_env import _R_to_quat

        a = obs_dict["agent"]
        proprio = torch.cat(
            [a["qpos"][:, :self.qdim], a["qvel"][:, :self.qdim]]
            + ([obs_dict["extra"]["tcp_pose"]] if self.tcp_obs else []), dim=-1)
        tcp = obs_dict["extra"]["tcp_pose"][:, :3]

        if self.point_obs or self.goal_delta is not None:
            # the raw points are ALSO needed by the goal-delta composition below:
            # composing with pose-reconstructed points re-injects the perceived
            # ORIENTATION error, which on a ~4-point cube is garbage. Caught by
            # the identity gate on PickCube: student 0.980, identity 0.039.
            op, sp, ko, _, obj_pts, scene_pts = self.perc.observe(obs_dict, points=True)
        else:
            op, sp, ko, _ = self.perc.observe(obs_dict)
            obj_pts = scene_pts = None
        # NOT `_keep_obj` -- that name is the synthetic occlusion mask used by
        # `reset_masks` and is a different dtype; clobbering it broke the reset.
        self._perc_keep = ko

        if self.goal_delta is not None:
            # goal points = the delta applied to the object's points AS PERCEIVED
            # NOW. The whole chain is camera -> points -> Kabsch -> delta ->
            # points: no simulator state anywhere. Composed ONCE per episode, as
            # a planner emits one goal; a scene with no prediction (NaN) falls
            # back to the identity transform, i.e. the planner abstaining, which
            # leaks nothing.
            stale = getattr(self, "_delta_stale", None)
            if self._delta_g7 is None or (stale is not None and bool(stale.any())):
                # RAW perceived world points, not a pose reconstruction. The
                # old form R(op_q) @ canon + op_p rotated the whole set by the
                # perceived orientation -- fine on a long peg (PCA axis stable),
                # catastrophic on a cube (~4 points, no stable axes): PickCube
                # identity read 0.039 against a 0.980 student. Masked points are
                # exact zeros, so they are excluded from the fit via weights.
                cur = obj_pts
                w_cur = (ko.to(cur.dtype)
                         * self.perc.keep0_obj.to(cur.dtype))
                few = w_cur.sum(dim=1) < 3
                if bool(few.any()):
                    # not enough live points to fit: fall back to the pose
                    # reconstruction for those scenes rather than a NaN goal
                    rec = torch.einsum("bij,bnj->bni", quat_to_R(op[:, 3:7]),
                                       self.perc.canon_obj) + op[:, None, :3]
                    cur = torch.where(few[:, None, None], rec, cur)
                    w_cur = torch.where(few[:, None],
                                        self.perc.keep0_obj.to(cur.dtype), w_cur)
                # K planner samples, each composed with the SAME perceived
                # points and solved through the same Kabsch; the goal_pose slot
                # takes their mean, the sd/k slots the cloud.
                g7_k = []
                for row in self.goal_delta:
                    dR, dt = row[0], row[1]
                    csrc = row[2] if len(row) > 2 else None
                    bad = ~(torch.isfinite(dR).all(dim=-1).all(dim=-1)
                            & torch.isfinite(dt).all(-1))
                    if csrc is not None:
                        bad = bad | ~torch.isfinite(csrc).all(-1)
                    eye = torch.eye(3, device=dR.device, dtype=dR.dtype)
                    dR = torch.where(bad[:, None, None], eye, dR)
                    dt = torch.where(bad[:, None], torch.zeros_like(dt), dt)
                    if csrc is not None:
                        # Re-anchor (train pool rows only; eval rows are
                        # scene-matched and pass through verbatim). The bank
                        # scene's dt misplaces a DIFFERENT scene's goal by
                        # (dR-I)@(c'-c_src); keep the sampled (rotation,
                        # centroid displacement) pair and rebuild dt about
                        # THIS scene's perceived centroid c'.
                        csrc = torch.where(bad[:, None], torch.zeros_like(csrc), csrc)
                        d = torch.einsum("bij,bj->bi", dR, csrc) + dt - csrc
                        cprime = ((cur * w_cur[..., None]).sum(1)
                                  / w_cur.sum(1, keepdim=True).clamp(min=1e-6))
                        dt = cprime + d - torch.einsum("bij,bj->bi", dR, cprime)
                    gpts = torch.einsum("bij,bnj->bni", dR, cur) + dt[:, None]
                    R, t, _ = kabsch(self.perc.canon_obj, gpts, weights=w_cur)
                    g7_k.append(torch.cat([t, _R_to_quat(R)], dim=-1))
                gs_new = torch.stack(g7_k, dim=1)                  # [B,K,7]
                g7_new, _ = self._summarise(gs_new)
                if self._delta_g7 is None:
                    self._delta_g7, self._delta_gs = g7_new, gs_new
                else:                      # refresh ONLY the envs that reset
                    self._delta_g7 = torch.where(stale[:, None], g7_new,
                                                 self._delta_g7)
                    self._delta_gs = torch.where(stale[:, None, None], gs_new,
                                                 self._delta_gs)
                if stale is not None:
                    stale.zero_()
            g7, gs = self._delta_g7, self._delta_gs
            if self.goal_delta_pool is not None and self._rel_a is not None:
                # TRAINING through the deployment path, under the same severity
                # curriculum the relbank route uses. a=0 must give EXACTLY the
                # true goal, so the warmup is a real warmup and not "a slightly
                # smaller planner error".
                raw = self.kp.raw_from_sim()
                gp0, gq0 = self.kp._goal_from_obs(raw)
                pts0 = to_world(self.perc.local_obj, gp0, gq0)
                R0, t0, _ = kabsch(self.perc.canon_obj, pts0)
                g_true = torch.cat([t0, _R_to_quat(R0)], dim=-1)
                a = self._rel_a
                gs = self._interp_pose(g_true[:, None].expand_as(gs), gs, a)
                g7, _ = self._summarise(gs)
        else:
            # The TRUE goal, carried by the same query points and solved against
            # the same canonical set -- so obj_pose and goal_pose are expressed in
            # ONE convention and their difference is meaningful. This is the
            # privileged column (3); column (4) replaces it with goal_delta above.
            raw = self.kp.raw_from_sim()
            gpos, gq = self.kp._goal_from_obs(raw)
            gs = None
            if self.goal_error_rel is not None:
                # The student is trained on the goal a PLANNER would emit here,
                # not on the true one. Applied to (gpos, gq) BEFORE the points
                # are built, so every corrupted sample travels through exactly
                # the same canonical-frame Kabsch as the true one and the
                # student cannot tell them apart by convention.
                gs_raw = self._apply_rel_error(gpos, gq)            # [B,K,7]
                B, K = gs_raw.shape[:2]
                gpts = to_world(self.perc.local_obj.repeat_interleave(K, 0),
                                gs_raw[..., :3].reshape(B * K, 3),
                                gs_raw[..., 3:].reshape(B * K, 4))
                R, t, _ = kabsch(self.perc.canon_obj.repeat_interleave(K, 0), gpts)
                gs = torch.cat([t, _R_to_quat(R)], dim=-1).reshape(B, K, 7)
                g7, _ = self._summarise(gs)
            else:
                gpts = to_world(self.perc.local_obj, gpos, gq)
                R, t, _ = kabsch(self.perc.canon_obj, gpts)
                g7 = torch.cat([t, _R_to_quat(R)], dim=-1)

        # MUST precede _pack_pose: that is what reads _sig_row(). Covers all
        # three goal sources -- planner delta (row 4), injected relbank error
        # (training) and the true goal (row 3, gs=None -> zero spread).
        self._set_cloud_sig(gs)
        if self.point_obs:
            gpts = torch.einsum("bij,bnj->bni", quat_to_R(g7[:, 3:7]),
                                self.perc.canon_obj) + g7[:, None, :3]
            return self._pack_points(proprio, obj_pts, gpts, scene_pts, ko,
                                     already_occluded=True)
        return self._pack_pose(proprio, op, g7, sp, tcp, gs=gs)

    def _sl_goal(self):
        """Slice of the goal pose inside the POSE layout, derived rather than
        hardcoded: `student.goal_slices` is the single source for the student
        side, and this is its env-side twin."""
        assert not self.point_obs, "goal slice is defined for the pose layout"
        start = 2 * self.qdim + (7 if self.tcp_obs else 0) + self.act_dim + 7
        return slice(start, start + 7)

    def set_prev_action(self, a):
        self._prev_act = a.detach()

    def close(self):
        self.kp.close()


def make_task_student_env(task, num_envs=256, num_kp=64, seed=0, **kw):
    """Builds the keypoint env exactly as the teacher saw it, then wraps it.

    `ignore_terminations` is FALSE here on purpose, opposite to teacher training:
    DAgger needs episodes to end where the task says they end, so the success
    rate logged during distillation is the real one.
    """
    from msppo.kp_teacher import TASKS, make_env
    spec = TASKS[task]
    student_kw = {k: kw.pop(k) for k in list(kw)
                  if k in ("kp_mask_ratio", "kp_noise_m", "kp_mask_height",
                           "kp_bias_m", "kp_bias_rand", "prev_action",
                           "scene_kp", "tcp_obs", "point_obs", "perception",
                           "track_px", "depth_noise_m", "query", "goal_delta",
                           "anchor", "goal_error_rel", "goal_err_scale",
                           "goal_rel_se3", "goal_sig", "goal_sig_bank",
                           "goal_form", "goal_k", "psi", "psi_bank")}
    perc = bool(student_kw.get("perception"))
    kpenv = make_env(task, num_envs=num_envs, num_kp=num_kp, seed=seed,
                     max_episode_steps=kw.pop("max_episode_steps",
                                              spec["horizon"]),
                     # caller (a distiller) passes the TEACHER's robot; the
                     # registry is only the fallback. See kp_teacher.TASKS.
                     robot=kw.pop("robot", None) or spec["robot"],
                     # FALSE by default on purpose (see the docstring). True is
                     # for scene-REPLAY training, where every env must run the
                     # full horizon and reset together so a banked per-scene
                     # planner goal keeps describing the scene it was solved for.
                     ignore_terminations=kw.pop("ignore_terminations", False),
                     perception=perc,
                     # Query pixel IDs only mean the same thing under ONE camera,
                     # so the perception path always renders through the task's
                     # fitted TraceGen camera (msgen/tasks.py). Without this the
                     # points the student tracks and the points TraceGen was asked
                     # about are different pixels, silently.
                     tracegen_camera=perc, **kw)
    return TaskStudentEnv(kpenv, qdim=spec["qdim"], task=task, seed=seed,
                          **student_kw)
