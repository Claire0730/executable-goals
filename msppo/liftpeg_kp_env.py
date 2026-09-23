"""Keypoint wrapper for LiftPegUpright-v1.

Stage 0 for the third task. Mirrors `peg_kp_env.py`'s structure so the same PPO,
the same four heads and the same student stack apply unchanged; everything below
is what the TASK forces to differ, and every item was read off the env or
measured against it rather than assumed.

WHAT THIS TASK ADDS TO THE SET. Peg and StackCube both have a full SE(3) goal
carried by a visible fixture (the hole, cubeB). This one has **no fixture at
all**: the goal is "standing up on the table", so `scene_kp` does not exist. That
makes it the clean test of whether the scene channel's measured importance
(SE(3) student 0.720 -> 0.000 without it on peg) is a property of the
ARCHITECTURE or of tasks that happen to have a fixture.

FOUR THINGS READ FROM THE ENV, NOT ASSUMED
──────────────────────────────────────────
1.  Geometry is FIXED, unlike peg. `peg_half_length = 0.12`, `peg_half_width =
    0.025` are class attributes (`lift_peg_upright.py:38-39`), not per-env draws,
    so the canonical point set is a fixed [K,3] -- the cube case, not the peg case.

2.  Success is a CONJUNCTION and the docstring in ManiSkill is wrong about it.
    `evaluate()` (`:88-99`) requires BOTH

        |‖euler_XYZ[2]‖ - pi/2| < 0.08        (orientation)
        |p_z - 0.12|            < 0.005       (position, +-5 mm)

    The class docstring says "y euler angle"; the code uses index 2. The code is
    the authority. An earlier note in this project recorded this task as having
    "no position condition" -- that was wrong, and the z term is the TIGHTER of
    the two (+-5 mm against 4.6 deg).

3.  THE ORIENTATION TEST IS GIMBAL-DEGENERATE, and it bites. An upright peg
    decomposes as euler_XYZ = (roll, -pi/2, roll) -- the classic XYZ singularity
    at pitch = -pi/2. Measured against the env's own `evaluate()` with the peg
    placed by hand at p_z = 0.12:

        roll  0 deg   ->  success 0.000     <- physically upright, still FAILS
        roll 90 deg   ->  success 1.000
        roll 15..165  ->  passes (scan, 15 deg steps)

    So `GOAL_Q` below is the roll-90 solution, NOT the obvious `euler2quat(0,
    -pi/2, 0)`. Picking the obvious one gives a goal the task can never score,
    and the failure is silent: keypoint distance goes to zero while success stays
    at 0.000.

4.  There is no `scene_kp`. `keypoints()` returns None for it and `has_scene` is
    False; the student driver must be run without a scene channel here.

GOAL DEFINITION. Success constrains z and orientation but leaves xy completely
free, so the goal is a 2-parameter family and one member has to be chosen. This
takes the object's xy AT EPISODE START and holds it for the episode:

  * committing per episode is what `EVAL_RESULTS_STRICT_20260814.md` §0 requires
    of any goal -- a planner emits ONE goal per episode, and re-deriving it every
    control step is exactly the protocol that inflated an earlier headline 4x;
  * following the object's live xy instead would make `obj -> goal` identically
    zero in xy, which hands the policy a degenerate channel.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from transforms3d.euler import euler2quat

import mani_skill.envs  # noqa: F401
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from msppo.peg_kp_env import quat_to_R, to_world, unit_box_keypoints

# Stock 32-D layout, dumped from obs_mode="state_dict" and verified:
#   agent/qpos(9) agent/qvel(9) extra/tcp_pose(7) extra/obj_pose(7)
PROPRIO = slice(0, 25)
TCP_P = slice(18, 21)
OBJ_P, OBJ_Q = slice(25, 28), slice(28, 32)

# `lift_peg_upright.py:38-39`. Fixed, not randomised -- see docstring item 1.
HALF_LENGTH = 0.12
HALF_WIDTH = 0.025
HALF = (HALF_LENGTH, HALF_WIDTH, HALF_WIDTH)

# Roll 90 about the vertical. Verified against the env's own evaluate():
# roll 0 scores 0.000 and roll 90 scores 1.000 at the same position -- item 3.
GOAL_Q = tuple(float(x) for x in euler2quat(0, -np.pi / 2, np.pi / 2))


class KeypointLiftPeg:
    """Composition, not gym.Wrapper: ManiSkillVectorEnv is a VectorEnv and fails
    gym.Wrapper's isinstance check -- same reason the other two compose."""

    has_scene = False
    RAW_DIM = 32

    def __init__(self, env, num_kp=64, reward="stock", w_kp=0.0, seed=0,
                 device="cuda", ap2ap_fields=True, last_action=False, sig_obs=False):
        assert reward == "stock", (
            "LiftPegUpright has no pose-reaching variant: its goal is not a full "
            "SE(3) target, so the peg reward's centre-distance term is undefined")
        self.env, self.base, self.device = env, env.unwrapped, device
        self.num_kp, self.reward_kind, self.w_kp = num_kp, reward, w_kp
        self.num_envs = self.base.num_envs

        self._unit = unit_box_keypoints(num_kp, seed).to(device)
        half = torch.tensor(HALF, device=device, dtype=torch.float32)
        # Fixed geometry, so the canonical set is [K,3] and shared across envs.
        self._canon = self._unit * half[None, :]
        self._goal_q = torch.tensor(GOAL_Q, device=device,
                                    dtype=torch.float32).expand(self.num_envs, 4)
        # Committed at reset, refreshed per env on done -- see docstring.
        self._goal_xy = torch.zeros((self.num_envs, 2), device=device)
        # The goal is NOT in the stock observation for this task, so the
        # slice-rewriting injector cannot reach it. Keep a clean copy and expose
        # a setter; `_goal_from_obs` then serves the perturbed goal to
        # `_extra_fields` and the goal keypoints, while the injector still builds
        # its approach frame from the clean one.
        self._goal_xy_clean = torch.zeros((self.num_envs, 2), device=device)
        self._goal_q_clean = self._goal_q.clone()

        self._last_action = torch.zeros(
            (self.num_envs, env.single_action_space.shape[0]), device=device)

        self.ap2ap_fields = ap2ap_fields
        # In perception mode the observation space is a DICT, so it has no
        # `.shape`. The teacher still consumes the flat stock vector, rebuilt by
        # `raw_from_sim()`, whose width is fixed and verified against
        # obs_mode="state" (32).
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

        # See `peg_kp_env` for why these 4 dims sit before the keypoints.
        self.sig_obs = bool(sig_obs)
        if self.sig_obs:
            base_dim += 4

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

    def _goal_from_obs_clean(self, base_obs):
        """The UNPERTURBED goal. `_goal_from_obs` reads `_goal_xy`, which the
        injector overwrites, so using it to build the approach frame would make
        the frame depend on the previous step's own noise."""
        p = torch.zeros_like(base_obs[:, OBJ_P])
        p[:, :2] = self._goal_xy_clean
        p[:, 2] = HALF_LENGTH
        return p, self._goal_q_clean

    def set_goal_offset(self, dp, dq):
        """Injector hook: shift the committed goal. xy only -- z is the peg half
        length and is a property of the object, not a prediction."""
        from msppo.peg_kp_env import quat_to_R  # noqa: F401  (import parity)
        from msppo.obs_noise import _qmul
        self._goal_xy = self._goal_xy_clean + dp[:, :2]
        self._goal_q = _qmul(dq, self._goal_q_clean)

    def _goal_from_obs(self, base_obs):
        """(p, q) of the goal peg pose. xy is the committed episode-start xy, z is
        the half LENGTH (upright), q is the verified roll-90 solution."""
        p = torch.zeros_like(base_obs[:, OBJ_P])
        p[:, :2] = self._goal_xy
        p[:, 2] = HALF_LENGTH
        return p, self._goal_q

    def keypoints(self, base_obs):
        """(obj_kp, goal_kp, None), [B,K,3] world coordinates, paired by index.

        Read from the OBSERVATION vector, never from `self.base.peg.pose`: the
        vector env auto-resets before returning, so the live property already
        holds the NEXT episode for any env that finished (`peg_kp_env.py:206`).
        """
        canon = self._canon_b(base_obs)
        obj = to_world(canon, base_obs[:, OBJ_P], base_obs[:, OBJ_Q])
        gp, gq = self._goal_from_obs(base_obs)
        goal = to_world(canon, gp, gq)
        return obj, goal, None

    def _extra_fields(self, base_obs):
        """The AP2AP fields the stock 32-D vector does not carry, in the same
        order `peg_kp_env._extra_fields` uses so the two are read the same way."""
        gp, gq = self._goal_from_obs(base_obs)
        return torch.cat([
            self.base.agent.is_grasping(self.base.peg).float()[:, None],   # 1
            base_obs[:, OBJ_P] - base_obs[:, TCP_P],                       # 3
            self.base.peg.linear_velocity, self.base.peg.angular_velocity,  # 6
            gp, gq,                                                         # 7
            gp - base_obs[:, OBJ_P],                                        # 3
        ], dim=-1)

    # TEACHER-SIDE OBSERVATION NOISE. Rewriting the pose slices here makes the
    # stock block, the AP2AP block and the keypoints all consistent, because all
    # three are derived from `obs` below. Default None = exactly the old path.
    def _noisy(self, obs):
        n = getattr(self, "obs_noise", None)
        if n is None:
            return obs
        n.tick(getattr(self, "noise_horizon", 0))
        return n.apply(obs)

    def _sig_row(self, b):
        """[b,4] noise-scale channel; zeros when no injector is attached."""
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

    def _augment(self, obs):
        obs = self._noisy(obs)
        obj, goal, _ = self.keypoints(obs)
        b = obs.shape[0]
        parts = [obs]
        if self.ap2ap_fields:
            parts.append(self._extra_fields(obs))
        if self.last_action_obs:
            parts.append(self._last_action)
        if self.sig_obs:
            parts.append(self._sig_row(b))
        parts += [obj.reshape(b, -1), goal.reshape(b, -1)]
        return torch.cat(parts, dim=-1), obj, goal

    def raw_from_sim(self):
        """[B,32] the stock state vector, rebuilt from the LIVE sim.

        Needed only in PERCEPTION mode: `obs_mode="rgb+depth+segmentation"` makes
        `_get_obs_extra` withhold `obj_pose` (it is gated on
        `obs_mode_struct.use_state`), which is the whole point -- nothing can
        accidentally read the object's true pose through the observation. But the
        TEACHER is privileged and must still see it.

        Reading live is safe HERE and only here, for the same reason
        `peg_student_env.teacher_observation` gives: DAgger takes both views
        before stepping, so the last auto-reset has already happened and the live
        sim holds exactly the state the returned observation describes.
        """
        a = self.base.agent
        return torch.cat([a.robot.get_qpos()[:, :9], a.robot.get_qvel()[:, :9],
                          a.tcp.pose.raw_pose, self.base.peg.pose.raw_pose],
                         dim=-1).float()

    def teacher_observation(self, raw=None):
        """[B,obs_dim] the privileged view, for distillation under perception."""
        return self._augment(raw if raw is not None else self.raw_from_sim())[0]

    # ------------------------------------------------------------------ gym --
    def _commit_goal(self, obs, idx=None):
        xy = obs[:, OBJ_P][:, :2]
        if idx is None:
            self._goal_xy = xy.clone(); self._goal_xy_clean = xy.clone()
        else:
            self._goal_xy[idx] = xy[idx]; self._goal_xy_clean[idx] = xy[idx]

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_action = torch.zeros_like(self._last_action)
        raw = obs if torch.is_tensor(obs) else self.raw_from_sim()
        self._commit_goal(raw)
        aug, _, _ = self._augment(raw)
        # In perception mode the caller needs the DICT (the camera lives there),
        # so hand back what the underlying env returned and let the caller ask
        # for `teacher_observation()` when it wants the privileged view.
        return (aug if torch.is_tensor(obs) else obs), info

    def step(self, action):
        self._last_action = action.detach()
        obs, rew, term, trunc, info = self.env.step(action)
        # In PERCEPTION mode the observation is a dict with no state vector (that
        # is the point -- nothing can read the object's pose through it), so the
        # privileged view is rebuilt from the live sim instead. Safe here for the
        # same reason `raw_from_sim` documents: auto_reset has already run, so the
        # live sim IS the state this observation describes.
        raw = obs if torch.is_tensor(obs) else self.raw_from_sim()
        # Refresh BEFORE augmenting: with auto_reset the returned obs already
        # belongs to the next episode for any env that finished, so its goal must
        # be re-committed from that obs or the first step of every episode after
        # the first would carry the previous episode's goal.
        done = (term | trunc).bool()
        if bool(done.any()):
            self._commit_goal(raw, done)
        aug, obj, goal = self._augment(raw)
        gd = (obj - goal).norm(dim=-1).mean(-1)
        info["kp_dist"] = gd
        # `ppo.py:137` logs `is_cubeA_grasped`; publish the grasp flag under that
        # name so the shared trainer's early-progress readout works here too.
        gr = self.base.agent.is_grasping(self.base.peg)
        info["is_peg_grasped"] = info["is_cubeA_grasped"] = gr
        if self.w_kp:
            # NOT the official reward any more; --w-kp 0 keeps it untouched.
            rew = rew + self.w_kp * (1 - torch.tanh(5.0 * gd))
        return (aug if torch.is_tensor(obs) else obs), rew, term, trunc, info

    def close(self):
        self.env.close()


def make_liftpeg_kp_env(num_envs=256, num_kp=64, reward="stock", w_kp=0.0, seed=0,
                        control="pd_joint_delta_pos", max_episode_steps=50,
                        reconfig_freq=0, reward_mode="normalized_dense",
                        ap2ap_fields=True, ignore_terminations=False,
                        robot="panda", last_action=False, perception=False,
                        tracegen_camera=False, sig_obs=False):
    """max_episode_steps defaults to the task's own 50 (`register_env`), not
    peg's 100. The horizon is part of the task definition here: the official RL
    rollouts in the demo package average 41.9 steps."""
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
        # The TraceGen dataset was rendered through the task's own FITTED camera
        # at 384 (msgen/tasks.py:95-108, swept 2026-08-14). Query pixel IDs only
        # mean the same thing under the same camera, so principle 4 requires
        # matching it here.
        from mani_skill.utils import sapien_utils
        from msgen.tasks import IMAGE_SIZE, get_task
        c = get_task("liftpeg")["camera"]
        extra["sensor_configs"] = {"base_camera": dict(
            pose=sapien_utils.look_at(eye=c["eye"], target=c["target"]),
            width=IMAGE_SIZE, height=IMAGE_SIZE, fov=c["fov"])}
    env = gym.make("LiftPegUpright-v1", num_envs=num_envs, obs_mode=mode,
                   control_mode=control, sim_backend="physx_cuda",
                   max_episode_steps=max_episode_steps, reward_mode=reward_mode,
                   reconfiguration_freq=reconfig_freq, robot_uids=robot, **extra)
    env = ManiSkillVectorEnv(env, auto_reset=True,
                             ignore_terminations=ignore_terminations)
    return KeypointLiftPeg(env, num_kp=num_kp, reward=reward, w_kp=w_kp,
                           seed=seed, ap2ap_fields=ap2ap_fields,
                           last_action=last_action, sig_obs=sig_obs)
