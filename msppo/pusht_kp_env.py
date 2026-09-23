"""Keypoint wrapper for PushT-v1 (not one of the five released tasks).

The object is a Tee, not a box, so `unit_box_keypoints` does not apply and a
canonical set has to be built from the actual collision geometry.

WHAT THIS TASK ADDS TO THE SET
──────────────────────────────
  * the only NON-BOX object -- the paired encoding has so far only ever been fed
    points sampled on a box, on every task in this project;
  * the only DIFFERENT EMBODIMENT -- `panda_stick`, 7 DoF, NO GRIPPER. Anything
    that assumes finger links or a grasp transform is undefined here, which is
    why `is_grasping` is never called below and why the proprio block is 21-D,
    not 25-D;
  * the only PURELY PLANAR task.

FIVE THINGS READ FROM THE ENV, NOT ASSUMED
──────────────────────────────────────────
1.  Layout is 31-D, dumped from obs_mode="state_dict" and verified:
        qpos(7) qvel(7) tcp_pose(7) goal_pos(3) obj_pose(7)
    Note qpos/qvel are 7, not 9 -- no gripper joints.

2.  THE GOAL ORIENTATION IS NOT IN THE OBSERVATION. `_get_obs_extra` exposes
    `goal_pos` (position only, `push_t.py:494`). The goal's rotation is the class
    constant `goal_z_rot = (5/3)*pi` (`:100`), identical in every episode. Taking
    it from the class is therefore reading a TASK CONSTANT, not simulator state
    -- the same status as the peg's calibrated insertion orientation. It is read
    from the live env below so a ManiSkill version bump cannot silently desync it.

3.  THE GOAL MARKER SITS AT z = 1e-3, THE OBJECT AT z = 0.021. The marker is a
    flat decal (`half_thickness = 1e-4` when `target=True`, `:181`). Placing the
    goal keypoints at the marker's own z would inject a spurious 20 mm vertical
    error into every single `obj -> goal` vector. Success is a 2-D area overlap
    computed in the goal frame (`pseudo_render_intersection`) and does not look
    at z at all, so the goal keypoints are placed at the OBJECT's z. This is the
    one place the goal pose is not simply "the marker's pose".

4.  Tee geometry, from `create_tee` (`:176-230`), centred on the centre of mass
    (com_y = 0.0375) so rotations apply about the COM:
        box1 (horizontal)  centre (0, -0.0375, 0)  half (0.100, 0.025, 0.02)
        box2 (vertical)    centre (0, +0.0625, 0)  half (0.025, 0.075, 0.02)
    `unit_tee_keypoints` samples both, area-weighted, corners first.

5.  Success is `intersection / goal_area >= 0.90` -- a 2-D overlap, not a
    distance. It is far stricter than it sounds: the Tee is nearly symmetric
    under a 180 deg z rotation but NOT symmetric, so a half-turn error scores
    well below threshold while a keypoint-distance metric would call it close.

SCENE CHANNEL. `scene_kp` is the goal marker's own points. That is not circular
even though it coincides with `goal_kp` here: in the student the two arrive by
different routes -- `scene_kp` is PERCEIVED (the marker is painted on the table
and visible), `goal_kp` is PREDICTED by the planner. It is the same relationship
the peg has between its hole rim and its insertion goal.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from msppo.peg_kp_env import quat_to_R, to_world

# Verified against obs_mode="state_dict" -- see docstring item 1.
PROPRIO = slice(0, 21)          # qpos 7 + qvel 7 + tcp_pose 7
TCP_P = slice(14, 17)
GOAL_P = slice(21, 24)
OBJ_P, OBJ_Q = slice(24, 27), slice(27, 31)

# `push_t.py:176-230`, in the Tee's own COM-centred frame -- docstring item 4.
TEE_BOXES = (
    ((0.0, -0.0375, 0.0), (0.100, 0.025, 0.02)),      # horizontal bar
    ((0.0, +0.0625, 0.0), (0.025, 0.075, 0.02)),      # vertical stem
)


def unit_tee_keypoints(k: int, seed: int = 0) -> torch.Tensor:
    """[k,3] points on the Tee's surface, in its own COM-centred frame.

    Corners of both boxes first (16 of them), then a surface sample split between
    the boxes in proportion to surface AREA rather than evenly -- an even split
    would over-represent the small stem and make the pooled descriptor rotate
    with it. Fixed seed and generated ONCE at construction: resampling per
    episode silently sends a StackCube policy out of distribution.

    Not scaled to a unit box, unlike `unit_box_keypoints`: the Tee's proportions
    are the object's identity, so the canonical set carries metric sizes and
    there is no per-env half-size to multiply by (PushT's Tee is not randomised).
    """
    g = torch.Generator().manual_seed(seed)
    corners = []
    for c, h in TEE_BOXES:
        c_t = torch.tensor(c, dtype=torch.float32)
        h_t = torch.tensor(h, dtype=torch.float32)
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    corners.append(c_t + h_t * torch.tensor([sx, sy, sz]))
    corners = torch.stack(corners)                      # [16,3]
    if k <= corners.shape[0]:
        return corners[:k]

    areas = []
    for _, h in TEE_BOXES:
        x, y, z = h
        areas.append(8.0 * (x * y + y * z + z * x))     # full box surface area
    areas = np.array(areas)
    n = k - corners.shape[0]
    n_each = np.floor(areas / areas.sum() * n).astype(int)
    n_each[0] += n - n_each.sum()                       # give the remainder away

    pts = []
    for (c, h), ni in zip(TEE_BOXES, n_each):
        if ni <= 0:
            continue
        c_t = torch.tensor(c, dtype=torch.float32)
        h_t = torch.tensor(h, dtype=torch.float32)
        p = torch.rand((int(ni), 3), generator=g) * 2 - 1
        face = torch.randint(0, 3, (int(ni),), generator=g)
        sign = torch.randint(0, 2, (int(ni),), generator=g) * 2.0 - 1.0
        p[torch.arange(int(ni)), face] = sign           # push onto a face
        pts.append(c_t + p * h_t)
    return torch.cat([corners] + pts, dim=0)


class KeypointPushT:
    """Composition, not gym.Wrapper: ManiSkillVectorEnv is a VectorEnv and fails
    gym.Wrapper's isinstance check."""

    has_scene = True
    RAW_DIM = 31

    def __init__(self, env, num_kp=64, reward="stock", w_kp=0.0, seed=0,
                 device="cuda", ap2ap_fields=True, last_action=False):
        assert reward == "stock", (
            "PushT has no pose-reaching variant: success is a 2-D area overlap, "
            "and a centre-distance reward is exactly the local optimum the "
            "task's own comment says the legacy reward got stuck in")
        self.env, self.base, self.device = env, env.unwrapped, device
        self.num_kp, self.reward_kind, self.w_kp = num_kp, reward, w_kp
        self.num_envs = self.base.num_envs

        self._canon = unit_tee_keypoints(num_kp, seed).to(device)
        # Read from the live env rather than hardcoded, so a ManiSkill version
        # that changes the constant cannot silently desync this -- item 2.
        z = float(self.base.goal_z_rot)
        self._goal_q = torch.tensor(
            [np.cos(z / 2), 0.0, 0.0, np.sin(z / 2)],
            device=device, dtype=torch.float32).expand(self.num_envs, 4)

        self._last_action = torch.zeros(
            (self.num_envs, env.single_action_space.shape[0]), device=device)

        self.ap2ap_fields = ap2ap_fields
        # In perception mode the observation space is a DICT, so it has no
        # `.shape`. The teacher still consumes the flat stock vector, rebuilt by
        # `raw_from_sim()`, whose width is fixed and verified against
        # obs_mode="state" (31).
        sp = self.env.single_observation_space
        base_dim = sp.shape[-1] if sp.shape is not None else self.RAW_DIM
        assert base_dim == self.RAW_DIM, (
            f"stock observation is {base_dim}, expected {self.RAW_DIM}")
        if ap2ap_fields:
            # No is_grasped: panda_stick has no gripper (docstring), so this is
            # 19 where peg and liftpeg are 20.
            base_dim += 19      # tcp_to_obj 3 + obj_vel 6 + goal_pose 7
                                # + obj_to_goal 3
        self.last_action_obs = last_action
        if last_action:
            base_dim += env.single_action_space.shape[0]

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
        """(p, q) of the goal Tee pose, DERIVED FROM THE OBSERVATION VECTOR.

        z comes from the OBJECT, not from the marker -- docstring item 3.
        """
        p = base_obs[:, GOAL_P].clone()
        p[:, 2] = base_obs[:, OBJ_P][:, 2]
        return p, self._goal_q

    def keypoints(self, base_obs):
        """(obj_kp, goal_kp, scene_kp), [B,K,3] world coordinates, paired by index.

        Read from the OBSERVATION vector, never from `self.base.tee.pose`: the
        vector env auto-resets before returning (`peg_kp_env.py:206`).
        """
        canon = self._canon_b(base_obs)
        obj = to_world(canon, base_obs[:, OBJ_P], base_obs[:, OBJ_Q])
        gp, gq = self._goal_from_obs(base_obs)
        goal = to_world(canon, gp, gq)
        # The marker as it actually sits on the table: its own z, not the
        # object's. This is the PERCEIVED fixture, so it must be where the camera
        # would see it.
        sp = base_obs[:, GOAL_P]
        scene = to_world(canon, sp, gq)
        return obj, goal, scene

    def _extra_fields(self, base_obs):
        gp, gq = self._goal_from_obs(base_obs)
        return torch.cat([
            base_obs[:, OBJ_P] - base_obs[:, TCP_P],                        # 3
            self.base.tee.linear_velocity, self.base.tee.angular_velocity,  # 6
            gp, gq,                                                          # 7
            gp - base_obs[:, OBJ_P],                                         # 3
        ], dim=-1)

    def _augment(self, obs):
        obj, goal, _ = self.keypoints(obs)
        b = obs.shape[0]
        parts = [obs]
        if self.ap2ap_fields:
            parts.append(self._extra_fields(obs))
        if self.last_action_obs:
            parts.append(self._last_action)
        parts += [obj.reshape(b, -1), goal.reshape(b, -1)]
        return torch.cat(parts, dim=-1), obj, goal

    def raw_from_sim(self):
        """[B,31] the stock state vector, rebuilt from the LIVE sim -- see
        `liftpeg_kp_env.raw_from_sim` for why this exists and why reading live is
        safe at the one point DAgger calls it."""
        a = self.base.agent
        return torch.cat([a.robot.get_qpos()[:, :7], a.robot.get_qvel()[:, :7],
                          a.tcp.pose.raw_pose, self.base.goal_tee.pose.p,
                          self.base.tee.pose.raw_pose], dim=-1).float()

    def teacher_observation(self):
        return self._augment(self.raw_from_sim())[0]

    # ------------------------------------------------------------------ gym --
    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_action = torch.zeros_like(self._last_action)
        raw = obs if torch.is_tensor(obs) else self.raw_from_sim()
        aug, _, _ = self._augment(raw)
        return (aug if torch.is_tensor(obs) else obs), info

    def step(self, action):
        self._last_action = action.detach()
        obs, rew, term, trunc, info = self.env.step(action)
        # Dict observation in perception mode -- see liftpeg_kp_env.step.
        raw = obs if torch.is_tensor(obs) else self.raw_from_sim()
        aug, obj, goal = self._augment(raw)
        gd = (obj - goal).norm(dim=-1).mean(-1)
        info["kp_dist"] = gd
        # No grasp flag exists on this embodiment, so `ppo.py`'s grasp readout
        # will log nan here BY DESIGN. The early-progress proxy for PushT is
        # kp_dist, not grasp rate.
        if self.w_kp:
            rew = rew + self.w_kp * (1 - torch.tanh(5.0 * gd))
        return (aug if torch.is_tensor(obs) else obs), rew, term, trunc, info

    def close(self):
        self.env.close()


def make_pusht_kp_env(num_envs=256, num_kp=64, reward="stock", w_kp=0.0, seed=0,
                      control="pd_joint_delta_pos", max_episode_steps=100,
                      reconfig_freq=0, reward_mode="normalized_dense",
                      ap2ap_fields=True, ignore_terminations=False,
                      robot="panda_stick", last_action=False, perception=False,
                      tracegen_camera=False):
    """robot_uids is panda_stick and there is no alternative: `SUPPORTED_ROBOTS =
    ["panda_stick"]` (`push_t.py:70`)."""
    mode = "rgb+depth+segmentation" if perception else "state"
    extra = {}
    if tracegen_camera:
        from mani_skill.utils import sapien_utils
        from msgen.tasks import IMAGE_SIZE, get_task
        c = get_task("pusht")["camera"]
        extra["sensor_configs"] = {"base_camera": dict(
            pose=sapien_utils.look_at(eye=c["eye"], target=c["target"]),
            width=IMAGE_SIZE, height=IMAGE_SIZE, fov=c["fov"])}
    env = gym.make("PushT-v1", num_envs=num_envs, obs_mode=mode,
                   control_mode=control, sim_backend="physx_cuda",
                   max_episode_steps=max_episode_steps, reward_mode=reward_mode,
                   reconfiguration_freq=reconfig_freq, robot_uids=robot, **extra)
    env = ManiSkillVectorEnv(env, auto_reset=True,
                             ignore_terminations=ignore_terminations)
    return KeypointPushT(env, num_kp=num_kp, reward=reward, w_kp=w_kp,
                         seed=seed, ap2ap_fields=ap2ap_fields,
                         last_action=last_action)
