"""Keypoint wrapper for PlaceSphere-v1 -- ZERO-SHOT probe of the pick-and-place family (2026-08-31).

Purpose: the four-task psi student has never seen a sphere or a bin; this adapter only builds the same pose interface so
the FROZEN student can be evaluated zero-shot (row 3 with the true goal = executor generality; row 4 needs a planner bank).
No teacher is trained here yet (the task has dense reward if we ever want one).

LAYOUT, 39-D flat state, dumped from obs_mode="state_dict" 2026-08-31 08:50 (dict order is the flatten order):
    qpos(9) qvel(9) is_grasped(1) tcp_pose(7) bin_pos(3) obj_pose(7) tcp_to_obj_pos(3)
Success (place_sphere.py): sphere centre within 5 mm xy of the bin centre AND z = bin bottom + radius (+-5 mm), static,
ungrasped -- the TIGHTEST positional tolerance in the family (5 mm vs PickCube's 25 mm).

GOAL: bin centre + [0, 0, block_half(0.0025) + radius(0.02)], orientation = the sphere's yaw at reset (orientation-free
task, same convention as PickCube). Keypoints: the canonical box cloud scaled to the sphere radius -- rotation from a
sphere's points is meaningless, but the POSITION interface (what the student consumes) is exact.
"""
from __future__ import annotations
import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from msppo.peg_kp_env import to_world, unit_box_keypoints

QPOS = slice(0, 9)
QVEL = slice(9, 18)
GRASPED = slice(18, 19)
TCP_P = slice(19, 22)
BIN_P = slice(26, 29)
OBJ_P, OBJ_Q = slice(29, 32), slice(32, 36)


class KeypointPlaceSphere:
    has_scene = False
    RAW_DIM = 39

    def __init__(self, env, num_kp=64, reward="stock", w_kp=0.0, seed=0,
                 device="cuda", ap2ap_fields=True, last_action=False,
                 fam_cfg=None, sig_obs=False):
        assert reward == "stock", reward
        self.env, self.base, self.device = env, env.unwrapped, device
        self.num_kp, self.reward_kind, self.w_kp = num_kp, reward, w_kp
        self.num_envs = self.base.num_envs
        r = float(getattr(self.base, "radius", 0.02))
        self._rest_z = r + float(getattr(self.base, "block_half_size", [0.0025])[0])
        self._canon = unit_box_keypoints(num_kp, seed).to(device) * r
        self._goal_q = torch.zeros((self.num_envs, 4), device=device)
        self._goal_q[:, 0] = 1.0
        self._goal_q_clean = self._goal_q.clone()
        self._goal_dp = torch.zeros((self.num_envs, 3), device=device)
        self._last_action = torch.zeros(
            (self.num_envs, env.single_action_space.shape[0]), device=device)
        self.ap2ap_fields = ap2ap_fields
        sp = self.env.single_observation_space
        base_dim = sp.shape[-1] if sp.shape is not None else self.RAW_DIM
        assert base_dim == self.RAW_DIM, (
            f"stock observation is {base_dim}, expected {self.RAW_DIM}")
        if ap2ap_fields:
            base_dim += 20
        self.last_action_obs = last_action
        if last_action:
            base_dim += env.single_action_space.shape[0]
        self.sig_obs = bool(sig_obs)
        if self.sig_obs:
            base_dim += 4
        self.kp_slice = slice(base_dim, base_dim + num_kp * 6)
        total = base_dim + num_kp * 6
        self.single_observation_space = gym.spaces.Box(-np.inf, np.inf, (total,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.num_envs, total), dtype=np.float32)
        self.single_action_space = self.env.single_action_space
        self.action_space = self.env.action_space

    # ------------------------------------------------------------ keypoints --
    @property
    def sphere(self):
        return getattr(self.base, "obj", None) or self.base.scene.actors["sphere"]

    def _canon_b(self, base_obs):
        return self._canon[None].expand(base_obs.shape[0], -1, -1)

    def set_goal_offset(self, dp, dq):
        from msppo.obs_noise import _qmul
        self._goal_dp = dp
        self._goal_q = _qmul(dq, self._goal_q_clean)

    def _goal_from_obs(self, base_obs):
        gp = base_obs[:, BIN_P].clone()
        gp[:, 2] = gp[:, 2] + self._rest_z
        return gp + self._goal_dp, self._goal_q

    def keypoints(self, base_obs):
        canon = self._canon_b(base_obs)
        obj = to_world(canon, base_obs[:, OBJ_P], base_obs[:, OBJ_Q])
        gp, gq = self._goal_from_obs(base_obs)
        return obj, to_world(canon, gp, gq), None

    def _extra_fields(self, base_obs):
        gp, gq = self._goal_from_obs(base_obs)
        return torch.cat([
            base_obs[:, GRASPED],
            base_obs[:, OBJ_P] - base_obs[:, TCP_P],
            self.sphere.linear_velocity, self.sphere.angular_velocity,
            gp, gq,
            gp - base_obs[:, OBJ_P],
        ], dim=-1)

    def _noisy(self, obs):
        n = getattr(self, "obs_noise", None)
        if n is None:
            return obs
        n.tick(getattr(self, "noise_horizon", 0))
        return n.apply(obs)

    def _sig_row(self, b):
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
        a = self.base.agent
        tcp, c = a.tcp.pose, self.sphere.pose
        bp = self.base.scene.actors["bin"].pose.p
        return torch.cat([
            a.robot.get_qpos()[:, :9], a.robot.get_qvel()[:, :9],
            a.is_grasping(self.sphere).float()[:, None],
            tcp.raw_pose, bp, c.raw_pose, c.p - tcp.p], dim=-1).float()

    def teacher_observation(self, raw=None):
        return self._augment(raw if raw is not None else self.raw_from_sim())[0]

    # ------------------------------------------------------------------ gym --
    def _commit_goal(self, raw, idx=None):
        q = raw[:, OBJ_Q]
        if idx is None:
            self._goal_q = q.clone(); self._goal_q_clean = q.clone()
        else:
            self._goal_q[idx] = q[idx]; self._goal_q_clean[idx] = q[idx]

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
        if bool(done.any()):
            self._commit_goal(raw, done)
        aug, obj, goal = self._augment(raw)
        info["kp_dist"] = (obj - goal).norm(dim=-1).mean(-1)
        if self.w_kp:
            rew = rew + self.w_kp * (1 - torch.tanh(5.0 * info["kp_dist"]))
        return (aug if torch.is_tensor(obs) else obs), rew, term, trunc, info

    def close(self):
        self.env.close()


def make_placesphere_kp_env(num_envs=256, num_kp=64, reward="stock", w_kp=0.0,
                            seed=0, control="pd_joint_delta_pos",
                            max_episode_steps=50, reconfig_freq=0,
                            reward_mode="normalized_dense", ap2ap_fields=True,
                            ignore_terminations=False, robot="panda",
                            last_action=False, perception=False,
                            tracegen_camera=False, fam_cfg=None, sig_obs=False):
    import os as _os
    _chan = "depth+segmentation" if _os.environ.get("MSGEN_NO_RGB") not in (None, "", "0") \
        else "rgb+depth+segmentation"
    mode = _chan if perception else "state"
    extra = {}
    if tracegen_camera:
        from mani_skill.utils import sapien_utils
        from msgen.tasks import IMAGE_SIZE, get_task
        c = get_task("placesphere")["camera"]
        extra["sensor_configs"] = {"base_camera": dict(
            pose=sapien_utils.look_at(eye=c["eye"], target=c["target"]),
            width=IMAGE_SIZE, height=IMAGE_SIZE, fov=c["fov"])}
    env = gym.make("PlaceSphere-v1", num_envs=num_envs, obs_mode=mode,
                   control_mode=control, sim_backend="physx_cuda",
                   max_episode_steps=max_episode_steps, reward_mode=reward_mode,
                   reconfiguration_freq=reconfig_freq, robot_uids=robot, **extra)
    env = ManiSkillVectorEnv(env, auto_reset=True,
                             ignore_terminations=ignore_terminations)
    return KeypointPlaceSphere(env, num_kp=num_kp, reward=reward, w_kp=w_kp,
                               seed=seed, ap2ap_fields=ap2ap_fields,
                               last_action=last_action, fam_cfg=fam_cfg, sig_obs=sig_obs)
