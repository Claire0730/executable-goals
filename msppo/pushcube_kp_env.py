"""Keypoint wrapper for PushCube-v1 -- the first non-prehensile task in the family.

Why it is here: this adapter gives the executor side a PushCube teacher through the same kp_teacher contract.

LAYOUT, dumped from obs_mode="state" and verified element-by-element:
    qpos(9) qvel(9) tcp_pose(7) goal_pos(3) obj_pose(7)          -> 35-D
    obs[18:21]=tcp.p  obs[25:28]=goal_region.p (z ~ 0.001, ON the table)  obs[28:31]=cube.p  obs[31:35]=cube.q
No is_grasped slice (unlike PickCube) and no relative-vector tail. Success (`push_cube.py`): cube xy within
goal_radius=0.1 of goal_region and cube still on the table; the gripper typically stays CLOSED and pushes.

GOAL KEYPOINTS. The task constrains the cube's xy only. goal_kp = canonical cube points at
(goal_x, goal_y, cube_half) with the cube's yaw committed at reset -- same convention as PickCube's positional goal
(demanding a yaw would invent a requirement the task does not have; z = rest height since the cube stays on the table).
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
TCP_P = slice(18, 21)
GOAL_P = slice(25, 28)
OBJ_P, OBJ_Q = slice(28, 31), slice(31, 35)


class KeypointPushCube:
    has_scene = False
    RAW_DIM = 35

    def __init__(self, env, num_kp=64, reward="stock", w_kp=0.0, seed=0,
                 device="cuda", ap2ap_fields=True, last_action=False,
                 fam_cfg=None, sig_obs=False):
        assert reward == "stock", "pushcube: stock reward only (the family reward is grasp-shaped)"
        self.env, self.base, self.device = env, env.unwrapped, device
        self.num_kp, self.reward_kind, self.w_kp = num_kp, reward, w_kp
        self.num_envs = self.base.num_envs
        half = float(self.base.cube_half_size) if hasattr(self.base, "cube_half_size") else 0.02
        self._half = half
        self._canon = unit_box_keypoints(num_kp, seed).to(device) * half
        self._goal_q = torch.zeros((self.num_envs, 4), device=device)
        self._goal_q[:, 0] = 1.0
        self._goal_q_clean = self._goal_q.clone()
        self._goal_dp = torch.zeros((self.num_envs, 3), device=device)   # injector hook (position offset)
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
        self.single_observation_space = gym.spaces.Box(
            -np.inf, np.inf, (total,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (self.num_envs, total), dtype=np.float32)
        self.single_action_space = self.env.single_action_space
        self.action_space = self.env.action_space

    # ------------------------------------------------------------ keypoints --
    @property
    def cube(self):
        """PushCubeEnv names the cube `obj`; fall back to the scene actor for robustness."""
        return getattr(self.base, "obj", None) or self.base.scene.actors["cube"]

    def _canon_b(self, base_obs):
        return self._canon[None].expand(base_obs.shape[0], -1, -1)

    def set_goal_offset(self, dp, dq):
        from msppo.obs_noise import _qmul
        self._goal_dp = dp
        self._goal_q = _qmul(dq, self._goal_q_clean)

    def _goal_from_obs(self, base_obs):
        """(p, q): the goal-region xy at the cube's rest height, cube yaw from reset."""
        gp = base_obs[:, GOAL_P].clone()
        gp[:, 2] = self._half
        return gp + self._goal_dp, self._goal_q

    def keypoints(self, base_obs):
        canon = self._canon_b(base_obs)
        obj = to_world(canon, base_obs[:, OBJ_P], base_obs[:, OBJ_Q])
        gp, gq = self._goal_from_obs(base_obs)
        return obj, to_world(canon, gp, gq), None

    def _extra_fields(self, base_obs):
        gp, gq = self._goal_from_obs(base_obs)
        gr = self.base.agent.is_grasping(self.cube).float()[:, None]
        return torch.cat([
            gr,                                                               # 1
            base_obs[:, OBJ_P] - base_obs[:, TCP_P],                          # 3
            self.cube.linear_velocity, self.cube.angular_velocity,  # 6
            gp, gq,                                                           # 7
            gp - base_obs[:, OBJ_P],                                          # 3
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
        tcp, c = a.tcp.pose, self.cube.pose
        g = self.base.scene.actors["goal_region"].pose.p
        return torch.cat([
            a.robot.get_qpos()[:, :9], a.robot.get_qvel()[:, :9],
            tcp.raw_pose, g, c.raw_pose], dim=-1).float()

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
        gd = (obj - goal).norm(dim=-1).mean(-1)
        info["kp_dist"] = gd
        if self.w_kp:
            rew = rew + self.w_kp * (1 - torch.tanh(5.0 * gd))
        return (aug if torch.is_tensor(obs) else obs), rew, term, trunc, info

    def close(self):
        self.env.close()


def make_pushcube_kp_env(num_envs=256, num_kp=64, reward="stock", w_kp=0.0,
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
        c = get_task("pushcube")["camera"]
        extra["sensor_configs"] = {"base_camera": dict(
            pose=sapien_utils.look_at(eye=c["eye"], target=c["target"]),
            width=IMAGE_SIZE, height=IMAGE_SIZE, fov=c["fov"])}
    env = gym.make("PushCube-v1", num_envs=num_envs, obs_mode=mode,
                   control_mode=control, sim_backend="physx_cuda",
                   max_episode_steps=max_episode_steps, reward_mode=reward_mode,
                   reconfiguration_freq=reconfig_freq, robot_uids=robot, **extra)
    env = ManiSkillVectorEnv(env, auto_reset=True,
                             ignore_terminations=ignore_terminations)
    return KeypointPushCube(env, num_kp=num_kp, reward=reward, w_kp=w_kp,
                            seed=seed, ap2ap_fields=ap2ap_fields,
                            last_action=last_action, fam_cfg=fam_cfg, sig_obs=sig_obs)
