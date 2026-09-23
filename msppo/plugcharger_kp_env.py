"""Keypoint wrapper for PlugCharger-v1 -- the hardest ManiSkill3 tabletop task, no official PPO, no dense reward
(SUPPORTED_REWARD_MODES = none/sparse), clearance ~0.5 mm, success = charger pose within 5 mm AND 0.2 rad of the
wall receptacle goal. Clean v0 (2026-09-01): custom staged dense reward modelled on PegInsertionSide's official one.

LAYOUT, 46-D state, verified element-by-element (probe 2026-09-01 07:1x):
    qpos(9) qvel(9) tcp_pose(7) charger_pose(7) receptacle_pose(7) goal_pose(7)
Charger spawns FLAT on the table (yaw +-60 deg), base half [20,15,12] mm, two pegs half [8,0.75,3.2] mm along +x;
receptacle on a wall at z=0.1 facing the robot; goal = receptacle * rotz(pi) -> insertion axis = goal frame +x.
The policy must pick the flat charger, reorient to horizontal, and insert at z=0.1.

REWARD (v0), normalised to ~[0,1]:
    r = [ (1 - tanh 4|tcp - base_grasp|)                      reach the charger base
        + is_grasped                                          grasp
        + 3 * (1 - tanh(0.5 s + 4.5 m)) * grasped             goal-frame yz alignment of front AND back point
        + 5 * (1 - tanh 5|x|) * grasped * pre_inserted        insertion progress along goal x once aligned (yz < 10 mm)
        ] / 10
"""
from __future__ import annotations
import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from msppo.peg_kp_env import to_world, unit_box_keypoints
from msppo.obs_noise import _qmul

QPOS = slice(0, 9)
QVEL = slice(9, 18)
TCP_P, TCP_Q = slice(18, 21), slice(21, 25)
OBJ_P, OBJ_Q = slice(25, 28), slice(28, 32)
REC_P = slice(32, 39)
GOAL_P, GOAL_Q = slice(39, 42), slice(42, 46)
BASE_HALF = torch.tensor([0.02, 0.015, 0.012])
PEG_LEN = 0.016            # peg half 8 mm -> full 16 mm forward of the charger origin


def _qrot(q, v):
    """rotate v [B,3] by quaternion q [B,4] (wxyz)."""
    w, x, y, z = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    u = torch.cat([x, y, z], -1)
    return v + 2 * torch.cross(u, torch.cross(u, v, dim=-1) + w * v, dim=-1)


def _qinv_rot(q, v):
    qc = q.clone(); qc[:, 1:] = -qc[:, 1:]
    return _qrot(qc, v)


class KeypointPlugCharger:
    has_scene = False
    RAW_DIM = 46

    def __init__(self, env, num_kp=64, reward="stock", w_kp=0.0, seed=0,
                 device="cuda", ap2ap_fields=True, last_action=False,
                 fam_cfg=None, sig_obs=False):
        self.env, self.base, self.device = env, env.unwrapped, device
        self.num_kp, self.reward_kind, self.w_kp = num_kp, reward, w_kp
        self.num_envs = self.base.num_envs
        canon = unit_box_keypoints(num_kp, seed).to(device)
        self._canon = canon * BASE_HALF.to(device)      # anisotropic charger-base cloud (orientation-bearing)
        self._goal_dp = torch.zeros((self.num_envs, 3), device=device)
        self._goal_dq = None
        self._last_action = torch.zeros((self.num_envs, env.single_action_space.shape[0]), device=device)
        self.ap2ap_fields = ap2ap_fields
        sp = self.env.single_observation_space
        base_dim = sp.shape[-1] if sp.shape is not None else self.RAW_DIM
        assert base_dim == self.RAW_DIM, f"stock observation is {base_dim}, expected {self.RAW_DIM}"
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
    def charger(self):
        return getattr(self.base, "charger", None) or self.base.scene.actors["charger"]

    def _canon_b(self, base_obs):
        return self._canon[None].expand(base_obs.shape[0], -1, -1)

    def set_goal_offset(self, dp, dq):
        self._goal_dp = dp; self._goal_dq = dq

    def _goal_from_obs(self, base_obs):
        gp = base_obs[:, GOAL_P] + self._goal_dp
        gq = base_obs[:, GOAL_Q]
        if self._goal_dq is not None:
            gq = _qmul(self._goal_dq, gq)
        return gp, gq

    def keypoints(self, base_obs):
        canon = self._canon_b(base_obs)
        obj = to_world(canon, base_obs[:, OBJ_P], base_obs[:, OBJ_Q])
        gp, gq = self._goal_from_obs(base_obs)
        return obj, to_world(canon, gp, gq), None

    def _extra_fields(self, base_obs):
        gp, gq = self._goal_from_obs(base_obs)
        gr = self.base.agent.is_grasping(self.charger).float()[:, None]
        return torch.cat([
            gr,
            base_obs[:, OBJ_P] - base_obs[:, TCP_P],
            self.charger.linear_velocity, self.charger.angular_velocity,
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
        return torch.cat([
            a.robot.get_qpos()[:, :9], a.robot.get_qvel()[:, :9],
            a.tcp.pose.raw_pose, self.charger.pose.raw_pose,
            self.base.receptacle.pose.raw_pose, self.base.goal_pose.raw_pose], dim=-1).float()

    def teacher_observation(self, raw=None):
        return self._augment(raw if raw is not None else self.raw_from_sim())[0]

    # ------------------------------------------------------------ reward v0 --
    def _dense_reward(self, base_obs):
        b = self.base
        cp, cq = base_obs[:, OBJ_P], base_obs[:, OBJ_Q]
        gp, gq = base_obs[:, GOAL_P], base_obs[:, GOAL_Q]
        # stage 1: reach the charger BASE centre (behind the origin along -x of the charger frame)
        back_off = torch.tensor([-0.02, 0.0, 0.0], device=cp.device).expand_as(cp)
        base_c = cp + _qrot(cq, back_off)
        d_reach = (base_obs[:, TCP_P] - base_c).norm(dim=-1)
        r = 1.0 - torch.tanh(4.0 * d_reach)
        grasped = b.agent.is_grasping(self.charger, max_angle=20).float()
        r = r + grasped
        # stage 3: goal-frame yz alignment of front tip and back point
        tip_off = torch.tensor([PEG_LEN, 0.0, 0.0], device=cp.device).expand_as(cp)
        tip_w = cp + _qrot(cq, tip_off)
        back_w = base_c
        tip_g = _qinv_rot(gq, tip_w - gp)
        back_g = _qinv_rot(gq, back_w - gp)
        yz_tip = tip_g[:, 1:].norm(dim=-1)
        yz_back = back_g[:, 1:].norm(dim=-1)
        s = yz_tip + yz_back
        m = torch.maximum(yz_tip, yz_back)
        # v1 (09-01): the two alignment points sit ON the x axis, leaving ROLL about the insertion axis
        # unconstrained -- and the dual prongs are laid out in +-y, so roll error makes insertion physically
        # impossible (v0 hovered 30 mm out at yz 2 mm, min|x| med 32.6 mm, success 0). Constrain the full
        # orientation with the quaternion geodesic angle, and give the last 30 mm a steep, larger incentive.
        dq = (cq * gq).sum(dim=-1).abs().clamp(max=1.0)
        ang = 2.0 * torch.arccos(dq)                                   # rad, [0, pi]
        r = r + 3.0 * (1.0 - torch.tanh(0.5 * s + 4.5 * m + 2.0 * ang)) * grasped
        pre = ((yz_tip < 0.01) & (yz_back < 0.01) & (ang < 0.35)).float()
        # stage 4: insertion progress -- steep gradient over the final centimetres (tanh 5x was too flat:
        # at x=30 mm it pays 85% of the maximum, so hovering was nearly free)
        org_g = _qinv_rot(gq, cp - gp)
        x_abs = org_g[:, 0].abs()
        r = r + 6.0 * (1.0 - torch.tanh(12.0 * x_abs)) * grasped * pre
        # v4 (09-01): HOLE-LEVEL alignment. v3 stalls at the faceplate because nothing rewards the sub-mm
        # search: yz gates are 10 mm coarse while the holes need <1 mm. Reward each peg TIP against ITS hole
        # (charger frame [2*peg_half_x, +-peg_gap, 0] -> goal frame [x, +-7 mm, 0]) with a sharp tanh(150 e)
        # slope that only really pays in the 0-5 mm band, gated near the plate.
        near = (x_abs < 0.03).float()
        for sgn in (1.0, -1.0):
            t_off = torch.tensor([2 * 0.008, sgn * 0.007, 0.0], device=cp.device).expand_as(cp)
            tg = _qinv_rot(gq, (cp + _qrot(cq, t_off)) - gp)
            e = (tg[:, 1:] - torch.tensor([sgn * 0.007, 0.0], device=cp.device)).norm(dim=-1)
            r = r + 1.0 * (1.0 - torch.tanh(150.0 * e)) * grasped * pre * near
        return r / 13.0

    # ------------------------------------------------------------------ gym --
    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_action = torch.zeros_like(self._last_action)
        raw = obs if torch.is_tensor(obs) else self.raw_from_sim()
        aug, _, _ = self._augment(raw)
        return (aug if torch.is_tensor(obs) else obs), info

    def step(self, action):
        # v6 (09-01): ACTION-RESOLUTION knob. v5 autopsy: failure set reaches 0.95 mm median hole error
        # (success needs ~0.3-0.5 mm) -- the joint-delta step size is the limiter, not the policy's intent.
        # Scaling actions by s<1 doubles effective resolution; horizon 200 absorbs the slower transport.
        _sc = getattr(self, "_act_scale", None)
        if _sc is None:
            import os as _os
            _sc = float(_os.environ.get("MSPPO_PLUG_ACTSCALE", "1.0")); self._act_scale = _sc
            if _sc != 1.0: print(f"[plug] action scale {_sc}", flush=True)
        if _sc != 1.0:
            action = action * _sc
        self._last_action = action.detach()
        obs, rew, term, trunc, info = self.env.step(action)
        raw = obs if torch.is_tensor(obs) else self.raw_from_sim()
        aug, obj, goal = self._augment(raw)
        rew = self._dense_reward(raw) + info["success"].float()   # success bonus on top
        info["kp_dist"] = (obj - goal).norm(dim=-1).mean(-1)
        return (aug if torch.is_tensor(obs) else obs), rew, term, trunc, info

    def close(self):
        self.env.close()


def make_plugcharger_kp_env(num_envs=256, num_kp=64, reward="stock", w_kp=0.0,
                            seed=0, control="pd_joint_delta_pos",
                            max_episode_steps=100, reconfig_freq=0,
                            reward_mode="normalized_dense", ap2ap_fields=True,
                            ignore_terminations=False, robot="panda_wristcam",
                            last_action=False, perception=False,
                            tracegen_camera=False, fam_cfg=None, sig_obs=False):
    import os as _os
    _chan = "depth+segmentation" if _os.environ.get("MSGEN_NO_RGB") not in (None, "", "0") \
        else "rgb+depth+segmentation"
    mode = _chan if perception else "state"
    # CLEARANCE CURRICULUM (v2, 09-01): the stock single-side clearance is 0.5 mm; v1 proved the policy presses
    # the peg tips against the faceplate at the right spot/angle but cannot find the holes (min|x| stalls at
    # PEG_LEN). Widening the holes lets it learn the full insertion motion; later stages anneal back down.
    _cl = _os.environ.get("MSPPO_PLUG_CLEARANCE")
    if _cl:
        from mani_skill.envs.tasks.tabletop.plug_charger import PlugChargerEnv
        PlugChargerEnv._clearance = float(_cl)
        print(f"[plug] receptacle clearance patched to {float(_cl)*1000:.1f} mm/side", flush=True)
    env = gym.make("PlugCharger-v1", num_envs=num_envs, obs_mode=mode,
                   control_mode=control, sim_backend="physx_cuda",
                   max_episode_steps=max_episode_steps, reward_mode="none",
                   reconfiguration_freq=reconfig_freq, robot_uids="panda_wristcam")
    env = ManiSkillVectorEnv(env, auto_reset=True, ignore_terminations=ignore_terminations)
    return KeypointPlugCharger(env, num_kp=num_kp, reward=reward, w_kp=w_kp,
                               seed=seed, ap2ap_fields=ap2ap_fields,
                               last_action=last_action, fam_cfg=fam_cfg, sig_obs=sig_obs)
