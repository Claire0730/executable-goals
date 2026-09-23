"""Keypoint wrapper for PickCube-v1 — gate T-G1 of the pretrained-executor spec.

`<private-repo>/docs/SPEC_PRETRAINED_EXECUTOR_20260816.md` §12 puts this first, and §10 risk #3
says why: the unified family reward introduces new free parameters (five weights
and `eps`), and PickCube is the cheapest place to tune them AND the only task in
the whole benchmark with a VERIFIED G1 pass (goal xy independently randomised,
`corr -0.005/+0.036`; the goal marker contributes 0 pixels because
`pick_cube.py:104` appends it to `_hidden_objects`; planner-free ceiling 0.98%).

So a number here is attributable to the reward, not to a leaked goal.

FIVE THINGS READ FROM THE ENV, NOT ASSUMED
──────────────────────────────────────────
1.  Layout is 42-D, dumped from obs_mode="state_dict" and verified:
        qpos(9) qvel(9) is_grasped(1) tcp_pose(7) goal_pos(3) obj_pose(7)
        tcp_to_obj_pos(3) obj_to_goal_pos(3)
    Note `is_grasped` sits at index 18, BEFORE tcp_pose -- PickCube is the only
    task in this set whose stock vector carries it, so the offsets do not match
    the other four.

2.  THE GOAL HAS NO ORIENTATION. `goal_site` is a sphere and `evaluate()` checks
    `‖goal_site.p - cube.p‖ <= 0.025` only. So `goal_kp` is the canonical cube
    points at the goal POSITION carrying the cube's CURRENT orientation -- the
    same choice `kp_env._set_oracle_goal` makes for StackCube, and for the same
    reason: demanding a yaw would invent a requirement the task does not have.

3.  SUCCESS ALSO REQUIRES A STATIC ROBOT (`is_robot_static(0.2)`), not just
    placement. That is the "hold" end-semantic the family reward deliberately
    does NOT encode -- spec §3 leaves stack-releases-vs-pick-holds to Stage 3.
    Expect the family reward alone to place the cube and then keep fidgeting.

4.  THERE IS NO SCENE OBJECT. The goal is a point in empty space. `has_scene` is
    False, so this task exercises the same no-fixture path LiftPegUpright does --
    and spec §2.1 requires the S0 scene slot to be ZEROED AT TRAINING TIME, not
    masked at eval, or the student sees a channel in distillation that will not
    exist at deployment.

5.  `cube_half_size = 0.02`, `goal_thresh = 0.025`, both read from the env rather
    than hardcoded, because `pick_cube_cfgs.py` parameterises them per variant.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from msppo.fam_reward import FamilyRewardCfg, family_reward, family_success
from msppo.peg_kp_env import to_world, unit_box_keypoints

# Verified against obs_mode="state_dict" -- docstring item 1.
QPOS = slice(0, 9)
QVEL = slice(9, 18)
GRASPED = slice(18, 19)
TCP_P = slice(19, 22)
GOAL_P = slice(26, 29)
OBJ_P, OBJ_Q = slice(29, 32), slice(32, 36)


class KeypointPickCube:
    has_scene = False
    RAW_DIM = 42

    def __init__(self, env, num_kp=64, reward="stock", w_kp=0.0, seed=0,
                 device="cuda", ap2ap_fields=True, last_action=False,
                 fam_cfg: FamilyRewardCfg | None = None, sig_obs=False):
        assert reward in ("stock", "family"), reward
        self.env, self.base, self.device = env, env.unwrapped, device
        self.num_kp, self.reward_kind, self.w_kp = num_kp, reward, w_kp
        self.num_envs = self.base.num_envs
        self.fam = fam_cfg or FamilyRewardCfg()

        half = float(self.base.cube_half_size)
        self._canon = unit_box_keypoints(num_kp, seed).to(device) * half
        self._goal_q = torch.zeros((self.num_envs, 4), device=device)
        self._goal_q[:, 0] = 1.0
        # Position lives in the stock slice GOAL_P and is perturbed there; the
        # ORIENTATION is committed at reset and needs the attribute route.
        self._goal_q_clean = self._goal_q.clone()
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

    def set_goal_offset(self, dp, dq):
        """Injector hook. Position is handled by the GOAL_P slice; only the
        committed orientation is set here."""
        from msppo.obs_noise import _qmul
        self._goal_q = _qmul(dq, self._goal_q_clean)

    def _goal_from_obs(self, base_obs):
        """(p, q): the goal POSITION with the cube's yaw committed at reset --
        docstring item 2."""
        return base_obs[:, GOAL_P], self._goal_q

    def keypoints(self, base_obs):
        canon = self._canon_b(base_obs)
        obj = to_world(canon, base_obs[:, OBJ_P], base_obs[:, OBJ_Q])
        gp, gq = self._goal_from_obs(base_obs)
        return obj, to_world(canon, gp, gq), None

    def _extra_fields(self, base_obs):
        gp, gq = self._goal_from_obs(base_obs)
        return torch.cat([
            base_obs[:, GRASPED],                                            # 1
            base_obs[:, OBJ_P] - base_obs[:, TCP_P],                         # 3
            self.base.cube.linear_velocity, self.base.cube.angular_velocity,  # 6
            gp, gq,                                                           # 7
            gp - base_obs[:, OBJ_P],                                          # 3
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
        a = self.base.agent
        tcp, c = a.tcp.pose, self.base.cube.pose
        g = self.base.goal_site.pose.p
        return torch.cat([
            a.robot.get_qpos()[:, :9], a.robot.get_qvel()[:, :9],
            a.is_grasping(self.base.cube).float()[:, None],
            tcp.raw_pose, g, c.raw_pose, c.p - tcp.p, g - c.p], dim=-1).float()

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
        gr = self.base.agent.is_grasping(self.base.cube)
        info["is_cube_grasped"] = info["is_cubeA_grasped"] = gr

        if self.reward_kind == "family":
            # The task's OWN reward and success are demoted to eval-only here
            # (spec §3): nothing below reads `rew` or `info["success"]`, so a
            # family number can never be quietly mixed with a native one.
            speed = self.base.cube.linear_velocity.norm(dim=-1)
            rew, placed = family_reward(
                gd, (raw[:, OBJ_P] - raw[:, TCP_P]).norm(dim=-1),
                gr, speed, self._last_action, self.fam)
            info["fam_placed"] = placed
            info["fam_success"] = family_success(gd, speed, self.fam)
        elif self.w_kp:
            rew = rew + self.w_kp * (1 - torch.tanh(5.0 * gd))
        return (aug if torch.is_tensor(obs) else obs), rew, term, trunc, info

    def close(self):
        self.env.close()


def make_pickcube_kp_env(num_envs=256, num_kp=64, reward="stock", w_kp=0.0,
                         seed=0, control="pd_joint_delta_pos",
                         max_episode_steps=50, reconfig_freq=0,
                         reward_mode="normalized_dense", ap2ap_fields=True,
                         ignore_terminations=False, robot="panda",
                         last_action=False, perception=False,
                         tracegen_camera=False, fam_cfg=None, sig_obs=False):
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
        c = get_task("pickcube")["camera"]
        extra["sensor_configs"] = {"base_camera": dict(
            pose=sapien_utils.look_at(eye=c["eye"], target=c["target"]),
            width=IMAGE_SIZE, height=IMAGE_SIZE, fov=c["fov"])}
    env = gym.make("PickCube-v1", num_envs=num_envs, obs_mode=mode,
                   control_mode=control, sim_backend="physx_cuda",
                   max_episode_steps=max_episode_steps, reward_mode=reward_mode,
                   reconfiguration_freq=reconfig_freq, robot_uids=robot, **extra)
    env = ManiSkillVectorEnv(env, auto_reset=True,
                             ignore_terminations=ignore_terminations)
    return KeypointPickCube(env, num_kp=num_kp, reward=reward, w_kp=w_kp,
                            seed=seed, ap2ap_fields=ap2ap_fields,
                            last_action=last_action, fam_cfg=fam_cfg, sig_obs=sig_obs)
