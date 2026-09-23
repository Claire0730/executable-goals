"""PegInsertionSide under the SHARED task contract.

`msppo/peg_kp_env.py` already wraps this task and every peg number in the project
rests on it, so it is NOT touched. This is a thin subclass that adds the four
methods the shared stack requires (`RAW_DIM`, `has_scene`, `_canon_b`,
`raw_from_sim`, `teacher_observation`) so peg can join a mixed-task student.

THE ONE THING THAT IS GENUINELY DIFFERENT FROM EVERY OTHER TASK HERE.
Peg geometry is RANDOMISED PER PARALLEL ENV at `_load_scene`
(`peg_insertion_side.py:114-115`: half-length U(0.085, 0.125), half-radius
U(0.015, 0.025)). So the canonical keypoint set is **[B,K,3], per env** -- not the
fixed [K,3] a cube allows. Everything downstream that assumed a shared canonical
set has to go through `_canon_b`, which is why the shared contract asks for that
method rather than a `_canon` attribute.

`_canon` is still exposed, as the NOMINAL template, because `reset_masks` draws
its occlusion side in the object's canonical frame and only needs a
representative shape. That is the same nominal template the 0.98 reference
teacher was trained on (`peg_kp_env._canon_peg`: "their kp64 is (64,3), SHARED
across envs ... a FIXED template at the distribution mid-point").
"""
from __future__ import annotations

import torch

from msppo.peg_kp_env import (HOLE_P, HOLE_Q, PEG_HALF, PEG_P, PEG_Q, TCP_P,
                              KeypointPegInsert, make_peg_kp_env)


class KeypointPegInsertShared(KeypointPegInsert):
    has_scene = True
    RAW_DIM = 43

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        # Nominal template, for the occlusion draw only -- see docstring.
        half = torch.tensor(self.NOMINAL_HALF, device=self.device,
                            dtype=torch.float32)
        self._canon = self._unit_peg * half[None, :]

    def _canon_b(self, base_obs):
        """[B,K,3] PER ENV -- peg geometry is not shared across envs."""
        return self._canon_peg(base_obs)

    def raw_from_sim(self):
        """[B,43] the stock state vector from the LIVE sim, for perception mode.

        Layout verified in `peg_kp_env`'s header:
            qpos(9) qvel(9) tcp_pose(7) peg_pose(7) peg_half_size(3)
            box_hole_pose(7) box_hole_radius(1)
        """
        b, a = self.base, self.base.agent
        return torch.cat([
            a.robot.get_qpos()[:, :9], a.robot.get_qvel()[:, :9],
            a.tcp.pose.raw_pose, b.peg.pose.raw_pose, b.peg_half_sizes,
            b.box_hole_pose.raw_pose, b.box_hole_radii.reshape(-1, 1),
        ], dim=-1).float()

    def teacher_observation(self, raw=None):
        return self._augment(raw if raw is not None else self.raw_from_sim())[0]

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_action = torch.zeros_like(self._last_action)
        raw = obs if torch.is_tensor(obs) else self.raw_from_sim()
        aug, _, _ = self._augment(raw)
        return (aug if torch.is_tensor(obs) else obs), info

    def step(self, action):
        self._last_action = action.detach()
        obs, rew, term, trunc, info = self.env.step(action)
        raw = obs if torch.is_tensor(obs) else self.raw_from_sim()
        aug, obj, goal = self._augment(raw)
        gd = (obj - goal).norm(dim=-1).mean(-1)
        info["kp_dist"] = gd
        gr = self.base.agent.is_grasping(self.base.peg)
        info["is_peg_grasped"] = info["is_cubeA_grasped"] = gr
        if self.reward_kind == "pose_reach":
            rew = self._pose_reach_reward(raw, obj, goal, info)
        elif self.w_kp:
            rew = rew + self.w_kp * (1 - torch.tanh(5.0 * gd))
        return (aug if torch.is_tensor(obs) else obs), rew, term, trunc, info


def make_peginsert_kp_env(num_envs=256, num_kp=64, reward="stock", w_kp=0.0,
                          seed=0, control="pd_joint_delta_pos",
                          max_episode_steps=100, reconfig_freq=0,
                          reward_mode="normalized_dense", ap2ap_fields=True,
                          ignore_terminations=False, robot="panda",
                          last_action=False, perception=False,
                          tracegen_camera=False, clearance=0.01, sig_obs=False,
                          priv_obs=False, **kw):
    import gymnasium as gym
    from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    PegInsertionSideEnv._clearance = clearance
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
        c = get_task("peg")["camera"]
        extra["sensor_configs"] = {"base_camera": dict(
            pose=sapien_utils.look_at(eye=c["eye"], target=c["target"]),
            width=IMAGE_SIZE, height=IMAGE_SIZE, fov=c["fov"])}
    env = gym.make("PegInsertionSide-v1", num_envs=num_envs, obs_mode=mode,
                   control_mode=control, sim_backend="physx_cuda",
                   max_episode_steps=max_episode_steps, reward_mode=reward_mode,
                   reconfiguration_freq=reconfig_freq, robot_uids=robot, **extra)
    env = ManiSkillVectorEnv(env, auto_reset=True,
                             ignore_terminations=ignore_terminations)
    return KeypointPegInsertShared(env, num_kp=num_kp, reward=reward, w_kp=w_kp,
                                   seed=seed, ap2ap_fields=ap2ap_fields,
                                   kp_template="nominal", last_action=last_action,
                                   sig_obs=sig_obs, priv_obs=priv_obs)
