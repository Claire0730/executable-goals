"""Keypoint wrapper for PegInsertionSide-v1, and the pose-reaching reward.

Separate from `kp_env.py`, which is hardcoded to StackCube in 36 places. Nothing
here touches the StackCube line.

WHY THIS EXISTS. Row 1 (`msppo/peg_ppo.py`: stock 43-D observation, stock staged
insertion reward, official recipe) measures **0.116 +/- 0.051** over 3 seeds. The
sibling workspace reaches **0.92** at the SAME num_envs (256), SAME budget (30M),
SAME clearance (0.01) and the SAME native success test. The difference is the
reward: theirs is a pose-reaching reward whose dominant term drives the object's
keypoints onto the GOAL's keypoints
(`integration/retrain/franka_ap2ap.py:318-330`), while the stock reward's final
insertion stage is sparse. This file makes that a one-variable comparison.

THREE THINGS THE CUBE CODE CANNOT BE REUSED FOR, all read from the env:

  * Peg geometry is RANDOMISED per parallel env in `_load_scene`
    (`peg_insertion_side.py:114-115`): half-length U(0.085, 0.125), half-radius
    U(0.015, 0.025). So the canonical keypoint set is **[B,K,3], per env**, not
    the fixed [K,3] a 2 cm cube allows. `msppo/kp.py`'s "the canonical set is
    FIXED, not resampled" is true for the cube and false here.
  * The goal is the peg pose putting the peg's HEAD at the hole centre, which the
    env already exposes as `goal_pose` (`:263-267`). Success checks head position
    in the hole frame only (`:269-283`) -- no static, no release.
  * `scene_kp` is the RECTANGULAR hole rim (`_build_box_with_hole` frames the
    opening with four boxes, so the opening is a rectangle of half-width
    `box_hole_radii`). Four rim corners are NOT collinear, so a pose solved from
    them is well conditioned -- unlike the peg's own traced points, which
    `msppo/peg_traceerr.py` measured to be exactly collinear.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
from mani_skill.utils.structs.pose import Pose
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# Stock 43-D layout, dumped from obs_mode="state_dict" and verified:
#   qpos(9) qvel(9) tcp_pose(7) peg_pose(7) peg_half_size(3)
#   box_hole_pose(7) box_hole_radius(1)
PROPRIO = slice(0, 25)
PEG_P, PEG_Q = slice(25, 28), slice(28, 32)
PEG_HALF = slice(32, 35)
HOLE_P, HOLE_Q = slice(35, 38), slice(38, 42)
TCP_P = slice(18, 21)


def unit_box_keypoints(k: int, seed: int = 0) -> torch.Tensor:
    """[k,3] points on the UNIT box [-1,1]^3: 8 corners then a surface sample.

    Kept in unit space so it can be scaled by each env's own half sizes. Fixed
    seed, generated once -- resampling per episode is what silently sent a
    StackCube policy out of distribution (88% -> 7%) in the earlier project.
    """
    g = torch.Generator().manual_seed(seed)
    corners = torch.tensor([[x, y, z] for x in (-1., 1.) for y in (-1., 1.)
                            for z in (-1., 1.)], dtype=torch.float32)
    n = max(k - 8, 0)
    if not n:
        return corners[:k]
    p = torch.rand((n, 3), generator=g) * 2 - 1
    face = torch.randint(0, 3, (n,), generator=g)
    sign = torch.randint(0, 2, (n,), generator=g) * 2. - 1.
    p[torch.arange(n), face] = sign
    return torch.cat([corners, p], dim=0)


def unit_rect_rim(k: int, seed: int = 0) -> torch.Tensor:
    """[k,3] points on the rim of a unit rectangle in the y-z plane (x = 0).

    The hole's opening; x is the insertion axis. Corners first so that even a
    heavily masked subset keeps points that are not collinear.
    """
    g = torch.Generator().manual_seed(seed)
    corners = torch.tensor([[0., y, z] for y in (-1., 1.) for z in (-1., 1.)],
                           dtype=torch.float32)
    n = max(k - 4, 0)
    if not n:
        return corners[:k]
    t = torch.rand((n,), generator=g) * 2 - 1
    edge = torch.randint(0, 4, (n,), generator=g)
    pts = torch.zeros((n, 3))
    pts[:, 1] = torch.where(edge < 2, t, torch.where(edge == 2, -1., 1.))
    pts[:, 2] = torch.where(edge < 2, torch.where(edge == 0, -1., 1.), t)
    return torch.cat([corners, pts], dim=0)


def quat_to_R(q: torch.Tensor) -> torch.Tensor:
    """SAPIEN (w,x,y,z) -> [...,3,3]."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def to_world(canon: torch.Tensor, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """canon [B,K,3] (already scaled) + pose -> [B,K,3] world points."""
    return torch.einsum("bij,bkj->bki", quat_to_R(q), canon) + p[:, None, :]


class KeypointPegInsert:
    """Composition, not gym.Wrapper: ManiSkillVectorEnv is a VectorEnv and fails
    gym.Wrapper's isinstance check -- same reason `KeypointStackCube` composes."""

    def __init__(self, env, num_kp=64, reward="stock", w_kp=0.125, seed=0,
                 device="cuda", ap2ap_fields=True, kp_template="nominal",
                 last_action=False, sig_obs=False, priv_obs=False, shaping_true_goal=False):
        assert reward in ("stock", "pose_reach"), reward
        self.env, self.base, self.device = env, env.unwrapped, device
        self.num_kp, self.reward_kind, self.w_kp = num_kp, reward, w_kp
        self.num_envs = self.base.num_envs

        # `SUCCESS_TH` at `franka_ap2ap.py:36` is 0.04. The placed gate and the
        # hold term both key off it, and it is what breaks the lift trap: lift
        # tops out at 1.5 while placed pays 5.0 + up to 2.0 for holding still.
        self.success_th = 0.04
        self._last_action = torch.zeros(
            (self.base.num_envs, env.single_action_space.shape[0]), device=device)
        self._unit_peg = unit_box_keypoints(num_kp, seed).to(device)
        self._unit_rim = unit_rect_rim(num_kp, seed).to(device)

        assert kp_template in ("nominal", "true"), kp_template
        self.ap2ap_fields, self.kp_template = ap2ap_fields, kp_template
        # In perception mode the observation space is a DICT and has no `.shape`.
        # This never arose while peg had only its own state-mode line; the shared
        # stack (msppo/peginsert_kp_env.py) needs the RGB-D path. In state mode
        # the assert below proves nothing changed.
        _sp = self.env.single_observation_space
        base_dim = _sp.shape[-1] if _sp.shape is not None else 43
        assert base_dim == 43, f"stock peg observation is {base_dim}, expected 43"
        if ap2ap_fields:
            base_dim += 20      # is_grasped 1 + tcp_to_obj 3 + obj_vel 6
                                # + goal_pose 7 + obj_to_goal 3
        # Dex4D's teacher takes the LAST ACTION as its own input branch
        # (Fig.2(a)); this observation never carried one. OPT-IN, because turning
        # it on changes obs_dim and every checkpoint in runs_rl/peg_kp_* and
        # peg_flat_* was trained without it.
        self.last_action_obs = last_action
        if last_action:
            base_dim += env.single_action_space.shape[0]

        # NOISE-SCALE CHANNEL (2026-08-26). 4 dims, appended AFTER every other
        # non-keypoint block and BEFORE the keypoints, because `kp_slice` and
        # `kp_teacher.branch_slices` both require the keypoints to stay last.
        # `PlainActorCritic` takes `obs_dim - num_kp*6` as its feature width, so
        # these 4 dims reach it with no head change. Opt-in: turning it on moves
        # `kp_slice` and every existing checkpoint was trained without it.
        self.sig_obs = bool(sig_obs)
        if self.sig_obs:
            base_dim += 4
        # PRIVILEGED BLOCK for an asymmetric critic: true obj_p + true goal_p
        # from the CLEAN observation. Sits after sig, before the keypoints.
        self.priv_obs = bool(priv_obs)
        self.shaping_true_goal = bool(shaping_true_goal)
        self.priv_slice = slice(base_dim, base_dim + 6) if self.priv_obs else None
        if self.priv_obs:
            base_dim += 6

        # obj_kp + goal_kp appended, mirroring the StackCube teacher's layout so
        # the same PairedKeypointActorCritic / PlainActorCritic heads apply.
        self.kp_slice = slice(base_dim, base_dim + num_kp * 6)
        total = base_dim + num_kp * 6
        self.single_observation_space = gym.spaces.Box(
            -np.inf, np.inf, (total,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (self.num_envs, total), dtype=np.float32)
        self.single_action_space = self.env.single_action_space
        self.action_space = self.env.action_space

    # ------------------------------------------------------------ keypoints --
    # Mid-points of the randomisation ranges in `peg_insertion_side.py:114-115`:
    # half-length U(0.085, 0.125) -> 0.105, half-radius U(0.015, 0.025) -> 0.020.
    NOMINAL_HALF = (0.105, 0.020, 0.020)

    def _canon_peg(self, base_obs):
        """Canonical peg points.

        `nominal` (default) matches the reference: their `kp64` is `(64, 3)` --
        SHARED across envs, not per-env -- and measures 209.7 x 40 x 40 mm, i.e. a
        FIXED template at the distribution mid-point, even though the real peg is
        170-250 mm long. Their keypoints therefore encode POSE ONLY.

        `true` scales by each env's actual `peg_half_sizes`, so the points encode
        pose AND size. That is strictly more information and is arguably the
        better representation, but it is NOT what the 0.92 run trained on, so it
        is an ablation rather than the reproduction.
        """
        if self.kp_template == "nominal":
            half = torch.tensor(self.NOMINAL_HALF, device=base_obs.device,
                                dtype=base_obs.dtype).expand(base_obs.shape[0], 3)
        else:
            half = base_obs[:, PEG_HALF]
        return self._unit_peg[None] * half[:, None, :]

    def _canon_rim(self, base_obs):
        r = base_obs[:, 42:43]                       # box_hole_radius
        s = torch.cat([torch.ones_like(r), r, r], dim=-1)
        return self._unit_rim[None] * s[:, None, :]

    def _goal_from_obs(self, base_obs):
        """The goal peg pose (p, q), DERIVED FROM THE OBSERVATION VECTOR.

        `self.base.goal_pose` is a live property, and ManiSkillVectorEnv
        auto-resets before returning, so for any env that just finished it
        already holds the NEXT episode's goal while `base_obs` holds the one that
        finished. Reading the live property mixed two episodes together in both
        the observation and the reward.

        `peg_insertion_side.py:267` defines
            goal_pose = box_hole_pose * peg_head_offsets.inv()
        and `:124-126` makes peg_head_offsets a PURE TRANSLATION [L, 0, 0] with
        L = peg_half_sizes[:, 0]. Both box_hole_pose and peg_half_sizes are in the
        43-D vector, so the goal is exactly reconstructible from it:
            q = hole_q,  p = hole_p + R(hole_q) @ [-L, 0, 0]
        Checked against the live property to 0.00 um right after reset.
        """
        q = base_obs[:, HOLE_Q]
        off = torch.zeros_like(base_obs[:, HOLE_P])
        off[:, 0] = -base_obs[:, PEG_HALF][:, 0]
        p = base_obs[:, HOLE_P] + torch.einsum("bij,bj->bi", quat_to_R(q), off)
        return p, q

    def keypoints(self, base_obs):
        """(obj_kp, goal_kp, scene_kp), all [B,K,3] in world coordinates.

        Read from the OBSERVATION vector rather than the live sim: the vector env
        auto-resets before returning, so `self.base.peg.pose` already holds the
        NEXT episode for any env that finished. This is the same timing hazard
        `kp_env.keypoints` documents.
        """
        canon = self._canon_peg(base_obs)
        obj = to_world(canon, base_obs[:, PEG_P], base_obs[:, PEG_Q])
        gp, gq = self._goal_from_obs(base_obs)
        goal = to_world(canon, gp, gq)
        scene = to_world(self._canon_rim(base_obs),
                         base_obs[:, HOLE_P], base_obs[:, HOLE_Q])
        return obj, goal, scene

    def _extra_fields(self, base_obs):
        """The AP2AP observation fields the stock 43-D vector does NOT carry.

        `franka_stock_tasks.py:50` reuses `FrankaAP2APEnv._get_obs_extra` verbatim
        ("418-dim extra dict"), whose fields are is_grasped / tcp_pose /
        tcp_to_obj / obj_pose / obj_vel / goal_pose / obj_to_goal / obj_kp /
        goal_kp (`franka_ap2ap.py:305-315`). Of those, the stock vector already
        has tcp_pose and obj_pose; these five it does not. The important one is
        `goal_pose`: the stock vector gives box_hole_pose and peg_half_size, from
        which the goal is DERIVABLE but not GIVEN, and their 0.92 run hands it
        over directly.
        """
        gp, gq = self._goal_from_obs(base_obs)
        return torch.cat([
            self.base.agent.is_grasping(self.base.peg).float()[:, None],   # 1
            base_obs[:, PEG_P] - base_obs[:, TCP_P],                       # 3 tcp_to_obj
            self.base.peg.linear_velocity, self.base.peg.angular_velocity,  # 6 obj_vel
            gp, gq,                                                         # 7 goal_pose
            gp - base_obs[:, PEG_P],                                        # 3 obj_to_goal
        ], dim=-1)

    # TEACHER-SIDE OBSERVATION NOISE. Rewriting the pose slices here makes the
    # stock block, the AP2AP block and the keypoints all consistent, because all
    # three are derived from `obs` below. Default None = exactly the old path.
    def _priv_row(self, clean):
        gp, _ = self._goal_from_obs(clean)
        return torch.cat([clean[:, PEG_P], gp], dim=-1)

    def _sig_row(self, b):
        """[b,4] noise-scale channel; all zeros when no injector is attached, so
        a sig_obs run with no noise is a clean control rather than an error."""
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
        n.tick(getattr(self, "noise_horizon", 0))
        return n.apply(obs)

    def _augment(self, obs):
        clean = obs
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
        if self.priv_obs:
            parts.append(self._priv_row(clean))
        parts += [obj.reshape(b, -1), goal.reshape(b, -1)]
        return torch.cat(parts, dim=-1), obj, goal

    # --------------------------------------------------------------- reward --
    def _pose_reach_reward(self, obs, obj, goal, info):
        """Ported from `integration/retrain/franka_ap2ap.py:318-343`.

        THE GOAL DISTANCE IS CENTRE DISTANCE, NOT KEYPOINT MEAN. Their code
        comments on exactly why (`:241-246`):

            "Keypoint-mean would secretly require orientation match -> after
             grasp+lift the cube is rotated, so kp-mean never < thresh even at
             correct position. This was why success stayed 0 while return hit 24."

        A first version here used the keypoint mean and reproduced that failure
        precisely: three seeds with reward climbing to 0.4 and success pinned at
        0.000. `goal` is still built from keypoints for the OBSERVATION -- it is
        only the reward's distance that is centre-to-centre.

        Terms and weights follow the source; the maximum is 25, hence /25.
        """
        tcp = obs[:, TCP_P]
        d_tcp = (obs[:, PEG_P] - tcp).norm(dim=-1).clamp(max=1.0)
        grasped = self.base.agent.is_grasping(self.base.peg).float()
        gd = (self._goal_from_obs(obs)[0] - obs[:, PEG_P]).norm(dim=-1)
        lifted = (obs[:, PEG_P][:, 2] - obs[:, PEG_HALF][:, 2]).clamp(0.0, 0.3)
        qvel = self.base.agent.robot.get_qvel()
        static = 1 - torch.tanh(5 * qvel.norm(dim=-1))

        gd = gd.clamp(max=1.0)
        r = 1 - torch.tanh(5 * d_tcp)                                  # reach
        r = r + grasped * 0.5                                          # grasp
        r = r + grasped * (lifted * 5.0)                               # lift
        r = r + grasped * (1 - torch.tanh(5 * gd)) * 2.0               # to-goal
        placed = (gd <= self.success_th).float()
        r = r + placed * static * 2.0                                  # hold at goal
        r = r + placed * 5.0                                           # placed
        r = r + info["success"].float() * 10.0                         # native insert
        r = r - 0.001 * (self._last_action ** 2).sum(-1)               # action penalty
        return torch.clamp(r, -5.0, 25.0) / 25.0

    # ------------------------------------------------------------------ gym --
    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        # A reset episode has no previous action; carrying the old one over would
        # hand the policy a step from the PREVIOUS episode at t=0.
        self._last_action = torch.zeros_like(self._last_action)
        aug, _, _ = self._augment(obs)
        return aug, info

    def step(self, action):
        self._last_action = action.detach()
        obs, rew, term, trunc, info = self.env.step(action)
        aug, obj, goal = self._augment(obs)
        gd = (obj - goal).norm(dim=-1).mean(-1)
        info["kp_dist"] = gd
        # `ppo.py` logs a grasp rate from `is_cubeA_grasped`, a StockCube field the
        # peg env does not have, so every peg run so far logged `grasp nan` -- i.e.
        # three rounds of 0.000 with no way to see whether the policy was even
        # picking the peg up. Publish it here under both names.
        gr = self.base.agent.is_grasping(self.base.peg)
        info["is_peg_grasped"] = info["is_cubeA_grasped"] = gr
        if self.reward_kind == "pose_reach":
            rew = self._pose_reach_reward(obs, obj, goal, info)
        elif self.w_kp:
            # NOT the stock reward: stock PLUS a keypoint-distance bonus. Their
            # 0.92 run uses the stock reward untouched (the MRO resolves
            # compute_dense_reward to PegInsertionSideEnv's), so a comparison
            # against it must pass --w-kp 0 and take this branch out.
            rew = rew + self.w_kp * (1 - torch.tanh(5.0 * gd))
        return aug, rew, term, trunc, info

    def close(self):
        self.env.close()


def make_peg_kp_env(num_envs=256, num_kp=64, reward="stock", w_kp=0.125, seed=0,
                    control="pd_joint_delta_pos", max_episode_steps=100,
                    clearance=0.01, reconfig_freq=0,
                    reward_mode="normalized_dense", ap2ap_fields=True,
                    ignore_terminations=False, kp_template="nominal",
                    robot="panda", last_action=False):
    PegInsertionSideEnv._clearance = clearance     # read inside _load_scene
    # robot_uids: the stock default is `panda_wristcam`, the reference passes
    # `panda` (`franka_stock_tasks.py`). Same 9-DoF proprio either way, but the
    # camera mount is an extra link on a contact-rich task, so match it.
    env = gym.make("PegInsertionSide-v1", num_envs=num_envs, obs_mode="state",
                   control_mode=control, sim_backend="physx_cuda",
                   max_episode_steps=max_episode_steps, reward_mode=reward_mode,
                   reconfiguration_freq=reconfig_freq, robot_uids=robot)
    # ignore_terminations=True is their `--no-partial_reset`: an insertion no
    # longer ends the episode, so the agent keeps being paid for holding it and
    # the value function sees past the success step.
    env = ManiSkillVectorEnv(env, auto_reset=True,
                             ignore_terminations=ignore_terminations)
    return KeypointPegInsert(env, num_kp=num_kp, reward=reward, w_kp=w_kp,
                             seed=seed, ap2ap_fields=ap2ap_fields,
                             kp_template=kp_template, last_action=last_action)
