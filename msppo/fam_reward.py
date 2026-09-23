"""The unified pick-and-place reward from `<private-repo>/docs/SPEC_PRETRAINED_EXECUTOR_20260816.md` §3.

ONE reward for the whole grasp-and-place family (PickCube / PickSingleYCB /
StackCube / LiftPegUpright / PegInsertionSide). Task semantics live in the GOAL,
not in the reward, which is the whole point of the pretrained-executor design:

    r = w_r · r_reach      (1 - tanh(k·‖tcp - obj‖))
      + w_g · r_grasp      (is_grasped)
      + w_t · r_transport  (is_grasped · (1 - tanh(k·kp_dist)))
      + w_s · r_settle     (placed · static)
      + B   · placed
      family success (TRAINING ONLY): kp_dist <= eps AND the object is static

`kp_dist` is the MEAN per-point distance between the object's canonical keypoints
carried by its current pose and the same points at the goal pose. It is the only
task-dependent quantity and it is supplied by the keypoint env, so this module
knows nothing about pegs, cubes or holes.

THREE THINGS THIS FILE IS DELIBERATELY NOT
──────────────────────────────────────────
1.  NOT the native reward. Each task's own `compute_dense_reward` and `success`
    are demoted to EVAL-ONLY (spec §3). Pretraining never touches them, so a
    family number and a native number can never be silently mixed.

2.  NOT a centre-distance reward. `peg_kp_env._pose_reach_reward` documents why
    the peg's TRANSPORT term uses centre distance: keypoint-mean secretly
    requires an orientation match, and after grasp+lift the object is rotated, so
    kp-mean never fell below threshold and success stayed 0.000 while return hit
    24. Here the requirement is the opposite -- LiftPegUpright's goal IS an
    orientation and peg's is 6-DOF -- so kp_dist is correct and the guard is
    `eps` being generous enough (0.05 m, from Dex4D's `goal_obj_dist <= 0.05`)
    rather than the metric being weakened.

3.  NOT free of `r_settle`. Spec §3: our StackCube reward once had a
    release-valley -- "stack RL always fails" turned out to be a hole in the
    reward, fixed by an ungrasp term. `r_settle` pays for arriving AND stopping.
    The end-of-episode semantics that genuinely differ (stack must RELEASE,
    PickCube must HOLD) stay out of here and are Stage 3's job.

The weights and `eps` are NEW FREE PARAMETERS -- spec §10 risk #3 names them as
such. They are tuned on PickCube first (cheapest, and the only task with a
verified G1 pass) before any mixed training; that is gate T-G1.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FamilyRewardCfg:
    """Defaults are the STARTING POINT for the T-G1 tuning pass, not a result."""
    w_reach: float = 1.0
    w_grasp: float = 1.0
    w_transport: float = 2.0
    w_settle: float = 2.0
    bonus: float = 5.0
    k: float = 5.0             # tanh sharpness, as every ManiSkill dense reward
    eps: float = 0.05          # Dex4D `goal_obj_dist <= 0.05`
    static_k: float = 5.0      # tanh sharpness on the object's speed
    act_penalty: float = 0.001
    # Max attainable is w_reach + w_grasp + w_transport + w_settle + bonus.
    # Normalising keeps the value function's scale comparable across tasks, which
    # matters because ONE critic serves all five.

    @property
    def rmax(self) -> float:
        return (self.w_reach + self.w_grasp + self.w_transport
                + self.w_settle + self.bonus)


def family_reward(kp_dist, tcp_to_obj, is_grasped, obj_speed, last_action,
                  cfg: FamilyRewardCfg = FamilyRewardCfg()):
    """All args [B]; `last_action` is [B,A]. Returns (reward [B], placed [B]).

    Every term is gated on `is_grasped` where physically required: transport
    without a grasp is the object being pushed, which is not what this family
    does, and paying for it invites the policy to bulldoze.
    """
    g = is_grasped.float()
    r = cfg.w_reach * (1 - torch.tanh(cfg.k * tcp_to_obj.clamp(max=1.0)))
    r = r + cfg.w_grasp * g
    r = r + cfg.w_transport * g * (1 - torch.tanh(cfg.k * kp_dist.clamp(max=1.0)))
    placed = (kp_dist <= cfg.eps).float()
    static = 1 - torch.tanh(cfg.static_k * obj_speed)
    r = r + cfg.w_settle * placed * static
    r = r + cfg.bonus * placed
    r = r - cfg.act_penalty * (last_action ** 2).sum(-1)
    return (r / cfg.rmax).clamp(-1.0, 1.0), placed


def family_success(kp_dist, obj_speed, cfg: FamilyRewardCfg = FamilyRewardCfg(),
                   static_thresh: float = 0.02):
    """The FAMILY success used during training. Never reported as a result --
    every published number is the task's NATIVE `success_once` (spec §3, and the
    project rule that training metrics never enter a table)."""
    return (kp_dist <= cfg.eps) & (obj_speed <= static_thresh)
