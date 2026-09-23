"""Canonical object keypoints and the paired object/goal construction.

Mirrors Dex4D: one fixed set of keypoints in the object's local frame is carried
to world by the CURRENT object pose and by the GOAL pose, so keypoint i of the
object and keypoint i of the goal are the same material point under two SE(3)
transforms (xarm6_leap_hand_ap2ap.py:992-993). Correspondence is by construction.

The canonical set is FIXED, not resampled per episode. The earlier project lost
StackCube 88 -> 7 under `reconfig_freq=1` because a per-reconfiguration keypoint
subset drifted and silently sent the policy out of distribution; a cube never
changes shape here, so there is nothing to resample.
"""
from __future__ import annotations

import torch

CUBE_HALF = 0.02


def canonical_keypoints(k: int = 128, half: float = CUBE_HALF,
                        seed: int = 0, device="cuda") -> torch.Tensor:
    """[k,3] points on the cube: the 8 corners, then a deterministic surface sample."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    corners = torch.tensor([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
                           dtype=torch.float32) * half
    n = max(k - len(corners), 0)
    if n:
        p = torch.rand((n, 3), generator=g) * 2 - 1          # in the cube
        face = torch.randint(0, 3, (n,), generator=g)        # push each onto a face
        sign = torch.randint(0, 2, (n,), generator=g) * 2 - 1
        p[torch.arange(n), face] = sign.float()
        pts = torch.cat([corners, p * half], dim=0)
    else:
        pts = corners[:k]
    return pts.to(device)


def quat_to_R(q: torch.Tensor) -> torch.Tensor:
    """SAPIEN quaternions are (w,x,y,z). q: [...,4] -> [...,3,3]."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)


def to_world(local_kp: torch.Tensor, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """local_kp [K,3]; p [B,3]; q [B,4] -> [B,K,3]."""
    R = quat_to_R(q)                                    # [B,3,3]
    return torch.einsum("bij,kj->bki", R, local_kp) + p[:, None, :]


def keypoint_distance(object_kp: torch.Tensor, goal_kp: torch.Tensor) -> torch.Tensor:
    """Dex4D's goal_obj_dist: mean over keypoints of the pairwise distance.

    No separate rotation term is needed anywhere downstream -- keypoints already
    encode orientation (xarm6_leap_hand_ap2ap.py:1388).
    """
    return torch.norm(goal_kp - object_kp, p=2, dim=-1).mean(dim=-1)
