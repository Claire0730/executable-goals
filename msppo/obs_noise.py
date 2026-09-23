"""Pose-level observation noise for TEACHER training.

WHY IT LIVES AT `base_obs` AND NOT ON THE KEYPOINTS. Every pose-derived field in
these envs is computed from the raw state vector -- `keypoints()` reads
`base_obs[:, OBJ_P/OBJ_Q]`, `_extra_fields()` reads the same and derives the
goal, and the stock block IS `base_obs`. Perturbing only the 384 keypoint dims
would leave the true `obj_pose` and `goal_pose` sitting in the stock and AP2AP
blocks for the policy to read, and the arm would be a no-op. Rewriting the pose
slices instead makes all three consistent by construction.

WHY POSE-LEVEL AND NOT PER-POINT IID. The student consumes a pose that its own
weighted Kabsch already solved, so its error is a RIGID displacement: the cloud
is the right shape in the wrong place. Per-point iid would hand the teacher a
non-rigid cloud, which the student never sees, and a Kabsch would average most
of it away anyway (5 mm iid -> ~2.9 mm of solved-pose error, measured).

WHY THE REWARD IS SAFE. With `reward="stock"` the reward is computed inside
ManiSkill from the true simulator state and never touches `base_obs`. So the
observation is corrupted while the reward stays honest -- the property the whole
experiment rests on, since it is what forces the policy to find the target
rather than trust the input.

MAGNITUDES ARE MEASURED, NOT CHOSEN. <private-repo>/experiments/20260824_objerr/o3_chamfer.py,
N=256 seed 999, on the perception path the student actually runs:

    task        Chamfer   translation   rotation(as Chamfer mm)   r_rms
    peginsert   16.8 mm      22.2            13.1                 57.4
    stack       11.7 (p50)   21.9            11.2                 22.3

Rotation is injected as a DISPLACEMENT, not an angle: the objects are symmetric
(box template, cylindrical peg), so an angle has no unique meaning while a cloud
displacement does -- which is also why the first two attempts at measuring this
(quaternion angle 113 deg, pointwise cloud 70 mm) were both void. The angle that
produces displacement d on a cloud of radius r is 2*asin(d / 2r).

⚠️ v1 IS ISOTROPIC. The goal error is drawn as a random direction of measured
magnitude, not replayed from the scene-relative relbank (frac / lat_v / lat_w /
dq) the student arms use. That decomposition is the more faithful model and is
the intended v2; isotropic matches what `kp_env.goal_error_range` already does,
so v1 stays consistent with existing project practice rather than inventing a
third convention.
"""
from __future__ import annotations

import math
import torch


def _qmul(a, b):
    """Hamilton product, SAPIEN (w,x,y,z). Same convention as
    `task_student_env._qmul`; kept identical so the two sides cannot drift."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw], dim=-1)


def _gauss_quat(n, deg_rms, gen, device):
    """[n,4] rotations drawn as an isotropic Gaussian ROTATION VECTOR.

    w ~ N(0, sigma^2 I) in so(3); angle = |w|, axis = w/|w|. This is the
    concentrated Gaussian on SO(3) and yields a DISTRIBUTION of angles, most of
    them small. The first version rotated by a FIXED angle about a random axis,
    i.e. a shell, so no episode was ever nearly correct.

    Composed multiplicatively. Adding noise to quaternion components would break
    the unit norm and is not a rotation at all.
    """
    if deg_rms <= 0:
        q = torch.zeros((n, 4), device=device)
        q[:, 0] = 1.0
        return q
    sig = math.radians(deg_rms) / math.sqrt(3.0)     # per-axis, so |w|_rms = deg_rms
    w = torch.randn((n, 3), generator=gen, device=device) * sig
    ang = w.norm(dim=-1, keepdim=True)
    axis = w / ang.clamp_min(1e-9)
    return torch.cat([torch.cos(ang / 2), axis * torch.sin(ang / 2)], dim=-1)


class PoseNoise:
    """Per-episode SE(3) perturbation of the object and goal pose slices."""

    def __init__(self, num_envs, slices, device="cuda", seed=0,
                 obj_mm=0.0, obj_rot_mm=0.0, goal_mm=0.0, goal_rot_mm=0.0,
                 r_rms_mm=1.0):
        self.n, self.sl, self.device = num_envs, slices, device
        self.gen = torch.Generator(device=device).manual_seed(seed + 90001)
        self.obj_mm, self.goal_mm = obj_mm, goal_mm
        # displacement -> angle, on a cloud of this radius
        ang = lambda d: math.degrees(2 * math.asin(min(1.0, d / (2 * max(r_rms_mm, 1e-6)))))
        self.obj_deg, self.goal_deg = ang(obj_rot_mm), ang(goal_rot_mm)
        self.dt_o = torch.zeros((num_envs, 3), device=device)
        self.dq_o = torch.zeros((num_envs, 4), device=device); self.dq_o[:, 0] = 1
        self.dt_g = torch.zeros((num_envs, 3), device=device)
        self.dq_g = torch.zeros((num_envs, 4), device=device); self.dq_g[:, 0] = 1
        self.resample()

    def _draw(self, k, mm, deg):
        """GAUSSIAN, not a fixed-magnitude shell.

        The first version drew a random DIRECTION and a FIXED length, so every
        episode carried exactly `mm` of error and the teacher never saw a
        nearly-correct goal. That is the mistake the student line already paid
        for: `mt4_ramp_s0` (a ~ U(0,1), so half the batch stays near-clean) beat
        both fixed-scale arms, `mt4_ns_n50` and `mt4_ns_n100`, on row 4 --
        keeping some easy episodes is what gives the policy a gradient to climb.

        `mm` is the RMS radius, so per-axis sigma = mm/sqrt(3); mean radius is
        then 0.92*mm and a real fraction of episodes land far inside it.
        """
        if mm <= 0:
            t = torch.zeros((k, 3), device=self.device)
        else:
            t = torch.randn((k, 3), generator=self.gen, device=self.device) * (
                mm / math.sqrt(3.0) / 1000.0)
        return t, _gauss_quat(k, deg, self.gen, self.device)

    def resample(self, idx=None):
        """One draw per episode. Re-drawing every step would average the error
        away and train against an easier problem than deployment."""
        if idx is None:
            idx = torch.arange(self.n, device=self.device)
        k = int(idx.numel())
        if k == 0:
            return
        t, q = self._draw(k, self.obj_mm, self.obj_deg)
        self.dt_o[idx], self.dq_o[idx] = t, q
        t, q = self._draw(k, self.goal_mm, self.goal_deg)
        self.dt_g[idx], self.dq_g[idx] = t, q

    def tick(self, horizon):
        """Called once per step. Teacher training uses ignore_terminations=1, so
        every env resets together at the horizon and a counter is an exact
        episode boundary -- no need to thread `done` through two env classes."""
        self._t = getattr(self, "_t", 0) + 1
        if horizon and self._t % horizon == 0:
            self.resample()

    def apply(self, base_obs):
        """Return a COPY of base_obs with the object and goal pose slices moved."""
        if not (self.obj_mm or self.goal_mm or self.obj_deg or self.goal_deg):
            return base_obs
        o = base_obs.clone()
        for key_p, key_q, dt, dq in (("obj_p", "obj_q", self.dt_o, self.dq_o),
                                     ("goal_p", "goal_q", self.dt_g, self.dq_g)):
            sp, sq = self.sl.get(key_p), self.sl.get(key_q)
            if sp is None:
                continue
            o[:, sp] = base_obs[:, sp] + dt
            if sq is not None:
                o[:, sq] = _qmul(dq, base_obs[:, sq])
        return o
