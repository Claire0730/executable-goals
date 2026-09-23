"""Teacher observation noise, v2 -- anisotropic goal, time-correlated object, sig channel.

v1 (`msppo/obs_noise.py`) stays byte-identical so the archived arms reproduce.
Everything below is what v1 got wrong, each backed by a measurement:

1. v1 IS ISOTROPIC. Measured on the planner's own output (relbank_visobj,
   N=242-256, approach frame u = obj->true goal, v = u x z, w = u x v):

       task        mu_u(mm)   sd_u    sd_v    sd_w   tail |z|>3
       pickcube      -47.2    74.3    45.3    25.8     1.17 %
       liftpeg       -16.1    16.0    68.5    46.8     0.26 %
       peginsert     -47.1    91.1    29.7    19.1     0.39 %
       stack         -66.7    78.5    21.0    19.8     1.79 %

   Isotropic is wrong by 2-5x in every direction (liftpeg the other way round).

2. v1 IS ZERO-MEAN. `mu_u` is negative for all four -- the planner systematically
   UNDERSHOOTS. That is the largest and most learnable component of the error,
   and a zero-mean draw deletes it. Evidence it matters: `mt4_ns_n100_s0` scored
   row3 0.000 / row4 0.875 on liftpeg by learning to invert exactly this bias.
   mu IS injected; it is NOT put in the sig channel (it is a learnable constant,
   so telling the policy would hand it the answer and remove the need to search).

3. v1 REDRAWS ONLY AT THE EPISODE BOUNDARY. Correct for the goal (the planner
   runs once) but wrong for the object, which is re-measured every step. k4
   (N=32, the perception path the student actually runs) measured the relative
   error GROWING within an episode and being highly persistent:

       task        t=0    mid    end   |  ac(1)   effective samples in T
       peginsert  11.5    9.4   10.9   |  0.97       1.5
       pickcube    0.0    3.3    8.5   |  0.81       5.4
       stack       0.0   58.9   94.5   |  0.92       2.1
       liftpeg    19.3   20.4   30.3   |  0.57      13.7

   So the object error is an OU process with a growing sigma, not a constant
   offset. Per-step iid would be averaged away by any filter and would train
   against an easier problem than deployment.

4. v1 IS GAUSSIAN-ONLY. peginsert/liftpeg are Gaussian (kurtosis -0.1/-0.2) but
   stack is not (kurtosis 2.9, 1.79% beyond 3 sigma vs 0.27% nominal, p99/p50 =
   6.4). A tail mixture at the measured outlier rate covers it. The sig channel
   carries the NOMINAL sigma only -- the tail is deliberately not announced,
   which is the realistic case where the reliability estimate is sometimes wrong.

WHY THE GOAL IS PERTURBED THROUGH THE FIXTURE SLICE. peginsert and stack have no
goal field in the stock observation; the goal is DERIVED (`_goal_from_obs`) from
the box hole / cubeB pose. Moving that pose moves the derived goal by the same
translation, so the fixture slice is the correct injection point and keeps the
stock block, the AP2AP block and the keypoints consistent by construction.

WHY THE REWARD IS STILL HONEST. All four stock rewards are computed inside
ManiSkill from the true simulator state and never read `base_obs`. The StackCube
cubeB-displacement penalty added in `stack_kp_env` likewise reads
`self.base.cubeB.pose`, not the observation. Observation corrupted, reward clean.

SIG IS LINEAR IN [0,1], NOT LOG. SPEC_TEACHER_V2 said log-normalise; that is
wrong here. The severity is drawn a ~ U(0,1) so the scale is already linear on
[0,1] by construction, and log(a) diverges as a -> 0. Linear it is.
"""
from __future__ import annotations

import math
import torch

from msppo.obs_noise import _qmul


def gauss_quat_per_env(deg_rms, gen, device):
    """[n] per-env rms angles -> [n,4]. `obs_noise._gauss_quat` takes ONE scalar
    angle for the whole batch; here every env carries its own severity, so the
    scalar version would silently collapse the ramp to its batch mean."""
    n = deg_rms.shape[0]
    sig = torch.deg2rad(deg_rms.clamp_min(0.0)) / math.sqrt(3.0)      # per-axis
    w = torch.randn((n, 3), generator=gen, device=device) * sig[:, None]
    ang = w.norm(dim=-1, keepdim=True)
    axis = w / ang.clamp_min(1e-9)
    return torch.cat([torch.cos(ang / 2), axis * torch.sin(ang / 2)], dim=-1)


# --- measured population models -------------------------------------------
# goal: (mu_u, sd_u, sd_v, sd_w) mm in the approach frame, tail rate, rot deg rms.
# rot deg is the mean |dq| angle of the same relbank; liftpeg's 165 deg is a
# symmetry artefact on a cylindrical peg and is NOT trustworthy.
GOAL_MODEL = {
    "pickcube":  dict(mu_u=-47.2, sd=(74.3, 45.3, 25.8), tail=0.0117, rot=45.6),
    "liftpeg":   dict(mu_u=-16.1, sd=(16.0, 68.5, 46.8), tail=0.0026, rot=45.0),
    "peginsert": dict(mu_u=-47.1, sd=(91.1, 29.7, 19.1), tail=0.0039, rot=35.4),
    "stack":     dict(mu_u=-66.7, sd=(78.5, 21.0, 19.8), tail=0.0179, rot=42.8),
    # pushcube: fitted 2026-08-31 from the DEPLOYED goals (T2K pos-only, seeds 999+997, n=512) in the same
    # approach frame (u = obj -> true goal). mu_u = -64.9 mm is the demo-stop convention made explicit: the
    # trajectories end where the cube ENTERS the 100 mm goal region, so the planner goal undershoots the
    # region centre along the push direction. rot from the solver rotation error (med 11.4 deg -> rms est).
    "pushcube":  dict(mu_u=-64.9, sd=(31.1, 27.6, 8.8), tail=0.0020, rot=14.0),
}

# SIG CHANNEL CORRUPTION (arm T-b). At deployment the policy cannot see the true
# severity; it sees a proxy -- the conformal-calibrated sqrt(tr Sigma_g) of the
# Kabsch Laplace solve -- whose measured rank correlation with the realised
# endpoint error is (k9_laplace.txt, base fit):
#       pickcube +0.356   liftpeg -0.294   peginsert +0.508   stack +0.521
# eta is the additive Gaussian corruption on the normalised readout that
# reproduces exactly that Spearman, solved numerically at N=200k.
# liftpeg's target is NEGATIVE (the proxy is anti-predictive there); the sign is
# applied so the arm is faithful rather than flattering.
# Solved by bisection against the REAL generator (n3_caleta.py, N=80k), not a
# surrogate: the first pass used a stand-in for |delta| and undershot every
# target. Verified on a held-out seed -- all four land on target to 3 decimals.
SIG_ETA = {"pickcube": 0.4637, "liftpeg": 0.6429, "peginsert": 0.2703, "stack": 0.1622,
           "pushcube": 0.35}   # pushcube proxy corruption NOT measured -- placeholder; sig_mode="true" never reads it beyond __init__
SIG_SIGN = {"pickcube": 1.0, "liftpeg": -1.0, "peginsert": 1.0, "stack": 1.0, "pushcube": 1.0}

# object: (sigma at t=0, sigma at horizon) mm, lag-1 autocorrelation, rotation as
# a cloud displacement in mm (o3_chamfer). Straight from k4 above.
# RECALIBRATED 2026-08-26 21:30 (n6_objonly.py, N=24, the perception path the
# student actually runs). The first version took these from k4, which measured
# the RELATIVE error (goal - obj); for StackCube the goal is derived from cubeB,
# so that number mixes cubeA and cubeB perception and cannot be charged to the
# object. It over-injected by 3.3x on stack, 4.8x on liftpeg, 2.7x on peg, and a
# controlled probe (`sc_probe_plain`, same task/head/hyper-parameters, noise off)
# reached grasp 0.485 at it 50 where the noised run was at 0.006 -- the object
# noise, not the head and not the goal noise, was what stopped StackCube.
#
# The value used is the FLUCTUATION, not the total error: the per-episode
# constant is a fixed offset between the perception centroid and the CAD origin,
# and a policy absorbs it. Measured object-only error, mm:
#       task        p50    constant   fluctuation
#       peginsert   21.3     21.0         4.0
#       pickcube    14.6      9.3        11.3
#       stack       13.7     49.1        28.6
#       liftpeg     21.5     20.4         6.3
# Flat (s0 == sT): the measurement gives one fluctuation per task, not a growth
# curve. The growth in the old numbers belonged to the relative error, which has
# now been shown not to be the object.
OBJ_MODEL = {
    "pickcube":  dict(s0=11.3, sT=11.3, rho=0.81, rot_mm=0.0),
    "pushcube":  dict(s0=11.3, sT=11.3, rho=0.81, rot_mm=0.0),   # same cube, same perception path as pickcube
    "liftpeg":   dict(s0=6.3,  sT=6.3,  rho=0.57, rot_mm=0.0),
    "peginsert": dict(s0=4.0,  sT=4.0,  rho=0.97, rot_mm=13.1),
    "stack":     dict(s0=28.6, sT=28.6, rho=0.92, rot_mm=11.2),
}


# DERIVED DIFFERENCE FIELDS. The stock observation of some tasks carries
# differences of the very poses being perturbed, computed by ManiSkill from the
# TRUE state. tcp is never perturbed, so `tcp_to_cubeA = cubeA - tcp` hands the
# true cubeA straight back and the whole injection becomes a no-op. Verified
# against the ManiSkill sources:
#   PegInsertionSide  peg_pose / peg_half_size / box_hole_pose / box_hole_radius
#                     -> NO difference fields, nothing to repair
#   StackCube         tcp_to_cubeA_pos, tcp_to_cubeB_pos, cubeA_to_cubeB_pos
#   PickCube          tcp_to_obj_pos, obj_to_goal_pos
#   LiftPegUpright    tcp_pose / obj_pose only -> nothing to repair
# Each entry is (destination slice, plus key, minus key); keys name point slices
# in `self.sl`, and "tcp_p" is read from the CLEAN observation because the tcp is
# proprioception and is not a perceived quantity.
DERIVED = {
    "peginsert": [],
    "liftpeg": [],
    "stack": [(slice(39, 42), "obj_p", "tcp_p"),
              (slice(42, 45), "goal_p", "tcp_p"),
              (slice(45, 48), "goal_p", "obj_p")],
    "pickcube": [(slice(36, 39), "obj_p", "tcp_p"),
                 (slice(39, 42), "goal_p", "obj_p")],
    "pushcube": [],   # stock obs carries no relative tail (pushcube_kp_env layout)
}
TCP_P = {"peginsert": slice(18, 21), "stack": slice(18, 21),
         "liftpeg": slice(18, 21), "pickcube": slice(19, 22), "pushcube": slice(18, 21)}


def axis_frame(u):
    """(v, w) completing a right-handed orthonormal frame on u. Identical to
    `peg_relbank.axis_frame` -- world +z reference, +x where u is near vertical
    -- so the injection frame and the relbank the parameters were fitted in are
    the same frame. They must not be allowed to drift apart."""
    ez = torch.zeros_like(u); ez[..., 2] = 1.0
    ex = torch.zeros_like(u); ex[..., 0] = 1.0
    ref = torch.where(u[..., 2:3].abs() > 0.9, ex, ez)
    v = torch.cross(u, ref, dim=-1)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return v, torch.cross(u, v, dim=-1)


class PoseNoiseV2:
    """Per-episode anisotropic goal error + per-step OU object error, with the
    scale exposed as a 4-vector for the sig channel.

    `goal_fn(base_obs) -> (gp, gq)` must be the env's own clean goal solver, so
    the approach frame is built from the TRUE geometry rather than a perturbed
    one. `slices` maps obj_p/obj_q/goal_p/goal_q onto the stock vector; for
    peginsert and stack the goal_* entries are the FIXTURE pose (box hole,
    cubeB), which is what the derived goal follows.
    """

    def __init__(self, num_envs, slices, task, goal_fn, horizon,
                 device="cuda", seed=0, goal_scale=1.0, obj_scale=1.0,
                 r_rms_mm=1.0, sig_mode="true", goal_attr_fn=None,
                 curriculum=0.0, a0=0.1, total_ticks=0, tail_scale=1.0,
                 z_reflect=False, goal_rot_scale=1.0, z_mode="reflect"):
        if task not in GOAL_MODEL:
            raise SystemExit(f"no measured noise model for task {task!r}")
        self.n, self.sl, self.device = num_envs, slices, device
        self.task, self.goal_fn, self.horizon = task, goal_fn, int(horizon)
        self.goal_attr_fn = goal_attr_fn
        # CURRICULUM. Anneal the UPPER BOUND of the uniform severity, never the
        # severity itself: a ~ U(0, a_max) with a_max rising a0 -> 1 over the
        # first `curriculum` fraction of training. Annealing the VALUE would end
        # training with every episode at a=1, which destroys the very property
        # that made the ramp arm win -- `mt4_ramp_s0` (a ~ U(0,1)) beat both
        # fixed-scale arms `mt4_ns_n50` and `mt4_ns_n100` on row 4 because some
        # near-clean episodes always remain for PPO to climb.
        # Progress is derived from the internal tick counter, so no change to
        # the training loop is needed.
        self.curriculum, self.a0 = float(curriculum), float(a0)
        self.total_ticks = int(total_ticks)
        self._dp_goal = torch.zeros((num_envs, 3), device=device)
        self.gm, self.om = GOAL_MODEL[task], OBJ_MODEL[task]
        if sig_mode not in ("true", "proxy"):
            raise SystemExit(f"sig_mode {sig_mode!r} not in true/proxy")
        self.sig_mode = sig_mode
        self.eta, self.sign = SIG_ETA[task], SIG_SIGN[task]
        self.tail_rate = float(tail_scale) * self.gm["tail"]   # 0 = no heavy tail
        # Z-REFLECTION (2026-08-27). For a goal DERIVED from a fixture with a
        # resting surface (StackCube: goal = cubeB top), a perceived goal below
        # the true surface is physically unreachable -- the policy pushes the
        # cube into cubeB, never perceives alignment, never releases. Measured:
        # at goal scale 0.3, 23.1% of draws sit >5 mm below the top (only 4.3%
        # above; the asymmetry is the undershoot's z-component). Reflect the z
        # component of the goal error to |dz|: same magnitude distribution, no
        # below-surface draws. Deployment can still produce them; this is a
        # training-time choice to keep those episodes solvable, recorded as such.
        self.z_reflect = bool(z_reflect)
        # z_mode: "reflect" maps dz -> |dz|, which removes the unreachable
        # below-surface draws but converts them into ABOVE-surface ones -- the
        # policy then releases from a height and the cube drops. Measured: with
        # reflection and NO object noise at all, stack caps at 0.746, far under
        # the analytic xy-only bound of 0.926, so that conversion costs real
        # episodes. "clamp" maps dz -> max(dz, 0) instead: below-surface draws
        # land exactly ON the surface, which is both reachable and correct.
        self.z_mode = z_mode
        # GOAL ROTATION SCALE. StackCube's success criterion does not look at
        # orientation at all (xy, z, static, ungrasped), yet the goal keypoints
        # carry it, so a 42.8 deg goal rotation error corrupts both the w_kp
        # keypoint shaping and the observation while corresponding to no real
        # requirement. 0 removes it; 1.0 is the measured value.
        self.goal_rot_scale = float(goal_rot_scale)
        # PER-ENV EPISODE PHASE. The first version counted global steps and
        # resampled everyone at the horizon, which is exact only when all envs
        # reset together (ignore_terminations=1). Under ignore_terminations=0
        # (stack) a success ends an episode early; that env then kept the old
        # draw AND was resampled mid-episode at the next global boundary. Now
        # the env passes its `done` mask and each env keeps its own phase.
        self.t_env = torch.zeros((num_envs,), device=device)
        self.goal_scale, self.obj_scale = float(goal_scale), float(obj_scale)
        self.gen = torch.Generator(device=device).manual_seed(seed + 90002)
        # displacement -> angle on a cloud of this radius (symmetric objects have
        # no unique angle, a cloud displacement does -- see obs_noise.py v1 doc)
        self._ang = lambda d: math.degrees(
            2 * math.asin(min(1.0, d / (2 * max(r_rms_mm, 1e-6)))))
        self._t = 0
        z3 = lambda: torch.zeros((num_envs, 3), device=device)
        self.a = torch.zeros((num_envs,), device=device)      # goal severity
        self.b = torch.zeros((num_envs,), device=device)      # object severity
        self.z_goal, self.tail = z3(), torch.zeros((num_envs, 1), device=device)
        self.s_hat = torch.zeros((num_envs, 1), device=device)   # proxy readout
        self.dq_g = torch.zeros((num_envs, 4), device=device); self.dq_g[:, 0] = 1
        self.d_obj = z3()
        self.dq_o = torch.zeros((num_envs, 4), device=device); self.dq_o[:, 0] = 1
        self.resample()

    # -- per-episode draws ---------------------------------------------------
    def resample(self, idx=None):
        if idx is None:
            idx = torch.arange(self.n, device=self.device)
        k = int(idx.numel())
        if k == 0:
            return
        rnd = lambda *s: torch.rand(s, generator=self.gen, device=self.device)
        nrm = lambda *s: torch.randn(s, generator=self.gen, device=self.device)
        # Uniform ramp. Measured: the ramp arm beat both fixed-scale arms on
        # row 4, because keeping some near-clean episodes gives a gradient to
        # climb. A FIXED severity would also make the sig channel a constant,
        # i.e. zero bits -- the channel only means something if this varies.
        self.a[idx] = rnd(k) * self._a_max()
        self.b[idx] = rnd(k)
        self.z_goal[idx] = nrm(k, 3)
        # 1x or 3x the sd (i.e. 9 Sigma) at the measured outlier rate. stack's
        # p99/p50 = 6.4 against a Gaussian 1.94, so the tail component has to be
        # ~3x the sd, not sqrt(3). mu is NOT multiplied by it -- the tail is a
        # property of the random part only.
        self.tail[idx] = (rnd(k, 1) < self.tail_rate).float() * 2.0 + 1.0
        self.dq_g[idx] = gauss_quat_per_env(
            self.gm["rot"] * self.a[idx] * self.goal_scale * self.goal_rot_scale,
            self.gen, self.device)
        # T-b readout. Built from the REALISED |delta| (that is what the
        # deployment proxy correlates with -- k9 measures Spearman(size, err),
        # not Spearman(size, scale)), then corrupted to the measured quality.
        sd = self.gm["sd"]
        dmag = (self.a[idx] * torch.sqrt(
            (self.z_goal[idx, 0] * sd[0] * self.tail[idx, 0]) ** 2
            + (self.z_goal[idx, 1] * sd[1] * self.tail[idx, 0]) ** 2
            + (self.z_goal[idx, 2] * sd[2] * self.tail[idx, 0]) ** 2))
        ref = float(sum(x * x for x in sd)) ** 0.5 * 1.5      # ~p99 of the clean draw
        r = (dmag / ref).clamp(0, 1)
        # A NEGATIVE target means the proxy is anti-predictive (liftpeg). Flip the
        # READOUT, not the noise -- adding signed noise leaves the correlation
        # positive, which is how the first attempt produced +0.304 for a -0.294
        # target.
        if self.sign < 0:
            r = 1.0 - r
        self.s_hat[idx, 0] = (r + self.eta * nrm(k)).clamp(0, 1)
        # Start the OU AT its stationary sd, not at zero. Starting from zero
        # costs ~1/(1-rho) steps to equilibrate (12 steps at rho=0.92), which
        # would silently under-noise the start of every episode -- wrong for
        # peginsert, whose sigma is flat at 11.5 mm from t=0.
        self.d_obj[idx] = nrm(k, 3) * (self.b[idx, None] * self.om["s0"]
                                       * self.obj_scale / 1000.0 / math.sqrt(3.0))
        self.dq_o[idx] = torch.tensor([1.0, 0, 0, 0], device=self.device)

    def _a_max(self):
        """Upper bound of the severity draw at the current point in training."""
        if self.curriculum <= 0 or self.total_ticks <= 0:
            return 1.0
        f = min(1.0, (self._t / self.total_ticks) / self.curriculum)
        return self.a0 + (1.0 - self.a0) * f

    # -- per-step -----------------------------------------------------------
    def tick(self, done=None):
        """Advance the OU object error one step; redraw at each env's episode
        boundary. `done` (bool [n]) marks envs that just reset. When the env
        passes nothing (or the legacy int horizon), fall back to the global
        horizon count -- exact for ignore_terminations=1, where all envs reset
        together."""
        self._t += 1
        self.t_env += 1
        if torch.is_tensor(done):
            done = done.bool()
            if bool(done.any()):
                idx = torch.nonzero(done).squeeze(-1)
                self.resample(idx)
                self.t_env[idx] = 0
        elif self.horizon and self._t % self.horizon == 0:
            self.resample()
            self.t_env.zero_()
            return
        rho = self.om["rho"]
        s = self._obj_sigma_m()                                     # [n,1] metres
        z = torch.randn((self.n, 3), generator=self.gen, device=self.device)
        # OU: preserves the marginal sd while reproducing the measured ac(1).
        self.d_obj = rho * self.d_obj + math.sqrt(max(1 - rho * rho, 0.0)) * s * z / math.sqrt(3.0)
        if self.om["rot_mm"] > 0:
            deg = self._ang(self.om["rot_mm"]) * self.b * self.obj_scale
            self.dq_o = gauss_quat_per_env(deg, self.gen, self.device)
    def _phase(self):
        """[n] per-env fraction of the episode elapsed."""
        H = max(self.horizon, 1)
        return (self.t_env % H) / H
    def _obj_sigma_m(self):
        """[n,1] object translation sigma in METRES at the current step."""
        mm = self.om["s0"] + (self.om["sT"] - self.om["s0"]) * self._phase()
        return (self.b * (mm * self.obj_scale / 1000.0))[:, None]
    # -- the sig channel -----------------------------------------------------
    def sig_row(self):
        """[n,4] = [obj_t, obj_r, goal_t, goal_r], each in [0,1].

        These are SCALE parameters, not realised magnitudes. Handing the policy
        |delta| would tell it the radius of the sphere the true goal sits on --
        far stronger than anything obtainable at deployment, and it would remove
        the need to search at all."""
        smax = max(self.om["s0"], self.om["sT"], 1e-9) * self.obj_scale / 1000.0
        # obj_scale = 0 (goal-only diagnostic) makes smax 0 -> 0/0 = NaN in the
        # observation. With no object noise the channel is simply zero.
        obj_t = (self._obj_sigma_m() / smax).clamp(0, 1) if smax > 0 else \
            torch.zeros((self.n, 1), device=self.device)
        obj_r = self.b[:, None] * (1.0 if self.om["rot_mm"] > 0 else 0.0)
        if self.sig_mode == "true":
            return torch.cat([obj_t, obj_r, self.a[:, None], self.a[:, None]], dim=-1)
        # T-b: the goal columns carry the corrupted readout. The OBJECT columns
        # are corrupted with the SAME eta, which is an ASSUMPTION, not a
        # measurement -- the object-side proxy quality (tracked-point count,
        # Kabsch conditioning) is listed as an open measurement in
        # SPEC_TEACHER_V2 section 9 and has not been quantified.
        g = self.s_hat
        ob_t = (obj_t + self.eta * torch.randn(obj_t.shape, generator=self.gen,
                                               device=self.device)).clamp(0, 1)
        ob_r = (obj_r + self.eta * torch.randn(obj_r.shape, generator=self.gen,
                                               device=self.device)).clamp(0, 1)             * (1.0 if self.om["rot_mm"] > 0 else 0.0)
        return torch.cat([ob_t, ob_r, g, g], dim=-1)

    # -- injection -----------------------------------------------------------
    def apply(self, base_obs):
        o = base_obs.clone()
        sp, sq = self.sl.get("obj_p"), self.sl.get("obj_q")
        gp_sl, gq_sl = self.sl.get("goal_p"), self.sl.get("goal_q")
        # goal: anisotropic in the approach frame built from the CLEAN geometry
        if self.goal_scale > 0:
            gp, _ = self.goal_fn(base_obs)
            d = gp - base_obs[:, sp]
            u = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            v, w = axis_frame(u)
            sd = self.gm["sd"]
            comp = (self.gm["mu_u"] + self.z_goal[:, 0:1] * sd[0] * self.tail) * u \
                 + (self.z_goal[:, 1:2] * sd[1] * self.tail) * v \
                 + (self.z_goal[:, 2:3] * sd[2] * self.tail) * w
            self._dp_goal = self.a[:, None] * comp * self.goal_scale / 1000.0
            if self.z_reflect:
                dz = self._dp_goal[:, 2:3]
                dz = dz.abs() if self.z_mode == "reflect" else dz.clamp_min(0.0)
                self._dp_goal = torch.cat([self._dp_goal[:, :2], dz], dim=-1)
            # LiftPegUpright has no goal slice at all -- the delta still has to be
            # computed here, because the attribute route below consumes it.
            if gp_sl is not None:
                o[:, gp_sl] = base_obs[:, gp_sl] + self._dp_goal
            if gq_sl is not None:
                o[:, gq_sl] = _qmul(self.dq_g, base_obs[:, gq_sl])
        # object: OU displacement already accumulated in tick()
        if sp is not None and self.obj_scale > 0:
            o[:, sp] = base_obs[:, sp] + self.d_obj
            if sq is not None:
                o[:, sq] = _qmul(self.dq_o, base_obs[:, sq])
        # repair the derived difference fields from the PERTURBED poses, or the
        # true poses stay recoverable and the arm is a no-op
        pt = {"tcp_p": base_obs[:, TCP_P[self.task]]}
        for key, sl_ in (("obj_p", sp), ("goal_p", gp_sl)):
            if sl_ is not None:
                pt[key] = o[:, sl_]
        for dst, a_, b_ in DERIVED[self.task]:
            if a_ in pt and b_ in pt:
                o[:, dst] = pt[a_] - pt[b_]
        # goal that lives in an ATTRIBUTE, not a slice (LiftPegUpright has no
        # goal in its stock observation at all; PickCube has the position but
        # commits the orientation at reset). The env exposes a setter; the
        # approach frame above is still built from the CLEAN goal.
        if self.goal_attr_fn is not None and self.goal_scale > 0:
            self.goal_attr_fn(self._dp_goal, self.dq_g)
        return o
