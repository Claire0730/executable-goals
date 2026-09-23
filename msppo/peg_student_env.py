"""Peg student environment: occluded perception in, solved SE(3) out.

Row 3 for peg. Mirrors `kp_env.py`'s `student_recon` mode, which is the arm that
won on StackCube (0.924 banked oracle against the raw-point student's 0.850, with
a 3-4x tighter seed spread), and keeps its exact 66-D layout so
`msppo.student.StudentTransformer(pose_obs=True)` is reused unchanged:

    proprio(25 = qpos 9 + qvel 9 + tcp_pose 7)
    prev_action(8)
    obj_pose(7) goal_pose(7) scene_pose(7)          <- SOLVED from points
    tcp->obj(3) tcp->goal(3) obj->goal(3) obj->scene(3)

Panda's proprio is 25-D on both tasks, so the layout is identical to StackCube's
by arithmetic, not by coincidence.

TEACHER LABELS come from the external reference checkout's frozen checkpoint via
`msppo.peg_teacher_ref` (0.98 native insertion, verified). `teacher_observation`
therefore emits THEIR 436-D vector, not ours -- see that module for why only two
data artifacts cross the boundary.

Three things that differ from the cube and are read from the env, not assumed:

  * `scene_kp` is the RECTANGULAR HOLE RIM, not a second object. Its four corners
    are non-collinear (sigma2/sigma0 = 0.897 measured), so a pose solved from an
    occluded subset stays conditioned. The peg's own TRACED points are exactly
    collinear (`msppo/peg_traceerr.py`), which is why the goal for row 4 must be
    carried by the rim rather than by points along the peg.
  * peg geometry is randomised per env, so the student's canonical set is
    [B,K,3]. The TEACHER's is the fixed nominal template it was trained on --
    two different canonical sets live side by side here on purpose.
  * success TERMINATES the stock env, so DAgger steps the scene directly rather
    than through the auto-resetting vector wrapper; a teacher queried after an
    auto-reset would be labelling the next episode's state.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from msppo.peg_kp_env import (HOLE_P, HOLE_Q, PEG_HALF, PEG_P, PEG_Q, TCP_P,
                              quat_to_R, to_world, unit_box_keypoints,
                              unit_rect_rim)
from msgen.tasks import NUM_KPS
from msppo.peg_teacher_ref import ref_kp64, ref_obs

PROPRIO_DIMS = slice(0, 25)          # qpos 9 + qvel 9 + tcp_pose 7

# Mean insertion orientation over 2048 episodes (seeds 0-3, 512 envs each).
# Error against the per-episode truth: mean 11.3 deg, p90 20.3, max 23.1.
# It is a fixture calibration, not simulator state -- a bolted-down jig in a real
# cell is measured once the same way.
GOAL_Q_CAL = (0.703246, 0.0, 0.0, 0.710947)


def _R_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Shepperd's method: pick the largest diagonal branch, so no square root is
    taken of a near-zero quantity. The naive w-branch loses precision exactly
    where a solved pose matters most, at large rotations."""
    m = R
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    q = torch.zeros(m.shape[:-2] + (4,), device=m.device, dtype=m.dtype)
    c0 = t > 0
    c1 = (~c0) & (m[..., 0, 0] >= m[..., 1, 1]) & (m[..., 0, 0] >= m[..., 2, 2])
    c2 = (~c0) & (~c1) & (m[..., 1, 1] >= m[..., 2, 2])
    c3 = (~c0) & (~c1) & (~c2)

    def fill(mask, s, w, x, y, z):
        if not bool(mask.any()):
            return
        q[mask] = torch.stack([w, x, y, z], dim=-1)[mask] / s[mask, None]

    s0 = torch.sqrt((t + 1).clamp_min(1e-12)) * 2
    fill(c0, s0, 0.25 * s0 ** 2, m[..., 2, 1] - m[..., 1, 2],
         m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1])
    s1 = torch.sqrt((1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2]).clamp_min(1e-12)) * 2
    fill(c1, s1, m[..., 2, 1] - m[..., 1, 2], 0.25 * s1 ** 2,
         m[..., 0, 1] + m[..., 1, 0], m[..., 0, 2] + m[..., 2, 0])
    s2 = torch.sqrt((1 + m[..., 1, 1] - m[..., 0, 0] - m[..., 2, 2]).clamp_min(1e-12)) * 2
    fill(c2, s2, m[..., 0, 2] - m[..., 2, 0], m[..., 0, 1] + m[..., 1, 0],
         0.25 * s2 ** 2, m[..., 1, 2] + m[..., 2, 1])
    s3 = torch.sqrt((1 + m[..., 2, 2] - m[..., 0, 0] - m[..., 1, 1]).clamp_min(1e-12)) * 2
    fill(c3, s3, m[..., 1, 0] - m[..., 0, 1], m[..., 0, 2] + m[..., 2, 0],
         m[..., 1, 2] + m[..., 2, 1], 0.25 * s3 ** 2)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


class PegStudentEnv:
    """Composition, not gym.Wrapper: ManiSkillVectorEnv is a VectorEnv and fails
    gym.Wrapper's isinstance check."""

    def __init__(self, env, num_kp=64, seed=0, device="cuda",
                 kp_mask_ratio=2, kp_noise_m=0.005, kp_mask_height=True,
                 prev_action=True, scene_kp=True, goal_error_bank=None,
                 goal_rot_error=True, zero_rot_goal=False, kp_bias_m=0.0,
                 kp_bias_rand=False, perception=False, track_px=0.0,
                 query="free", goal_rot_from_scene=False, goal_override=None,
                 goal_err_commit=False, goal_error_rel=None,
                 goal_delta=None, sam2_masks=None, goal_err_scale=None,
                 point_obs=False, tcp_obs=True):
        self.env, self.base, self.device = env, env.unwrapped, device
        self.num_kp, self.prev_action, self.scene_kp = num_kp, prev_action, scene_kp
        self.kp_mask_ratio, self.kp_noise_m = kp_mask_ratio, kp_noise_m
        self.kp_mask_height, self.kp_bias_m = kp_mask_height, kp_bias_m
        self.kp_bias_rand = kp_bias_rand
        self._bias_obj = self._bias_scene = None
        self.num_envs = self.base.num_envs
        self.goal_error_bank = goal_error_bank
        self.goal_rot_error, self.zero_rot_goal = goal_rot_error, zero_rot_goal
        # [num_envs,7] TraceGen-predicted goal pose PER ENV INDEX, replacing the
        # simulator's true goal. Distinct from `goal_error_bank`, which perturbs
        # the true goal with an error measured on some OTHER scene: this is the
        # planner's actual answer to this scene. Rows may be NaN -- a scene with
        # under 3 traced peg points has no prediction at all, which is a planner
        # failure and is reported as such rather than being backfilled.
        self.goal_override = goal_override
        # `_sample_goal_error` draws a fresh bank index on EVERY call, and it is
        # called once per control step, so the replayed goal error is white noise
        # in time. `reset_masks` already rejects that treatment for occlusion --
        # "resampling every step would average the occlusion away and make the
        # task easier than the real one" -- and a planner emits ONE goal per
        # episode, not a new one per step. Committing the index per episode is
        # therefore the faithful protocol; it is opt-in so the earlier numbers
        # stay reproducible under the protocol that produced them.
        self.goal_err_commit = goal_err_commit
        self._goal_err_idx = None
        # SCENE-RELATIVE planner error (msppo/peg_relbank.py): the measured
        # end-to-end error re-expressed in the peg->goal frame, so it transports to
        # whatever layout an episode draws. Always committed per episode -- a
        # planner emits one goal. This is what makes the planner's goal quality
        # part of the TRAINING distribution instead of a test-time surprise.
        self.goal_error_rel = goal_error_rel
        self._rel_idx = None
        # How much of the sampled error to apply, per EPISODE. None/1.0 = the full
        # measured error, which is what the first retrain used -- and it collapsed the
        # true-goal ceiling from 0.776 to 0.368, because a policy that only ever sees
        # the worst goal learns to ignore the channel. "uniform" draws a ~ U(0,1) per
        # episode instead, so the policy sees the whole quality range, which is what
        # domain randomisation means. A float pins one scale, for diagnosis.
        self.goal_err_scale = goal_err_scale
        self._rel_a = None
        # DEPLOYABLE goal channel: ([B,3,3], [B,3]) the rigid transform TraceGen
        # predicts for the traced points, composed here with the PERCEIVED object
        # pose. `goal_override` composes the same delta with the SIMULATOR's object
        # pose upstream, which puts privileged state in the policy's input; this
        # path never reads an object pose out of the simulator.
        self.goal_delta = goal_delta
        # Composed ONCE, on the first frame after a reset, then held. The delta is
        # "from where the object is NOW to where it should end up", so re-composing
        # it against the CURRENT pose every step makes the goal recede by one delta
        # as the object advances -- a carrot the policy can never reach. That read
        # 0.000 on every run and bank until it was caught; a planner commits to one
        # endpoint, it does not re-aim at every control step.
        self._delta_g7 = None
        self._gen = torch.Generator(device=device).manual_seed(seed)

        # PAIRED-POINT student (Dex4D Fig.2(b)/3(c)): the observation carries the
        # occluded point SETS instead of the SE(3) solved from them. Dex4D's own
        # evidence for Paired Point Encoding (Table II, 0.057 -> 0.203 -> 0.600)
        # is measured on the STUDENT under masking, not on the teacher, which
        # sees unoccluded points -- so this is where the claim is testable.
        # OPT-IN: every existing peg student checkpoint is the 66-D pose layout.
        self.point_obs = point_obs
        # Dex4D's student list is joint angle / joint velocity / last action /
        # masked paired points. tcp_pose is NOT on it. It is not privileged
        # (qpos through FK gives it), so dropping it costs the policy a
        # learned FK rather than information -- which is exactly what a
        # literal port asks it to do.
        self.tcp_obs = tcp_obs
        self.perception, self.goal_rot_from_scene = perception, goal_rot_from_scene
        self.GOAL_Q_CAL = torch.tensor(GOAL_Q_CAL, device=device)
        if perception:
            from msppo.peg_perception import PegPerception
            self.perc = PegPerception(env, num_kp=num_kp, seed=seed,
                                      device=device, track_px=track_px,
                                      query=query, sam2_masks=sam2_masks)
        self._unit_peg = unit_box_keypoints(num_kp, seed).to(device)
        self._unit_rim = unit_rect_rim(num_kp, seed).to(device)
        self._teacher_kp = ref_kp64(device)          # the FIXED nominal template

        act = env.single_action_space.shape[0]
        self._prev_act = torch.zeros((self.num_envs, act), device=device)
        self._keep_obj = torch.ones((self.num_envs, num_kp), device=device)
        self._keep_scene = torch.ones((self.num_envs, num_kp), device=device)

        if point_obs:
            # mirrors StudentTransformer(pose_obs=False).sl exactly:
            # proprio 25 | prev_action | obj_kp | goal_kp | scene_kp
            #
            # The POINT COUNT is not num_kp in every mode. `--query grid` hands
            # back TraceGen's whole 20x20 grid (400 points, of which ~4.9 land on
            # the peg and the rest are masked to zero), because principle 4
            # requires the tracked points and TraceGen's query points to be the
            # same pixel IDs. `--query free` samples num_kp points ON the object.
            # Declaring num_kp here in grid mode silently mis-sizes the student.
            self.n_pts = (NUM_KPS if (perception and query == "grid") else num_kp)
            total = ((25 if tcp_obs else 18) + (act if prev_action else 0)
                     + self.n_pts * 3 * (3 if scene_kp else 2))
        else:
            if not tcp_obs:
                raise SystemExit("tcp_obs=False is only defined for the point "
                                 "encoding; the pose layout's `rel` block is built "
                                 "from tcp and would be undefined without it.")
            total = (25 + (act if prev_action else 0)
                     + 7 * (3 if scene_kp else 2) + 3 * (4 if scene_kp else 3))
        self.single_observation_space = gym.spaces.Box(
            -np.inf, np.inf, (total,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (self.num_envs, total), dtype=np.float32)
        self.single_action_space = env.single_action_space
        self.action_space = env.action_space

    # ----------------------------------------------------------- keypoints --
    def _canon_peg(self, base_obs):
        """[B,K,3] scaled by THIS env's peg. The student perceives the real peg,
        so unlike the teacher's fixed template this one tracks the true size."""
        return self._unit_peg[None] * base_obs[:, PEG_HALF][:, None, :]

    def _canon_rim(self, base_obs):
        r = base_obs[:, 42:43]
        s = torch.cat([torch.ones_like(r), r, r], dim=-1)
        return self._unit_rim[None] * s[:, None, :]

    def _goal_from_obs(self, base_obs):
        """goal_pose from the OBSERVATION, never the live property: the vector env
        auto-resets before returning. `peg_insertion_side.py:267` with
        `peg_head_offsets` a pure translation [L,0,0] (`:124-126`)."""
        q = base_obs[:, HOLE_Q]
        off = torch.zeros_like(base_obs[:, HOLE_P])
        off[:, 0] = -base_obs[:, PEG_HALF][:, 0]
        p = base_obs[:, HOLE_P] + torch.einsum("bij,bj->bi", quat_to_R(q), off)
        return p, q

    def keypoints(self, base_obs, apply_error=True):
        """(obj, goal, scene) [B,K,3] world points, student-side canonical sets."""
        canon = self._canon_peg(base_obs)
        obj = to_world(canon, base_obs[:, PEG_P], base_obs[:, PEG_Q])
        gp, gq = self._goal_from_obs(base_obs)
        if self.goal_delta is not None:
            # Deliberately NOT supported on the state path: composing the delta
            # needs the PERCEIVED object pose, and this student's obj channel is
            # solved against a CAD template from simulator state (PEG_P/PEG_Q), so
            # there is no perceived pose here to compose with. Supporting it would
            # mean composing against simulator state -- the exact contamination the
            # delta representation exists to remove.
            raise SystemExit("goal_delta requires a perception student "
                             "(--perception --query grid); the state student has no "
                             "perceived object pose to compose the delta with")
        if self.goal_override is not None:
            # same substitution as in the perception path: TraceGen's answer to
            # THIS scene replaces the simulator's goal outright. Kept here too so
            # the state student can serve as a bridge row between the old
            # error-replay protocol and the end-to-end one.
            ov = self.goal_override
            bad = ~torch.isfinite(ov).all(dim=-1)
            gp = torch.where(bad[:, None], base_obs[:, PEG_P], ov[:, :3])
            gq = torch.where(bad[:, None], base_obs[:, PEG_Q], ov[:, 3:7])
        if apply_error and self.goal_error_rel is not None:
            gp, gq = self._apply_rel_error(base_obs[:, PEG_P], gp, gq)
        if apply_error and self.zero_rot_goal:
            # The trivial baseline the planner must beat: "predict no rotation",
            # i.e. leave the peg's CURRENT orientation as the target. Its error
            # equals the true required rotation, so a planner whose rotation error
            # exceeds that is worse than doing nothing.
            gq = base_obs[:, PEG_Q]
        if apply_error and self.goal_error_bank is not None:
            dp, dq = self._sample_goal_error(base_obs.shape[0])
            gp = gp + dp
            if dq is not None:
                gq = self._qmul(dq, gq)
        goal = to_world(canon, gp, gq)
        scene = to_world(self._canon_rim(base_obs),
                         base_obs[:, HOLE_P], base_obs[:, HOLE_Q])
        return obj, goal, scene

    def _sample_goal_error(self, n):
        """(dp [n,3], dq [n,4]) drawn from the measured prediction-error bank.

        `results/peg_bank_n50.npz` holds one (dp, dR) pair per real scene, from
        `msppo/peg_traceerr.py`: dp is the world offset of the predicted endpoint
        from the true one, dR the relative rotation Rp Rg^T that carries the true
        goal orientation onto the predicted one. Replaying real pairs beats
        sampling from a fitted distribution -- position and rotation error are
        correlated per scene, and a factorised sample would break that.
        """
        b = self.goal_error_bank
        if self.goal_err_commit and self._goal_err_idx is not None:
            i = self._goal_err_idx                    # one draw per EPISODE
        else:
            i = torch.randint(0, b["dp"].shape[0], (n,), generator=self._gen,
                              device=b["dp"].device)
        dp = b["dp"][i]
        dq = _R_to_quat(b["dR"][i]) if self.goal_rot_error else None
        return dp, dq

    def _apply_rel_error(self, peg_p, gp, gq):
        """Replace the true goal with what the planner would have predicted here.

        Sampled tuple = (fraction of the required travel, two lateral offsets,
        relative rotation). Expanded against THIS episode's peg and goal, so a
        scene needing 250 mm and one needing 470 mm get proportionate errors --
        the world-frame bank gave both the same offset.
        """
        from msppo.peg_relbank import expand, scale_error
        b = self.goal_error_rel
        i = (self._rel_idx if self._rel_idx is not None
             else torch.randint(0, b["frac"].shape[0], (gp.shape[0],),
                                generator=self._gen, device=self.device))
        a = (self._rel_a if self._rel_a is not None
             else torch.ones((gp.shape[0],), device=self.device))
        frac, lv, lw, dq = scale_error(b, i, a)
        gp2 = expand(peg_p, gp, frac, lv, lw)
        gq2 = self._qmul(dq, gq) if self.goal_rot_error else gq
        return gp2, gq2

    @staticmethod
    def _qmul(a, b):
        """Hamilton product, SAPIEN (w,x,y,z)."""
        aw, ax, ay, az = a.unbind(-1)
        bw, bx, by, bz = b.unbind(-1)
        return torch.stack([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw], dim=-1)

    # -------------------------------------------------------------- solving --
    def _occlude(self, pts, keep, bias=None):
        """Per-step height mask + perception error, then ZERO the dropped points --
        upstream zeroes rather than removes (`utils/util.py:387`) and the token
        encoders expect a fixed-length set.

        TWO error components, and the split is the whole point:

          `kp_noise_m`  PER-POINT iid. Kabsch averages it over the ~32 surviving
                        points, so it shrinks by about sqrt(32) = 5.7x. 5 mm of
                        this lands as 2.9 mm of solved-pose error -- measured.
          `kp_bias_m`   PER-OBJECT, PER-EPISODE, applied to every point of that
                        object identically. Kabsch CANNOT average it out: it
                        passes straight through into the solved translation.

        Real 3D tracking error has both. Camera-extrinsic error, depth scale
        error and tracker drift move the whole point set together, and modelling
        perception as iid noise alone is optimistic in exactly the way that makes
        a sim number fail to transfer. A deployable number needs the correlated
        term.
        """
        from msppo.masking import mask_height
        if self.kp_mask_height:
            keep = mask_height(pts, keep, self._gen)
        out = pts
        if self.kp_noise_m > 0:
            out = out + torch.randn(out.shape, generator=self._gen,
                                    device=out.device) * self.kp_noise_m
        if bias is not None:
            out = out + bias[:, None, :]
        return out * keep[..., None], keep

    def _resample_bias(self):
        """One correlated offset per object per episode.

        `kp_bias_rand` draws the MAGNITUDE per episode from U(0, kp_bias_m) as
        well, instead of fixing it. Training that way is what makes a policy
        deployable without first knowing the tracker's true accuracy: it covers
        the range rather than betting on one value. Evaluation fixes the
        magnitude so the sensitivity curve stays readable.
        """
        if not self.kp_bias_m:
            self._bias_obj = self._bias_scene = None
            return
        sh = (self.num_envs, 3)
        sig = self.kp_bias_m
        if self.kp_bias_rand:
            sig = torch.rand((self.num_envs, 1), generator=self._gen,
                             device=self.device) * self.kp_bias_m
        self._bias_obj = torch.randn(sh, generator=self._gen,
                                     device=self.device) * sig
        self._bias_scene = torch.randn(sh, generator=self._gen,
                                       device=self.device) * sig

    def _solve_pose(self, kp, canon, keep=None):
        """Weighted Kabsch. The weights are not cosmetic: masked points are ZEROED
        rather than removed, and an unweighted fit treats those zeros as real
        observations at the origin -- measured 94.4 mm error unweighted against
        2.7 mm weighted on the cube."""
        from msppo.kabsch import kabsch
        w = None if keep is None else keep.to(kp.dtype)
        R, t, _ = kabsch(canon, kp, weights=w)
        return torch.cat([t, _R_to_quat(R)], dim=-1)

    # ------------------------------------------------------------ the views --
    def student_observation(self, base_obs, apply_error=True):
        """[B,66] the student's own view: everything solved from occluded points."""
        from msppo.masking import mask_oneside
        obj, goal, scene = self.keypoints(base_obs, apply_error=apply_error)
        canon = self._canon_peg(base_obs)
        obj_o, ko = self._occlude(obj, self._keep_obj, self._bias_obj)
        scene_o, ks = self._occlude(scene, self._keep_scene, self._bias_scene)
        if self.point_obs:
            pro = base_obs[:, PROPRIO_DIMS if self.tcp_obs else slice(0, 18)]
            return self._pack_points(pro, obj_o, goal, scene_o)
        op = self._solve_pose(obj_o, canon, ko)
        # goal_kp is a PREDICTION, not a perception: it carries its own error and
        # is never occluded, so it is solved unweighted.
        gp = self._solve_pose(goal, canon)
        tcp = base_obs[:, TCP_P]
        parts = [base_obs[:, PROPRIO_DIMS]]
        if self.prev_action:
            parts.append(self._prev_act)
        parts += [op, gp]
        if self.scene_kp:
            sp = self._solve_pose(scene_o, self._canon_rim(base_obs), ks)
            parts.append(sp)
        parts += [op[:, :3] - tcp, gp[:, :3] - tcp, gp[:, :3] - op[:, :3]]
        if self.scene_kp:
            parts.append(sp[:, :3] - op[:, :3])
        return torch.cat(parts, dim=-1)

    def _pack_pose(self, proprio, op, gp, sp, tcp):
        """[B, 66 or 56] the POSE layout, for the perception path.

        Was inlined twice and BOTH copies appended `sp` and `sp - op`
        unconditionally, so `scene_kp=False` was silently ignored here while the
        state path honoured it -- the declared observation space said 56 and the
        method returned 66. Only the point path had an assert, which is why this
        survived. Both are routed through here now, and the assert covers both.
        """
        parts = [proprio]
        if self.prev_action:
            parts.append(self._prev_act)
        parts += [op, gp]
        if self.scene_kp:
            parts.append(sp)
        parts += [op[:, :3] - tcp, gp[:, :3] - tcp, gp[:, :3] - op[:, :3]]
        if self.scene_kp:
            parts.append(sp[:, :3] - op[:, :3])
        out = torch.cat(parts, dim=-1)
        assert out.shape[1] == self.single_observation_space.shape[0], (
            f"pose layout {out.shape[1]} != declared "
            f"{self.single_observation_space.shape[0]}")
        return out

    def _pack_points(self, proprio, obj, goal, scene):
        """[B, 25 + act + K*3*(2 or 3)] -- the paired-point student layout.

        Order must mirror `StudentTransformer(pose_obs=False).sl` exactly
        (`student.py:115-119`): obj, then goal, then scene. `student.tokens`
        stacks obj and goal into one 6-channel block for `PointNetToken`, which
        is the paired encoding: channel i of a point is [current_xyz, target_xyz]
        of THE SAME query point, so correspondence survives the pooling.
        A silent permutation here would still train, and would still be a
        DECOUPLED encoding wearing a paired label -- Dex4D measures that variant
        at 0.203 against 0.600, so it is worth the assert below.
        """
        b = proprio.shape[0]
        parts = [proprio]
        if self.prev_action:
            parts.append(self._prev_act)
        parts += [obj.reshape(b, -1), goal.reshape(b, -1)]
        if self.scene_kp:
            parts.append(scene.reshape(b, -1))
        out = torch.cat(parts, dim=-1)
        assert out.shape[1] == self.single_observation_space.shape[0], (
            f"point layout {out.shape[1]} != declared "
            f"{self.single_observation_space.shape[0]}")
        return out

    def teacher_observation(self):
        """[B,436] the REFERENCE teacher's vector, read from the LIVE sim.

        Live is correct here and only here: DAgger drives the scene directly so
        nothing has auto-reset between the action and this call. It also always
        reflects the TRUE goal -- a teacher shown the student's displaced goal
        would act on the same wrong target and supervise nothing.
        """
        return ref_obs(self.base, self._teacher_kp)

    # ------------------------------------------------------------- episode --
    def reset_masks(self):
        """One occlusion side per EPISODE, as upstream does -- resampling every
        step would average the occlusion away and make the task easier than the
        real one."""
        from msppo.masking import mask_oneside
        obj = self._unit_peg[None].expand(self.num_envs, -1, -1)
        self._keep_obj = mask_oneside(obj, self.kp_mask_ratio, self._gen)
        rim = self._unit_rim[None].expand(self.num_envs, -1, -1)
        self._keep_scene = mask_oneside(rim, self.kp_mask_ratio, self._gen)
        self._prev_act = torch.zeros_like(self._prev_act)
        self._resample_bias()
        if self.goal_err_commit and self.goal_error_bank is not None:
            self._goal_err_idx = torch.randint(
                0, self.goal_error_bank["dp"].shape[0], (self.num_envs,),
                generator=self._gen, device=self.device)
        if self.goal_error_rel is not None:
            self._rel_idx = torch.randint(
                0, self.goal_error_rel["frac"].shape[0], (self.num_envs,),
                generator=self._gen, device=self.device)
            if self.goal_err_scale == "uniform":
                self._rel_a = torch.rand((self.num_envs,), generator=self._gen,
                                         device=self.device)
            elif self.goal_err_scale is None:
                self._rel_a = torch.ones((self.num_envs,), device=self.device)
            else:
                self._rel_a = torch.full((self.num_envs,), float(self.goal_err_scale),
                                         device=self.device)

    def set_prev_action(self, a):
        self._prev_act = a.detach()

    # ------------------------------------------------------- perception mode --
    def perc_reset(self, obs_dict):
        """Pick query points on the first frame. Call right after env.reset."""
        self.perc.reset(obs_dict)
        self._delta_g7 = None            # a new episode re-composes the goal

    def student_observation_perc(self, obs_dict, apply_error=True):
        """[B,66] with obj/scene/goal ALL solved from the rendered RGB-D frame.

        Same 66-D layout as `student_observation`, but nothing here reads an
        object pose out of the simulator except the two places that legitimately
        cannot come from perception yet:

          * the GOAL configuration, which is the planner's job -- TraceGen will
            supply those points; until then they are the true goal points, and
            the TraceGen error bank is replayed on top exactly as before.
          * the rigid carry-forward of query points between frames, i.e. the
            tracker's job. See `peg_perception`'s `track_px`.

        The poses live in the PCA frame of the t=0 point set, NOT in any CAD
        frame, so a policy trained on `student_observation` cannot be evaluated
        here -- the two use different conventions for the same geometry. The
        comparison has to be between two separately distilled students.
        """
        from msppo.kabsch import kabsch
        a = obs_dict["agent"]
        proprio = torch.cat([a["qpos"], a["qvel"]]
                            + ([obs_dict["extra"]["tcp_pose"]] if self.tcp_obs else []),
                            dim=-1)
        tcp = obs_dict["extra"]["tcp_pose"][:, :3]

        if self.point_obs:
            op, sp, _, _, obj_pts, scene_pts = self.perc.observe(obs_dict, points=True)
        else:
            op, sp, _, _ = self.perc.observe(obs_dict)

        # goal: the SAME query points, carried to the goal configuration, then
        # solved against the SAME canonical set -- so goal_pose and obj_pose are
        # expressed in one convention and their difference is meaningful.
        gp = self.base.goal_pose
        gpos, gq = gp.p, gp.q
        if self.goal_delta is not None:
            # goal points = delta applied to the object's points as PERCEIVED now.
            # `op` is the pose solved from this frame's depth measurement, so the
            # whole chain is camera -> points -> Kabsch -> delta -> points: no
            # simulator state anywhere. Scenes with no prediction (NaN delta) fall
            # back to the identity transform -- "no motion predicted" -- which is
            # the planner abstaining, and leaks nothing.
            if self._delta_g7 is None:
                dR, dt = self.goal_delta
                bad = ~(torch.isfinite(dR).all(dim=-1).all(dim=-1)
                        & torch.isfinite(dt).all(-1))
                eye = torch.eye(3, device=dR.device, dtype=dR.dtype)
                dR = torch.where(bad[:, None, None], eye, dR)
                dt = torch.where(bad[:, None], torch.zeros_like(dt), dt)
                cur = torch.einsum("bij,bnj->bni", quat_to_R(op[:, 3:7]),
                                   self.perc.canon_obj) + op[:, None, :3]
                goal_pts = torch.einsum("bij,bnj->bni", dR, cur) + dt[:, None]
                R, t, _ = kabsch(self.perc.canon_obj, goal_pts,
                                 weights=self.perc.keep0_obj.to(goal_pts.dtype))
                self._delta_g7 = torch.cat([t, _R_to_quat(R)], dim=-1)
            g7 = self._delta_g7
            if self.point_obs:
                goal_pts = torch.einsum("bij,bnj->bni", quat_to_R(g7[:, 3:7]),
                                        self.perc.canon_obj) + g7[:, None, :3]
                return self._pack_points(proprio, obj_pts, goal_pts, scene_pts)
            return self._pack_pose(proprio, op, g7, sp, tcp)
        if self.goal_override is not None:
            # TraceGen's own answer to THIS scene. Where it produced none (too few
            # traced peg points), fall back to the peg's current pose -- the
            # degenerate "no motion predicted" output, which is also what trace
            # step 0 gives, and which leaks nothing about the hole. Those rows are
            # counted separately so the row can be reported both ways.
            ov = self.goal_override
            bad = ~torch.isfinite(ov).all(dim=-1)
            gpos = torch.where(bad[:, None], self.base.peg.pose.p, ov[:, :3])
            gq = torch.where(bad[:, None], self.base.peg.pose.q, ov[:, 3:7])
        if apply_error and self.goal_error_rel is not None:
            gpos, gq = self._apply_rel_error(self.base.peg.pose.p, gpos, gq)
        if apply_error and self.zero_rot_goal:
            gq = self.base.peg.pose.q
        if apply_error and self.goal_error_bank is not None:
            dp, dq = self._sample_goal_error(gpos.shape[0])
            gpos = gpos + dp
            if dq is not None:
                gq = self._qmul(dq, gq)
        if self.goal_rot_from_scene:
            # RETRACTED as first written. Taking the goal orientation from the
            # scene channel's solved quaternion means taking the PCA frame of the
            # box's visible grid points, which is NOT the hole axis: the sampled
            # surface changes between episodes and the principal axes flip.
            # Measured 106.0 deg of error against the true insertion orientation,
            # and the PCA-to-hole offset itself scatters by 90.5 deg across envs,
            # so no one-time calibration recovers it. That arm read 0.000 success
            # at 13.2k iterations while its siblings were at 0.44-0.51.
            #
            # What does work is a CALIBRATION CONSTANT. The true insertion
            # orientation varies only 18.6 deg across episodes, so a single fixed
            # quaternion, measured once over 2048 episodes, lands at 11.3 deg mean
            # (p90 20.3) -- seven times better than TraceGen's 80.5 deg. In a real
            # cell the fixture is bolted down and calibrated once, which is
            # exactly this. Peg's own grid points keep responsibility for
            # position, where they measure 0.9 mm.
            gq = self.GOAL_Q_CAL[None].expand(gq.shape[0], -1)
        goal_pts = torch.einsum("bij,bnj->bni", quat_to_R(gq), self.perc.local_obj) \
            + gpos[:, None]
        R, t, _ = kabsch(self.perc.canon_obj, goal_pts,
                         weights=self.perc.keep0_obj.to(goal_pts.dtype))
        g7 = torch.cat([t, _R_to_quat(R)], dim=-1)

        if self.point_obs:
            return self._pack_points(proprio, obj_pts, goal_pts, scene_pts)
        return self._pack_pose(proprio, op, g7, sp, tcp)

    def close(self):
        self.env.close()


def make_peg_student_env(num_envs=256, num_kp=64, seed=0, clearance=0.01,
                         control="pd_joint_delta_pos", max_episode_steps=100,
                         robot="panda", ignore_terminations=False,
                         perception=False, tracegen_camera=False,
                         point_obs=False, tcp_obs=True, **kw):
    # robot: the reference teacher was trained on `panda`, and it is the teacher
    # that is frozen here, so the robot is not ours to choose.
    PegInsertionSideEnv._clearance = clearance
    # perception mode needs the camera; the 43-D state vector is then NOT
    # produced, which is the point -- nothing can accidentally read it.
    mode = "rgb+depth+segmentation" if perception else "state"
    extra = {}
    if tracegen_camera:
        # The TraceGen dataset was rendered through the task's own camera at 384
        # (msgen/replay.py:42-59). Query pixel IDs only mean the same thing under
        # the same camera, so principle 4 requires matching it here too.
        from mani_skill.utils import sapien_utils
        from msgen.tasks import IMAGE_SIZE, get_task
        c = get_task("peg")["camera"]
        extra["sensor_configs"] = {"base_camera": dict(
            pose=sapien_utils.look_at(eye=c["eye"], target=c["target"]),
            width=IMAGE_SIZE, height=IMAGE_SIZE, fov=c["fov"])}
    env = gym.make("PegInsertionSide-v1", num_envs=num_envs, obs_mode=mode,
                   control_mode=control, sim_backend="physx_cuda",
                   max_episode_steps=max_episode_steps,
                   reward_mode="normalized_dense", reconfiguration_freq=0,
                   robot_uids=robot, **extra)
    env = ManiSkillVectorEnv(env, auto_reset=True,
                             ignore_terminations=ignore_terminations)
    return PegStudentEnv(env, num_kp=num_kp, seed=seed,
                         point_obs=point_obs, tcp_obs=tcp_obs,
                         perception=perception, **kw)
