"""Perception for the peg student: RGB-D in, SE(3) out. No simulator state.

WHY. The student's 66-D observation has 33 dimensions (obj_pose, goal_pose,
scene_pose and their relative vectors) that were being built from the
SIMULATOR'S ground-truth object poses plus a synthetic noise model. That is a
proxy, and its two parameters (`kp_noise_m`, `kp_bias_m`) were guesses. This
module removes the proxy: every one of those dimensions is derived from a
rendered RGB-D frame through the same arithmetic a real rig would run.

WHAT IS AND IS NOT REAL HERE

  real   camera intrinsics/extrinsics, pixel quantisation, depth quantisation
         (int16 millimetres), points leaving the frame, and OCCLUSION BY ACTUAL
         GEOMETRY via a depth test -- which replaces `mask_oneside`, a synthetic
         one-side plane mask, with the occlusion the scene actually produces.
  real   the canonical point set is the t=0 unprojected points, not a CAD box.
         The project's correspondence principle is "correspondence by construction: TraceGen
         query points == CoTracker3 query points, locked to the same pixel IDs on
         a single timestamped reference frame" -- that is exactly this.
  proxy  point-to-point correspondence across time is assumed exact. A real
         tracker drifts. Drift belongs in PIXEL space, not as a 3D offset, and is
         left as a separate knob (`track_px`) so it can be calibrated against a
         real CoTracker3 measurement later rather than guessed now.
  proxy  segmentation is used ONCE, at t=0, only to CHOOSE which pixels become
         query points. It is never read again. The real-world equivalent is SAM2
         (or an operator click) on the first frame, which is a normal deployment
         step -- unlike per-frame segmentation, which would not be.

THE FRAME PROBLEM, AND WHY THE CANONICAL SET CANNOT SIMPLY BE THE t=0 POINTS

If the canonical set were the t=0 points in the OBJECT's t=0 frame, the solved
transform would be the object's MOTION since t=0, not its pose -- and for the
box, which never moves, that is the identity for every episode. The channel that
tells the policy where the hole is would carry no information at all.

So the canonical frame is derived FROM THE POINTS: centroid at the origin, axes
from the PCA of the t=0 point set. That is perception-derivable, needs no CAD,
and gives a pose that varies across episodes the way the true one does. PCA axes
carry a sign ambiguity, which would make the frame flip between episodes and
present the policy with two different conventions for the same geometry; the
signs are therefore pinned by requiring a positive dot product with fixed world
axes, and the third axis is set by a right-handedness constraint.
"""
from __future__ import annotations

import torch

from msppo.kabsch import kabsch


def _R_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Shepperd's method -- see `peg_student_env._R_to_quat` for why not the
    naive w-branch."""
    from msppo.peg_student_env import _R_to_quat as f
    return f(R)


class PegPerception:
    """Query points chosen once at t=0, then tracked and unprojected each step."""

    def __init__(self, env, num_kp=64, seed=0, device="cuda", cam="base_camera",
                 depth_tol_m=0.01, track_px=0.0, depth_noise_m=0.0,
                 scene_static=True, query="free", sam2_masks=None,
                 obj_name="peg", scene_name="box"):
        self.base = env.unwrapped
        self.n = self.base.num_envs
        # The ONLY task-specific things in this module: which actor is the object
        # and which is the fixture. Defaulting to peg/box means every existing peg
        # run reaches identical code, so the 0.720 that rests on this file cannot
        # move. `scene_name=None` is a task with NO fixture (LiftPegUpright: the
        # goal is "standing on the table"); the scene tensors are then filled with
        # the object's so shapes stay valid, and the caller MUST drop the scene
        # channel -- `TaskStudentEnv` asserts that.
        self.obj_name, self.scene_name = obj_name, scene_name
        self.num_kp, self.device, self.cam = num_kp, device, cam
        self.depth_tol_m = depth_tol_m
        self.track_px, self.depth_noise_m = track_px, depth_noise_m
        assert query in ("free", "grid"), query
        self.scene_static, self.query = scene_static, query
        # SAM2 masks for t=0 instead of the simulator's segmentation, the last
        # offline-removable simulator input in this pipeline
        # (msppo/peg_sam2_masks.py). Supplied as {"peg": [B,H,W] bool,
        # "box": [B,H,W] bool} and consumed by SYNTHESISING a segmentation image
        # with labels 1 and 2, so `_pick`/`_pick_grid` are untouched -- their
        # occlusion and depth logic is what the row-3 numbers rest on and is not
        # worth re-deriving for this.
        self.sam2_masks = sam2_masks
        self.g = torch.Generator(device=device).manual_seed(seed)
        self.canon_obj = self.canon_scene = None
        self.T0_obj = self.T0_scene = None

    @property
    def _obj(self):
        a = getattr(self.base, self.obj_name, None)
        if a is None:                       # envs that expose the actor only in the scene dict (PlaceSphere: `obj` attr, "sphere" actor)
            a = self.base.scene.actors[self.obj_name]
        return a

    @property
    def _scene(self):
        """The fixture, or the object again when the task has none. Callers with
        no fixture must set scene_kp=False; returning the object keeps every
        tensor the right shape without inventing a second body."""
        name = self.scene_name or self.obj_name
        a = getattr(self.base, name, None)
        if a is None:
            a = self.base.scene.actors[name]
        return a

    @property
    def has_scene(self):
        return self.scene_name is not None

    # --------------------------------------------------------------- camera --
    @staticmethod
    def _unproject(u, v, z, K, E):
        """[B,N] pixels + depth -> [B,N,3] world points. K [B,3,3], E [B,3,4]."""
        fx, fy = K[:, 0, 0:1], K[:, 1, 1:2]
        cx, cy = K[:, 0, 2:3], K[:, 1, 2:3]
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z
        pc = torch.stack([x, y, z], dim=-1)                      # camera frame
        R, t = E[:, :, :3], E[:, :, 3]
        return torch.einsum("bij,bnj->bni", R.transpose(1, 2), pc - t[:, None, :])

    @staticmethod
    def _project(pw, K, E):
        """[B,N,3] world -> ([B,N] u, [B,N] v, [B,N] z_camera)."""
        R, t = E[:, :, :3], E[:, :, 3]
        pc = torch.einsum("bij,bnj->bni", R, pw) + t[:, None, :]
        z = pc[..., 2].clamp_min(1e-6)
        fx, fy = K[:, 0, 0:1], K[:, 1, 1:2]
        cx, cy = K[:, 0, 2:3], K[:, 1, 2:3]
        return pc[..., 0] / z * fx + cx, pc[..., 1] / z * fy + cy, z

    def _cam(self, obs):
        d = obs["sensor_data"][self.cam]
        p = obs["sensor_param"][self.cam]
        # int16 MILLIMETRES; -32768 marks no hit. Verified against the peg: pixels
        # on it unproject to 87-89 mm from its centre and its half-length is 85.4.
        depth = d["depth"][..., 0].float() / 1000.0
        valid = d["depth"][..., 0] > 0
        return depth, valid, d["segmentation"][..., 0], p["intrinsic_cv"], p["extrinsic_cv"]

    # ---------------------------------------------------------------- t = 0 --
    def _pick_grid(self, seg, depth, valid, sid, K, E):
        """Query points restricted to TraceGen's own 20x20 pixel grid.

        The correspondence principle requires the TraceGen query points and the
        tracked points to be THE SAME pixel IDs on one reference frame. Sampling
        freely on the object breaks that: the goal (from TraceGen's grid) and the
        object (from free samples) would then be solved against different
        canonical sets and their difference would be meaningless.

        The cost is measured and severe for this object. Of 400 grid points, only
        5.1 land on the peg (min 2, max 9) because the peg covers 1.3% of the
        384x384 frame, and those few lie along its axis, so rotation about that
        axis is close to unobservable:

            64 free points on the object   61.8 kept   0.7 mm   0.3 deg
            grid points on the object       4.9 kept   0.9 mm   6.9 deg

        Position survives; rotation degrades 23x. That is the geometric reason
        behind the 80.5 deg goal-rotation error measured for peg, and it is a
        property of grid density against a thin object, not of the flow model.
        """
        from msgen.labels import grid_pixels
        B, H, W = seg.shape
        px = grid_pixels()
        u = torch.as_tensor(px[:, 0], device=self.device).float()[None].expand(B, -1)
        v = torch.as_tensor(px[:, 1], device=self.device).float()[None].expand(B, -1)
        bi = torch.arange(B, device=self.device)[:, None]
        ui, vi = u.long().clamp(0, W - 1), v.long().clamp(0, H - 1)
        z = depth[bi, vi, ui]
        keep = ((seg[bi, vi, ui] == sid[:, None]) & (z > 0) & valid[bi, vi, ui]).float()
        return self._unproject(u, v, z, K, E), keep

    def _pick(self, seg, depth, valid, sid, K, E):
        """Choose `num_kp` query pixels on body `sid` and unproject them.

        Returns (points [B,K,3], keep [B,K]). Envs showing fewer than num_kp
        pixels of the body are padded and masked rather than silently repeated:
        a repeated point is a duplicated constraint and would quietly reweight
        the Kabsch fit.
        """
        B, H, W = seg.shape
        on = (seg == sid[:, None, None]) & valid
        u = torch.zeros((B, self.num_kp), device=self.device)
        v = torch.zeros_like(u)
        keep = torch.zeros_like(u)
        flat = on.reshape(B, -1).float()
        cnt = flat.sum(-1)
        for i in range(B):                     # per-env: the counts differ
            c = int(cnt[i])
            if c == 0:
                continue
            k = min(c, self.num_kp)
            idx = torch.multinomial(flat[i], k, replacement=False, generator=self.g)
            v[i, :k] = (idx // W).float()
            u[i, :k] = (idx % W).float()
            keep[i, :k] = 1.0
        z = depth[torch.arange(B, device=self.device)[:, None],
                  v.long().clamp(0, H - 1), u.long().clamp(0, W - 1)]
        return self._unproject(u, v, z, K, E), keep

    @staticmethod
    def _pca_frame(pts, keep):
        """[B,K,3] -> (R [B,3,3], c [B,3]): a frame defined BY the points.

        Sign ambiguity is resolved by forcing each axis to point along the world
        axis it is most aligned with; without that the frame flips between
        episodes and the policy sees two conventions for one geometry.
        """
        w = keep[..., None]
        c = (pts * w).sum(1) / w.sum(1).clamp_min(1e-6)
        q = (pts - c[:, None]) * w
        _, _, Vt = torch.linalg.svd(q.transpose(1, 2) @ q)
        R = Vt.transpose(1, 2)                                   # columns = axes
        sign = torch.sign(torch.diagonal(R, dim1=1, dim2=2))
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        R = R * sign[:, None, :]
        # right-handed: flip the last axis if the determinant went negative
        det = torch.linalg.det(R)
        R = torch.cat([R[:, :, :2], R[:, :, 2:] * torch.sign(det)[:, None, None]], dim=2)
        return R, c

    def reset(self, obs, idx=None):
        """Pick query points and fix the canonical frames.

        `idx` is a boolean mask of envs to re-pick. The vector env auto-resets
        only the envs that finished, and those envs have a NEW scene: their old
        query points refer to a peg and a box that no longer exist there. Envs
        still mid-episode must keep theirs, because the canonical frame defines
        the pose convention -- re-picking under a running policy would jump the
        convention mid-trajectory and the observation would be discontinuous.
        """
        depth, valid, seg, K, E = self._cam(obs)
        if idx is not None and self.canon_obj is not None:
            keep_old = ~idx
            saved = (self.canon_obj.clone(), self.canon_scene.clone(),
                     self.keep0_obj.clone(), self.keep0_scene.clone(),
                     self.local_obj.clone(), self.world_scene.clone(),
                     tuple(t.clone() for t in self.T0_obj),
                     tuple(t.clone() for t in self.T0_scene))
            self._reset_all(depth, valid, seg, K, E)
            m = keep_old[:, None, None]
            m2 = keep_old[:, None]
            self.canon_obj = torch.where(m, saved[0], self.canon_obj)
            self.canon_scene = torch.where(m, saved[1], self.canon_scene)
            self.keep0_obj = torch.where(m2, saved[2], self.keep0_obj)
            self.keep0_scene = torch.where(m2, saved[3], self.keep0_scene)
            self.local_obj = torch.where(m, saved[4], self.local_obj)
            self.world_scene = torch.where(m, saved[5], self.world_scene)
            self.T0_obj = (torch.where(m, saved[6][0], self.T0_obj[0]),
                           torch.where(m2, saved[6][1], self.T0_obj[1]))
            self.T0_scene = (torch.where(m, saved[7][0], self.T0_scene[0]),
                             torch.where(m2, saved[7][1], self.T0_scene[1]))
            return
        self._reset_all(depth, valid, seg, K, E)

    def _reset_all(self, depth, valid, seg, K, E):
        pick = self._pick_grid if self.query == "grid" else self._pick
        peg_id, box_id = self._obj.per_scene_id, self._scene.per_scene_id
        if self.sam2_masks is not None:
            m = self.sam2_masks
            # peg wins any overlap: it is the thin foreground object and losing its
            # few pixels to the box would drop the scene below a solvable count
            seg = torch.where(m[self.obj_name], 1,
                              torch.where(m[self.scene_name], 2, 0)).to(seg.dtype)
            one = torch.ones_like(peg_id)
            peg_id, box_id = one, one * 2
        po, ko = pick(seg, depth, valid, peg_id, K, E)
        ps, ks = pick(seg, depth, valid, box_id, K, E)
        self.keep0_obj, self.keep0_scene = ko, ks

        Ro, co = self._pca_frame(po, ko)
        Rs, cs = self._pca_frame(ps, ks)
        self.T0_obj, self.T0_scene = (Ro, co), (Rs, cs)
        # canonical = the t=0 points in their own PCA frame
        self.canon_obj = torch.einsum("bij,bnj->bni", Ro.transpose(1, 2), po - co[:, None])
        self.canon_scene = torch.einsum("bij,bnj->bni", Rs.transpose(1, 2), ps - cs[:, None])
        # where each point sits on the body, so its world position can be
        # predicted at time t without re-reading the simulator's pose for it
        pp, pq = self._obj.pose.p, self._obj.pose.q
        from msppo.peg_kp_env import quat_to_R
        self.local_obj = torch.einsum("bij,bnj->bni", quat_to_R(pq).transpose(1, 2),
                                      po - pp[:, None])
        self.world_scene = ps                       # the box never moves

    # ---------------------------------------------------------------- t > 0 --
    def observe(self, obs, points=False):
        """(obj_pose [B,7], scene_pose [B,7], keep_obj, keep_scene).

        `points=True` appends the raw measured point sets (obj_pts, scene_pts),
        for the PAIRED-POINT student, which consumes the points themselves rather
        than the SE(3) solved from them. They are the SAME tensors the Kabsch
        solve is fed, so the two student encodings see identical perception --
        only the interface differs. Masked points arrive as exact zeros (the
        `measure` closure multiplies by `keep`), which is what the PointNet
        tokenizer's centre-and-append expects (`student.py:50-58`).

        The object's query points are carried forward rigidly (the tracker's job)
        and then RE-MEASURED through the camera: projected to pixels, read off the
        current depth map, unprojected. A point whose rendered depth disagrees
        with its expected depth is occluded and is masked -- that is real
        occlusion by scene geometry, not a synthetic plane mask.
        """
        from msppo.peg_kp_env import quat_to_R
        depth, valid, _, K, E = self._cam(obs)
        B, H, W = depth.shape
        pp, pq = self._obj.pose.p, self._obj.pose.q
        true_obj = torch.einsum("bij,bnj->bni", quat_to_R(pq), self.local_obj) + pp[:, None]

        def measure(true_pts, keep0):
            u, v, zexp = self._project(true_pts, K, E)
            if self.track_px:
                u = u + torch.randn(u.shape, generator=self.g, device=u.device) * self.track_px
                v = v + torch.randn(v.shape, generator=self.g, device=v.device) * self.track_px
            inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            # ROUND, not truncate. `.long()` truncates toward zero, which biases
            # the sampled depth pixel by about half a pixel and shows up as a
            # systematic pose error. It was visible only as an absurdity: adding
            # 0.25 px of tracking jitter measured MORE accurate (1.4 mm) than no
            # jitter at all (2.3 mm), because the jitter was dithering the
            # truncation bias away.
            ui = u.round().long().clamp(0, W - 1)
            vi = v.round().long().clamp(0, H - 1)
            bi = torch.arange(B, device=self.device)[:, None]
            zobs = depth[bi, vi, ui]
            ok = valid[bi, vi, ui] & inb & (zobs > 0)
            # depth test: a nearer surface at that pixel means the point is hidden
            vis = ok & ((zobs - zexp).abs() < self.depth_tol_m)
            if self.depth_noise_m:
                zobs = zobs + torch.randn(zobs.shape, generator=self.g,
                                          device=zobs.device) * self.depth_noise_m
            pts = self._unproject(u, v, zobs, K, E)
            keep = keep0 * vis.float()
            return pts * keep[..., None], keep

        obj_pts, ko = measure(true_obj, self.keep0_obj)
        if self.scene_static:
            # The box is a STATIC fixture within an episode. Re-solving its pose
            # every frame from whatever survives occlusion is not what a real rig
            # does and is actively harmful here: once the arm reaches over it the
            # visible count falls from 64 to under 20, the survivors are close to
            # coplanar, and the solved pose wanders. Measured 46 mm of apparent
            # motion in 20 steps for an object that never moved -- the policy
            # would read that as the hole sliding away.
            #
            # Localising the fixture once, on the unoccluded first frame, needs no
            # simulator access and is the normal deployment procedure. Set
            # scene_static=False to re-solve per frame and see the drift.
            Rs, cs = self.T0_scene
            out = (self._solve(self.canon_obj, obj_pts, ko),
                   torch.cat([cs, _R_to_quat(Rs)], dim=-1), ko, self.keep0_scene)
            # the box is localised once at t=0, so its point set is the t=0 one
            # held fixed -- the same convention the pose branch above uses
            return out + ((obj_pts, self.world_scene * self.keep0_scene[..., None])
                          if points else ())
        scene_pts, ks = measure(self.world_scene, self.keep0_scene)

        out = (self._solve(self.canon_obj, obj_pts, ko),
               self._solve(self.canon_scene, scene_pts, ks), ko, ks)
        return out + ((obj_pts, scene_pts) if points else ())

    @staticmethod
    def _solve(canon, pts, keep):
        R, t, _ = kabsch(canon, pts, weights=keep.to(pts.dtype))
        return torch.cat([t, _R_to_quat(R)], dim=-1)
