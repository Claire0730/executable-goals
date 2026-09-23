"""Synthesize exact sim ground-truth 3D traces and write them in TraceGen's
episode format.

For a query frame t0, each of the 400 grid pixels becomes a trace:

  1. segmentation id -> owning rigid body; depth -> unproject to a world point p0
  2. for each future step t, apply that body's rigid motion T(t) . T(t0)^-1 to p0
  3. reproject to (pixel_x, pixel_y, camera_depth)

Static bodies keep their point fixed. Pixels with no owning body (id 0, or an
id absent from the body table) are written as -inf, which is how the dataloader
builds `trajectory_mask` (datasets.py:542).

Occlusion is deliberately NOT treated as invalidity: a trace describes the 3D
motion of a material point, which stays well-defined once the point is hidden.

Output layout, one directory per clip:
    <out>/<clip>/images/<stem>.png
    <out>/<clip>/depth/<stem>_raw.npz     key 'depth', METRES
    <out>/<clip>/samples/<stem>.npz       keys 'keypoints','traj','valid_steps'
    <out>/<clip>/three_instructions.json

Run:
  python -m msgen.labels --raw data/raw/peg --out data/ds/peg_train --task peg
"""
from __future__ import annotations

import argparse
import json
import os

import imageio.v2 as imageio
import numpy as np

from msgen.tasks import GRID, IMAGE_SIZE, NUM_KPS, TASKS, TRAJ_STEPS, get_task

# Depth outside this range is sensor void / the far plane, not geometry.
DEPTH_MIN_M, DEPTH_MAX_M = 0.05, 3.0

# Minimum total image-space path (px) for a query frame to be worth emitting.
# Matches the trainer's movement_bool threshold of 0.1 normalized (~38 px at 384).
MIN_PATH_PX = 45.0


def grid_pixels() -> np.ndarray:
    """The 400 query pixels, ROW-MAJOR: index k -> (y=k//20, x=k%20).

    This ordering is mandatory. cogvideox_flow.py:393 reshapes [B,400,32,3] into
    [B,20,20,32,3]; any other ordering scrambles the decoder's spatial latent
    with no error raised.
    """
    step = IMAGE_SIZE / GRID
    centers = (np.arange(GRID) + 0.5) * step
    ys, xs = np.meshgrid(centers, centers, indexing="ij")   # ij -> row-major
    return np.stack([xs.ravel(), ys.ravel()], axis=1)       # [400,2] as (x,y)


def _fps(xs: np.ndarray, ys: np.ndarray, n: int, seed: int):
    """Farthest-point sampling over mask pixels: the best-conditioned sampler measured.

    `<private-repo>/experiments/20260814_kpsample/s0_collinearity.py`, peg test clips, same point count as
    the grid: the share of frames whose 3D point cloud is ill-conditioned for a rotation fit
    (sigma2/sigma1 < 0.1) goes 38% (uniform grid) -> 13% (uniform over the mask) -> 4% (this).
    Degenerate frames (< 0.05) go 31% -> 5% -> 0%.
    """
    P = np.stack([xs, ys], 1).astype(np.float64)
    if len(P) <= n:
        return xs, ys
    rng = np.random.default_rng(seed)
    sel = [int(rng.integers(len(P)))]
    d = np.linalg.norm(P - P[sel[0]], axis=1)
    for _ in range(n - 1):
        i = int(np.argmax(d))
        sel.append(i)
        d = np.minimum(d, np.linalg.norm(P - P[i], axis=1))
    s = np.asarray(sel)
    return xs[s], ys[s]


def _decoy_cluster(ox, oy, seg, target_id):
    """The object's point cluster, moved onto the background with its shape intact.

    The control for the re-lattice cost. `object_aware_pixels` does two things at once: it puts
    points ON the target (the intended benefit) and it re-sorts all 400 queries onto the lattice
    (the cost -- the query interface is index-bound and `patch_size 2` binds neighbouring 2x2
    queries into one token, so a clustered insertion makes the lattice rows non-uniform and
    splits pairs that used to share a token). Those two have never been measured apart.

    A decoy cluster keeps the SHAPE, SIZE, SPREAD and COUNT of the object cluster and only moves
    it off the target, so it pays the identical re-lattice cost with none of the benefit.

    Placement, in order, first candidate whose points all miss the target mask:
      1. point reflection through the image centre
      2. horizontal mirror
      3. vertical mirror
      4. translations along the image diagonal
    Returns None if the target covers so much of the frame that no placement is clean, in which
    case the caller falls back to the uniform grid and the clip is excluded from this arm.
    """
    cx = (IMAGE_SIZE - 1) / 2.0
    cands = [(2 * cx - ox, 2 * cx - oy), (2 * cx - ox, oy), (ox, 2 * cx - oy)]
    for s in (0.25, -0.25, 0.4, -0.4):
        cands.append((ox + s * IMAGE_SIZE, oy + s * IMAGE_SIZE))
    for dx, dy in cands:
        dx = np.clip(dx, 0, IMAGE_SIZE - 1)
        dy = np.clip(dy, 0, IMAGE_SIZE - 1)
        if not (seg[dy.astype(int), dx.astype(int)] == target_id).any():
            return dx, dy
    return None


def object_aware_pixels(seg: np.ndarray, target_id: int, n_obj: int, seed: int,
                        decoy: bool = False) -> np.ndarray:
    """The 400 query pixels with `n_obj` of them placed ON the target, topology preserved.

    WHY NOT JUST OVERWRITE COORDINATES. `cogvideox_flow.py:393` reshapes [B,400,32,3] into
    [B,20,20,32,3] and CogVideoX then applies a 2D patch embedding (patch_size 2) plus 3D
    sincos over that grid, so the query INDEX ORDER is read as an image-like 2D layout.
    TraceForge's own non-uniform paths (`--query_alloc_ckpt`, `--query_object_mask red`) write
    the new coordinates straight into the grid's slots, which keeps every shape and silently
    scrambles that layout. So the union of background and object points is re-assigned onto the
    lattice here: sorted into GRID rows by y, then by x within each row. Neighbours in the index
    stay neighbours in the image.

    Reproduces `grid_pixels()` EXACTLY at n_obj=0, which is the identity test.

    Only the target moves the rotation estimate, and only elongated targets are ill-conditioned
    in the first place (peg 0.14, cubes 0.77-0.85), so the background lattice is kept: it is
    what the model has always been trained on, and thinning it is a second change.
    """
    base = grid_pixels()
    if n_obj <= 0:
        return base
    ys, xs = np.nonzero(seg == target_id)
    if len(xs) < 3:
        return base
    ox, oy = _fps(xs, ys, min(n_obj, len(xs)), seed)
    if decoy:
        moved = _decoy_cluster(ox, oy, seg, target_id)
        if moved is None:
            return base
        ox, oy = moved
    # drop the background points nearest the object, so the count stays at NUM_KPS
    keep = np.ones(len(base), dtype=bool)
    if len(ox) > 0:
        d = np.linalg.norm(base[:, None, :] - np.stack([ox, oy], 1)[None, :, :], axis=2).min(1)
        keep[np.argsort(d)[: len(ox)]] = False
    pts = np.concatenate([base[keep], np.stack([ox, oy], 1).astype(np.float64)], 0)
    # re-assign onto the lattice: GRID rows by y, then by x inside each row
    order = np.argsort(pts[:, 1], kind="stable")
    pts = pts[order]
    rows = [pts[i * GRID:(i + 1) * GRID] for i in range(GRID)]
    rows = [r[np.argsort(r[:, 0], kind="stable")] for r in rows]
    return np.concatenate(rows, 0)


def quat_to_R(q: np.ndarray) -> np.ndarray:
    """SAPIEN raw_pose quaternion is (w,x,y,z). Returns [...,3,3]."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(*q.shape[:-1], 3, 3)


def unproject(px: np.ndarray, depth_m: np.ndarray, K: np.ndarray, ext: np.ndarray) -> np.ndarray:
    """Pixels + camera depth -> world points. ext is 3x4 world->camera [R|t]."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (px[:, 0] - cx) / fx * depth_m
    y = (px[:, 1] - cy) / fy * depth_m
    p_cam = np.stack([x, y, depth_m], axis=1)
    R, t = ext[:, :3], ext[:, 3]
    return (p_cam - t) @ R                                   # R^T (p_cam - t)


def project(p_world: np.ndarray, K: np.ndarray, ext: np.ndarray):
    """World points -> (pixel[N,2], camera depth[N])."""
    R, t = ext[:, :3], ext[:, 3]
    p_cam = p_world @ R.T + t
    z = p_cam[..., 2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    safe = np.where(np.abs(z) < 1e-6, 1e-6, z)
    u = p_cam[..., 0] / safe * fx + cx
    v = p_cam[..., 1] / safe * fy + cy
    return np.stack([u, v], axis=-1), z


def future_indices(t0: int, n_frames: int, n_steps: int = TRAJ_STEPS - 1) -> np.ndarray:
    """Uniform-in-TIME resampling of the clip's remaining frames.

    Kept as the `--time-mode uniform` alternative. It is a poor default: demo
    motion is very non-uniform in speed (fast reach, slow manipulation), so the
    reach phase produces per-step deltas that saturate the decoder's +/-1 clamp
    for 5-10% of moving point-steps.
    """
    return np.linspace(t0, n_frames - 1, n_steps + 1)[1:]


def arclen_indices(uv: np.ndarray, valid: np.ndarray,
                   n_steps: int = TRAJ_STEPS - 1) -> np.ndarray:
    """Fractional frame indices spaced by equal SCENE MOTION, not equal time.

    uv is [400, N, 2] image-space positions over the remaining frames. We take
    one global time-warp for the whole sample -- all 400 points must share the
    same 32 future timestamps, since the trace is a spatiotemporal field over
    the grid, not 400 independent curves.

    Equalizing motion per step keeps the largest per-step displacement near the
    mean, which is what keeps the labels inside the action-scale clamp.
    """
    n = uv.shape[1]
    if n < 2:
        return np.zeros(n_steps)
    step = np.linalg.norm(np.diff(uv[valid], axis=1), axis=-1)   # [Nvalid, N-1]
    moving = step.sum(axis=1) > 1e-6
    speed = step[moving].mean(axis=0) if moving.any() else step.mean(axis=0)

    s = np.concatenate([[0.0], np.cumsum(speed)])
    if s[-1] <= 1e-9:                       # nothing moves; fall back to time
        return np.linspace(0, n - 1, n_steps + 1)[1:]
    targets = np.linspace(0, s[-1], n_steps + 1)[1:]
    return np.interp(targets, s, np.arange(n))


def tf_robust_seglen(uv: np.ndarray, valid: np.ndarray, top_percent: float = 0.02):
    """Per-step scene motion the way TraceForge measures it: the mean of the FASTEST tracks.

    `TraceForge/infer.py:204-208` takes `ceil(top_percent * N)` largest per-segment lengths and
    averages those, where our `arclen_indices` averages over every MOVING track. With ~14% of
    queries moving, those two speeds differ: TraceForge's is set by the object and the gripper,
    ours by the average moving point.
    """
    step = np.linalg.norm(np.diff(uv[valid], axis=1), axis=-1)     # [Nvalid, N-1]
    if step.size == 0:
        return None
    N = step.shape[0]
    k = max(1, int(np.ceil(top_percent * N)))
    return np.partition(step, N - k, axis=0)[N - k:, :].mean(axis=0)   # [N-1]


def tf_retarget_indices(uv: np.ndarray, valid: np.ndarray, interval: float,
                        n_steps: int = TRAJ_STEPS - 1) -> np.ndarray:
    """TraceForge's time convention, ported onto our exact sim ground truth.

    The difference this exists to test. TraceGen was pretrained on labels where a step is a
    FIXED AMOUNT OF MOTION (`interval` along the robust cumulative arc length,
    `infer.py:219-222`), so a slow clip simply produces fewer steps and the rest are padded with
    -inf. Our `arclen_indices` instead always emits exactly `n_steps` covering the whole
    remaining clip, i.e. a step means "1/32 of whatever is left". Same arc-length family,
    different meaning per step -- and the pretrained weights learned the first one.

    `interval` MUST be a constant across the dataset, otherwise this degenerates into our own
    fixed-count rule with a different speed estimate. Calibrate it once with
    `<private-repo>/experiments/20260816_tfconv/t0_calibrate.py` and pass the value in.

    Returns up to `n_steps` fractional frame indices; a short clip returns fewer, and
    `synth_clip` leaves the remaining steps at -inf, which is how TraceForge pads.
    """
    n = uv.shape[1]
    if n < 2:
        return np.zeros(0)
    robust = tf_robust_seglen(uv, valid)
    if robust is None:
        return np.linspace(0, n - 1, n_steps + 1)[1:]
    s = np.concatenate([[0.0], np.cumsum(robust)])
    total = float(s[-1])
    if total <= 1e-9:
        return np.linspace(0, n - 1, n_steps + 1)[1:]
    n_out = min(int(np.floor(total / interval)), n_steps)
    if n_out < 1:
        n_out = 1
    targets = interval * np.arange(1, n_out + 1, dtype=np.float64)
    targets[-1] = min(targets[-1], total)
    return np.interp(targets, s, np.arange(n))


def body_transforms(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """poses [T,B,7] (p + wxyz) -> R [T,B,3,3], p [T,B,3]."""
    return quat_to_R(poses[..., 3:7]), poses[..., :3]


def _positions_all_frames(raw, t0, px, valid, row, p0):
    """Reproject every grid point at every remaining frame: [400, N, 3]."""
    K, ext = raw["K"], raw["extrinsic_cv"]
    R_all, T_all = body_transforms(raw["poses"])
    n_frames = len(raw["seg"])
    r0 = np.where(valid, row, 0)

    R0, t0_p = R_all[t0][r0], T_all[t0][r0]
    local = np.einsum("nji,nj->ni", R0, p0 - t0_p)

    out = np.zeros((len(px), n_frames - t0, 3))
    for j, t in enumerate(range(t0, n_frames)):
        p_t = np.einsum("nij,nj->ni", R_all[t][r0], local) + T_all[t][r0]
        uv, z = project(p_t, K, ext)
        out[:, j, :2] = uv
        out[:, j, 2] = z
    return out


def synth_clip(raw: dict, t0: int, px: np.ndarray, time_mode: str = "arclen",
               tf_interval: float = 0.0):
    """Traces for one query frame.

    Returns traj [400,33,3] with (x_pixel, y_pixel, z_metres); invalid -> -inf.
    """
    K, ext = raw["K"], raw["extrinsic_cv"]
    seg, depth_mm, poses = raw["seg"], raw["depth_mm"], raw["poses"]
    body_ids = raw["body_ids"]
    n_frames = len(seg)

    depth0 = depth_mm[t0].astype(np.float64) / 1000.0
    ix = np.clip(px[:, 0].astype(int), 0, IMAGE_SIZE - 1)
    iy = np.clip(px[:, 1].astype(int), 0, IMAGE_SIZE - 1)
    d0 = depth0[iy, ix]
    sid = seg[t0][iy, ix]

    # Map each query pixel to a row in the body table; -1 when unattributable.
    id_to_row = {int(b): r for r, b in enumerate(body_ids)}
    row = np.array([id_to_row.get(int(s), -1) for s in sid], dtype=np.int64)

    valid = (row >= 0) & (d0 > DEPTH_MIN_M) & (d0 < DEPTH_MAX_M)

    p0 = unproject(px, d0, K, ext)                       # [400,3] world at t0

    traj = np.full((NUM_KPS, TRAJ_STEPS, 3), -np.inf, dtype=np.float64)
    traj[valid, 0, :2] = px[valid]
    traj[valid, 0, 2] = d0[valid]

    # Unattributable points get a FINITE step 0 and -inf afterwards, rather than
    # -inf everywhere. `fill_traj_with_last_valid` (datasets.py:628) forward-fills
    # from the last finite step but skips points with no finite step at all, so
    # all-(-inf) points survive into the metric's cumsum and, since 0 * nan = nan,
    # turn the masked trajectory MSE/MAE into nan. One finite step is enough for
    # the forward-fill to make the row finite while `trajectory_mask` (built as
    # traj != -inf) still excludes steps 1..32 from every loss and metric.
    traj[~valid, 0, :2] = px[~valid]
    traj[~valid, 0, 2] = 0.0

    # Positions at every remaining frame, then a single global time-warp.
    pos = _positions_all_frames(raw, t0, px, valid, row, p0)     # [400,N,3]
    if time_mode == "tfretarget":
        fidx = tf_retarget_indices(pos[:, :, :2], valid, tf_interval)
    elif time_mode == "arclen":
        fidx = arclen_indices(pos[:, :, :2], valid)
    else:
        fidx = future_indices(t0, n_frames) - t0

    n = pos.shape[1]
    for s, tf in enumerate(fidx, start=1):
        lo = int(np.floor(tf))
        hi = min(lo + 1, n - 1)
        w = tf - lo
        # Linear interpolation between adjacent frames is sufficient here: the
        # points are already reprojected, and consecutive kept frames are close.
        p = (1 - w) * pos[:, lo, :] + w * pos[:, hi, :]
        traj[valid, s, :] = p[valid]

    return traj, valid


def write_clip(raw_path: str, out_dir: str, task: str, stride: int, min_future: int,
               time_mode: str = "arclen", n_obj: int = 0, body_names=None,
               tf_interval: float = 0.0,
               decoy: bool = False,
               clip_target_id=None):
    raw = dict(np.load(raw_path))
    clip = os.path.splitext(os.path.basename(raw_path))[0]
    n_frames = len(raw["seg"])
    px = grid_pixels()
    # Object-aware sampling is per QUERY FRAME, because the object moves; the uniform grid was
    # a per-clip constant. `keypoints` is already stored per sample, so downstream consumers
    # that READ it are unaffected -- only those that RECOMPUTE `grid_pixels()` need updating
    # (msgen/objmetrics.py:25-35 is one).
    target_id = None
    if n_obj > 0:
        from msgen.objmetrics import target_id_for
        # `target_id_for` prefers the id the clip generator recorded, which is the only thing
        # that works for a task whose object actor is named per scene (PickSingleYCB).
        target_id = target_id_for(
            dict(body_names=list(body_names or []), body_ids=raw["body_ids"],
                 target_id=clip_target_id), task)
        if target_id is None:
            print(f"  [{clip}] target body not resolvable; falling back to the grid")

    d = f"{out_dir}/{clip}"
    for sub in ("images", "depth", "samples"):
        os.makedirs(f"{d}/{sub}", exist_ok=True)

    instr = get_task(task)["instructions"]
    json.dump({f"instruction_{i+1}": s for i, s in enumerate(instr)},
              open(f"{d}/three_instructions.json", "w"), indent=2)

    written = 0
    # Query frames need enough motion left to define a trace at all.
    for t0 in range(0, n_frames - min_future, stride):
        if target_id is not None:
            # seed on t0 so a rebuild is reproducible and two frames do not share a draw
            px = object_aware_pixels(raw["seg"][t0], target_id, n_obj, seed=t0,
                                     decoy=decoy)
        traj, valid = synth_clip(raw, t0, px, time_mode, tf_interval)
        if valid.sum() < 10:
            continue
        # A query frame with no remaining motion has no trace to predict, and
        # the trainer would drop it anyway via movement_bool (which needs one
        # point whose normalized path exceeds 0.1, i.e. ~38 px at 384).
        #
        # Padded rows (tfretarget writes < TRAJ_STEPS steps and leaves an -inf
        # tail) used to poison this guard: diff across the -inf boundary is
        # inf/nan, `nan < MIN_PATH_PX` is False, and the DEGENERATE sample was
        # written while fully-finite low-motion ones were correctly skipped --
        # the exact inversion that voided an earlier build.
        # Sum each row's path over its finite prefix only.
        d2 = np.abs(np.diff(traj[valid][:, :, :2], axis=1))
        path = np.nansum(np.where(np.isfinite(d2), d2, np.nan), axis=(1, 2))
        if path.size == 0 or np.nanmax(path) < MIN_PATH_PX:
            continue
        stem = f"{t0:05d}"
        imageio.imwrite(f"{d}/images/{stem}.png", raw["rgb"][t0])
        np.savez(f"{d}/depth/{stem}_raw.npz",
                 depth=(raw["depth_mm"][t0].astype(np.float32) / 1000.0))
        np.savez(f"{d}/samples/{stem}.npz",
                 keypoints=px.astype(np.float32),
                 traj=traj.astype(np.float32),
                 valid_steps=np.ones(TRAJ_STEPS, dtype=bool))
        written += 1
    return clip, written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="dir of replay .npz clips")
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--clips", type=int, default=None, help="use only the first N clips")
    ap.add_argument("--stride", type=int, default=2, help="query-frame stride")
    ap.add_argument("--min-future", type=int, default=8,
                    help="skip query frames with fewer than this many frames left")
    ap.add_argument("--time-mode", choices=["arclen", "uniform", "tfretarget"],
                    default="arclen",
                    help="space the 32 future steps by equal scene motion (default) or equal time")
    ap.add_argument("--tf-interval", type=float, default=0.0,
                    help="required for --time-mode tfretarget: the constant arc-length step, "
                         "calibrated offline; pass the value explicitly")
    ap.add_argument("--decoy", action="store_true",
                    help="control arm: move the object cluster onto the BACKGROUND, keeping its "
                         "shape, so the re-lattice cost is paid with none of the benefit")
    ap.add_argument("--n-obj", type=int, default=0,
                    help="place this many of the 400 queries ON the target via farthest-point "
                         "sampling of its mask, replacing the nearest background points and "
                         "re-assigning the union onto the 20x20 lattice. 0 = the uniform grid, "
                         "byte-identical to before. Conditioning saturates by n=8 on peg "
                         "(measured offline).")
    args = ap.parse_args()

    manifest = json.load(open(f"{args.raw}/manifest.json"))
    clips = sorted(f for f in os.listdir(args.raw) if f.endswith(".npz"))
    if args.clips is not None:
        clips = clips[:args.clips]
    os.makedirs(args.out, exist_ok=True)

    total, per_clip = 0, []
    for c in clips:
        meta = next((m for m in manifest["clips"]
                     if m["clip"] == os.path.splitext(c)[0]), None)
        name, n = write_clip(f"{args.raw}/{c}", args.out, args.task,
                             args.stride, args.min_future, args.time_mode,
                             n_obj=args.n_obj, decoy=args.decoy,
                             tf_interval=args.tf_interval,
                             body_names=(meta or {}).get("body_names"),
                             clip_target_id=(meta or {}).get("target_id"))
        per_clip.append(dict(clip=name, samples=n))
        total += n
        print(f"[{name}] {n} samples", flush=True)

    # GRID is recorded so a consumer cannot silently index a 400-point grid into a
    # 1600-point dataset: msgen.tasks reads it from MSGEN_GRID, so a process launched
    # without the right value would produce a different grid_pixels() with no error.
    json.dump(dict(task=args.task, grid=GRID, num_kps=NUM_KPS, n_obj=args.n_obj,
                   tf_interval=args.tf_interval,
                   decoy=bool(args.decoy),
                   source_raw=args.raw, stride=args.stride,
                   time_mode=args.time_mode,
                   min_future=args.min_future, traj_steps=TRAJ_STEPS,
                   n_clips=len(clips), n_samples=total, clips=per_clip,
                   source_manifest=manifest["clips"][:len(clips)]),
              open(f"{args.out}/dataset.json", "w"), indent=2)
    print(f"wrote {total} samples from {len(clips)} clips -> {args.out}")


if __name__ == "__main__":
    main()
