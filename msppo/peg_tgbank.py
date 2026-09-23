"""Scene-matched TraceGen goals for the peg eval set -- the missing row 4.

Row 4 has never been run end to end. Its goal has always been the SIMULATOR's
true goal with an error drawn from `results/peg_bank_n50.npz`, i.e. an error
measured on a DIFFERENT scene (peg_student_env.py:359). This module closes that:
it predicts on the very 256 scenes the eval runs, so the goal is TraceGen's answer
to the scene in front of it.

The StackCube analogue is `tracebank.py`; the three stages are the same, because
TraceGen inference (~0.8 s/sample) cannot live inside an RL loop and the two
repos are in disjoint conda envs (trace_gen has no mani_skill, maniskill has no
omegaconf/wandb -- measured, not assumed):

    render  ($PM)  the eval scenes -> a TraceGen episode dataset
    predict ($PG)  msgen.predict over that dataset
    solve   ($PM)  traces -> a predicted peg pose PER TRACE STEP

Two departures from `tracebank`, both measured in
`<private-repo>/experiments/20260812_e2e_replan`:

  * the solve keeps EVERY step, not just the endpoint. Endpoint error is 65.0 mm
    while step-8 error is 14.6 mm and step-4 is 8.2 mm, because error grows
    roughly with travelled distance (ratio 0.66 -> 0.22). The endpoint is the
    worst point on the trace to use as a subgoal, and every previous peg row 4
    used exactly that point. Which step to consume becomes an eval-time knob
    costing no extra inference.
  * scenes must be rendered at the SAME num_envs as the eval. Geometry is drawn
    once at construction under `reconfiguration_freq=0`, so num_envs is part of
    the scene identity: at 256 envs the pose hashes are bit-identical across
    processes, at 64 they differ (m0_envmatch.py).

    $PM -m msppo.peg_tgbank render --out data/bank/peg --num-envs 256 --seed 999
    $PG -m msgen.predict --dataset data/bank/peg --tag peg_e2e \
        --ckpt runs/peg_n50/ckpt/20260803_085308/best_model.pth
    $PM -m msppo.peg_tgbank solve --bank data/bank/peg \
        --pred results/preds/peg_e2e.npz --out results/peg_goals_e2e.npz
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from msgen.labels import grid_pixels, quat_to_R, unproject
from msgen.tasks import IMAGE_SIZE, TRAJ_STEPS, get_task


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def render(out_dir, num_envs=256, seed=999, clearance=0.01, cam="base_camera"):
    """Dump the eval scenes as a TraceGen episode dataset, one clip per env.

    Built through `make_peg_student_env` with the flags a grid perception student
    is evaluated under, so the scenes are the eval's scenes by construction
    rather than by a re-derivation that could drift.
    """
    from msppo.peg_student_env import make_peg_student_env
    import imageio.v2 as imageio

    env = make_peg_student_env(
        num_envs=num_envs, num_kp=64, seed=seed, clearance=clearance,
        max_episode_steps=100, perception=True, query="grid",
        tracegen_camera=True)
    base = env.base
    obs, _ = env.env.reset(seed=seed)

    sd, sp = obs["sensor_data"][cam], obs["sensor_param"][cam]
    rgb = _np(sd["rgb"]).astype(np.uint8)                       # [B,H,W,3]
    depth_m = _np(sd["depth"])[..., 0].astype(np.float32) / 1000.0
    seg = _np(sd["segmentation"])[..., 0].astype(np.int32)
    K = _np(sp["intrinsic_cv"]).astype(np.float64)              # [B,3,3]
    E = _np(sp["extrinsic_cv"]).astype(np.float64)              # [B,3,4]

    peg = _np(base.peg.pose.raw_pose)[:, :7].astype(np.float64)
    box = _np(base.box.pose.raw_pose)[:, :7].astype(np.float64)
    goal = np.concatenate([_np(base.goal_pose.p), _np(base.goal_pose.q)],
                          axis=-1).astype(np.float64)
    peg_id = _np(base.peg.per_scene_id).reshape(-1).astype(np.int64)
    box_id = _np(base.box.per_scene_id).reshape(-1).astype(np.int64)
    half = _np(base.peg_half_sizes).astype(np.float64)

    px = grid_pixels()
    ix = np.clip(px[:, 0].astype(int), 0, IMAGE_SIZE - 1)
    iy = np.clip(px[:, 1].astype(int), 0, IMAGE_SIZE - 1)
    instr = get_task("peg")["instructions"]
    os.makedirs(out_dir, exist_ok=True)

    cfgs = []
    for i in range(num_envs):
        clip = f"cfg_{i:05d}"
        d = f"{out_dir}/{clip}"
        for s in ("images", "depth", "samples"):
            os.makedirs(f"{d}/{s}", exist_ok=True)
        json.dump({f"instruction_{j+1}": s for j, s in enumerate(instr)},
                  open(f"{d}/three_instructions.json", "w"), indent=2)
        imageio.imwrite(f"{d}/images/00000.png", rgb[i])
        np.savez(f"{d}/depth/00000_raw.npz", depth=depth_m[i])
        # kept so the solve can pick the object's grid points without re-rendering
        np.savez_compressed(f"{d}/seg.npz", seg=seg[i])

        # A dummy target, exactly as in tracebank: the loader DROPS any sample
        # whose `movement_bool` is zero (trainer.py:297), so a static placeholder
        # would be filtered out and silently yield no prediction for this scene.
        # `predict_trajectory` never sees the target, so its content cannot leak.
        d0 = depth_m[i][iy, ix]
        valid = (d0 > 0.05) & (d0 < 3.0)
        traj = np.full((len(px), TRAJ_STEPS, 3), -np.inf, dtype=np.float32)
        ramp = np.linspace(0, 60.0, TRAJ_STEPS)[None, :]
        traj[valid, :, 0] = px[valid, 0:1] + ramp
        traj[valid, :, 1] = px[valid, 1:2]
        traj[valid, :, 2] = d0[valid, None]
        np.savez(f"{d}/samples/00000.npz", keypoints=px.astype(np.float32),
                 traj=traj, valid_steps=np.ones(TRAJ_STEPS, dtype=bool))

        cfgs.append(dict(clip=clip, env_idx=i, seed=seed,
                         peg=peg[i].tolist(), box=box[i].tolist(),
                         goal=goal[i].tolist(), peg_seg_id=int(peg_id[i]),
                         box_seg_id=int(box_id[i]), peg_half=half[i].tolist(),
                         K=K[i].tolist(), extrinsic_cv=E[i].tolist(),
                         n_grid_on_peg=int((seg[i][iy, ix] == peg_id[i]).sum())))
        if (i + 1) % 64 == 0:
            print(f"  rendered {i+1}/{num_envs}", flush=True)

    json.dump(dict(n=len(cfgs), num_envs=num_envs, seed=seed,
                   clearance=clearance, configs=cfgs),
              open(f"{out_dir}/configs.json", "w"))
    env.close()
    n_on = np.array([c["n_grid_on_peg"] for c in cfgs])
    print(f"wrote {len(cfgs)} scenes -> {out_dir}")
    print(f"  grid points on the peg: mean {n_on.mean():.1f} "
          f"min {n_on.min()} max {n_on.max()} | <3 points: {(n_on < 3).sum()} scenes")


def _R_to_quat(R):
    """[3,3] -> (w,x,y,z), via the largest-diagonal branch so no case is
    ill-conditioned. tracebank's version used the trace branch alone, which loses
    precision when the trace is near -1."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
             (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s,
             (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s,
             (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
             (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def kabsch_np(P, Q):
    """P,Q [N,3] -> (R,t) minimising ||R P + t - Q||. Unweighted: these are
    predicted points, all equally trusted, unlike the occlusion-masked solve in
    peg_student_env where zeroed points MUST be down-weighted."""
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, Qc - R @ Pc


def kabsch_wdisp(P, Q):
    """Displacement-WEIGHTED Kabsch: weight_i = ||Q_i - P_i|| / max_j ||Q_j - P_j|| + 0.05.

    Ported unchanged from <private-repo>/experiments/20260824_wallsel/w9_solvers.py (k_wdisp + _fit) so
    the production solve reproduces the `sv_*_wdisp_goals.npz` files bit for bit. Why it
    works: 15% of the traced object points are predicted near-static (the misclassified
    mode) and drag the rigid fit toward "no motion"; weighting by predicted displacement
    lets the points that actually move decide the transform. Measured on
    mix4gvwallsel_n800 (base -> wdisp, endpoint median): pickcube 40.5 -> 37.4 mm,
    stack 26.8 -> 23.2, peginsert 48.7 -> 35.1, liftpeg 272.8 -> 281.0."""
    d = np.linalg.norm(Q - P, axis=1)
    w = d / max(d.max(), 1e-9) + 0.05
    w = np.maximum(w, 1e-9)
    w = w / w.sum()
    Pc = (w[:, None] * P).sum(0)
    Qc = (w[:, None] * Q).sum(0)
    H = (P - Pc).T @ (w[:, None] * (Q - Qc))
    U, _, Vt = np.linalg.svd(H)
    dd = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, dd]) @ U.T
    return R, Qc - R @ Pc


def ransac_kabsch(src, dst, tol_m=0.03, iters=64, seed=0):
    """Largest subset of correspondences consistent with ONE rigid transform.

    The median-relative filter this replaces fails exactly where it is needed. In the
    worst scene 3 of 5 traced points are predicted near-static while 2 travel 300 mm,
    so the MEDIAN displacement is itself near-static, the threshold collapses to ~0,
    and nothing is dropped -- the least-squares fit still splits the difference and
    lands 271 mm out. Consensus does not care which group is the majority.

    Minimal sample is 3 points, the fewest that determine a rigid transform from
    non-collinear correspondences. With only ~5 points the sample space is tiny, so
    the iterations are effectively exhaustive.
    """
    n = len(src)
    if n < 3:
        return kabsch_np(src, dst), np.ones(n, dtype=bool)
    rng = np.random.default_rng(seed)
    best = (None, np.zeros(n, dtype=bool), -1)
    for _ in range(iters):
        idx = rng.choice(n, size=3, replace=False)
        try:
            R, t = kabsch_np(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        res = np.linalg.norm((R @ src.T).T + t - dst, axis=-1)
        inl = res < tol_m
        if inl.sum() > best[2]:
            best = ((R, t), inl, int(inl.sum()))
    inl = best[1]
    if inl.sum() < 3:
        return kabsch_np(src, dst), np.ones(n, dtype=bool)
    # refit on the full consensus set, which is what makes RANSAC better than the
    # minimal sample it was found with
    return kabsch_np(src[inl], dst[inl]), inl


def solve(bank_dir, pred_npz, out_npz, min_pts=3, robust=False,
          sam2_masks=None, ransac=False, ransac_tol=0.03, maxrel=0.0):
    """Traces -> a predicted peg pose at EVERY trace step, per env index.

    The traced object points at step 0 and at step s are exact correspondences,
    so Kabsch gives the SE(3) the object is predicted to have undergone by s.
    Applied to the peg's current pose that is a predicted peg pose, in the same
    (position, quaternion) form the env's own `goal_pose` has, which is what makes
    the two swappable at one hook (peg_student_env.py:355).

    Step 0 is NOT a prediction -- predict.py:61 prepends the GT step and the model
    emits displacements -- so step 0 is the identity transform by construction and
    is kept only to make the array's indexing match the trace's.
    """
    meta = json.load(open(f"{bank_dir}/configs.json"))
    cfgs = {c["clip"]: c for c in meta["configs"]}
    d = np.load(pred_npz, allow_pickle=True)
    pred, eps = d["pred"], d["episode_id"]     # [n, 400, TRAJ_STEPS, 3]
    # The query pixels come from THE BANK, not from grid_pixels(). They are no longer
    # the same for every scene: an object-aware bank places n_obj of the 400 queries on
    # that scene's peg (`<private-repo>/experiments/20260815_trace_e2e/b0_rebank_objaware.py`), and the
    # object-aware checkpoint expects exactly those. Recomputing a uniform grid here
    # would index the segmentation at pixels the model was never asked about -- silently,
    # since every shape still matches. Cross-validated cost of the wrong pairing:
    # 45.56 deg instead of 22.34 deg one way, 174.9 mm centroid error the other.
    def bank_px(clip):
        z = np.load(f"{bank_dir}/{clip}/samples/00000.npz")
        return (z["keypoints"].astype(np.float64) if "keypoints" in z
                else grid_pixels())          # older banks carry no keypoints

    sam = None
    if sam2_masks:
        # which traced points are on the peg, from SAM2 rather than the simulator's
        # segmentation. Precomputed per env index by msppo/peg_sam2_masks.py.
        z = np.load(sam2_masks)
        sam = z["peg_grid_sel"]
        if sam.shape[0] != meta["num_envs"]:
            raise SystemExit(f"sam2 masks cover {sam.shape[0]} scenes, bank has "
                             f"{meta['num_envs']}")
    n_env = meta["num_envs"]
    T = pred.shape[-2]
    goal_p = np.full((n_env, T, 3), np.nan, dtype=np.float64)
    goal_q = np.full((n_env, T, 4), np.nan, dtype=np.float64)
    # The DELTA is what TraceGen actually predicts: the rigid transform the traced
    # points undergo. Composing it with the simulator's object pose to get an
    # absolute goal (as goal_p/goal_q below do) puts privileged state into the
    # channel the policy consumes. Storing the delta lets the env compose it with
    # the PERCEIVED object pose instead, which is what a real cell has.
    dR = np.full((n_env, T, 3, 3), np.nan, dtype=np.float64)
    dt = np.full((n_env, T, 3), np.nan, dtype=np.float64)
    n_pts = np.zeros(n_env, dtype=np.int32)
    solved = np.zeros(n_env, dtype=bool)

    for i in range(len(pred)):
        clip = str(eps[i])
        if clip not in cfgs:
            continue
        c = cfgs[clip]
        j = int(c["env_idx"])
        K, E = np.array(c["K"]), np.array(c["extrinsic_cv"])
        segmap = np.load(f"{bank_dir}/{clip}/seg.npz")["seg"]
        px = bank_px(clip)
        ix = np.clip(px[:, 0].astype(int), 0, IMAGE_SIZE - 1)
        iy = np.clip(px[:, 1].astype(int), 0, IMAGE_SIZE - 1)
        p = pred[i]
        on_obj = sam[j] if sam is not None else (segmap[iy, ix] == c["peg_seg_id"])
        on = on_obj & np.isfinite(p).all(axis=(1, 2))
        n_pts[j] = int(on.sum())
        if on.sum() < min_pts:
            continue

        src = unproject(p[on, 0, :2], p[on, 0, 2].astype(np.float64), K, E)
        if robust:
            # Drop points TraceGen predicts as near-static while their peers move.
            # 15.8% of traced peg points are such (m7_perpoint.py), and a
            # least-squares rigid fit splits the difference between movers and
            # non-movers, landing short of the goal by about that share -- which is
            # exactly the measured 0.840 travel fraction. Membership is decided ONCE
            # at the endpoint, not per step, so the point set stays fixed along the
            # trace. This is 3DMF's Kabsch+RANSAC idea in its simplest form.
            end = unproject(p[on, -1, :2], p[on, -1, 2].astype(np.float64), K, E)
            disp = np.linalg.norm(end - src, axis=-1)
            keep = disp >= 0.33 * np.median(disp)
            if keep.sum() >= min_pts:
                idx = np.where(on)[0][keep]
                on = np.zeros_like(on)
                on[idx] = True
                src = src[keep]
                n_pts[j] = int(keep.sum())
        peg_p = np.array(c["peg"][:3])
        peg_R = quat_to_R(np.array(c["peg"][3:7]))
        inl_end = None
        if maxrel > 0:
            # The per-point displacement distribution is BIMODAL, measured on the test
            # clips with ground truth: 81.5% of truly-moving points have amplitude
            # 0.75-1.25x correct (median 0.986) and 15.1% are predicted near-static.
            # So the two groups separate cleanly -- but only if the threshold is taken
            # relative to the MAXIMUM. Relative to the median it collapses whenever the
            # non-movers are the majority, which is exactly the worst case: with
            # displacements [300, 290, 2, 1, 3] mm the median is 3 mm and nothing is
            # dropped, leaving the fit 271 mm out.
            end = unproject(p[on, -1, :2], p[on, -1, 2].astype(np.float64), K, E)
            disp = np.linalg.norm(end - src, axis=-1)
            keep = disp >= maxrel * disp.max()
            if keep.sum() >= min_pts:
                inl_end = keep
                n_pts[j] = int(keep.sum())
        if ransac:
            end = unproject(p[on, -1, :2], p[on, -1, 2].astype(np.float64), K, E)
            _, inl_end = ransac_kabsch(src, end, tol_m=ransac_tol)
            n_pts[j] = int(inl_end.sum())
        for s in range(T):
            dst = unproject(p[on, s, :2], p[on, s, 2].astype(np.float64), K, E)
            if inl_end is not None:
                # consensus decided ONCE at the endpoint, so the point set stays
                # fixed along the trace rather than changing membership per step
                R, t = kabsch_np(src[inl_end], dst[inl_end])
            else:
                R, t = kabsch_np(src, dst)
            dR[j, s], dt[j, s] = R, t
            # kept for the open-loop error tables and for backward compatibility;
            # the DEPLOYABLE path consumes dR/dt, never these.
            goal_p[j, s] = R @ peg_p + t
            goal_q[j, s] = _R_to_quat(R @ peg_R)
        solved[j] = True

    true_goal = np.array([c["goal"] for c in meta["configs"]])
    np.savez(out_npz, goal_p=goal_p, goal_q=goal_q, dR=dR, dt=dt,
             n_pts=n_pts, solved=solved,
             true_goal=true_goal, num_envs=n_env, seed=meta["seed"],
             clearance=meta["clearance"])

    print(f"solved {solved.sum()}/{n_env} scenes ({(~solved).sum()} under "
          f"{min_pts} traced peg points) -> {out_npz}")
    print(f"  grid points on the peg: mean {n_pts[solved].mean():.1f} "
          f"min {n_pts[solved].min()} max {n_pts[solved].max()}")
    print("\n  predicted pose at step s vs the TRUE goal pose "
          "(what row 4 consumes):")
    print(f"  {'step':>5} {'pos mm':>9} {'rot deg':>9}  {'<20mm&15deg':>12}")
    for s in [4, 8, 12, 16, 24, T - 1]:
        if s >= T:
            continue
        e = np.linalg.norm(goal_p[solved, s] - true_goal[solved, :3], axis=-1) * 1000
        dq = np.abs(np.einsum("ij,ij->i", goal_q[solved, s], true_goal[solved, 3:]))
        dr = np.degrees(2 * np.arccos(np.clip(dq, -1, 1)))
        print(f"  {s:>5} {e.mean():>9.1f} {dr.mean():>9.1f}  "
              f"{((e < 20) & (dr < 15)).mean():>11.1%}")
    print("\n  NOTE open-loop. The closed-loop student sees the real hole and has"
          "\n  read 0.708 where this kind of number predicted ~0.01."
          "\n  Do not infer success from the table above.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--out", required=True)
    r.add_argument("--num-envs", type=int, default=256)
    r.add_argument("--seed", type=int, default=999)
    r.add_argument("--clearance", type=float, default=0.01)
    s = sub.add_parser("solve")
    s.add_argument("--bank", required=True)
    s.add_argument("--pred", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--min-pts", type=int, default=3)
    s.add_argument("--sam2-masks", default=None,
                   help="take the on-peg grid selection from SAM2 instead of the "
                        "simulator's segmentation")
    s.add_argument("--maxrel", type=float, default=0.0,
                   help="keep traced points whose displacement is at least this "
                        "fraction of the scene's MAXIMUM. Exploits the measured "
                        "bimodality (a spike near 0, a bulk near 1) and is immune to "
                        "the non-movers being the majority, unlike --robust")
    s.add_argument("--ransac", action="store_true",
                   help="largest consistent subset instead of the median-relative "
                        "drop, which fails when the non-movers are the majority")
    s.add_argument("--ransac-tol", type=float, default=0.03)
    s.add_argument("--robust", action="store_true",
                   help="drop near-static traced points before the fit: 84.2 -> "
                        "63.7 mm open loop (m7_perpoint.py)")
    a = ap.parse_args()
    if a.cmd == "render":
        render(a.out, a.num_envs, a.seed, a.clearance)
    else:
        solve(a.bank, a.pred, a.out, a.min_pts, a.robust, a.sam2_masks,
              a.ransac, a.ransac_tol, a.maxrel)


if __name__ == "__main__":
    main()
