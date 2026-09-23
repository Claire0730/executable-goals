"""Scene-matched TraceGen goals for ANY task in the shared stack.

Generalises `msppo/peg_tgbank.py`, which is hardcoded to the peg in its render
stage (peg/box actors, peg_half_sizes, the env's own `goal_pose` property). The
SOLVE stage was already almost task-agnostic and is reused verbatim in spirit.

Three stages, in two mutually exclusive conda envs -- `trace_gen` has no
`mani_skill` and `maniskill` has no `omegaconf`/`wandb`, so the interface between
them is a directory of files and one npz:

    render   ($PM)  the eval scenes -> a TraceGen episode dataset
    predict  ($PG)  msgen.predict over that dataset -> results/preds/<tag>.npz
    solve    ($PM)  traces -> a rigid transform dR/dt PER TRACE STEP, per env

    $PM -m msppo.task_tgbank render --task stack --out data/bank/stack \
        --num-envs 256 --seed 999
    $PG -m msgen.predict --dataset data/bank/stack --tag stack_e2e \
        --ckpt $CK_MIX4
    $PM -m msppo.task_tgbank solve --task stack --bank data/bank/stack \
        --pred results/preds/stack_e2e.npz --out results/stack_goals.npz

WHAT dR/dt IS AND WHY IT IS THE ONLY DEPLOYABLE FORM. TraceGen predicts where the
traced points GO. Kabsch on (points at step 0, points at step s) gives the rigid
transform the object is predicted to undergo -- and that is scene-relative, so
the env can compose it with the PERCEIVED object pose. Composing it here with the
simulator's object pose instead would produce an absolute goal that carries
privileged state into the policy's input; `peg_eval_student.py:55-58` marks that
form as valid only as an UPPER BOUND. This module therefore stores dR/dt as the
product and treats the absolute pose as a diagnostic.

⚠️ num_envs IS PART OF SCENE IDENTITY. Object geometry and layout are drawn once
at construction under `reconfiguration_freq=0`, so a bank rendered at 64 envs
does NOT describe the scenes an eval at 256 envs will visit. `task_eval_student`
refuses the mismatch rather than silently scoring the wrong scenes.

⚠️ EVERY STEP IS KEPT, not just the endpoint. Error grows with travelled
distance, so the endpoint is the worst point on the trace to use as a subgoal.
Which step to consume becomes an eval-time knob costing no extra inference.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from msgen.labels import grid_pixels, unproject
from msgen.tasks import IMAGE_SIZE, TRAJ_STEPS, get_task
from msppo.peg_tgbank import _R_to_quat, kabsch_np, kabsch_wdisp, ransac_kabsch

# task -> (msgen task key, env attribute of the object, of the fixture or None)
TASK_MAP = {
    "liftpeg": ("liftpeg", "peg", None),
    "pusht": ("pusht", "tee", "goal_tee"),
    "stack": ("stack", "cubeA", "cubeB"),
    # goal_site is in _hidden_objects (pick_cube.py:104): no scene channel, and
    # the goal reaches the student ONLY through the planner.
    "pickcube": ("pickcube", "cube", None),
    # PushCube-v1 names the cube `obj`; the goal region is a flat visual, no scene channel.
    "pushcube": ("pushcube", "obj", None),
    # PegInsertionSide through the SHARED constructor. `msppo/peg_tgbank.py`
    # already renders a peg bank, but through the OLD peg line -- different
    # gym.make call, different wrapper, and peg randomises its geometry per
    # scene, so scene i of that bank is not guaranteed to be scene i of a
    # `task_student_env` eval. A bank must be rendered by the same constructor
    # the eval uses or the goals describe different objects.
    "peginsert": ("peg", "peg", "box"),
    # Zero-shot probe: PlaceSphere-v1 names the sphere `obj`
    # (place_sphere.py:141). scn_attr must be None: placesphere_kp_env has no
    # fixture-perception channel (raises on scene_kp=True); the goal is derived
    # from the bin's pose by `_goal_from_obs`, so nothing is lost for `solve`.
    "placesphere": ("placesphere", "obj", None),
}


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def render(task, out_dir, num_envs=256, seed=999, num_kp=64, cam="base_camera",
           show_goal=False, robot=None):
    """Dump the eval scenes as a TraceGen episode dataset, one clip per env.

    Built through `make_task_student_env` with the flags a grid perception
    student is evaluated under, so the scenes ARE the eval's scenes by
    construction rather than by a re-derivation that could drift.
    """
    import imageio.v2 as imageio

    from msppo.kp_teacher import TASKS
    from msppo.task_student_env import make_task_student_env

    mkey, obj_attr, scn_attr = TASK_MAP[task]
    spec = dict(TASKS[task])
    if robot:
        spec["robot"] = robot
    env = make_task_student_env(
        task, num_envs=num_envs, num_kp=num_kp, seed=seed,
        max_episode_steps=spec["horizon"], perception=True, query="grid",
        scene_kp=scn_attr is not None, robot=spec["robot"])
    base = env.base
    # `env.kp.reset()`, NOT `env.kp.env.reset()`. The inner call resets the
    # ManiSkillVectorEnv and BYPASSES `KeypointX.reset`, which is the only place
    # `_commit_goal` runs. Bypassing it leaves the committed goal state at its
    # constructor value -- identity quaternion for StackCube/PickCube, zeros for
    # LiftPegUpright's goal xy -- so the `goal` written into configs.json is not
    # the goal the EVAL will use.
    obs, _ = env.kp.reset(seed=seed)
    if show_goal:
        # Same two-step unhide as msgen.replay.unhide_goal (remove from the
        # hidden list AND show_visual -- the hidden flag survives removal),
        # then RE-CAPTURE: the reset's obs above was rendered with it hidden.
        from msgen.replay import unhide_goal
        unhide_goal(base)
        obs = base.get_obs()

    sd, sp = obs["sensor_data"][cam], obs["sensor_param"][cam]
    rgb = _np(sd["rgb"]).astype(np.uint8)
    depth_m = _np(sd["depth"])[..., 0].astype(np.float32) / 1000.0
    seg = _np(sd["segmentation"])[..., 0].astype(np.int32)
    K = _np(sp["intrinsic_cv"]).astype(np.float64)
    E = _np(sp["extrinsic_cv"]).astype(np.float64)

    raw = env.kp.raw_from_sim()
    obj_pose = _np(getattr(base, obj_attr).pose.raw_pose)[:, :7].astype(np.float64)
    gp, gq = env.kp._goal_from_obs(raw)
    goal = np.concatenate([_np(gp), _np(gq)], axis=-1).astype(np.float64)
    obj_id = _np(getattr(base, obj_attr).per_scene_id).reshape(-1).astype(np.int64)
    scn_id = (_np(getattr(base, scn_attr).per_scene_id).reshape(-1).astype(np.int64)
              if scn_attr else np.full(num_envs, -1, dtype=np.int64))

    px = grid_pixels()
    ix = np.clip(px[:, 0].astype(int), 0, IMAGE_SIZE - 1)
    iy = np.clip(px[:, 1].astype(int), 0, IMAGE_SIZE - 1)
    instr = get_task(mkey)["instructions"]
    if os.path.exists(f"{out_dir}/configs.json"):
        old = json.load(open(f"{out_dir}/configs.json"))
        if old.get("task") != task:
            raise SystemExit(
                f"{out_dir} already holds a bank for {old.get('task')!r} (or an "
                f"older schema -- `msppo/tracebank.py` writes to data/bank/stack). "
                f"Pick a different --out.")
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
        np.savez_compressed(f"{d}/seg.npz", seg=seg[i])

        # A dummy target, exactly as in peg_tgbank: the loader DROPS any sample
        # whose `movement_bool` is zero, so a static placeholder would be
        # filtered out and silently yield no prediction for this scene.
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
                         obj=obj_pose[i].tolist(), goal=goal[i].tolist(),
                         obj_seg_id=int(obj_id[i]), scn_seg_id=int(scn_id[i]),
                         K=K[i].tolist(), extrinsic_cv=E[i].tolist(),
                         n_grid_on_obj=int((seg[i][iy, ix] == obj_id[i]).sum())))
        if (i + 1) % 64 == 0:
            print(f"  rendered {i+1}/{num_envs}", flush=True)

    json.dump(dict(task=task, n=len(cfgs), num_envs=num_envs, seed=seed,
                   horizon=spec["horizon"], configs=cfgs),
              open(f"{out_dir}/configs.json", "w"))
    env.close()
    n_on = np.array([c["n_grid_on_obj"] for c in cfgs])
    print(f"wrote {len(cfgs)} scenes -> {out_dir}")
    print(f"  grid points on the object: mean {n_on.mean():.1f} min {n_on.min()} "
          f"max {n_on.max()} | <3 points: {(n_on < 3).sum()} scenes")
    if (n_on < 3).sum():
        print("  ⚠ scenes under 3 points have NO rigid solution and are reported "
              "as planner failures, never backfilled.")


def solve(task, bank_dir, pred_npz, out_npz, min_pts=3, maxrel=0.0,
          ransac=False, ransac_tol=0.03, weight=None):
    """Traces -> the rigid transform dR/dt at EVERY trace step, per env index.

    weight: None = plain Kabsch (used by scripts/30_goals.sh); "disp" =
    displacement-weighted Kabsch (`peg_tgbank.kabsch_wdisp`, optional).
    Recorded in the npz as `weight`."""
    fit = {None: kabsch_np, "none": kabsch_np, "disp": kabsch_wdisp}[weight]
    meta = json.load(open(f"{bank_dir}/configs.json"))
    # A bank written by `tracebank.py` has a different schema: it
    # writes {clip, cubeA, cubeB, cubeA_seg_id, K, extrinsic_cv} and no
    # `num_envs` / `env_idx` / `obj_seg_id`. Solving that with this code would
    # index fields that are not there, or worse, silently pair the wrong scenes.
    # Fail loudly and name the fix.
    missing = [k for k in ("task", "num_envs", "configs") if k not in meta]
    if missing or "env_idx" not in meta["configs"][0]:
        raise SystemExit(
            f"{bank_dir}/configs.json is not a task_tgbank bank (missing "
            f"{missing or ['env_idx']}). `msppo/tracebank.py` writes a DIFFERENT "
            f"schema to data/bank/stack. Render a fresh one to its own directory:\n"
            f"  $PM -m msppo.task_tgbank render --task {task} "
            f"--out data/bank/task_{task} --num-envs <N> --seed 999")
    if meta["task"] != task:
        raise SystemExit(f"bank is for task {meta['task']!r}, asked for {task!r}")
    cfgs = {c["clip"]: c for c in meta["configs"]}
    d = np.load(pred_npz, allow_pickle=True)
    pred, eps = d["pred"], d["episode_id"]

    def bank_px(clip):
        """Query pixels come from THE BANK, not from grid_pixels(). An
        object-aware bank places its queries on that scene's object, and the
        object-aware checkpoint expects exactly those; recomputing a uniform grid
        would index the segmentation at pixels the model was never asked about --
        silently, since every shape still matches."""
        z = np.load(f"{bank_dir}/{clip}/samples/00000.npz")
        return (z["keypoints"].astype(np.float64) if "keypoints" in z
                else grid_pixels())

    n_env = meta["num_envs"]
    T = pred.shape[-2]
    # The Kabsch convention here is p1 = R p0 + t, so `t` absorbs the rotation ABOUT THE WORLD
    # ORIGIN. Any consumer that wants to alter R while preserving the object's own motion needs
    # the source centroid to re-derive t; without it, swapping R in silently displaces the goal
    # by (R - I) c_src -- tens of centimetres at a typical table distance. Stored so that
    # rotation ablations are expressible at all. Purely additive: existing readers name their
    # keys and are unaffected.
    c_src = np.full((n_env, 3), np.nan)
    dR = np.full((n_env, T, 3, 3), np.nan)
    dt = np.full((n_env, T, 3), np.nan)
    goal_p = np.full((n_env, T, 3), np.nan)
    goal_q = np.full((n_env, T, 4), np.nan)
    n_pts = np.zeros(n_env, dtype=np.int32)
    n_inl = np.zeros(n_env, dtype=np.int32)
    _z = np.load(pred_npz, allow_pickle=True)
    _pred_ckpt = str(_z["prov_ckpt"]) if "prov_ckpt" in _z.files else ""
    del _z
    solved = np.zeros(n_env, dtype=bool)

    from msgen.labels import quat_to_R
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
        on = (segmap[iy, ix] == c["obj_seg_id"]) & np.isfinite(p).all(axis=(1, 2))
        n_pts[j] = int(on.sum())
        if on.sum() < min_pts:
            continue

        src = unproject(p[on, 0, :2], p[on, 0, 2].astype(np.float64), K, E)
        inl = None
        if maxrel > 0 or ransac:
            end = unproject(p[on, -1, :2], p[on, -1, 2].astype(np.float64), K, E)
            if ransac:
                _, inl = ransac_kabsch(src, end, tol_m=ransac_tol)
            else:
                # The per-point displacement distribution is BIMODAL: 81.5% of
                # truly-moving points land at 0.75-1.25x correct and 15.1% are
                # predicted near-static. Taking the threshold relative to the
                # MAXIMUM survives the non-movers being the majority; relative to
                # the median it collapses exactly when it is needed.
                disp = np.linalg.norm(end - src, axis=-1)
                keep = disp >= maxrel * disp.max()
                inl = keep if keep.sum() >= min_pts else None
            if inl is not None:
                n_inl[j] = int(inl.sum())   # inlier count; n_pts keeps the meaning 'input points'

        c_src[j] = (src[inl] if inl is not None else src).mean(axis=0)
        op = np.array(c["obj"][:3])
        oR = quat_to_R(np.array(c["obj"][3:7]))
        for s in range(T):
            dst = unproject(p[on, s, :2], p[on, s, 2].astype(np.float64), K, E)
            # consensus decided ONCE at the endpoint, so the point set stays
            # fixed along the trace rather than changing membership per step
            R, t = (fit(src[inl], dst[inl]) if inl is not None
                    else fit(src, dst))
            dR[j, s], dt[j, s] = R, t
            # diagnostic only: this composes with the SIMULATOR's object pose and
            # is therefore an upper bound, never what the policy consumes
            goal_p[j, s] = R @ op + t
            goal_q[j, s] = _R_to_quat(R @ oR)
        solved[j] = True

    true_goal = np.array([c["goal"] for c in meta["configs"]])
    # PROVENANCE. Record which bank and which prediction file the goals came
    # from. Additive only.
    import os as _os
    np.savez(out_npz, dR=dR, dt=dt, c_src=c_src, goal_p=goal_p, goal_q=goal_q,
             n_pts=n_pts, n_inl=n_inl, solved=solved, true_goal=true_goal,
             num_envs=n_env, seed=meta["seed"], task=meta["task"],
             weight=str(weight or "none"),
             prov_bank_dir=_os.path.abspath(bank_dir),
             prov_pred_npz=_os.path.abspath(pred_npz),
             prov_ransac=str(bool(ransac)), prov_maxrel=str(maxrel),
             prov_pred_ckpt=str(_pred_ckpt))
    print(f"solved {solved.sum()}/{n_env} scenes ({(~solved).sum()} under "
          f"{min_pts} traced object points) -> {out_npz}")
    if solved.sum():
        print(f"  grid points on the object: mean {n_pts[solved].mean():.1f} "
              f"min {n_pts[solved].min()} max {n_pts[solved].max()}")
        print("\n  predicted pose at step s vs the TRUE goal pose:")
        print(f"  {'step':>5} {'pos mm':>9} {'rot deg':>9}  {'<20mm&15deg':>12}")
        for s in [4, 8, 12, 16, 24, T - 1]:
            if s >= T:
                continue
            e = np.linalg.norm(goal_p[solved, s] - true_goal[solved, :3],
                               axis=-1) * 1000
            dq = np.abs(np.einsum("ij,ij->i", goal_q[solved, s],
                                  true_goal[solved, 3:]))
            dr = np.degrees(2 * np.arccos(np.clip(dq, -1, 1)))
            print(f"  {s:>5} {e.mean():>9.1f} {dr.mean():>9.1f}  "
                  f"{((e < 20) & (dr < 15)).mean():>11.1%}")
        print("\n  NOTE: open-loop diagnostic only; do not infer closed-loop "
              "success.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--task", required=True, choices=sorted(TASK_MAP))
    r.add_argument("--out", required=True)
    r.add_argument("--num-envs", type=int, default=256)
    r.add_argument("--seed", type=int, default=999)
    r.add_argument("--num-kp", type=int, default=64)
    r.add_argument("--robot", default=None,
                   help="override the registry robot for this bank. For testing "
                        "a domain shift, not normal use: the peg PLANNER was "
                        "trained on frames of a panda_wristcam while the shared "
                        "executor line runs a plain panda, and the arm fills much "
                        "of the frame. Object and goal placement do not depend on "
                        "the robot -- verified on StackCube, where both robots "
                        "gave identical cube poses at one seed and differed only "
                        "in qpos[6] -- so the other robot still describes the "
                        "SAME scenes.")
    r.add_argument("--show-goal", action="store_true",
                   help="render the goal marker into the sensor frames "
                        "(msgen.replay.unhide_goal); configs.json unchanged")
    s = sub.add_parser("solve")
    s.add_argument("--task", required=True, choices=sorted(TASK_MAP))
    s.add_argument("--bank", required=True)
    s.add_argument("--pred", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--min-pts", type=int, default=3)
    s.add_argument("--maxrel", type=float, default=0.0)
    s.add_argument("--ransac", action="store_true")
    s.add_argument("--ransac-tol", type=float, default=0.03)
    s.add_argument("--weight", default=None, choices=["none", "disp"],
                   help="disp = displacement-weighted Kabsch (optional; the released "
                        "pipeline uses --ransac without it)")
    a = ap.parse_args()
    if a.cmd == "render":
        render(a.task, a.out, a.num_envs, a.seed, a.num_kp,
               show_goal=a.show_goal, robot=a.robot)
    else:
        solve(a.task, a.bank, a.pred, a.out, a.min_pts, a.maxrel,
              a.ransac, a.ransac_tol, weight=a.weight)


if __name__ == "__main__":
    main()
