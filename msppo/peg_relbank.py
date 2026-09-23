"""Scene-RELATIVE goal error, so the planner's error can be injected while training.

Row 4 has always injected a WORLD-frame offset drawn from another scene
(`peg_traceerr.py` writes dp, dq; `peg_student_env._sample_goal_error` replays it).
Two things are wrong with that as a model of the planner:

  * it is decorrelated from the scene. The measured end-to-end error is not an
    offset at all -- the prediction covers 0.840 +/- 0.191 of the required travel,
    with 36.4 mm of lateral scatter (m2_shortfall.py). A scene needing 250 mm and
    one needing 470 mm get the same 65 mm offset, which describes neither.
  * a fixed offset cannot be re-expressed in a new scene's geometry, so it can
    only be sampled, never transported. Training needs transport: every DAgger
    reset draws a new peg and hole layout.

So the error is stored in the frame the task defines -- along the peg->goal axis,
plus two lateral components, plus the relative rotation -- and expanded against
whatever geometry the current episode has. `export` and `expand` live in the same
module on purpose: the frame convention has to match exactly on both sides, and
`verify` round-trips the export to prove it does.

    python -m msppo.peg_relbank export --goals results/peg_goals_e2e_s999.npz \
        --bank data/bank/peg_e999 --out results/peg_relbank_e2e.npz
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch


def axis_frame(u):
    """[...,3] unit axis -> (v, w) completing a right-handed orthonormal frame.

    World +z is the reference, falling back to +x where the axis is nearly
    vertical. Both branches are evaluated and selected with a where, so the same
    code runs batched on GPU during training and on numpy at export time.
    """
    ez = torch.zeros_like(u)
    ez[..., 2] = 1.0
    ex = torch.zeros_like(u)
    ex[..., 0] = 1.0
    ref = torch.where((u[..., 2:3].abs() > 0.9), ex, ez)
    v = torch.cross(u, ref, dim=-1)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    w = torch.cross(u, v, dim=-1)
    return v, w


def _axis(peg_p, goal_p):
    d = goal_p - peg_p
    n = d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return d / n, n.squeeze(-1)


def to_relative(peg_p, true_goal_p, pred_goal_p):
    """(frac, lat_v, lat_w): the prediction decomposed in the task's own frame."""
    u, n = _axis(peg_p, true_goal_p)
    v, w = axis_frame(u)
    dp = pred_goal_p - peg_p
    frac = (dp * u).sum(-1) / n
    return frac, (dp * v).sum(-1), (dp * w).sum(-1)


def expand(peg_p, true_goal_p, frac, lat_v, lat_w):
    """Inverse of `to_relative`, against a DIFFERENT scene's geometry."""
    u, n = _axis(peg_p, true_goal_p)
    v, w = axis_frame(u)
    return (peg_p + (frac * n)[:, None] * u
            + lat_v[:, None] * v + lat_w[:, None] * w)


def export(goals_npz, bank_dir, out_npz):
    z = np.load(goals_npz)
    solved = z["solved"].astype(bool)
    goal_p, goal_q, true_goal = z["goal_p"], z["goal_q"], z["true_goal"]
    T = goal_p.shape[1]
    cfg = {c["env_idx"]: c for c in json.load(open(f"{bank_dir}/configs.json"))["configs"]}
    # `peg` is what the peg bank calls the moving body; the shared banks written
    # by task_tgbank call it `obj`. Try both so this file works for any task
    # without changing a single number it has already produced for peg.
    okey = "peg" if "peg" in cfg[0] else "obj"
    peg0 = np.array([cfg[i][okey][:3] for i in range(len(solved))])

    t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float64))  # noqa: E731
    pp, tg = t(peg0[solved]), t(true_goal[solved, :3])
    pg = t(goal_p[solved, T - 1])
    frac, lv, lw = to_relative(pp, tg, pg)

    # rotation stays a relative quaternion, as the world-frame bank already did:
    # dq carries the TRUE goal orientation onto the predicted one, which is
    # frame-independent and so needs no re-expression.
    from msppo.peg_student_env import _R_to_quat
    qt, qp = t(true_goal[solved, 3:]), t(goal_q[solved, T - 1])
    Rt = _quat_to_R(qt)
    Rp = _quat_to_R(qp)
    dR = Rp @ Rt.transpose(-1, -2)
    dq = _R_to_quat(dR)

    # round-trip: expanding against the SAME geometry must return the prediction
    back = expand(pp, tg, frac, lv, lw)
    err = (back - pg).norm(dim=-1).max().item() * 1000
    if err > 1e-3:
        raise SystemExit(f"round-trip failed by {err:.4f} mm -- the frame "
                         f"conventions in to_relative/expand disagree")

    np.savez(out_npz, frac=frac.numpy().astype(np.float32),
             lat_v=lv.numpy().astype(np.float32),
             lat_w=lw.numpy().astype(np.float32),
             dq=dq.numpy().astype(np.float32))
    f = frac.numpy()
    print(f"exported {len(f)} scene-relative error tuples -> {out_npz}")
    print(f"  round-trip max error {err:.2e} mm")
    print(f"  frac  mean {f.mean():.3f} sd {f.std():.3f} "
          f"p10 {np.percentile(f,10):+.3f} p90 {np.percentile(f,90):.3f}")
    print(f"  lat_v sd {lv.numpy().std()*1000:.1f} mm   "
          f"lat_w sd {lw.numpy().std()*1000:.1f} mm")


def _rel_one(goals_npz, bank_dir):
    """(solved[N] bool, frac, lat_v, lat_w, dq) over ALL N scenes of one goals
    file; unsolved rows are NaN. Same frame convention as `export`, round-trip
    checked the same way, but the scene index is preserved."""
    z = np.load(goals_npz)
    solved = z["solved"].astype(bool)
    goal_p, goal_q, true_goal = z["goal_p"], z["goal_q"], z["true_goal"]
    T = goal_p.shape[1]
    cfg = {c["env_idx"]: c for c in json.load(open(f"{bank_dir}/configs.json"))["configs"]}
    okey = "peg" if "peg" in cfg[0] else "obj"
    obj0 = np.array([cfg[i][okey][:3] for i in range(len(solved))])
    t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float64))  # noqa: E731
    N = len(solved)
    frac = torch.full((N,), float("nan"), dtype=torch.float64)
    lv, lw = frac.clone(), frac.clone()
    dq = torch.full((N, 4), float("nan"), dtype=torch.float64)
    if solved.any():
        pp, tg, pg = t(obj0[solved]), t(true_goal[solved, :3]), t(goal_p[solved, T - 1])
        f, v, w = to_relative(pp, tg, pg)
        from msppo.peg_student_env import _R_to_quat
        Rt, Rp = _quat_to_R(t(true_goal[solved, 3:])), _quat_to_R(t(goal_q[solved, T - 1]))
        q = _R_to_quat(Rp @ Rt.transpose(-1, -2))
        err = (expand(pp, tg, f, v, w) - pg).norm(dim=-1).max().item() * 1000
        if err > 1e-3:
            raise SystemExit(f"{goals_npz}: round-trip failed by {err:.4f} mm")
        m = torch.from_numpy(solved)
        frac[m], lv[m], lw[m], dq[m] = f, v, w, q
    return solved, frac, lv, lw, dq


def export_k(goals_list, bank_dir, out_npz):
    """K goal files of the SAME bank -> one relbank with K columns per scene.

    Rows are the INTERSECTION of the K `solved` masks, and `scene_idx` records
    which scenes survived: the single-sample `export` drops unsolved scenes and
    forgets the index, which is fine for a bag of independent draws but not
    for K draws that must stay grouped by scene (spec 2026-08-25 §4)."""
    parts = [_rel_one(g, bank_dir) for g in goals_list]
    keep = np.logical_and.reduce([p[0] for p in parts])
    idx = np.nonzero(keep)[0]
    km = torch.from_numpy(keep)
    stack = lambda j: torch.stack([p[j][km] for p in parts], dim=1)  # [N,K,...]  # noqa: E731
    np.savez(out_npz,
             frac=stack(1).numpy().astype(np.float32),
             lat_v=stack(2).numpy().astype(np.float32),
             lat_w=stack(3).numpy().astype(np.float32),
             dq=stack(4).numpy().astype(np.float32),
             scene_idx=idx.astype(np.int32), K=len(goals_list))
    f = stack(1).numpy()
    print(f"exported {len(idx)} scenes x K={len(goals_list)} -> {out_npz}")
    print(f"  frac mean {f.mean():.3f}  within-scene sd {f.std(1).mean():.3f}  across-scene sd {f.mean(1).std():.3f}")
    return int(len(idx))


def _quat_to_R(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], dim=-2)


def slerp_from_identity(dq, a):
    """[N,4] (w,x,y,z) rotated a fraction `a` of the way from IDENTITY toward dq.

    `a` is [N] so each episode can draw its own scale. Sign-aligned first: SAPIEN
    quaternions are not canonicalised, and slerping from identity to -q travels the
    long way round, which would make a "smaller" scale a LARGER rotation.
    """
    q = torch.where(dq[:, :1] < 0, -dq, dq)
    w = q[:, 0].clamp(-1.0, 1.0)
    th = torch.arccos(w)                       # half-angle
    s = torch.sin(th)
    small = s < 1e-8
    sd = s.clamp_min(1e-8)
    c0 = (torch.sin((1 - a) * th) + torch.sin(a * th) * q[:, 0]) / sd
    out = torch.stack([torch.where(small, torch.ones_like(w), c0)]
                      + [torch.where(small, torch.zeros_like(w),
                                     torch.sin(a * th) * q[:, j] / sd)
                         for j in (1, 2, 3)], dim=-1)
    return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def scale_error(bank, idx, a):
    """Interpolate the sampled error toward NO error. a=0 -> exact true goal,
    a=1 -> the measured error unchanged.

    Same three-part interpolation as the requirement curve
    (`<private-repo>/experiments/20260812_e2e_replan/m12_reqcurve.py`), so a training scale and a
    curve point with the same `a` mean the same thing.
    """
    frac, lv, lw, dq = bank["frac"][idx], bank["lat_v"][idx], bank["lat_w"][idx], bank["dq"][idx]
    if bank.get("K", 1) > 1:                      # [B,K] / [B,K,4]; a is [B], broadcast to K
        aK = a[:, None].expand_as(frac)
        B, K = frac.shape
        q = slerp_from_identity(dq.reshape(B * K, 4), aK.reshape(B * K)).reshape(B, K, 4)
        return 1.0 + aK * (frac - 1.0), aK * lv, aK * lw, q
    return 1.0 + a * (frac - 1.0), a * lv, a * lw, slerp_from_identity(dq, a)


def load(path, device="cuda"):
    """The four error components, plus the two RELIABILITY signals when the bank
    carries them (`*_relbank_visobj_sig.npz`, written by
    <private-repo>/experiments/20260823_relsig/b1_addsig.py).

    They live here rather than being computed at deployment because injection
    replaces the goal AFTER the solve: a synthetic error has no rmse of its own,
    so a signal computed on the spot would describe the clean solve and be
    uncorrelated with the error actually injected. Storing them means replaying
    an error replays ITS reliability, and the pair stays intact in training.

    Measured 2026-08-23 (<private-repo>/experiments/20260823_relsig): rmse ranks scenes by
    POSITION error on PegInsert -- worst vs best quartile 1.89-2.71x across five
    planners, and 2.32-2.81x INSIDE a fixed n_pts stratum, so it is not a proxy
    for point count. s21 = sigma2/sigma1 of the solve's cross-covariance ranks
    ROTATION error on PickCube (3.0-3.7x) and StackCube (2.3-4.2x). n_pts ranks
    nothing and is constant in the object-aware banks, so it is not carried."""
    z = np.load(path)
    d = {k: torch.from_numpy(z[k]).to(device).float()
         for k in ("frac", "lat_v", "lat_w", "dq")}
    # K-column bank (`export_k`): frac/lat_* are [N,K], dq is [N,K,4]. A legacy
    # single-sample bank has no `K` key and reads as K=1 with unchanged shapes.
    d["K"] = int(z["K"]) if "K" in z.files else 1
    d["sig"] = (torch.from_numpy(np.stack([z["rmse"], z["s21"]], -1)).to(device).float()
                if ("rmse" in z.files and "s21" in z.files) else None)
    return d


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--goals", required=True)
    e.add_argument("--bank", required=True)
    e.add_argument("--out", required=True)
    ek = sub.add_parser("export-k", help="K goal files of ONE bank -> K columns per scene")
    ek.add_argument("--goals", required=True, help="comma-separated K goal npz")
    ek.add_argument("--bank", required=True)
    ek.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.goals, a.bank, a.out)
    else:
        export_k(a.goals.split(","), a.bank, a.out)


if __name__ == "__main__":
    main()
