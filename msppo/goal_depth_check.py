"""Depth cross-check (DCC) -- a generic pipeline stage that validates/refines a predicted terminal goal against the
t=0 depth image.

Generalises the StackCube contact snap (<private-repo>/experiments/20260831_t2k/r2_contact_snap.py): every hardcoded number is now
derived from the scene and from the planner's own T2K outputs, and the stage decides PER SCENE whether it applies --
otherwise it abstains and returns the goal unchanged.  Design rule: this stage may only ever use what deployment has
(depth image, intrinsics/extrinsics, the planner's own outputs); no simulator state.

GATE (all three from the planner's own outputs, so the planner decides where the check is meaningful):
  1. contact_mode == grasp (T2K)  (a placed object; push/none goals have no support structure to localise)
  2. the trajectory ENDS BELOW A RECENT PEAK: max(z of the last 8 steps) - z(final) > DROP_MIN -- a place
     descends from carry height; measured separation (t2kpos banks, 999): stack 0.95 vs pickcube 0.03 /
     peg 0.02 / pushcube 0.11.  (The terminal twist axis is too diluted: transport+descend fuse into one
     line segment, net direction mostly lateral -- axis_z median -0.12 on stack.)
  3. support height = goal_z - obj_half is > MIN_H above the table (else the "support" is the table itself
     and a plateau search would snap to arbitrary table points; obj_half := object's resting z at t=0)

CHECK (when gated in): a plateau of depth points at the support height (band +-BAND_HW) within WIN metres (xy) of the
predicted goal, excluding the object's own start (they are at the same height for same-sized objects).  >= MIN_PTS
points -> snap goal xy to the plateau centroid (goal z kept).  Otherwise abstain with a reason -- a gated-in scene
with NO support in the depth image is also a flag that the goal itself is suspect (usable to re-rank K samples).
"""
from __future__ import annotations
import numpy as np

DROP_MIN, MIN_H, BAND_HW, WIN, MIN_PTS, EXCL_R = 0.02, 0.015, 0.012, 0.06, 20, 0.035
MODES = ("none", "grasp", "push")


def unproject_full(depth, K, ext):
    H, W = depth.shape
    vv, uu = np.mgrid[0:H, 0:W]
    z = depth.ravel().astype(np.float64)
    x = (uu.ravel() - K[0, 2]) * z / K[0, 0]
    y = (vv.ravel() - K[1, 2]) * z / K[1, 1]
    cam = np.stack([x, y, z], -1)
    R, t = ext[:3, :3], ext[:3, 3]
    return (cam - t) @ R          # world = R^T (cam - t)


def gate(goal_p, obj_p, traj_z, contact_mode_logits):
    """traj_z: the planner's per-step object z (last steps suffice)."""
    mode = MODES[int(np.argmax(contact_mode_logits))]
    if mode != "grasp":
        return False, f"abstain: contact_mode={mode}"
    tz = np.asarray(traj_z, np.float64)
    drop = float(tz[-8:].max() - tz[-1])
    if drop < DROP_MIN:
        return False, f"abstain: no terminal descent (drop={drop*1000:.0f}mm)"
    obj_half = float(obj_p[2])                     # resting object: centre height == half height
    support_h = float(goal_p[2]) - obj_half
    if support_h < MIN_H:
        return False, f"abstain: support at table level (h={support_h*1000:.0f}mm)"
    return True, f"support_h={support_h*1000:.0f}mm"


def cross_check(goal_p, obj_p, traj_z, contact_mode_logits, depth, K, ext):
    """-> (goal_refined [3], info dict). Abstains (returns goal unchanged) unless gated in AND a plateau is found."""
    ok, why = gate(goal_p, obj_p, traj_z, contact_mode_logits)
    if not ok:
        return np.asarray(goal_p, np.float64), dict(applied=False, reason=why)
    W3 = unproject_full(np.asarray(depth, np.float64), np.asarray(K, np.float64), np.asarray(ext, np.float64))
    support_h = float(goal_p[2]) - float(obj_p[2])
    near = np.linalg.norm(W3[:, :2] - np.asarray(goal_p[:2]), axis=-1) < WIN
    band = near & (np.abs(W3[:, 2] - support_h) < BAND_HW)
    band &= np.linalg.norm(W3[:, :2] - np.asarray(obj_p[:2]), axis=-1) > EXCL_R
    n = int(band.sum())
    if n < MIN_PTS:
        return np.asarray(goal_p, np.float64), dict(applied=False, reason=f"gated in, no support plateau ({n} pts) -- goal suspect")
    cxy = W3[band, :2].mean(0)
    g = np.array([cxy[0], cxy[1], goal_p[2]], np.float64)
    return g, dict(applied=True, reason=f"snapped {np.linalg.norm(g[:2]-goal_p[:2])*1000:.1f}mm ({n} pts)", n_pts=n)


def aerial_island_check(goal_p, obj_p, traj_z, contact_mode_logits, depth, K, ext,
                        win=0.08, z_hw=0.03, min_pts=15, max_pts=500, min_h=0.04):
    """PickCube-family variant of the depth cross-check: an AERIAL goal (rendered marker) leaves a
    small isolated depth island in mid-air.  Gate: grasp mode + the trajectory ends ASCENDING (rise > 20 mm over
    the last 8 steps -- the mirror of the place-family descent gate) + goal well above the table.  Check: depth
    points within `win` (xy) and +-z_hw of the PREDICTED goal; a small cluster (min_pts..max_pts) is the marker
    (thousands = the robot arm intruding -> abstain); its centroid replaces the goal.  Measured basis: the head's
    error is 84% along the camera ray (monocular depth ambiguity) while the island centroid sits at 18.4 mm med
    vs the head's 34.2 -- the island IS the missing depth fix.

    VERDICT (three iterations, kept as a recorded NEGATIVE): geometry-only marker extraction loses to
    the head.  (1) window point-count cap -> robot kills 190/256 windows; (2) image-connectivity clusters -> the
    marker merges with the robot whenever they overlap in the image (most scenes) and fingertips fake the size
    test (snap errors 74-81 mm); (3) 3D-voxel components + isolation -> the marker's visible depth fragment is
    3D-adjacent to robot surfaces in ~80% of scenes (10/256 clean snaps).  The marker is an APPEARANCE-defined
    object sitting visually against the robot; the reliable cue is its colour, which would be a task-specific
    appearance key, not geometry.  The structural answers remain: a physical-goal task (PlaceSphere) or a second
    view (the head's error is 84% along the camera ray -- parallax removes exactly that axis)."""
    import numpy as np
    mode = MODES[int(np.argmax(contact_mode_logits))]
    if mode != "grasp":
        return np.asarray(goal_p, np.float64), dict(applied=False, reason=f"abstain: contact_mode={mode}")
    tz = np.asarray(traj_z, np.float64)
    rise = float(tz[-1] - tz[-8:].min())
    if rise < 0.02:
        return np.asarray(goal_p, np.float64), dict(applied=False, reason=f"abstain: no terminal ascent (rise={rise*1000:.0f}mm)")
    if float(goal_p[2]) - float(obj_p[2]) < min_h:
        return np.asarray(goal_p, np.float64), dict(applied=False, reason="abstain: goal not aerial")
    W3 = unproject_full(np.asarray(depth, np.float64), np.asarray(K, np.float64), np.asarray(ext, np.float64))
    # search the whole MID-AIR COLUMN around the predicted xy: the head's error is 84% along the camera ray, so
    # the predicted z-band would be mis-centred by construction, while the perpendicular (xy) error is only
    # ~17 mm med -- the 8 cm window reliably contains the true marker.
    m = (W3[:, 2] > float(obj_p[2]) + 0.05) & (W3[:, 2] < 0.40) &         (np.linalg.norm(W3[:, :2] - np.asarray(goal_p[:2]), axis=-1) < win)
    if int(m.sum()) < min_pts:
        return np.asarray(goal_p, np.float64), dict(applied=False, reason=f"gated in, no island ({int(m.sum())} px) -- occluded/suspect")
    # image-connectivity merges the marker with the robot whenever they overlap in the image, and compact robot
    # parts (fingertips) fake the size test.  The marker's UNIQUE property is 3D isolation: it floats with empty
    # space all around, while a fingertip connects upward into the arm.  So: 3D voxel connected components over
    # ALL scene points (not just the window), then accept a component only if it is marker-sized, compact, AND
    # isolated (its 4 cm expanded shell contains almost nothing else).
    from scipy import ndimage
    vox = 0.02
    allpts = W3[np.isfinite(W3).all(-1) & (W3[:, 2] > float(obj_p[2]) + 0.05) & (W3[:, 2] < 0.40)]
    if len(allpts) < min_pts:
        return np.asarray(goal_p, np.float64), dict(applied=False, reason="gated in, no mid-air points -- occluded/suspect")
    lo = allpts.min(0) - vox
    ijk = np.floor((allpts - lo) / vox).astype(int)
    grid = np.zeros(ijk.max(0) + 1, dtype=bool)
    grid[tuple(ijk.T)] = True
    lab, nlab = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    comp = lab[tuple(ijk.T)]
    best, best_d, best_n = None, 1e9, 0
    for li in range(1, nlab + 1):
        sel = comp == li
        npix = int(sel.sum())
        if not (min_pts <= npix <= max_pts):
            continue
        pts = allpts[sel]
        if np.linalg.norm(pts.max(0) - pts.min(0)) > 0.07:
            continue
        cen = pts.mean(0)
        dd = float(np.linalg.norm(cen[:2] - np.asarray(goal_p[:2], np.float64)))
        if dd > win:
            continue
        if dd < best_d:
            best, best_d, best_n = cen, dd, npix
    if best is None:
        return np.asarray(goal_p, np.float64), dict(applied=False, reason="abstain: no isolated marker-sized component")
    return best, dict(applied=True, reason=f"snapped {best_d*1000:.1f}mm ({best_n} px)", n_pts=best_n)


# ---------------------------------------------------------------------------
# DCC v2: unified measurement layer.
#
# The support-plateau snap above and the pickcube marker localizer are two
# instantiations of one principle -- cross-check the network's goal ESTIMATE
# against a direct geometric MEASUREMENT from the same RGB-D frame, and take
# the measurement whenever its own validity gates pass. This registry gives
# them one interface without touching the locked stack path (`cross_check` is
# wrapped as-is; its numbers are frozen in the main table).
#
#   measure_goal("stack",    pred, depth=..., K=..., ext=..., obj_p=..., traj_z=..., contact_mode_logits=...)
#   measure_goal("pickcube", pred, depth=..., K=..., ext=..., rgb=..., gmap_logits=...)   # no gmap_logits -> abstains
#
# Returns (goal_p, info); info["applied"] says whether the measurement won.

MARKER_RADIUS_M = 0.0185     # goal-marker surface->centre offset along the camera ray,
                             # fitted on isolation seeds 990-996 (n=411, median).
MARKER_MIN_PX = 5


def op_marker(pred_goal, rgb, depth, K, ext):
    """Retired colour rule; kept for reference only, no longer dispatched.
    pickcube: localize the rendered goal marker (pure green, RGB~(5,168,5)) from
    colour + depth. Deployment-legitimate: consumes only the planner's own RGB-D."""
    m = (rgb[:, :, 1] > 100) & (rgb[:, :, 0] < 60) & (rgb[:, :, 2] < 60)
    if int(m.sum()) < MARKER_MIN_PX:
        return pred_goal, {"applied": False, "reason": "no_marker_pixels"}
    vs, us = np.nonzero(m)
    dep = depth[vs, us]
    ok = dep > 0.05
    if int(ok.sum()) < MARKER_MIN_PX:
        return pred_goal, {"applied": False, "reason": "marker_depth_holes"}
    u, v = us[ok].mean(), vs[ok].mean()
    z = float(np.median(dep[ok])) + MARKER_RADIUS_M
    pc = np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
    E3 = ext[:3] if ext.shape[0] == 4 else ext
    pw = E3[:3, :3].T @ (pc - E3[:3, 3])
    return pw, {"applied": True, "n_px": int(ok.sum())}


# ---- SAM2 marker locator: the colour rule above is retired (project rule:
# no colour thresholds anywhere). The prompt is the gmap head's peak patch -- a learned, appearance-based
# localiser -- and SAM2 turns it into a mask; the geometric gates below decide whether to trust it.
SAM2_MASK_AREA = (20, 4000)      # px; measured offline: close-range markers reach ~2100 px, so 2000 rejected 3/256
SAM2_DEPTH_AGREE_M = 0.04        # mask median depth must agree with the prompt pixel's depth
SAM2_BEHIND_MAX_M = 0.03         # v4: markers just behind a cube edge sit up to ~25 mm behind their ring (v3 used 0 and lost 3-7
                                 # good masks per seed). Robot/cube surface masks sit at ~0 too, so this gate is weak against them;
                                 # SAM2_MIN_SCORE is the effective false-accept gate. (marker floats or rests on a surface; robot/cube
                                 # surface masks are continuous with their ring). Seed-999 sweep: front>=0 & score>=0.8
                                 # keeps 202/209 good masks, rejects 16/18 false accepts; the earlier >=30 mm rule lost 15 good ones.
SAM2_MIN_SCORE = 0.8             # SAM2 mask score; false accepts (robot/cube masks) median 0.21, true markers median 0.98 (AUC 0.98)
SAM2_RANK_FALLBACK = False        # try lower-ranked candidates when the best fails gate_mask. Off by default: row4
                                 # .832/.844 (off) vs .828/.824 (on) -- the recall gain was paid back by false accepts on
                                 # scenes where the marker is invisible (confident wrong goals are worse than abstention).
SAM2_PROMPT_JITTERS_PX = (6, 12)  # v4: extra single-point prompts at centre +-jitter (u,v): offline misses had the peak 5-18 px off the marker
SAM2_ASPECT_MAX = 2.0            # bbox aspect; the arm/finger masks are elongated
SAM2_RING_PX = 6
GMAP_GRID, GMAP_PATCH = 24, 16
_SAM2 = None


def _sam2_predictor():
    """Module-level singleton; imports SAM2 lazily so this module stays importable without it."""
    global _SAM2
    if _SAM2 is None:
        import os, sys, torch
        sam2_dir = os.environ.get("SAM2_DIR", "third_party/sam2")
        sys.path.insert(0, sam2_dir)
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        m = build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml", os.environ.get("SAM2_CKPT", f"{sam2_dir}/checkpoints/sam2.1_hiera_small.pt"),
                       device="cuda" if torch.cuda.is_available() else "cpu")
        _SAM2 = SAM2ImagePredictor(m)
    return _SAM2


def gmap_peak_prompts(gmap_logits, depth):
    """argmax patch of the 24x24 gmap -> two positive point prompts: patch centre, and the patch's
    nearest valid-depth pixel (the marker is in front of whatever is behind it). Returns (pts[2,2] (u,v), labels[2], (r,c))."""
    p = int(np.argmax(np.asarray(gmap_logits).ravel())); r, c = p // GMAP_GRID, p % GMAP_GRID
    r0, c0 = r * GMAP_PATCH, c * GMAP_PATCH
    # nearest valid depth searched over the 3x3 patch neighbourhood (v4): the marker is often a few px outside the peak patch
    R0, C0 = max(0, r0 - GMAP_PATCH), max(0, c0 - GMAP_PATCH)
    blk = depth[R0:r0 + 2 * GMAP_PATCH, C0:c0 + 2 * GMAP_PATCH]
    val = blk > 0.05
    if val.any():
        k = int(np.argmin(np.where(val, blk, np.inf))); dv, du = divmod(k, blk.shape[1]); dv += R0 - r0; du += C0 - c0
    else:
        dv = du = GMAP_PATCH // 2
    pts = np.array([[c0 + GMAP_PATCH / 2, r0 + GMAP_PATCH / 2], [c0 + du, r0 + dv]], float)
    return pts, np.array([1, 1], int), (r, c)


def select_mask(masks, scores, pts, depth, cand_pts=None, area=None):
    """Rank SAM2 candidates: keep those with SAM2 score >= SAM2_MIN_SCORE, area in SAM2_MASK_AREA, containing >=1 of their
    own prompts (cand_pts[i], default pts), and whose median depth agrees (SAM2_DEPTH_AGREE_M) with the depth at a contained
    prompt (or, if the contained prompts are depth holes, with the depth at any valid prompt of that candidate). Returns
    (mask, score, reason, ranked) with the best mask first and `ranked` = list of (mask, score, idx) sorted by score, so the
    caller can fall back down the ranking when the best fails gate_mask. Abstains 'prompt_depth_invalid' when no pair prompt
    has valid depth."""
    if not any(depth[int(v), int(u)] > 0.05 for u, v in pts): return None, -1.0, "prompt_depth_invalid", []
    ranked = []
    for i, (m, s) in enumerate(zip(masks, scores)):
        if float(s) < SAM2_MIN_SCORE: continue
        m = m.astype(bool); a = int(m.sum()); cp = pts if cand_pts is None else cand_pts[i]
        lo, hi = area if area is not None else SAM2_MASK_AREA
        if not (lo <= a <= hi): continue
        contained = [(u, v) for (u, v) in cp if m[int(v), int(u)]]
        if not contained: continue
        ref = [depth[int(v), int(u)] for (u, v) in contained if depth[int(v), int(u)] > 0.05] or \
              [depth[int(v), int(u)] for (u, v) in cp if depth[int(v), int(u)] > 0.05]
        dm = depth[m]; dm = dm[dm > 0.05]
        if len(dm) == 0 or not ref or min(abs(float(np.median(dm)) - d) for d in ref) > SAM2_DEPTH_AGREE_M: continue
        ranked.append((m, float(s), i))
    ranked.sort(key=lambda t: -t[1])
    if not ranked: return None, -1.0, "no_mask", []
    return ranked[0][0], ranked[0][1], "ok", ranked


def gate_mask(mask, depth, aspect_max=None):
    """Geometric gates: enough valid depth; not elongated; not a hole/background (at most SAM2_BEHIND_MAX_M behind its ring)."""
    dm = depth[mask]; ok = dm > 0.05
    if int(ok.sum()) < MARKER_MIN_PX: return False, "marker_depth_holes"
    vs, us = np.nonzero(mask); h, w = vs.max() - vs.min() + 1, us.max() - us.min() + 1
    if max(h, w) / max(1, min(h, w)) >= (aspect_max if aspect_max is not None else SAM2_ASPECT_MAX): return False, "aspect"
    from scipy.ndimage import binary_dilation
    ring = binary_dilation(mask, iterations=SAM2_RING_PX) & ~mask
    dr = depth[ring]; dr = dr[dr > 0.05]
    if len(dr) == 0 or float(np.median(dm[ok])) - float(np.median(dr)) > SAM2_BEHIND_MAX_M: return False, "behind"
    return True, "ok"


def op_marker_sam2(pred_goal, rgb, depth, K, ext, gmap_logits, predictor=None, radius_m=None, area=None, aspect_max=None):
    """pickcube: SAM2 mask from the gmap-peak prompt ensemble -> centroid + median depth + MARKER_RADIUS_M along the ray.
    Zero colour rules. Abstains (returns pred_goal) when no candidate passes select_mask + gate_mask. Task parameters:
    radius_m (surface->centre offset along the ray; default MARKER_RADIUS_M for the pickcube sphere, 0 for a flat disc),
    area (px range), aspect_max -- pushcube's 10 cm disc is larger and foreshortened, so it passes (0.0, (200, 12000), 3.0). Input contract (asserted): depth [H,W] metres with H == W == GMAP_GRID*GMAP_PATCH,
    gmap_logits of length GMAP_GRID**2 laid out row-major on that image, rgb [H,W,3]."""
    depth = np.asarray(depth); gmap_logits = np.asarray(gmap_logits).ravel()
    assert depth.shape[:2] == (GMAP_GRID * GMAP_PATCH, GMAP_GRID * GMAP_PATCH), f"depth must be {GMAP_GRID * GMAP_PATCH}^2, got {depth.shape}"
    assert rgb.shape[:2] == depth.shape[:2], f"rgb {rgb.shape} vs depth {depth.shape}"
    assert gmap_logits.shape[0] == GMAP_GRID * GMAP_GRID, f"gmap_logits must have {GMAP_GRID * GMAP_GRID} entries, got {gmap_logits.shape}"
    _dv = depth[depth > 0.05]
    assert _dv.size == 0 or 0.1 <= float(np.median(_dv)) <= 5.0, "depth does not look like metres (median outside 0.1-5 m)"
    pts, labels, rc = gmap_peak_prompts(gmap_logits, depth)
    pr = predictor if predictor is not None else _sam2_predictor()
    pr.set_image(np.ascontiguousarray(rgb[:, :, :3]))
    # prompt ensemble: the (centre, nearest-depth) pair, then single points at centre +- jitter; the image is embedded once.
    H, W = depth.shape[:2]; c = pts[0]
    singles = [np.array([[min(max(c[0] + du, 0), W - 1), min(max(c[1] + dv, 0), H - 1)]])
               for jit in (0,) + tuple(SAM2_PROMPT_JITTERS_PX) for du, dv in ((0, 0), (jit, 0), (-jit, 0), (0, jit), (0, -jit)) if not (jit and du == 0 and dv == 0)]
    trials = [(pts, labels)] + [(sp, np.array([1], int)) for sp in singles]
    masks, scores, tpts = [], [], []
    for tp, tl in trials:
        mk, sc, _ = pr.predict(point_coords=tp, point_labels=tl, multimask_output=True)
        for m_, s_ in zip(np.asarray(mk), np.asarray(sc, float)):
            masks.append(m_); scores.append(s_); tpts.append(tp)
    masks = np.asarray(masks); scores = np.asarray(scores, float)
    info = {"applied": False, "prompt_uv": pts.tolist(), "patch_rc": rc, "n_trials": len(trials)}
    m, s, reason, ranked = select_mask(masks, scores, pts, depth, cand_pts=tpts, area=area)
    if m is None:
        info["reason"] = reason; return pred_goal, info
    chosen = None; last_reason = "no_mask"
    for m, s, i in (ranked if SAM2_RANK_FALLBACK else ranked[:1]):   # optionally fall back down the score ranking
        ok, last_reason = gate_mask(m, depth, aspect_max=aspect_max)
        if ok: chosen = (m, s, i); break
    if chosen is None:
        info["reason"] = last_reason; info["score"] = ranked[0][1]; return pred_goal, info
    m, s, i = chosen
    vs, us = np.nonzero(m); dep = depth[vs, us]; okd = dep > 0.05
    u, v = us[okd].mean(), vs[okd].mean(); z = float(np.median(dep[okd])) + (MARKER_RADIUS_M if radius_m is None else radius_m)
    pc = np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
    E3 = ext[:3] if ext.shape[0] == 4 else ext
    pw = E3[:3, :3].T @ (pc - E3[:3, 3])
    info.update(applied=True, n_px=int(okd.sum()), score=s, reason="ok", win_prompt_uv=np.asarray(tpts[i]).tolist(), rank=[r[2] for r in ranked].index(i))
    return pw, info


def measure_goal(task, pred_goal, **kw):
    """Dispatch to the task's measurement operator; unknown task -> estimate passes through."""
    if task == "pickcube":
        gl = kw.get("gmap_logits")
        if gl is None:
            return pred_goal, {"applied": False, "reason": "no_gmap_logits"}
        return op_marker_sam2(pred_goal, kw["rgb"], kw["depth"], kw["K"], kw["ext"], gl, kw.get("predictor"))
    if task == "stack":
        return cross_check(pred_goal, kw["obj_p"], kw["traj_z"], kw["contact_mode_logits"],
                           kw["depth"], kw["K"], kw["ext"])
    return pred_goal, {"applied": False, "reason": f"no_operator_for_{task}"}
