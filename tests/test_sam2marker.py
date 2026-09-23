import sys, numpy as np, pytest
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from msppo import goal_depth_check as G

K = np.array([[484.92407, 0, 192], [0, 484.92407, 192], [0, 0, 1]], float)
EXT = np.eye(4)[:3]                      # camera == world, so world point = camera point


def scene(disc=True, disc_z=0.55, table_z=0.70, arm=False):
    rgb = np.zeros((384, 384, 3), np.uint8); dep = np.full((384, 384), table_z, np.float32)
    yy, xx = np.mgrid[:384, :384]
    m_disc = ((xx - 200) ** 2 + (yy - 150) ** 2) <= 10 ** 2
    if disc: dep[m_disc] = disc_z
    m_arm = (xx >= 195) & (xx <= 205) & (yy >= 120) & (yy <= 260)   # 11x141 = 1551 px (within the area cap), aspect ~13
    if arm: dep[m_arm] = disc_z
    gm = np.full(576, -10.0); r, c = 150 // 16, 200 // 16; gm[r * 24 + c] = 10.0   # peak patch at the disc
    return rgb, dep, gm, m_disc, m_arm


class Stub:
    def __init__(self, masks, scores): self.masks, self.scores = masks, scores
    def set_image(self, rgb): pass
    def predict(self, point_coords, point_labels, multimask_output=True): return self.masks, self.scores, None


def test_prompts_at_peak_and_nearest_depth():
    rgb, dep, gm, m_disc, _ = scene()
    pts, labels, (r, c) = G.gmap_peak_prompts(gm, dep)
    assert (r, c) == (150 // 16, 200 // 16) and labels.tolist() == [1, 1]
    assert abs(pts[0, 0] - (c * 16 + 8)) < 1e-6 and abs(pts[0, 1] - (r * 16 + 8)) < 1e-6
    assert m_disc[int(pts[1, 1]), int(pts[1, 0])]            # nearest-depth pixel lies on the disc


def test_gate_rejects_behind_and_aspect():
    rgb, dep, gm, m_disc, m_arm = scene()
    flat = np.full_like(dep, 0.70); assert G.gate_mask(m_disc, flat) == (True, "ok")          # resting on the table is allowed
    hole = np.full_like(dep, 0.70); hole[m_disc] = 0.80; assert G.gate_mask(m_disc, hole) == (False, "behind")
    rgb2, dep2, _, _, m_arm2 = scene(arm=True); assert G.gate_mask(m_arm2, dep2) == (False, "aspect")
    assert G.gate_mask(m_disc, dep) == (True, "ok")
    tiny = np.zeros_like(m_disc); tiny[150, 200] = True; assert G.gate_mask(tiny, dep)[1] == "marker_depth_holes"


def test_select_mask_prefers_valid_candidate():
    rgb, dep, gm, m_disc, m_arm = scene(arm=True)
    huge = np.ones_like(m_disc)                                        # area > 2000 -> rejected
    masks = np.stack([huge, m_arm, m_disc]); scores = np.array([0.9, 0.85, 0.82])                # all above SAM2_MIN_SCORE
    pts, labels, _ = G.gmap_peak_prompts(gm, dep)
    m, s, reason, ranked = G.select_mask(masks, scores, pts, dep)
    assert m is not None and (m == m_arm).all() and reason == "ok" and [r[2] for r in ranked] == [1, 2]   # arm ranks first, disc second
    m2, _, r2, _ = G.select_mask(np.stack([huge]), np.array([0.9]), pts, dep); assert m2 is None and r2 == "no_mask"
    m3, _, r3, _ = G.select_mask(np.stack([m_disc]), np.array([0.5]), pts, dep); assert m3 is None and r3 == "no_mask"   # low score rejected


def test_operator_end_to_end_with_stub_matches_colour_formula():
    rgb, dep, gm, m_disc, _ = scene()
    stub = Stub(np.stack([m_disc, m_disc, m_disc]), np.array([0.5, 0.6, 0.9]))
    goal, info = G.op_marker_sam2(np.zeros(3), rgb, dep, K, EXT, gm, predictor=stub)
    assert info["applied"] and info["n_px"] >= 5
    vs, us = np.nonzero(m_disc); u, v = us.mean(), vs.mean(); z = 0.55 + G.MARKER_RADIUS_M
    exp = np.array([(u - 192) * z / 484.92407, (v - 192) * z / 484.92407, z])
    assert np.allclose(goal, exp, atol=1e-6)
    hole = np.full_like(dep, 0.7); hole[m_disc] = 0.8
    goal2, info2 = G.op_marker_sam2(np.array([9., 9., 9.]), rgb, hole, K, EXT, gm, predictor=stub)
    assert not info2["applied"] and info2["reason"] == "behind" and np.allclose(goal2, [9, 9, 9])
    nodep = np.zeros_like(dep); goal3, info3 = G.op_marker_sam2(np.array([9., 9., 9.]), rgb, nodep, K, EXT, gm, predictor=stub)
    assert not info3["applied"] and info3["reason"] == "prompt_depth_invalid"


def test_operator_falls_back_down_ranking_and_measure_goal_tolerant():
    rgb, dep, gm, m_disc, m_arm = scene(arm=True)
    stub = Stub(np.stack([m_arm, m_disc, m_disc]), np.array([0.95, 0.9, 0.85]))     # best-scoring candidate fails the aspect gate
    goal, info = G.op_marker_sam2(np.zeros(3), rgb, dep, K, EXT, gm, predictor=stub)
    assert not info["applied"] and info["reason"] == "aspect"                 # default: top candidate only
    G.SAM2_RANK_FALLBACK = True
    try:
        goal, info = G.op_marker_sam2(np.zeros(3), rgb, dep, K, EXT, gm, predictor=stub)
    finally:
        G.SAM2_RANK_FALLBACK = False
    assert info["applied"] and info["rank"] >= 1 and info["n_px"] == int(m_disc.sum())    # the stub repeats its 3 masks per trial, so the first disc sits behind all arm copies
    g2, i2 = G.measure_goal("pickcube", np.array([1., 2., 3.]), rgb=rgb, depth=dep, K=K, ext=EXT)     # legacy callers: no gmap_logits
    assert not i2["applied"] and i2["reason"] == "no_gmap_logits" and np.allclose(g2, [1, 2, 3])
    with pytest.raises(AssertionError):
        G.op_marker_sam2(np.zeros(3), rgb, (dep * 1000).astype(np.uint16), K, EXT, gm, predictor=stub)   # millimetre depth rejected


def test_prompts_at_image_border():
    rgb, dep, gm, _, _ = scene()
    gm[:] = -10.0; gm[0] = 10.0                                              # peak patch (r=0,c=0)
    pts, labels, (r, c) = G.gmap_peak_prompts(gm, dep)
    assert (r, c) == (0, 0) and 0 <= pts[1, 0] < 32 and 0 <= pts[1, 1] < 32
    gm[:] = -10.0; gm[575] = 10.0                                            # peak patch (23,23)
    pts, labels, (r, c) = G.gmap_peak_prompts(gm, dep)
    assert (r, c) == (23, 23) and 352 <= pts[1, 0] <= 383 and 352 <= pts[1, 1] <= 383
