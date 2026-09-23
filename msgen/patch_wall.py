"""Bounded scene backdrop. Enable with MSGEN_WALL=1.

WHY. The sim scenes have no background geometry: behind the table the ground
plane recedes to the camera far plane, so the depth image spans 0.9-32.2 m while
the entire workspace occupies 4.1% of that range (cube-vs-table contrast:
0.12%). `preprocess_depth` being a no-op, those raw metres reach the frozen
SigLIP tower unscaled -- the strongest depth structure in the frame is the void
horizon, not the task. The Generalist was pretrained on REAL indoor scenes
(bounded, textured backgrounds), so an infinite void is the out-of-distribution
input, not the fix. This patch bounds depth AT THE SCENE LEVEL: no transform
changes under the warm start (the dnorm lesson), the input's value range simply
becomes finite because the world is.

WHAT IT ADDS. A static, collision-free backdrop of horizontal bands of a slate-indigo family (deliberately NOT white/grey: the Franka is white) (the colour steps give the RGB streams edges to
lock onto -- a uniform white wall would bound depth but feed DINOv3/SigLIP a
textureless field). The wall is placed PER TASK, perpendicular to that task's
fitted camera axis at WALL_D metres beyond the look-at target, sized to cover
the frustum. Deterministic constants: it consumes no RNG, so scene identity
(seed -> layout) and the scenegate are unaffected.

    MSGEN_WALL=1 ...
"""
from __future__ import annotations

import os

_APPLIED = False
BANDS = [  # (half_height, rgb) bottom -> top; 6 x 0.7 m bands, z in [-1.05, +3.15].
    # SLATE-INDIGO family, chosen against the scene palette so nothing is
    # confusable: the Franka is WHITE/light grey (so no whites/greys here), the
    # table is wood brown, cubes are red/green, the peg box is beige, the peg
    # halves are red/white/BLUE -- the wall's blue-violet is kept desaturated
    # and mid-dark so the saturated peg blue still stands apart.
    (0.35, (0.23, 0.26, 0.36)),
    (0.35, (0.42, 0.46, 0.58)),
    (0.35, (0.29, 0.33, 0.45)),
    (0.35, (0.50, 0.54, 0.66)),
    (0.35, (0.25, 0.29, 0.40)),
    (0.35, (0.36, 0.40, 0.52)),
]
RADIUS = 2.0        # metres, the enclosure's half-extent around the workspace
THICK = 0.05


def maybe_patch() -> bool:
    global _APPLIED, BANDS
    if _APPLIED or os.environ.get("MSGEN_WALL", "0") in ("0", "", None):
        return _APPLIED
    # MSGEN_WALL_COLOR=white -- uniform matte white wall (same geometry, no bands). This is the texture
    # ablation the header argues AGAINST; added for the domain-shift probe series.
    if os.environ.get("MSGEN_WALL_COLOR", "").lower() == "white":
        BANDS = [(hz, (0.92, 0.92, 0.92)) for hz, _ in BANDS]
        print("[wall] colour override: uniform white", flush=True)
    import numpy as np
    import sapien
    from mani_skill.envs.tasks.tabletop import (lift_peg_upright, peg_insertion_side,
                                                pick_cube, push_cube, stack_cube)
    from mani_skill.utils.building import actors
    
    def wrap(cls, mkey):
        orig = cls._load_scene

        def _load_scene(self, options):
            orig(self, options)
            # A single backdrop proved insufficient: the stack camera's oblique
            # view leaks past one wall's edge (measured 6.89 m rays). Enclose the
            # workspace on all four sides instead; every horizontal ray is then
            # bounded by <= RADIUS*sqrt(2) + slack, task-independently.
            import itertools
            for side, (px, py, yaw) in enumerate([
                    ( RADIUS, 0.0, 0.0), (-RADIUS, 0.0, 0.0),
                    (0.0,  RADIUS, 1.5707963), (0.0, -RADIUS, 1.5707963)]):
                q = [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]
                # The TABLETOP is z=0 but the room floor sits at z~-0.91; a wall
                # starting at 0 leaves a slot under it through which rays past
                # the table edge reach the distant floor (measured: a full-width
                # >5 m band). Start below the floor.
                z0 = -1.05
                for i, (hz, rgb) in enumerate(BANDS):
                    actors.build_box(
                        self.scene, half_sizes=[THICK, RADIUS + 2 * THICK, hz],
                        color=list(rgb) + [1.0],
                        name=f"msgen_wall_s{side}_b{i}",
                        body_type="static", add_collision=False,
                        initial_pose=sapien.Pose(p=[px, py, z0 + hz], q=q))
                    z0 += 2 * hz

        cls._load_scene = _load_scene
        print(f"[wall] {cls.__name__}: 4-sided enclosure at +-{RADIUS} m, "
              f"{len(BANDS)} colour bands, no collision", flush=True)

    wrap(pick_cube.PickCubeEnv, "pickcube")
    wrap(stack_cube.StackCubeEnv, "stack")
    wrap(peg_insertion_side.PegInsertionSideEnv, "peg")
    wrap(lift_peg_upright.LiftPegUprightEnv, "liftpeg")
    # PushCube banks of the paper were rendered without the wall (see the release's KNOWN_ISSUES, item 7;
    # the protocol inconsistency was caught during the white-wall probe). The wrapper below covers it.
    wrap(push_cube.PushCubeEnv, "pushcube")
    _APPLIED = True
    return True
