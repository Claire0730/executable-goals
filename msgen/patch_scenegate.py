"""Resample scenes the camera cannot properly see. Enable with MSGEN_SCENE_GATE.

WHY. A manual review of the contact sheets flagged 23 of 256 PickCube eval scenes
and 68 of 256 StackCube eval scenes as unusable: the goal or the object is outside the frame,
half outside, or occluded. That matches the automatic measurement -- PickCube's goal is drawn up
to 30 cm above the cube (pick_cube.py:129) and leaves the frame 10.9% of the time; StackCube
places both cubes in a 0.2 x 0.4 m region (stack_cube.py:86) that is wider than the view.

WHAT IT DOES. `_initialize_episode(env_idx, options)` accepts a SUBSET of environments, so after
the task's own randomisation runs, the scenes whose bodies fall outside the usable frame are
re-randomised -- only those, repeatedly, until they pass or the attempt budget runs out. The
task's sampler is untouched; this is rejection sampling on top of it.

⚠️ THIS CHANGES THE TASK DISTRIBUTION. Numbers produced under the gate are NOT comparable with
ManiSkill's published PickCube/StackCube results, nor with any number this project measured
before, because those rest on the unconstrained distribution. It is opt-in per run and it prints
what it rejected, so a gated run can never be mistaken for an ungated one in a log.

CRITERION. Every body the task manipulates must project inside [M, RES-M] and sit within a depth
band. The margin exists because a centre just inside the edge still means a clipped object, and
because the goal is drawn with a tolerance sphere whose projected radius is ~25 px for PickCube
at typical depth.

    MSGEN_SCENE_GATE=1 [MSGEN_SCENE_GATE_MARGIN=40] [MSGEN_SCENE_GATE_Z=0.5,2.5] ...
"""
from __future__ import annotations

import os

_APPLIED = False
RES = 384
# attribute names on the env for the bodies whose visibility the task depends on
BODIES = {"PickCube": ("cube", "goal_site"), "StackCube": ("cubeA", "cubeB"),
          "PegInsertionSide": ("peg", "box"), "LiftPegUpright": ("peg",)}
# the first entry of BODIES is the TARGET, whose query-point coverage is checked
GRID = 20
# half-extent in metres, for the projected footprint the query lattice is counted against.
# Read from the env when it exposes one (`cube_half_size`), else this fallback.
HALF = {"PickCube": 0.02, "StackCube": 0.02, "PegInsertionSide": 0.025, "LiftPegUpright": 0.05}


def _cfg():
    m = float(os.environ.get("MSGEN_SCENE_GATE_MARGIN", "40"))
    z = os.environ.get("MSGEN_SCENE_GATE_Z", "0.5,2.5").split(",")
    return m, float(z[0]), float(z[1]), int(os.environ.get("MSGEN_SCENE_GATE_TRIES", "40"))


def maybe_patch() -> bool:
    global _APPLIED
    if _APPLIED or os.environ.get("MSGEN_SCENE_GATE", "0") in ("0", "", None):
        return _APPLIED
    import torch
    from mani_skill.envs.tasks.tabletop import (lift_peg_upright, peg_insertion_side,
                                                pick_cube, stack_cube)
    margin, zmin, zmax, tries = _cfg()
    minpts = int(os.environ.get("MSGEN_SCENE_GATE_MINPTS", "4"))
    step = RES / GRID
    lat = (torch.arange(GRID, dtype=torch.float64) + 0.5) * step   # 9.6, 28.8, ... 374.4
    stats = {"resampled": 0, "scenes": 0, "gave_up": 0}

    def wrap(cls, bodies, half):
        orig = cls._initialize_episode

        def _initialize_episode(self, env_idx, options: dict):
            orig(self, env_idx, options)
            cam = (getattr(self, "_sensors", {}) or {}).get("base_camera")
            if cam is None:
                return
            idx = env_idx
            for _ in range(tries):
                p = cam.get_params()
                K = p["intrinsic_cv"].to(torch.float64)
                E = p["extrinsic_cv"].to(torch.float64)
                bad = torch.zeros(len(idx), dtype=torch.bool, device=idx.device)
                for name in bodies:
                    b = getattr(self, name, None)
                    if b is None:
                        continue
                    w = b.pose.p.to(torch.float64)[idx]                   # [n,3] world
                    pc = torch.einsum("bij,bj->bi", E[idx, :, :3], w) + E[idx, :, 3]
                    z = pc[:, 2].clamp_min(1e-6)
                    u = pc[:, 0] / z * K[idx, 0, 0] + K[idx, 0, 2]
                    v = pc[:, 1] / z * K[idx, 1, 1] + K[idx, 1, 2]
                    bad |= ((u < margin) | (u > RES - margin) | (v < margin)
                            | (v > RES - margin) | (z < zmin) | (z > zmax))
                # QUERY-POINT COVERAGE, counted geometrically because the segmentation does not
                # exist yet at initialisation. Depth alone cannot predict it: measured
                # corr(depth, points) = -0.36 on PickCube, and the failing scenes' depth range
                # (1.067-1.187 m) sits entirely inside the passing range (1.041-1.193). The
                # variation is LATTICE PHASE -- the grid is 19.2 px apart and the cube is ~39 px
                # across, so whether it covers 4 points or 9 depends on where its centre lands.
                # Counting lattice points inside the projected footprint captures exactly that.
                tb = getattr(self, bodies[0], None)
                if tb is not None and minpts > 0:
                    # `cube_half_size` is a scalar on PickCube but a [3] tensor on
                    # StackCube; `tensor or half` asks for the tensor's truth value and
                    # raises. Take the largest extent, fall back to the registry HALF.
                    _hs = getattr(self, "cube_half_size", None)
                    if _hs is None:
                        hs = half
                    elif hasattr(_hs, "flatten"):
                        hs = float(_hs.flatten().max())
                    else:
                        hs = float(_hs)
                    w = tb.pose.p.to(torch.float64)[idx]
                    pc = torch.einsum("bij,bj->bi", E[idx, :, :3], w) + E[idx, :, 3]
                    zz = pc[:, 2].clamp_min(1e-6)
                    cu = pc[:, 0] / zz * K[idx, 0, 0] + K[idx, 0, 2]
                    cv = pc[:, 1] / zz * K[idx, 1, 1] + K[idx, 1, 2]
                    r = K[idx, 0, 0] * hs / zz                      # projected half-extent, px
                    L = lat.to(cu.device)
                    nu = ((L[None, :] >= (cu - r)[:, None]) & (L[None, :] <= (cu + r)[:, None])).sum(1)
                    nv = ((L[None, :] >= (cv - r)[:, None]) & (L[None, :] <= (cv + r)[:, None])).sum(1)
                    bad |= (nu * nv) < minpts
                if not bool(bad.any()):
                    break
                idx = idx[bad]
                stats["resampled"] += int(idx.numel())
                # `table_scene.initialize(env_idx)` ignores env_idx for the robot and writes
                # qpos for the WHOLE batch (scene_builder/table/scene_builder.py:102), so a
                # subset call dies with "value tensor of shape [n,9] cannot be broadcast to
                # indexing result of shape [N,9]". The resample only needs the object and goal
                # re-placed -- the robot's initial pose was set on the first pass and is not
                # what we are rejecting -- so the table/robot reset is skipped for the retry.
                # Writes inside _initialize_episode are scoped by scene._reset_mask, which the
                # OUTER reset set to the original env_idx (sapien_env.py:878-881). Calling the
                # initialiser on a subset without narrowing that mask makes set_pose try to
                # broadcast [n,7] into [N,7] and die. Narrow it, then restore.
                ts = getattr(self, "table_scene", None)
                prev = self.scene._reset_mask.clone()
                self.scene._reset_mask[:] = False
                self.scene._reset_mask[idx] = True
                if ts is not None:
                    _keep = ts.initialize
                    ts.initialize = lambda *_a, **_k: None
                try:
                    orig(self, idx, options)
                finally:
                    if ts is not None:
                        ts.initialize = _keep
                    self.scene._reset_mask = prev
            else:
                stats["gave_up"] += int(idx.numel())
                print(f"[scenegate] {cls.__name__}: {idx.numel()} scenes still outside the "
                      f"frame after {tries} tries -- the camera cannot cover this task's "
                      f"placement region, a gate cannot fix that", flush=True)
            stats["scenes"] += int(len(env_idx))

        cls._initialize_episode = _initialize_episode

    for mod, cname in ((pick_cube, "PickCube"), (stack_cube, "StackCube"),
                       (peg_insertion_side, "PegInsertionSide"),
                       (lift_peg_upright, "LiftPegUpright")):
        for attr in dir(mod):
            c = getattr(mod, attr)
            # only classes that DEFINE the method: PickCubeSO100Env/WidowXAIEnv inherit it,
            # so wrapping them too would run the gate twice per reset
            if (isinstance(c, type) and attr.endswith("Env") and cname in attr
                    and "_initialize_episode" in c.__dict__):
                wrap(c, BODIES[cname], HALF[cname])
                print(f"[scenegate] wrapped {attr} bodies={BODIES[cname]}", flush=True)
    _APPLIED = True
    print(f"[scenegate] active: margin={margin:.0f}px depth=[{zmin},{zmax}]m "
          f"min_query_points={minpts} tries={tries}. "
          f"THIS CHANGES THE TASK DISTRIBUTION -- results are not comparable with ungated runs.",
          flush=True)
    import atexit
    atexit.register(lambda: print(f"[scenegate] resampled {stats['resampled']} draws over "
                                  f"{stats['scenes']} scenes, gave up on {stats['gave_up']}",
                                  flush=True))
    return True
