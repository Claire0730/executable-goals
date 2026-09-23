"""Task registry for TraceGen x ManiSkill3 trace prediction.

Two stock ManiSkill3 tasks, each with a fixed oblique camera and three natural
language instructions (TraceGen samples one at random per training sample).
"""
from __future__ import annotations

import glob
import os

# Rendering / label geometry. 384 is TraceGen's SigLIP input resolution.
#
# GRID is an EXPERIMENTAL PARAMETER, not a constant of the architecture:
# `cogvideox_flow.py:393` derives the decoder's latent side as
# `int(math.sqrt(num_kps))` at runtime, so a denser query grid needs only
# `num_kps` in the config to match. It is overridable by environment variable so
# both grids can coexist without editing code between runs -- and because it is
# global, EVERY process touching a dataset must be given the same value. A
# mismatch is silent: `grid_pixels()` would return 400 points for a 1600-point
# dataset and index the wrong queries with nothing raised. Datasets built by
# `msgen.labels` therefore record their grid, and consumers check it.
#
# Motivation, measured: only 4.9 of 400 grid points land on the peg (it covers
# 1.3% of the frame), which is why a pose solved from them carries 6.9 deg of
# rotation error against 0.3 deg from 64 free points. 4x the density gives ~20
# points on the object, and a direction error fitted from n points falls as
# sqrt(n): 7.3 deg -> ~3.6 deg, i.e. 47 mm -> 23 mm of lateral offset at peg's
# 367 mm of travel, inside the +-30 mm success window.
IMAGE_SIZE = 384
GRID = int(os.environ.get("MSGEN_GRID", "20"))
NUM_KPS = GRID * GRID          # 400 at GRID=20, 1600 at GRID=40
TRAJ_STEPS = 33                # index 0 absolute + 32 future (datasets.py:672)
MAX_FRAMES = 64                # frames kept per replayed demo clip

# Demos live in two places: the ManiSkill default root (peg and stack were downloaded there)
# and the task-scan directory (the other six). Both are searched rather than
# consolidated, so nothing is duplicated and neither location has to move.
DEMO_ROOTS = tuple(r for r in (
    os.environ.get("MS_DEMO_DIR"),                       # explicit override (public release)
    os.path.expanduser("~/.maniskill/demos"),            # mani_skill.utils.download_demo default
) if r)
DEMO_ROOT = DEMO_ROOTS[0]          # kept for callers that predate the search

# Which subdirectory of a demo package to replay. Verified by opening the h5 files:
# the `rl/` packages carry `env_states` (actors + articulations) exactly like the
# `motionplanning/` ones, with ~1000 successful episodes each, so `msgen/replay.py`'s
# state-driven replay works from either. That matters because LiftPegUpright, PushT and
# PokeCube ship NO motionplanning demos at all.
#   PickCube is set to `rl` deliberately: its downloaded
#   `motionplanning/trajectory.h5` is TRUNCATED (h5py: "eof = 96, stored_eof = 2048") while
#   its rl package has 997-1022 usable episodes.
DEFAULT_DEMO_SOURCE = "motionplanning"

TASKS = {
    "stack": dict(
        env_id="StackCube-v1",
        # Oblique 3/4 view, zoomed onto the manipulation workspace. A 2 cm cube
        # against TraceGen's fixed 20x20 grid only ever covers a couple of grid
        # points, so the framing is as tight as it can be while still holding
        # both cubes (y in [-0.15, 0.16]) and the gripper approach (z up to 0.29).
        camera=dict(eye=[0.30, 0.26, 0.26], target=[-0.03, 0.0, 0.06], fov=0.85),
        target_body="cubeA",
        instructions=[
            "stack the red cube on top of the green cube",
            "pick up the red block and place it onto the green block",
            "put the red cube onto the green cube",
        ],
    ),
    "peg": dict(
        env_id="PegInsertionSide-v1",
        # The peg sweeps ~0.39 m along +y into a box fixed at y=0.29. Viewing
        # from +x (perpendicular to that sweep) keeps the whole insertion in
        # frame instead of letting the box occlude it, as a corner view does.
        # Swept over candidates: this one clips the peg at t=0 in 1/18 episodes,
        # vs 1/4 at fov 1.25-1.30. Narrower views lose the peg's spawn spread.
        camera=dict(eye=[0.50, 0.0, 0.33], target=[0.0, 0.03, 0.06], fov=1.40),
        target_body="peg_0",
        instructions=[
            "insert the peg into the hole in the box",
            "pick up the peg and push it sideways into the box hole",
            "align the peg with the hole and insert it",
        ],
    ),
    # ---- the extended eight-task benchmark set (<private-repo>/docs/TASKSET_FINAL_20260814.md).
    # Cameras are not hand-picked: each is the best ADMISSIBLE candidate of the sweep in
    # `<private-repo>/experiments/20260814_camscan/c2b_calibrated.json`, printed by `c3_registry.py`.
    # Admissible = the target is visible in every episode, its mask touches the frame border
    # in <=10% of episodes (peg's hand-tuned camera sits at 1/18 = 6%), and the object's spawn
    # AND goal both project inside the frame in >=90%. That last constraint exists because
    # `msgen/replay.py:58` installs ONE camera for the whole clip, so a framing that holds the
    # object at t=0 can still lose it mid-episode.
    # The two anchors above keep their hand-tuned cameras: changing them would break
    # comparability with every existing number. The sweep's role for them was calibration --
    # it reproduces peg's (5.1 points, 6% clip vs the tuned 4.9 / 1-in-18).
    "liftpeg": dict(
        env_id="LiftPegUpright-v1",
        # swept: 29.6 of 400 grid points on the target (6x peg's regime, never measured before),
        # 0% of episodes under the 3-point Kabsch floor, 6% clip, spawn+goal in frame 100%.
        # fov_scale 1.60, distance 4.5 bbox radii, elevation 30 deg, from 8 admissible.
        camera=dict(eye=[0.003, 0.551, 0.397], target=[0.003, -0.007, 0.075], fov=0.70),
        target_body="peg",
        demo_source="rl",          # ships no motionplanning demos; rl/ carries env_states
        instructions=[
            "lift the peg and stand it upright",
            "rotate the peg so that it stands vertically on the table",
            "make the peg stand up on its end",
        ],
    ),
    "pusht": dict(
        env_id="PushT-v1",
        # swept: 26.9 points, 0% under 3, 9% clip, endpoints 100%.
        # fov_scale 1.35, distance 2.1 radii, elevation 50 deg, from 9 admissible.
        camera=dict(eye=[0.094, -0.056, 0.359], target=[-0.156, -0.056, 0.061], fov=1.20),
        target_body="Tee",         # actor name is capitalised (verified by discovery run)
        demo_source="rl",
        instructions=[
            "push the T-shaped block onto the target outline",
            "slide the tee until it covers the goal outline",
            "move the T block to the marked target pose",
        ],
    ),
    "pickcube": dict(
        env_id="PickCube-v1",
        # CHOSEN OVER FRAMES, not at t=0 (<private-repo>/experiments/20260814_camscan/c4_pickcube.json).
        # 5.1 of 400 points on the cube with 1% of frames under the 3-point Kabsch floor --
        # better than peg's own dataset (3.9, 14%). Two corrections got here, both measured:
        #
        # 1. ELEVATION. The t=0 sweep picked 50 deg reading 4.0 points; the dataset it built
        #    measured 1.9 with 64% of FRAMES under the floor, because the gripper closes over
        #    the cube from above ("the thing that moves the object is the thing that hides it",
        #    HANDOFF_PLANNER_20260813 §3). Over 80 replayed frames: 50 deg -> 80% under 3,
        #    30 -> 28%, 20 -> 14%, 10 -> 10%. (PokeCube runs the OPPOSITE way; see below.)
        # 2. WHERE THE CAMERA LOOKS. At 10 deg the cube still fell below the bottom edge in
        #    50% of episodes at t=0, because the target sat at z=0.22 -- the centre of the
        #    spawn-union-goal bbox, which the airborne goal pulls upward, pushing the table
        #    surface under the frame. Aiming at the MIDPOINT OF THE OBJECT'S VERTICAL TRAVEL
        #    (z=0.10) fixes it: 3.7 points/18% -> 5.1/1%.
        camera=dict(eye=[0.003, 1.070, 0.289], target=[0.003, -0.003, 0.10], fov=0.35),
        target_body="cube",
        demo_source="rl",          # its motionplanning trajectory.h5 as downloaded is TRUNCATED
        instructions=[
            "pick up the red cube and move it to the goal position",
            "grasp the cube and lift it to the target point",
            "move the red cube to the goal marker",
        ],
    ),
    "pokecube": dict(
        env_id="PokeCube-v1",
        # CHOSEN OVER FRAMES (<private-repo>/experiments/20260814_camscan/c4_pokecube.json): 4.2 points with
        # 5% of frames under the 3-point floor, against 3.6 / 22% for the t=0 pick.
        # NOTE the elevation trend is the OPPOSITE of PickCube's -- 45 deg is best here, 10 deg
        # is worst (15%), while PickCube goes 10 deg best (10%) and 45 deg worst (66%). PokeCube
        # pushes the cube with a held peg, so a side view puts the peg in front of the cube,
        # whereas PickCube's gripper closes over it from above. Occlusion geometry is per-task;
        # there is no "lower is better" rule, which is why this sweep measures instead.
        camera=dict(eye=[0.163, -0.898, 0.959], target=[0.163, -0.002, 0.063], fov=0.35),
        target_body="cube",
        demo_source="rl",
        instructions=[
            "poke the cube with the peg into the goal region",
            "use the rod to push the cube onto the target",
            "push the cube with the peg into the marked area",
        ],
    ),
    "pushcube": dict(
        env_id="PushCube-v1",
        # swept: 5.7 points, 3% under 3, 6% clip, endpoints 94%.
        # fov_scale 0.90 at 2.1 radii, elevation 50 deg, from 26 admissible.
        camera=dict(eye=[0.103, 0.298, 0.423], target=[0.103, -0.007, 0.06], fov=0.80),
        target_body="cube",
        demo_source="motionplanning",
        instructions=[
            "push the cube onto the goal region",
            "slide the red cube to the target circle",
            "move the cube into the marked area",
        ],
    ),
    # Zero-shot probe: never collected as training data; registered for bank rendering only.
    "placesphere": dict(
        env_id="PlaceSphere-v1",
        camera=dict(eye=[0.103, 0.298, 0.423], target=[0.103, -0.007, 0.06], fov=0.80),
        target_body="sphere",
        demo_source="motionplanning",
        instructions=[
            "place the sphere into the bin",
            "pick up the ball and put it in the container",
            "move the sphere into the small bin",
        ],
    ),
    # PickSingleYCB-v1 is deliberately absent: it ships NO demos (its clips must come from RL
    # rollouts) and its object actor is named per scene (`062_dice-0`, `033_spatula-1`, ...), so
    # `target_body` cannot be a constant. Swept at 11.1 points, the best of the eight.
}


# ── opt-in camera override ───────────────────────────────────────────────────────────────────
# A camera is the single source of truth for BOTH the training render (replay.py:44-58) and the
# student env the eval banks are captured from (e.g. pickcube_kp_env.py:209-210). Editing the
# table above directly would therefore re-aim every future render AND every future evaluation,
# silently invalidating the comparison between any existing bank/checkpoint and anything new --
# a finetuned planner loses 46% of its accuracy for a 2.1 deg re-aim (<private-repo>/experiments/20260822_frac/
# c5_summary.txt), so an old checkpoint scored at a new camera reads as a regression that is
# really a mismatch. Opting in per run keeps both cameras available and makes the choice appear
# in the log.
#
#   MSGEN_CAM_DZ_PICKCUBE=0.06        raise that task's look-at target by 6 cm
#   MSGEN_CAM_EYE_PEG=0.3323,-0.3448,0.33    move the camera itself
#   MSGEN_CAM_FOV_PEG=1.20                   change its vertical field of view, radians
#
# Measured for pickcube (<private-repo>/experiments/20260824_camfix/c7_pickcube.json): +0.06 takes the GOAL
# from 89.1% to 100% inside the frame over the 256 eval scenes, while object points rise 4.4 ->
# 4.6 and frames under the 3-point Kabsch floor fall 3.3% -> 0.0%. Nothing gets worse.
#
# Measured for peg (<private-repo>/experiments/20260824_camfix/c7_peg_finalists.json), against the recorded
# finding that the box's HOLE is invisible in 36 of 100 training clips and 156 of the
# 256 eval scenes. The box yaw is drawn from [pi/2 - pi/8, pi/2 + pi/8] with x and y locked, so
# the hole axis is always within 22.5 deg of world +y while the registry camera sits at +x --
# viewing the hole edge-on by construction. Orbiting toward -y looks into the entrance the peg
# actually enters, with the peg NEARER the camera than the box; orbiting to +y puts the box in
# between and collapses everything (points 3.9 -> 1.0, frames with both bodies rendered
# 100% -> 73%). At -45 deg with the FOV narrowed to 1.20, every column improves at once:
#     REGISTRY      3.9 pts, 21.0% of frames under the Kabsch floor, hole visible  39.1%
#     az-45 f1.20   7.0 pts, 17.1% under the floor,                  hole visible 100.0%
# with peg and goal in frame 100%/100% and both bodies rendered in 100% of frames, unchanged.
# Narrower still (1.05, 0.90) buys more points but drops the peg out of frame in 4-22% of eval
# scenes, so 1.20 is the boundary, not a preference.
for _t in list(TASKS):
    _dz = os.environ.get(f"MSGEN_CAM_DZ_{_t.upper()}")
    if _dz:
        _c = TASKS[_t]["camera"]
        _old = list(_c["target"])
        _c["target"] = [_old[0], _old[1], _old[2] + float(_dz)]
        print(f"[tasks] {_t}: camera target {_old} -> {_c['target']} "
              f"(MSGEN_CAM_DZ_{_t.upper()}={_dz})", flush=True)
    for _k, _f in (("EYE", list), ("TARGET", list), ("FOV", float)):
        _v = os.environ.get(f"MSGEN_CAM_{_k}_{_t.upper()}")
        if not _v:
            continue
        _c = TASKS[_t]["camera"]
        _key = _k.lower()
        _old = _c[_key]
        _c[_key] = [float(x) for x in _v.split(",")] if _f is list else float(_v)
        print(f"[tasks] {_t}: camera {_key} {_old} -> {_c[_key]} "
              f"(MSGEN_CAM_{_k}_{_t.upper()}={_v})", flush=True)


def get_task(name: str) -> dict:
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; known: {sorted(TASKS)}")
    return TASKS[name]


def demo_dir(name: str) -> str:
    """First existing `<root>/<env_id>/<source>` that actually holds a trajectory.h5.

    Raises rather than returning a missing path, because the previous failure mode here was
    a downstream FileNotFoundError several frames away from the cause.
    """
    cfg = get_task(name)
    src = cfg.get("demo_source", DEFAULT_DEMO_SOURCE)
    tried = []
    for root in DEMO_ROOTS:
        d = f"{root}/{cfg['env_id']}/{src}"
        tried.append(d)
        if os.path.isfile(f"{d}/trajectory.h5") or glob.glob(f"{d}/*.h5"):
            return d
    raise FileNotFoundError(
        f"no {src} demos for {name} ({cfg['env_id']}); looked in {tried}. "
        f"Download with: python -m mani_skill.utils.download_demo {cfg['env_id']} "
        f"-o {DEMO_ROOTS[-1]}")
