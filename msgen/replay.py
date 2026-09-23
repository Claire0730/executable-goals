"""Replay official ManiSkill motion-planning demos and record everything the
sim-GT trace synthesis needs.

We drive the sim from the demos' recorded per-step `env_states` (actor poses +
articulation state) via `set_state_dict`, NOT by re-stepping the recorded
actions. Action replay was tried first and is unreliable here: `env.reset()`
between episodes does not restore scene state, so episode N+1 silently began
from episode N's final pose and produced clips with zero motion. Driving state
directly is exact, order-independent, and lets us index any frame. It also never
imports mplib (mplib 0.1.1 segfaults on this box).

Per kept frame we store RGB, metric depth, segmentation and the pose of every
rigid body in the scene. That is sufficient to propagate any pixel's 3D point
forward in time analytically -- see labels.py.

Run:
  python -m msgen.replay --task peg --episodes 18 --out data/raw/peg
"""
from __future__ import annotations

import argparse
import json
import os

import gymnasium as gym
import h5py
import numpy as np
import torch

import mani_skill.envs  # noqa: F401  (registers the envs)
from mani_skill.utils import sapien_utils

from msgen.tasks import IMAGE_SIZE, MAX_FRAMES, TASKS, demo_dir, get_task

CAM = "base_camera"


def _np(x):
    return x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def make_env(task: str, control_mode: str):
    cfg = get_task(task)
    c = cfg["camera"]
    cam_cfg = dict(
        pose=sapien_utils.look_at(eye=c["eye"], target=c["target"]),
        width=IMAGE_SIZE,
        height=IMAGE_SIZE,
        fov=c["fov"],
    )
    return gym.make(
        cfg["env_id"],
        obs_mode="rgb+depth+segmentation",
        control_mode=control_mode,
        render_mode="rgb_array",
        num_envs=1,
        sim_backend="physx_cpu",
        sensor_configs={CAM: cam_cfg},
    )


def body_table(env):
    """Every rigid body we can attribute a segmentation pixel to.

    Returns (names, ids) where ids are the per-scene segmentation ids. Both the
    loose actors (cubes, peg, table) and every robot link are included, so a
    grid point landing on the gripper is propagated by the gripper's motion.
    """
    ue = env.unwrapped
    names, ids, handles = [], [], []
    for name, actor in ue.scene.actors.items():
        names.append(name)
        ids.append(int(_np(actor.per_scene_id).reshape(-1)[0]))
        handles.append(actor)
    for art_name, art in ue.scene.articulations.items():
        for link in art.links:
            names.append(f"{art_name}/{link.name}")
            ids.append(int(_np(link.per_scene_id).reshape(-1)[0]))
            handles.append(link)
    return names, np.asarray(ids, dtype=np.int32), handles


def capture(env, handles):
    """One frame: rgb uint8, depth mm uint16, seg uint16, poses [B,7] (p + wxyz)."""
    obs = env.unwrapped.get_obs()
    sd = obs["sensor_data"][CAM]
    rgb = _np(sd["rgb"])[0].astype(np.uint8)
    depth = _np(sd["depth"])[0, ..., 0].astype(np.uint16)      # millimetres
    seg = _np(sd["segmentation"])[0, ..., 0].astype(np.uint16)
    poses = np.stack([_np(h.pose.raw_pose).reshape(-1)[:7] for h in handles])
    return rgb, depth, seg, poses.astype(np.float64)


def cam_params(env):
    obs = env.unwrapped.get_obs()
    p = obs["sensor_param"][CAM]
    K = _np(p["intrinsic_cv"]).reshape(3, 3).astype(np.float64)
    ext = _np(p["extrinsic_cv"]).reshape(3, 4).astype(np.float64)
    return K, ext


def unhide_goal(env, name: str = "goal_site"):
    """Make the goal marker visible to the SENSOR cameras.

    PickCube's goal_site (green sphere, radius = goal_thresh = 25 mm,
    pick_cube.py:96-104) sits in `_hidden_objects`, and `sapien_env.py:601`
    re-hides every member before each sensor capture. Two steps are both
    required: remove it from the list (stops future hides) AND call
    `show_visual()` (clears the already-applied hidden flag; verified in
    <private-repo>/experiments/20260820_goalvis/v0_peek.py -- removing alone still renders 0
    pixels). Must run AFTER every reset that reconfigures, because
    `_load_scene` re-appends the recreated actor.
    """
    ue = env.unwrapped
    tgt = ue.scene.actors.get(name)
    if tgt is None:
        raise SystemExit(f"--show-goal: no actor named {name!r} in this task")
    ue._hidden_objects = [o for o in ue._hidden_objects if o.name != name]
    tgt.show_visual()


def set_state(env, states, t):
    """Drive the sim to the demo's recorded state at step t."""
    sd = {cat: {k: torch.as_tensor(np.asarray(v[t])[None], dtype=torch.float32)
                for k, v in group.items()}
          for cat, group in states.items()}
    env.unwrapped.set_state_dict(sd)


def replay_episode(env, handles, states, n_states):
    """Visit MAX_FRAMES uniformly spaced recorded states and capture each."""
    steps = np.unique(np.linspace(0, n_states - 1, min(MAX_FRAMES, n_states)).astype(int))
    rgbs, depths, segs, poses, succ = [], [], [], [], []

    for t in steps:
        set_state(env, states, int(t))
        r, d, s, p = capture(env, handles)
        rgbs.append(r); depths.append(d); segs.append(s); poses.append(p)
        succ.append(bool(_np(env.unwrapped.evaluate()["success"]).reshape(-1)[0]))

    return (np.stack(rgbs), np.stack(depths), np.stack(segs), np.stack(poses),
            steps.astype(np.int32), np.asarray(succ))


def resolve_demo_files(ddir: str, prefer: str | None = None):
    """Find the (h5, json) pair in a demo directory.

    `motionplanning/` packages hold a plain `trajectory.h5`, but `rl/` packages name theirs
    `trajectory.none.<control_mode>.physx_cuda.h5` -- so the previously hardcoded
    `trajectory.h5` cannot open an rl package at all, and rl is the ONLY source for
    LiftPegUpright, PushT and PokeCube.

    `pd_joint_delta_pos` is preferred by default because it is the control mode all of the
    rl packages provide, which keeps the replayed clips consistent across tasks.
    """
    import glob as _glob

    cands = sorted(_glob.glob(f"{ddir}/*.h5"))
    if not cands:
        raise FileNotFoundError(f"no .h5 in {ddir}")
    pick = None
    if prefer:
        pick = next((c for c in cands if prefer in c), None)
        if pick is None:
            raise FileNotFoundError(
                f"no demo with control mode {prefer!r} in {ddir}; have "
                f"{[os.path.basename(c) for c in cands]}")
    if pick is None:
        pick = next((c for c in cands if "pd_joint_delta_pos" in c), cands[0])
    # The control-mode heuristic silently preferred `trajectory.teacher5cm.pd_joint_delta_pos.h5` (a 5 cm
    # probe file dropped next to StackCube's motionplanning demos on 09-07) over the 4 cm `trajectory.h5`. An explicit
    # override (basename or full path) makes the demo source a stated choice rather than a directory-listing accident.
    want = os.environ.get("MSGEN_DEMO_H5", "")
    if want:
        pick = want if os.path.isabs(want) else f"{ddir}/{want}"
        if not os.path.isfile(pick):
            raise FileNotFoundError(f"MSGEN_DEMO_H5={want!r} not found in {ddir}")
    js = pick[:-3] + ".json"
    if not os.path.isfile(js):
        raise FileNotFoundError(f"{pick} has no matching .json")
    return pick, js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--control-mode", default=None,
                    help="substring selecting which demo file to replay, e.g. "
                         "pd_joint_delta_pos; default prefers that mode when present")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--start", type=int, default=0, help="index into the demo episode list")
    ap.add_argument("--episode-ids", default=None,
                    help="JSON file with a list of POSITIONS into the success-"
                         "filtered episode list (the same positions a sequential "
                         "replay would have assigned as clip indices -- i.e. a "
                         "pool manifest's pool_clip numbers). Output clips are "
                         "written in the given order, so clip_k of this pool "
                         "corresponds 1:1 to entry k of the manifest that "
                         "supplied the ids. Overrides --start/--episodes.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--show-goal", action="store_true",
                    help="render the goal marker (e.g. PickCube's goal_site) "
                         "into the sensor frames instead of hiding it; poses "
                         "and manifest schema are unchanged")
    args = ap.parse_args()

    ddir = demo_dir(args.task)
    traj_h5, traj_json = resolve_demo_files(ddir, args.control_mode)
    print(f"demos: {traj_h5}")
    meta = json.load(open(traj_json))
    h5 = h5py.File(traj_h5, "r")

    eps = [e for e in meta["episodes"] if e.get("success", True)]
    if args.episode_ids:
        ids = json.load(open(args.episode_ids))
        missing = [i for i in ids if i >= len(eps)]
        if missing:
            raise SystemExit(f"episode positions out of range: {missing[:5]} (pool has {len(eps)})")
        eps = [eps[i] for i in ids]
    else:
        if args.episodes is None:
            raise SystemExit("--episodes is required unless --episode-ids is given")
        eps = eps[args.start:args.start + args.episodes]
    if args.episodes is not None and not args.episode_ids and len(eps) < args.episodes:
        raise SystemExit(f"only {len(eps)} episodes available from index {args.start}")

    control_mode = eps[0]["control_mode"]
    env = make_env(args.task, control_mode)
    os.makedirs(args.out, exist_ok=True)

    manifest = []
    for i, ep in enumerate(eps):
        # Reconfigure per episode: PegInsertionSide randomizes the peg's size and
        # the box's hole geometry at reconfiguration, so replaying episode N's
        # states into episode 0's geometry renders the wrong peg and fails the
        # task's own success check. Handles and segmentation ids are rebuilt
        # afterwards because reconfiguring recreates the scene objects.
        rk = dict(ep["reset_kwargs"])
        rk["options"] = {**rk.get("options", {}), "reconfigure": True}
        env.reset(**rk)
        if args.show_goal:
            unhide_goal(env)
        names, ids, handles = body_table(env)
        K, ext = cam_params(env)

        g = h5[f"traj_{ep['episode_id']}"]["env_states"]
        states = {cat: {k: g[cat][k] for k in g[cat]} for cat in g}
        n_states = len(next(iter(states["actors"].values())))
        rgb, depth, seg, poses, steps, succ = replay_episode(env, handles, states, n_states)

        clip = f"clip_{i:03d}"
        np.savez(
            f"{args.out}/{clip}.npz",
            rgb=rgb, depth_mm=depth, seg=seg, poses=poses, steps=steps, success=succ,
            body_ids=ids, K=K, extrinsic_cv=ext,
        )
        # Record the target's segmentation id while the env handle is still in scope. Every
        # consumer then reads it instead of matching a name, which is the only thing that works
        # for a task whose object actor is named per scene (see msgen/objmetrics.target_id_for).
        tname = get_task(args.task).get("target_body")
        tid = int(ids[names.index(tname)]) if tname in names else None

        manifest.append(dict(clip=clip, episode_id=int(ep["episode_id"]),
                             target_name=tname, target_id=tid,
                             seed=ep["reset_kwargs"].get("seed"),
                             n_states=int(n_states), n_frames=int(len(steps)),
                             success_at_end=bool(succ[-1]),
                             body_names=names, body_ids=ids.tolist()))
        print(f"[{clip}] episode_id={ep['episode_id']} states={n_states} "
              f"kept={len(steps)} success_end={succ[-1]}", flush=True)

    json.dump(dict(task=args.task, env_id=get_task(args.task)["env_id"],
                   control_mode=control_mode, body_names=names,
                   body_ids=ids.tolist(), clips=manifest),
              open(f"{args.out}/manifest.json", "w"), indent=2)
    print(f"wrote {len(manifest)} clips -> {args.out}")


if __name__ == "__main__":
    main()
