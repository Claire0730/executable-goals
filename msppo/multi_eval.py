"""Evaluate a mixed-task student ONE TASK AT A TIME, native success, held-out seed.

A mixed mean is not a result here. If a two-task student scores 0.50 that can be
0.50/0.50 or 1.00/0.00, and those are opposite conclusions about the hypothesis
under test. So every task gets its own row and the mean is printed last, marked.

The student is rebuilt from its own run.json (`tasks`, `lang`, `lang_dim`,
`scene_kp`, `point_obs`) -- never from flags. The language row is FIXED to
instruction 0 at eval: training samples one of three per episode, but a
deterministic eval must not have a hidden random input.

    $PM -m msppo.multi_eval --run runs_rl/mt_m0_s0 --episodes 256
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from msppo.kp_teacher import TASKS
from msppo.student import goal_extra_slices, StudentTransformer, goal_slices
from msppo.task_student_env import TaskStudentEnv, make_task_student_env

T5_NPZ = "assets/instr_t5.npz"


def lang_row(cfg, task, n, dev):
    """[n, lang_dim] for ONE task, or None."""
    if not cfg["lang_dim"]:
        return None
    tasks = cfg["tasks"]
    if cfg["lang"] == "onehot":
        v = torch.zeros(len(tasks), device=dev)
        v[tasks.index(task)] = 1.0
        return v.expand(n, -1)
    z = np.load(T5_NPZ)
    # instruction 0, fixed: a deterministic eval cannot carry a random input.
    return torch.as_tensor(z[task][0], device=dev).expand(n, -1)


def env_patches():
    """Apply the OPT-IN scene patches the deployment planner was trained against
    (bounded wall backdrop, camera-visibility scene gate, per-task camera
    overrides) and return what took effect, so every JSON records the env it
    was scored in. All are env-var gated and process-wide:
        MSGEN_WALL=1  MSGEN_SCENE_GATE=1  MSGEN_CAM_DZ_<TASK>  MSGEN_CAM_EYE_PEG  MSGEN_CAM_FOV_PEG
    The scene gate is all-or-nothing per process, so a run that wants it on
    PickCube/StackCube only must score those tasks in a separate process
    (<private-repo>/experiments/20260824_wallsel/w12_eval800.sh does exactly that)."""
    import os
    from msgen.patch_scenegate import maybe_patch as gate
    from msgen.patch_wall import maybe_patch as wall
    w, g = bool(wall()), bool(gate())
    if os.environ.get("MSGEN_WALL", "0") not in ("0", "") and not w:
        raise SystemExit("MSGEN_WALL requested but the patch did not apply")
    if os.environ.get("MSGEN_SCENE_GATE", "0") not in ("0", "") and not g:
        raise SystemExit("MSGEN_SCENE_GATE requested but the patch did not apply")
    cam = {k: v for k, v in os.environ.items() if k.startswith("MSGEN_CAM_")}
    return dict(wall=w, scene_gate=g, camera=cam, base_pose=os.environ.get("MSPPO_BASE_POSE") or None)   # recorded in every JSON


def load_delta(path, episodes, step, dev):
    """One task's prediction(s). `a+b+c+d` = K sample files of the SAME bank,
    returned as a list of (dR, dt); a single path returns a one-element list.
    The env composes each sample with the same perceived points and takes the
    mean for the goal_pose slot (and the cloud for the sd/k slots)."""
    return [_load_one(p, episodes, step, dev) for p in path.split("+")]


def load_psi(path, episodes, dev):
    """[N,4] psi bank (k1 unit vector [N,3], h metres [N]) from the planner trace (<private-repo>/experiments/20260830_k3/psi_bank.py),
    rows in env order like the goal bank."""
    import numpy as np
    z = np.load(path)
    if z["k1"].shape[0] != episodes:
        raise SystemExit(f"{path} covers {z['k1'].shape[0]} scenes but this eval runs {episodes}")
    return torch.as_tensor(np.concatenate([z["k1"], z["h"][:, None]], 1), device=dev, dtype=torch.float32)


def _load_one(path, episodes, step, dev):
    """One task's TraceGen prediction, as the rigid transform the env composes
    with the PERCEIVED object pose. `msppo.task_tgbank solve` writes it.

    The bank must have been rendered at the SAME num_envs as this eval: geometry
    is drawn once under reconfiguration_freq=0, so num_envs is part of scene
    identity and a bank of a different size describes different scenes."""
    import numpy as np
    z = np.load(path)
    st = step if step is not None else z["dR"].shape[1] - 1
    if z["dR"].shape[0] != episodes:
        raise SystemExit(
            f"{path} covers {z['dR'].shape[0]} scenes but this eval runs "
            f"{episodes}; render the bank at the same count")
    return (torch.as_tensor(z["dR"][:, st], device=dev, dtype=torch.float32),
            torch.as_tensor(z["dt"][:, st], device=dev, dtype=torch.float32))


LAST_DIAG = {}   # task -> per-episode diagnostics of the last score() call (failure analysis; written to the json as per_env)


def score(cfg, task, episodes, seed, student, dev, goal_delta=None, sig_bank=None,
          zero_block=None, psi_bank=None):
    if goal_delta is not None:
        want = cfg.get("goal_k", 4) if cfg.get("goal_form", "mean") != "mean" else None
        if want is not None and len(goal_delta) != want:
            # a K-form student fed the wrong number of samples would silently
            # see a repeated single goal (sd = 0) -- the identity case, not the
            # planner. Refuse instead.
            raise SystemExit(f"{task}: goal_form {cfg['goal_form']} run needs exactly "
                             f"K={want} sample files (a+b+c+d), got {len(goal_delta)}")
    spec = TASKS[task]
    env = make_task_student_env(
        task, num_envs=episodes, num_kp=cfg["num_kp"], seed=seed,
        max_episode_steps=spec["horizon"], kp_mask_ratio=cfg["kp_mask_ratio"],
        kp_noise_m=cfg["kp_noise_m"], kp_bias_m=cfg["kp_bias_m"],
        kp_bias_rand=False, point_obs=cfg["point_obs"],
        # Must reproduce the PER-TASK mode training used, not the student's
        # single declared flag: a fixture-less task in a mixed run was trained
        # with the block ZEROED, and evaluating it with a real fixture pose would
        # hand the policy an input it never saw.
        # A run trained with --no-scene has NO fixture channel for any task, and
        # this expression would otherwise hand peg and StackCube a real fixture
        # pose they never saw in training -- the exact train/eval skew the
        # per-task comment below exists to prevent, one level up.
        scene_kp=(False if cfg.get("no_scene") else
                  (True if TaskStudentEnv.ACTORS[task][1] is not None
                   else ("zero" if cfg["scene_kp"] else False))),
        tcp_obs=True, perception=True, goal_delta=goal_delta,
        # From the RUN, never a default: flipping either of these silently
        # re-scores every older run (handover rule 6).
        goal_rel_se3=cfg.get("goal_rel_se3", False),
        goal_sig=cfg.get("goal_sig", False), goal_sig_bank=sig_bank,
        goal_form=cfg.get("goal_form", "mean"), goal_k=cfg.get("goal_k", 4),
        psi=bool(cfg.get("psi", False)), psi_bank=psi_bank)
    kp = env.kp
    lr = lang_row(cfg, task, episodes, dev)
    goal_sl, goal_rel_sl = goal_slices(student)
    extra_sls = goal_extra_slices(student)   # K-sample slots, zeroed with the goal
    lo = torch.as_tensor(env.single_action_space.low, device=dev)
    hi = torch.as_tensor(env.single_action_space.high, device=dev)

    obs, _ = kp.reset(seed=seed)
    env.perc_reset(obs)
    env.reset_masks()
    live = torch.ones(episodes, dtype=torch.bool, device=dev)
    ever = torch.zeros(episodes, dtype=torch.bool, device=dev)
    # per-episode diagnostics (frozen at the episode's first done, like `ever`): grasped ever, fixture bumped > 10 mm
    # (stack: cubeB), last object-to-TRUE-goal distance while live, steps lived. Never allowed to change the score.
    _obj_name, _fix_name = TaskStudentEnv.ACTORS.get(task, (None, None))
    _obj = getattr(kp.base, _obj_name, None) if _obj_name else None
    if _obj is None and _obj_name and _obj_name in getattr(kp.base.scene, "actors", {}):
        _obj = kp.base.scene.actors[_obj_name]
    _fix = getattr(kp.base, _fix_name) if _fix_name and hasattr(kp.base, _fix_name) else None
    _fix0 = _fix.pose.p.clone() if _fix is not None else None
    d_grasp = torch.zeros(episodes, dtype=torch.bool, device=dev); d_bump = torch.zeros(episodes, dtype=torch.bool, device=dev)
    d_err = torch.full((episodes,), float("nan"), device=dev); d_steps = torch.zeros(episodes, device=dev)
    d_tcpmin = torch.full((episodes,), 9.9, device=dev)
    with torch.no_grad():
        for _ in range(spec["horizon"]):
            s = env.student_observation_perc(obs).float()
            if lr is not None:
                s = torch.cat([s, lr], dim=-1)
            if zero_block is not None:
                # CHANNEL INERTNESS TEST (student side). Same score with the
                # block zeroed means the student never learned to read it.
                # Mirrors kp_teacher_eval --zero-block on the teacher side.
                # a LIST of slices: "goal_all" must zero the rel block too
                for _z in zero_block:
                    s[:, _z] = 0.0
            if cfg.get("no_trace"):
                s[:, goal_sl] = 0.0
                if goal_rel_sl.stop > goal_rel_sl.start:
                    s[:, goal_rel_sl] = 0.0
                for e in extra_sls:
                    s[:, e] = 0.0
            a = student.actor_mean(s)
            obs, _, term, trunc, info = kp.step(a.clamp(lo, hi))
            env.set_prev_action(a)
            done = (term | trunc).bool()
            sv = info["success"].bool().clone()
            fi = info.get("final_info")
            if fi is not None and "success" in fi:
                sv[done] = fi["success"].bool()[done]
            ever |= sv & live
            try:
                # the sim state at a DONE step already belongs to the auto-reset next episode: exclude it
                _lv = live & ~done
                if _obj is not None:
                    d_grasp |= kp.base.agent.is_grasping(_obj).bool() & _lv
                    _gp, _ = kp._goal_from_obs(kp.raw_from_sim())
                    d_err = torch.where(_lv, (_obj.pose.p - _gp).norm(dim=-1), d_err)
                if _obj is not None:
                    d_tcpmin = torch.where(_lv, torch.minimum(d_tcpmin, (kp.base.agent.tcp.pose.p - _obj.pose.p).norm(dim=-1)), d_tcpmin)
                if _fix is not None:
                    d_bump |= ((_fix.pose.p - _fix0).norm(dim=-1) > 0.01) & _lv
                d_steps += _lv.float()
            except Exception:
                _obj = _fix = None
            live &= ~done
            if bool(done.any()):
                env.reset_masks(done)
                env.perc_reset(obs, idx=done)
            if not live.any():
                break
    env.close()
    LAST_DIAG[task] = dict(success=ever.int().tolist(), grasped=d_grasp.int().tolist(), bumped=d_bump.int().tolist(),
                           final_err_mm=[round(v * 1000, 1) if v == v else None for v in d_err.tolist()], steps=d_steps.int().tolist(),
                           tcp_min_mm=[round(v * 1000, 1) for v in d_tcpmin.tolist()])
    return float(ever.float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--ckpt", default="final", choices=["final", "best"])
    ap.add_argument("--goal-sig-bank", default=None,
                    help="task=npz,... per-scene [rmse, s21]; required when the "
                         "run has goal_sig and --goal-delta names that task")
    ap.add_argument("--goal-delta", default=None,
                    help="THE STRICT ROW-4 INTERFACE for a mixed student: "
                         "comma-separated task=path.npz, one per task that has a "
                         "prediction. Tasks left out keep the TRUE goal, and the "
                         "output records which was which -- a table that mixes "
                         "the two silently is the failure this flag exists to "
                         "prevent. Example: "
                         "pickcube=results/pickcube_goals_mix4.npz,stack=...")
    ap.add_argument("--psi-bank", default=None,
                    help="task=path.npz,... psi (k1,h) per scene from the planner trace, for psi students (row 4)")
    ap.add_argument("--goal-step", type=int, default=None,
                    help="which trace step to consume (default: the last)")
    ap.add_argument("--zero-block", default=None,
                    choices=("goal_sig", "goal", "goal_rel", "goal_all"),
                    help="CHANNEL INERTNESS TEST: zero this student observation "
                         "block before the policy sees it. Same score means the "
                         "channel was never learned. Slices come from the "
                         "student model itself, not from hardcoded offsets.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None,
                    help="comma-separated subset of the run's tasks to score "
                         "(the scene gate is per-process, so gated and ungated "
                         "tasks must be scored in separate invocations)")
    args = ap.parse_args()
    patches = env_patches()
    print(f"[env] wall={patches['wall']} scene_gate={patches['scene_gate']} cam={patches['camera']}")

    gd_paths = {}
    if args.goal_delta:
        for part in args.goal_delta.split(","):
            if "=" not in part:
                raise SystemExit(f"--goal-delta wants task=path, got {part!r}")
            t, path = part.split("=", 1)
            gd_paths[t.strip()] = path.strip()

    cfg = json.load(open(f"{args.run}/run.json"))
    dev = "cuda"
    if cfg.get("psi"):
        # the student carries the psi block: row 3 takes the frame-patched env's own prior command (as the teacher's
        # clean_prior eval), row 4 the trace-derived bank (--psi-bank task=npz with k1 [N,3], h [N])
        from msppo.frame_patches import apply as _fp
        print(f"[psi] frame patches applied for {_fp(cfg['tasks'])}")
    pb_paths = {}
    if getattr(args, "psi_bank", None):
        for part in args.psi_bank.split(","):
            if "=" not in part:
                raise SystemExit(f"--psi-bank wants task=path, got {part!r}")
            t_, p_ = part.split("=", 1); pb_paths[t_.strip()] = p_.strip()
        if not cfg.get("psi"):
            raise SystemExit("--psi-bank given but this run has no psi block")
    # Width/depth from the RUN, not from this file's defaults. Older runs have no
    # such field and get the 128/4/4 they were trained with; a wider run that
    # this file rebuilt at the default would fail load_state_dict AFTER an
    # 11-hour training run, which is the failure this project keeps paying for.
    student = StudentTransformer(
        num_kp=cfg["num_kp"], act_dim=8, pose_obs=cfg["pose_obs"],
        scene_kp=cfg["scene_kp"], tcp=cfg["tcp_obs"], qdim=cfg["qdim"],
        lang_dim=cfg["lang_dim"], d=cfg.get("d_model", 128),
        layers=cfg.get("layers", 4), heads=cfg.get("heads", 4),
        rel_token=cfg.get("rel_token", False),
        goal_sig=cfg.get("goal_sig", False),
        goal_form=cfg.get("goal_form", "mean"), goal_k=cfg.get("goal_k", 4),
        psi=bool(cfg.get("psi", False)), psi_token=bool(cfg.get("psi_token", False)), prev_action=True).to(dev)
    student.load_state_dict(torch.load(
        f"{args.run}/student.pt" if args.ckpt == "final"
        else f"{args.run}/student_best.pt"))
    student.eval()

    unknown = set(gd_paths) - set(cfg["tasks"])
    if unknown:
        raise SystemExit(f"--goal-delta names tasks this run does not have: {unknown}")

    sig_paths = {}
    if getattr(args, "goal_sig_bank", None):
        for part in args.goal_sig_bank.split(","):
            k, v = part.split("=", 1); sig_paths[k.strip()] = v.strip()
    # A planner goal with the "perfect fit" default would be a LIE: (0,1) means
    # zero residual and perfect conditioning, i.e. trust it completely. Refuse
    # rather than quietly feed the student a reliability it does not have.
    if cfg.get("goal_sig"):
        # A K>1 delta needs no bank: the reliability is the SAMPLING CLOUD's own
        # spread, computed per step from the K samples (`_set_cloud_sig`). A K=1
        # delta has no spread, so the channel would read 0 -- "trust it
        # completely" -- which is the lie this guard exists to prevent.
        miss = [t for t, v in gd_paths.items()
                if t not in sig_paths and len(v.split("+")) < 2]
        if miss:
            raise SystemExit(
                f"goal_sig run with a K=1 --goal-delta on {miss} and no "
                f"--goal-sig-bank. Either pass the K samples as "
                f"'a.npz+b.npz+c.npz+d.npz' so the cloud spread is real, or "
                f"supply --goal-sig-bank.")

    zb = None
    if args.zero_block:
        sl = getattr(student, "sl", {})
        key = {"goal_sig": "goal_sig", "goal": "goal_pose", "goal_rel": "goal_rel"}
        gsl, grl = goal_slices(student)
        if args.zero_block == "goal_all":
            # goal_pose ALONE is not enough: `goal_slices` documents that with
            # obj_pose present the goal position is exactly recoverable from the
            # rel block (measured 3.07e-05 mm). Zeroing only the pose slot is the
            # ablation that was mislabelled for the whole project.
            zb = [gsl] + ([grl] if grl.stop > grl.start else [])
        else:
            zb = {"goal_sig": sl.get("goal_sig"), "goal": gsl, "goal_rel": grl}[args.zero_block]
            if zb is None or zb.stop <= zb.start:
                raise SystemExit(f"this student has no {args.zero_block!r} block")
            zb = [zb]
        print(f"[inert] zeroing {args.zero_block} = {zb}")

    only = [t.strip() for t in args.only.split(",")] if args.only else list(cfg["tasks"])
    bad = set(only) - set(cfg["tasks"])
    if bad:
        raise SystemExit(f"--only names tasks this run does not have: {bad}")
    per, src = {}, {}
    for t in only:
        gd = (load_delta(gd_paths[t], args.episodes, args.goal_step, dev)
              if t in gd_paths else None)
        src[t] = gd_paths.get(t, "true goal")
        sb = None
        if t in sig_paths:
            zz = np.load(sig_paths[t])
            sb = torch.as_tensor(np.stack([zz["rmse"], zz["s21"]], -1),
                                 device=dev, dtype=torch.float32)
        pb = load_psi(pb_paths[t], args.episodes, dev) if t in pb_paths else None
        per[t] = score(cfg, t, args.episodes, args.seed, student, dev,
                       goal_delta=gd, sig_bank=sb, zero_block=zb, psi_bank=pb)
        print(f"  {t:10s} success_once = {per[t]:.4f}   goal={src[t]}   "
              f"(N={args.episodes}, seed {args.seed} != train seed {cfg['seed']})")
    m = float(np.mean(list(per.values())))
    print(f"  {'MEAN':10s} {m:.4f}   (the mean is not a result; the per-task rows are)")

    out = dict(run=args.run, arm="multitask_student", tasks=cfg["tasks"],
               lang=cfg["lang"], lang_dim=cfg["lang_dim"],
               no_trace=cfg.get("no_trace", False), ckpt=args.ckpt,
               goal_rel_se3=cfg.get("goal_rel_se3", False),
               rel_token=cfg.get("rel_token", False),
               goal_form=cfg.get("goal_form", "mean"), goal_k=cfg.get("goal_k", 4),
               episodes=args.episodes, eval_seed=args.seed,
               train_seed=cfg["seed"], per_task=per,
               # WHICH GOAL EACH CELL SAW. A row 3 number and a row 4 number look
               # identical once they are in a table; this is what tells them apart.
               goal_source=src, goal_step=args.goal_step, success_once=m,
               # THE ENV EACH CELL WAS SCORED IN. A wall/gated cell and a plain
               # one are different numbers (mt4_ramp_s0 x n800 0.305 vs 0.102).
               env_patches=patches, only=only, psi_bank=pb_paths or None,
               per_env={t: LAST_DIAG.get(t) for t in only})
    path = args.out or f"{args.run}/eval_student.json"
    json.dump(out, open(path, "w"), indent=2)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
