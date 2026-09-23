"""ONE student, MANY teachers: mixed-task DAgger for the grasp-and-place family.

THE HYPOTHESIS UNDER TEST. At the executor level the input is the object's SE(3)
and the goal's SE(3), and the behaviour is always the same shape: approach,
grasp, transport, release. If that is true, PickCube / StackCube /
PegInsertionSide are one task and one student should serve all of them.

WHY THIS IS NOT THE SPEC'S STAGE 1, AND IS CHEAPER.
`<private-repo>/docs/SPEC_PRETRAINED_EXECUTOR_20260816.md` §4 trains ONE NEW mixed-task teacher
by RL in a new `fam_env` -- multi-task PPO, a new asset pool, and the biggest
unknown in the plan. This file keeps the EXISTING per-task teachers and distils
them into one student. DAgger is supervised and each teacher only ever labels its
own scenes, so none of the multi-task RL risk applies. `fam_env.py` is not needed.

THE THREE ARMS, PRE-REGISTERED
──────────────────────────────
    M0  --lang none     obj+goal SE(3) only. The hypothesis as stated.
    M1  --lang onehot   + a K-way task indicator. The CONTROL for M2.
    M2  --lang t5       + t5-small instruction embedding, the same frozen encoder
                        TraceGen conditions on, looked up from assets/instr_t5.npz.

⚠️ M2 is not more INFORMATIVE than M1. The instruction set is fifteen fixed
strings, so the channel carries at most log2(n_tasks) bits either way. What M2
buys is extensibility (a new task needs no new output slot), and that claim
cannot be tested at n=2. Any statement that "language helps" must be M2 vs M1,
never M2 vs M0.

⚠️ THE LEAKAGE CHECK IS MANDATORY. A perfectly separable task channel invites the
student to learn "task id -> behaviour" and ignore the goal entirely -- the exact
failure that mislabelled this project's no-goal ablation by 5.5x. Every arm
therefore has a `--no-trace` twin that zeroes all 13 goal-carrying dimensions via
`goal_slices()`. If a no-goal twin still scores well, the goal channel is
decoration and the planner has no value.

MECHANICS. K environments, `num_envs // K` each, stepped in lockstep. Each task's
own frozen teacher labels its own slice; the slices are concatenated into one
batch and one optimiser updates one student. Per-task success is logged
separately, because a mixed mean can hide one task at zero.

    $PM -m msppo.multi_distill --tasks pickcube,liftpeg \
        --teachers fam_pc_native,lp_paired_s0 --tag mt_m0_s0 --lang none
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from msppo.kp_teacher import TASKS, build_agent
from msppo.student import goal_extra_slices, StudentTransformer, goal_slices
from msppo.peg_relbank import load as load_relbank
from msppo.multi_eval import env_patches
from msppo.task_student_env import make_task_student_env
from msppo.teacher_perc import teacher_obs_from_perception

T5_NPZ = "assets/instr_t5.npz"


def load_teacher(run, kpenv, qdim):
    """Rebuild the head from the teacher's OWN run.json, never from flags."""
    cfg = json.load(open(f"{run}/run.json"))
    obs_dim = kpenv.single_observation_space.shape[0]
    act_dim = kpenv.single_action_space.shape[0]

    class _A:
        token_dim = cfg.get("token_dim", 128)
        hidden = cfg.get("hidden", "1024,1024,512,512")
        pn_hidden = cfg.get("pn_hidden", "128,128")

    agent, _ = build_agent(cfg["head"], kpenv, obs_dim, act_dim,
                           cfg.get("num_kp", 64),
                           bool(cfg.get("last_action", 0)), qdim, _A)
    # `obs_dim` is written by `kp_teacher.py` but NOT by the older per-task
    # drivers (`peg_kp_ppo.py`, `train_rl.py`), so a teacher that predates the
    # shared registry has no such field -- `cfg["obs_dim"]` raised KeyError and
    # killed all three arms of a mixed run in under a second.
    #
    # The checkpoint is the better guard anyway: it fails on a width mismatch
    # whether or not run.json recorded one. The run.json check stays as the
    # EARLIER, more legible error when the field is there.
    want = cfg.get("obs_dim")
    if want is not None and obs_dim != want:
        raise SystemExit(
            f"{run} trained on obs_dim {want} but this env builds {obs_dim}; "
            f"the student env must match the teacher's num_kp / ap2ap_fields / "
            f"last_action")
    try:
        agent.load_state_dict(torch.load(f"{run}/agent.pt"))
    except RuntimeError as e:
        raise SystemExit(
            f"{run}: checkpoint does not fit an env of obs_dim {obs_dim} "
            f"(head={cfg.get('head')}, num_kp={cfg.get('num_kp')}, "
            f"last_action={cfg.get('last_action')}, ap2ap={cfg.get('ap2ap_fields')})"
            f"\n{e}")
    agent.eval()
    for p in agent.parameters():
        p.requires_grad_(False)
    return agent, cfg


def lang_vectors(mode, tasks, device, gen):
    """[K, lang_dim] one row per task, or None.

    t5 mode keeps all THREE instructions per task and samples one per episode
    batch, matching how TraceGen trains (`msgen/tasks.py:4`: "TraceGen samples one
    at random per training sample"). Pinning a single string would make the
    channel a one-hot with 512 wasted dimensions.
    """
    if mode == "none":
        return None, 0
    if mode == "onehot":
        return torch.eye(len(tasks), device=device), len(tasks)
    z = np.load(T5_NPZ)
    # The npz predates the shared task registry and keys PegInsertionSide as
    # `peg`, while `kp_teacher.TASKS` calls it `peginsert`. Without this alias a
    # four-task t5 run dies at once with "has no embedding for ['peginsert']" --
    # the embedding is there, only the name is not. Alias rather than rewrite the
    # asset, so every earlier run that read `peg` still reads the same vectors.
    ALIAS = {"peginsert": "peg"}
    key = lambda t: ALIAS.get(t, t)
    missing = [t for t in tasks if key(t) not in z]
    if missing:
        raise SystemExit(f"{T5_NPZ} has no embedding for {missing}; run "
                         f"$PG -m experiments.20260817_multitask.t5_instr")
    return ([torch.as_tensor(z[key(t)], device=device) for t in tasks],
            int(z[key(tasks[0])].shape[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="comma-separated")
    ap.add_argument("--teachers", required=True, help="comma-separated run tags")
    ap.add_argument("--init-from", default=None,
                    help="warm-start the STUDENT from another student.pt (strict). "
                         "The reference 4-task student already solves the tasks; DAgger "
                         "then only has to adapt to the new teachers' labels.")
    ap.add_argument("--teacher-perc", type=int, default=0,
                    help="label from the student's perceived observation instead of sim truth; "
                         "required for noise-trained teachers, else the search behaviour is averaged away")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lang", default="none", choices=["none", "onehot", "t5"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-envs", type=int, default=256, help="TOTAL, split evenly")
    ap.add_argument("--iterations", type=int, default=40000)
    ap.add_argument("--num-kp", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--w-future", type=float, default=0.1)
    ap.add_argument("--kp-mask-ratio", type=int, default=2)
    ap.add_argument("--kp-noise-m", type=float, default=0.005)
    ap.add_argument("--kp-bias-m", type=float, default=0.0)
    ap.add_argument("--point-obs", action="store_true")
    ap.add_argument("--goal-err-rel", default=None,
                    help="TRAIN on a noisy goal, per task: task=path.npz,... "
                         "Each npz is that task's OWN measured planner error "
                         "(`msppo.peg_relbank export`), decomposed in the scene's "
                         "object->goal frame and drawn once per episode. Tasks "
                         "left out keep the true goal, and run.json records "
                         "which was which.")
    ap.add_argument("--goal-err-scale", default=None,
                    help="'uniform' draws severity a~U(0,1) per episode (a CURRICULUM: "
                         "halves the mean error vs deployment); 'bern<p>' draws "
                         "a in {0,1} with P(1)=p, i.e. the EXACT deployed error "
                         "distribution on noisy episodes plus clean ones; a float "
                         "pins it (0.5 = half the measured error); the default "
                         "applies it unscaled.")
    ap.add_argument("--goal-rel-se3", action="store_true",
                    help="L2b: the 7 goal slots carry the OBJECT-FRAME relative "
                         "transform (R_obj^T(t_g-t_o), q_obj^-1 (x) q_g) instead "
                         "of the goal's absolute world pose. Layout, token map, "
                         "parameter count and goal_slices() are unchanged, so the "
                         "arm isolates the representation. Both quantities are "
                         "BILINEAR in the absolute pair, which the affine "
                         "tokenizer cannot form -- this is not a re-parameterisation.")
    ap.add_argument("--goal-sig", action="store_true",
                    help="give the student the RELIABILITY of the goal it is "
                         "being shown: [rmse (m), sigma2/sigma1] of the solve "
                         "that produced it, read per episode from the relbank "
                         "(needs a *_sig.npz bank). Measured 2026-08-23: rmse "
                         "ranks PegInsert position error 1.89-2.71x worst-vs-"
                         "best quartile (2.32-2.81x inside a fixed n_pts "
                         "stratum, so not a point-count proxy); s21 ranks "
                         "PickCube/StackCube rotation error 2.3-4.2x. Both are "
                         "computed from the planner's own output, never from "
                         "simulator truth, so they exist on a real robot.")
    ap.add_argument("--goal-form", default="mean", choices=["mean", "sd", "k", "ell"],
                    help="K-sample goal form (spec 2026-08-25). 'mean' = the legacy "
                         "layout (goal_pose holds the K-mean when the relbank has K "
                         "columns); 'sd' adds the K samples' diag position sd (3); "
                         "'k' adds the K sample poses as order-free tokens (7K). "
                         "Non-mean forms need a K relbank (`peg_relbank export-k`) "
                         "for EVERY task in --goal-err-rel.")
    ap.add_argument("--goal-k", type=int, default=4)
    ap.add_argument("--psi-token", action="store_true", help="psi as its own token (Linear 4->d) instead of inside the objgoal token")
    ap.add_argument("--psi", action="store_true",
                    help="psi block (k1 approach dir + h carry height) in the student obs, taken from the frame-patched "
                         "teacher envs (msppo.frame_patches); teachers must be framework teachers (runs_rl/*_v9_*)")
    ap.add_argument("--goal-delta-train", default=None,
                    help="PIPELINE-IDENTITY TRAINING: train through the same "
                         "code path deployment uses -- compose the planner "
                         "transform with the PERCEIVED points -- instead of "
                         "applying a relbank error to the true goal pose. "
                         "Same syntax as multi_eval --goal-delta "
                         "(task=a.npz+b.npz+...). Mutually exclusive with "
                         "--goal-err-rel. Use a bank seed the eval never sees.")
    ap.add_argument("--goal-err-warmup", type=float, default=0.0,
                    help="leading fraction of training with NO goal error "
                         "(a_max=0), so the task is learned before the error")
    ap.add_argument("--goal-err-curriculum", type=float, default=0.0,
                    help="fraction of training over which a_max ramps 0->1 "
                         "after the warmup; 0 = step straight to full")
    ap.add_argument("--rel-token", action="store_true",
                    help="L1: give the `rel` block its own token. Under "
                         "--no-scene it is otherwise read by NOTHING (measured "
                         "2026-08-22: d(action)/d(rel) = 0.000e+00). Adds no "
                         "information -- the observation is byte-identical -- so "
                         "it is the control that L2 needs.")
    ap.add_argument("--no-scene", action="store_true",
                    help="drop the fixture/scene channel for EVERY task, not "
                         "just the ones that have no fixture. Tests whether the "
                         "executor can work with no reference object at all.")
    ap.add_argument("--no-trace", action="store_true",
                    help="THE LEAKAGE CHECK: zero all 13 goal-carrying dims via "
                         "goal_slices(). A mixed student that still works here "
                         "has learned the task from its identity channel, not "
                         "from the goal.")
    # CAPACITY. The default 128-wide, 4-layer trunk is 0.8M parameters and was
    # sized on ONE task. Four tasks share it, and the interference shows up
    # entirely on the hardest one: PegInsert reads 0.633 in the four-task mix
    # against 0.734 for its own single-task student, while the other three lose
    # nothing. More width is the first thing to try against that.
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()

    tasks = args.tasks.split(",")
    teachers = args.teachers.split(",")
    assert len(tasks) == len(teachers), "one teacher per task"
    K = len(tasks)
    dev = "cuda"
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    out = f"runs_rl/{args.tag}"
    os.makedirs(out, exist_ok=True)

    qd = {TASKS[t]["qdim"] for t in tasks}
    ad = {8}
    if len(qd) != 1:
        raise SystemExit(f"tasks disagree on proprio width {qd}; a single student "
                         f"cannot serve both (PushT's panda_stick is 7, not 9)")
    qdim = qd.pop()

    # Both tasks in the first experiment have NO fixture, so the scene channel is
    # absent for both and the layout is consistent by construction. Adding stack
    # or peg later needs the scene slot PRESENT and ZEROED for the fixture-less
    # tasks (spec §2.1) -- masking it only at eval would be train/deploy skew.
    # PER-TASK scene mode. A single student needs ONE observation width, so when
    # the mix contains both a fixture task (peg's hole rim, StackCube's cubeB) and
    # a fixture-less one (PickCube, LiftPegUpright), the block stays in the layout
    # for everybody and is ZEROED where no fixture exists -- at TRAINING time, not
    # masked at eval (spec §2.1; the `arm_t.py:56-59` no-trace lesson).
    from msppo.task_student_env import TaskStudentEnv
    has_fix = {t: TaskStudentEnv.ACTORS[t][1] is not None for t in tasks}
    if args.no_scene:
        # NO REFERENCE OBJECT AT ALL. Not "zero it for the tasks that lack one" --
        # drop it everywhere, including peg's hole rim and StackCube's cubeB.
        #
        # Why this is worth measuring rather than assumed dead: single-task
        # students died without it, 0.000 in 3/3 seeds on peg and 0.720 -> 0.000
        # on StackCube. A MIXED student has never been asked. It shares one trunk
        # with two tasks that have no fixture and still reach 0.97-0.98, so there
        # is a mechanism by which it could carry over.
        #
        # And if it works, the whole benchmark changes character: peg's and
        # StackCube's goals are recoverable from their fixtures (std 12 mm and
        # exactly 0), which is why neither can currently demonstrate planner
        # value. Remove the fixture and all four tasks become clean tests.
        has_fix = {t: False for t in tasks}
    any_fix = any(has_fix.values())
    scene_mode = {t: (True if has_fix[t] else ("zero" if any_fix else False))
                  for t in tasks}
    scene_kp = any_fix          # what the STUDENT declares
    if any_fix and not all(has_fix.values()):
        print(f"[mix] mixed fixtures -> scene slot kept, zeroed for tasks without a fixture: "
              f"{ {t: ('real' if has_fix[t] else 'ZERO') for t in tasks} }")

    rel_paths = {}
    if args.goal_err_rel:
        for part in args.goal_err_rel.split(","):
            if "=" not in part:
                raise SystemExit(f"--goal-err-rel wants task=path, got {part!r}")
            t_, p_ = part.split("=", 1)
            rel_paths[t_.strip()] = p_.strip()
        unknown = set(rel_paths) - set(tasks)
        if unknown:
            raise SystemExit(f"--goal-err-rel names tasks not in this mix: {unknown}")
        print(f"[goal-noise] scale={args.goal_err_scale or 1.0} "
              f"{ {t: rel_paths.get(t, 'TRUE GOAL') for t in tasks} }", flush=True)

    patches = env_patches()
    print(f"[env] wall={patches['wall']} scene_gate={patches['scene_gate']} cam={patches['camera']}")
    _gw, _gc = args.goal_err_warmup, args.goal_err_curriculum
    if args.goal_delta_train and args.goal_err_rel:
        raise SystemExit("--goal-delta-train and --goal-err-rel are two ways to "
                         "do the same thing; pick one")
    per = args.num_envs // K
    envs, tchs = [], []
    if args.psi:
        if args.teacher_perc:
            raise SystemExit("--psi with --teacher-perc is not wired (teacher_obs_from_perception carries no psi block)")
        from msppo.frame_patches import apply as _fp
        print(f"[psi] frame patches applied for {_fp(tasks)}", flush=True)
    for t, tt in zip(tasks, teachers):
        tcfg = json.load(open(f"runs_rl/{tt}/run.json"))
        e = make_task_student_env(
            t, num_envs=per, num_kp=args.num_kp, seed=args.seed,
            max_episode_steps=tcfg["max_episode_steps"], control=tcfg["control"],
            ap2ap_fields=bool(tcfg.get("ap2ap_fields", 1)),
            last_action=bool(tcfg.get("last_action", 0)),
            robot=tcfg.get("robot"), w_kp=0.0,
            # the v3 teachers carry a 4-dim sig block and a StackCube contact
            # block; the kp env must build the SAME layout or load_teacher's
            # obs_dim assert fires (450 vs 446)
            **({"sig_obs": True} if tcfg.get("sig_obs") else {}),
            **({"contact_obs": True} if tcfg.get("contact_obs") else {}),
            kp_mask_ratio=args.kp_mask_ratio, kp_noise_m=args.kp_noise_m,
            kp_bias_m=args.kp_bias_m, kp_bias_rand=True,
            point_obs=args.point_obs, scene_kp=scene_mode[t], tcp_obs=True,
            perception=True, psi=args.psi,
            goal_error_rel=(load_relbank(rel_paths[t]) if t in rel_paths else None),
            goal_err_scale=args.goal_err_scale,
            goal_rel_se3=args.goal_rel_se3, goal_sig=args.goal_sig,
            goal_form=args.goal_form, goal_k=args.goal_k)
        envs.append(e)
        tchs.append(load_teacher(f"runs_rl/{tt}", e.kp, qdim)[0])

    # A relbank without the signals makes _sig_row fall back to (0.0, 1.0) --
    # "perfect goal" -- for every episode, so the channel would be a constant and
    # the arm would look like a null for a plumbing reason. Fail loudly instead.
    if args.goal_sig:
        # A K>1 relbank needs no `sig` columns: the reliability is the SAMPLING
        # CLOUD's own spread, computed per step in `_set_cloud_sig` from the K
        # samples themselves. That is the deployable quantity (a planner emits K
        # samples; their spread needs no truth) and it also reaches the teacher
        # through `perc_sig_row`, so both sides agree. Only a K=1 bank with no
        # rmse/s21 leaves the channel constant, which is what this guards.
        bad = [t for t, e in zip(tasks, envs)
               if e.goal_error_rel is not None
               and e.goal_error_rel.get("sig") is None
               and int(e.goal_error_rel.get("K", 1)) <= 1]
        if bad:
            raise SystemExit(
                f"--goal-sig but the relbank for {bad} carries no rmse/s21. Use the "
                f"*_relbank_visobj_sig.npz banks (<private-repo>/experiments/20260823_relsig/b1_addsig.py).")
        if not rel_paths:
            print("[goal-sig] no injection: every episode gets the true-goal "
                  "reliability (0.0, 1.0), i.e. a constant channel")

    # A K-form student trained on a K=1 bank would see K copies of one goal
    # (sd = 0, identical tokens) on every episode and learn nothing about the
    # cloud -- a null for a plumbing reason. Every task must carry a K bank.
    if args.goal_form != "mean":
        bad = [t for t, e in zip(tasks, envs)
               if e.goal_error_rel is None or e.goal_error_rel.get("K", 1) != args.goal_k]
        if bad:
            raise SystemExit(f"--goal-form {args.goal_form} needs a K={args.goal_k} "
                             f"relbank (peg_relbank export-k) for {bad}")

    lang, lang_dim = lang_vectors(args.lang, tasks, dev, gen)
    act_dim = envs[0].single_action_space.shape[0]
    base_dim = envs[0].single_observation_space.shape[0]
    for e in envs[1:]:
        if e.single_observation_space.shape[0] != base_dim:
            raise SystemExit("tasks build different student observation widths")

    student = StudentTransformer(num_kp=args.num_kp, act_dim=act_dim,
                                 pose_obs=not args.point_obs, scene_kp=scene_kp,
                                 tcp=True, qdim=qdim, lang_dim=lang_dim,
                                 d=args.d_model, layers=args.layers,
                                 heads=args.heads, prev_action=True,
                                 rel_token=args.rel_token,
                                 goal_sig=args.goal_sig,
                                 goal_form=args.goal_form,
                                 goal_k=args.goal_k, psi=args.psi, psi_token=args.psi_token).to(dev)
    assert student.obs_dim == base_dim + lang_dim, (
        f"student {student.obs_dim} != env {base_dim} + lang {lang_dim}")
    goal_sl, goal_rel_sl = goal_slices(student)
    extra_sls = goal_extra_slices(student)   # K-sample slots, zeroed with the goal
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)

    logf = open(f"{out}/train.log", "w", buffering=1)

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")

    from msppo import wb
    wbrun = wb.maybe_init(args.tag, dict(vars(args), trainer="multi_distill"),
                          group="mt%d" % len(tasks))

    log(f"tag={args.tag} tasks={tasks} teachers={teachers} lang={args.lang} "
        f"lang_dim={lang_dim} student_obs={student.obs_dim} scene_kp={scene_kp} "
        f"qdim={qdim} envs={per}x{K} iters={args.iterations} "
        f"point_obs={args.point_obs} no_trace={args.no_trace} seed={args.seed} "
        f"params={sum(p.numel() for p in student.parameters())/1e6:.3f}M")

    lo = torch.as_tensor(envs[0].single_action_space.low, device=dev)
    hi = torch.as_tensor(envs[0].single_action_space.high, device=dev)

    obs = []
    for e in envs:
        o, _ = e.kp.reset(seed=args.seed)
        e.perc_reset(o)
        e.reset_masks()
        obs.append(o)
    # One language row per ENV, resampled per episode: with three strings per task
    # a fixed choice would collapse the channel to a one-hot.
    lrow = [None] * K
    if lang_dim and args.lang == "t5":
        for k in range(K):
            i = torch.randint(0, lang[k].shape[0], (per,), generator=gen, device=dev)
            lrow[k] = lang[k][i]
    elif lang_dim:
        for k in range(K):
            lrow[k] = lang[k].expand(per, -1)

    B, N = args.batch, per * K
    obs_b = torch.zeros((B, N, student.obs_dim), device=dev)
    act_b = torch.zeros((B, N, act_dim), device=dev)
    fq_b = torch.zeros((B, N, qdim), device=dev)
    fv_b = torch.zeros((B, N, qdim), device=dev)
    ok_b = torch.zeros((B, N), dtype=torch.bool, device=dev)
    succ = [deque(maxlen=200) for _ in range(K)]
    if args.init_from:
        sd = torch.load(args.init_from, map_location=dev)
        own = student.state_dict()
        # A source that predates a new INPUT TOKEN (e.g. goal_sig) is missing
        # exactly that token's tensors. Those are freshly initialised; every
        # other tensor must still match, and any shape mismatch is a real error,
        # never something to paper over.
        miss = [k for k in own if k not in sd]
        bad = [k for k in sd if k in own and own[k].shape != sd[k].shape]
        if bad:
            raise SystemExit(f"--init-from shape mismatch: {bad}")
        extra = [k for k in sd if k not in own]
        if extra:
            raise SystemExit(f"--init-from has tensors this student lacks: {extra}")
        student.load_state_dict(sd, strict=not miss)
        log(f"[init-from] loaded {args.init_from} "
            f"({len(sd)} tensors" + (f", {len(miss)} newly initialised: "
            f"{sorted(miss)}" if miss else ", strict") + ")")
    hist, best, t0 = [], -1.0, time.time()
    bc_l = fut_l = 0.0

    for e in envs:
        e.goal_err_warmup, e.goal_err_curriculum = _gw, _gc
    if args.goal_delta_train:
        import numpy as _np
        paths = dict(p.split("=", 1) for p in args.goal_delta_train.split(","))
        for t, e in zip(tasks, envs):
            if t not in paths:
                raise SystemExit(f"--goal-delta-train has no bank for {t!r}")
            pool = []
            for one in paths[t].split("+"):
                z = _np.load(one.strip())
                if int(z["seed"]) == 999:
                    raise SystemExit(
                        f"{one} is seed 999 -- that is the eval seed; training "
                        f"on it would be training on the test set")
                st = z["dR"].shape[1] - 1
                # NO num_envs check here, unlike multi_eval._load_one: the row is
                # drawn at random per episode, so bank scenes need not correspond
                # to training scenes. The ROTATION transfers verbatim, but dt
                # absorbs the rotation about the WORLD origin for the bank scene
                # (review 20260828 E3): c_src ships along so the env can re-anchor
                # dt about the training scene's own centroid.
                if "c_src" not in z.files:
                    raise SystemExit(f"{one} has no c_src; cannot re-anchor "
                                     "(E3) -- rebuild the bank")
                pool.append((torch.as_tensor(z["dR"][:, st], device=dev, dtype=torch.float32),
                             torch.as_tensor(z["dt"][:, st], device=dev, dtype=torch.float32),
                             torch.as_tensor(z["c_src"], device=dev, dtype=torch.float32)))
            e.goal_delta_pool = pool
            e.resample_goal_delta()
            log(f"[goal-delta-train] {t}: K={len(pool)} bank rows={pool[0][0].shape[0]} "
                f"<- {paths[t]}")
    if _gw or _gc:
        log(f"[goal-err] warmup={_gw} curriculum={_gc} "
            f"(a_max reaches 1.0 at iter {int((_gw + _gc) * args.iterations)})")

    for it in range(1, args.iterations + 1):
        for e in envs:
            e.set_noise_progress(it / args.iterations)
        so, ao, fq, fv, ok = [], [], [], [], []
        for k, (e, tc) in enumerate(zip(envs, tchs)):
            s = e.student_observation_perc(obs[k]).float()
            if lang_dim:
                s = torch.cat([s, lrow[k]], dim=-1)
            if args.no_trace:
                s[:, goal_sl] = 0.0
                if goal_rel_sl.stop > goal_rel_sl.start:
                    s[:, goal_rel_sl] = 0.0
                for e in extra_sls:
                    s[:, e] = 0.0
            with torch.no_grad():
                if args.teacher_perc:
                    # the teacher labels from the STUDENT's degraded view, not
                    # from simulator truth -- see msppo/teacher_perc.py
                    t_obs = teacher_obs_from_perception(e, tasks[k], s)
                else:
                    t_obs = e.kp.teacher_observation()
                a_t = tc.actor_mean(tc.encode(t_obs.float()))
                a_s = student.actor_mean(s)
            nobs, _, term, trunc, info = e.kp.step(a_s.clamp(lo, hi))
            e.set_prev_action(a_s)
            done = (term | trunc).bool()
            so.append(s); ao.append(a_t)
            fq.append(nobs["agent"]["qpos"][:, :qdim].float())
            fv.append(nobs["agent"]["qvel"][:, :qdim].float())
            ok.append(~done)
            if bool(done.any()):
                sv = info["success"]
                fi = info.get("final_info")
                if fi is not None and "success" in fi:
                    sv = fi["success"]
                succ[k].extend(sv[done].float().tolist())
                e.reset_masks(done)
                e.perc_reset(nobs, idx=done)
                if lang_dim and args.lang == "t5":
                    i = torch.randint(0, lang[k].shape[0], (per,),
                                      generator=gen, device=dev)
                    lrow[k] = torch.where(done[:, None], lang[k][i], lrow[k])
            obs[k] = nobs

        s_i = (it - 1) % B
        obs_b[s_i] = torch.cat(so); act_b[s_i] = torch.cat(ao)
        fq_b[s_i] = torch.cat(fq); fv_b[s_i] = torch.cat(fv)
        ok_b[s_i] = torch.cat(ok)

        if it % B == 0:
            fo = obs_b.reshape(-1, student.obs_dim)
            fa = act_b.reshape(-1, act_dim)
            fqq, fvv = fq_b.reshape(-1, qdim), fv_b.reshape(-1, qdim)
            fok = ok_b.reshape(-1)
            idx = np.arange(fo.shape[0])
            mb = fo.shape[0] // args.minibatches
            bc_l = fut_l = 0.0
            for _ in range(args.epochs):
                np.random.shuffle(idx)
                for c in range(0, fo.shape[0], mb):
                    j = torch.as_tensor(idx[c:c + mb], device=dev)
                    mean, fut = student(fo[j])
                    loss = F.l1_loss(mean, fa[j])
                    bc_l += float(loss)
                    v = fok[j]
                    if bool(v.any()):
                        aux = (F.l1_loss(fut["qpos"][v], fqq[j][v])
                               + F.l1_loss(fut["qvel"][v], fvv[j][v]))
                        fut_l += float(aux)
                        loss = loss + args.w_future * aux
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

        if it % args.log_every == 0:
            per_task = [float(np.mean(s)) if s else float("nan") for s in succ]
            m = float(np.nanmean(per_task))
            n = args.epochs * args.minibatches
            log(f"it {it}/{args.iterations} success {m:.3f} "
                + " ".join(f"{t}={v:.3f}" for t, v in zip(tasks, per_task))
                + f" bc {bc_l/max(n,1):.4f} fut {fut_l/max(n,1):.4f} "
                f"{it*N/(time.time()-t0):.0f} sps")
            if m == m and m > best:
                best = m
                torch.save(student.state_dict(), f"{out}/student_best.pt")
            hist.append(dict(it=it, success=m,
                             per_task=dict(zip(tasks, per_task)),
                             bc=bc_l / max(n, 1)))
            wb.log(wbrun, dict(success=m, best=best,
                               bc=bc_l / max(n, 1), fut=fut_l / max(n, 1),
                               **{f"task/{t}": v for t, v in zip(tasks, per_task)}),
                   step=it)

    torch.save(student.state_dict(), f"{out}/student.pt")
    wandb_id = wb.finish(wbrun)
    json.dump(dict(tag=args.tag, arm="multitask_student", tasks=tasks,
                   teachers=teachers, lang=args.lang, lang_dim=lang_dim,
                   obs="student" if args.point_obs else "student_recon",
                   pose_obs=not args.point_obs, point_obs=args.point_obs,
                   n_pts=args.num_kp, num_kp=args.num_kp, qdim=qdim,
                   scene_kp=scene_kp, tcp_obs=True, perception=True,
                   no_trace=args.no_trace, no_scene=args.no_scene,
                   goal_rel_se3=args.goal_rel_se3, rel_token=args.rel_token,
                   goal_sig=args.goal_sig,
                   goal_form=args.goal_form, goal_k=args.goal_k, psi=args.psi, psi_token=args.psi_token,
                   env_patches=patches,
                   goal_err_rel=args.goal_err_rel,
                   goal_err_scale=args.goal_err_scale,
                   goal_delta_train=args.goal_delta_train,
                   goal_err_warmup=args.goal_err_warmup,
                   goal_err_curriculum=args.goal_err_curriculum, seed=args.seed,
                   d_model=args.d_model, layers=args.layers, heads=args.heads,
                   params=sum(p.numel() for p in student.parameters()),
                   kp_mask_ratio=args.kp_mask_ratio, kp_noise_m=args.kp_noise_m,
                   kp_bias_m=args.kp_bias_m, num_envs=args.num_envs,
                   envs_per_task=per, iterations=args.iterations, lr=args.lr,
                   best_success=best, history=hist, wandb_id=wandb_id),
              open(f"{out}/run.json", "w"), indent=2)
    log(f"saved -> {out}  best {best:.3f}")
    for e in envs:
        e.close()


if __name__ == "__main__":
    main()
