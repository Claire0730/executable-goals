"""Stage 1 driver for the tasks that had no teacher: LiftPegUpright and PushT.

One driver, two tasks, four heads. Everything downstream of the env is already
task-agnostic (`ppo.py`, `ppe.py`, `ppe_strict.py` carry no task name), so this
file is only the wiring plus the two things that genuinely differ per task: the
proprio width and the default horizon.

    $PM -m msppo.kp_teacher --task liftpeg --tag lp_paired_s0 --seed 0 --head paired
    $PM -m msppo.kp_teacher --task pickcube --tag pc_paired_s0 --seed 0 --head paired

WHY THE DEFAULTS ARE WHAT THEY ARE

  * `--total-steps` defaults per task, not globally. PushT has an official 25M
    PPO run; LiftPegUpright has released RL ROLLOUTS but NO published
    hyperparameters, so 25M is a
    budget guess anchored on PokeCube, which converges at 5M on the same 50-step
    horizon. Expect to spend one tuning pass on liftpeg and none on pusht.
  * `--num-steps` matches the task horizon (50 / 100), as `ppo.py`'s comment
    requires: one whole stock episode per rollout segment.
  * `--target-kl` defaults to **0.1**, not peg's 0.0. `ppo.py:40-45` records that
    the reference implementation's PPO defaults to 0.1 and never overrides it, and
    that believing an earlier note to the contrary is what sent the peg runs out
    with KL 3-6x the reference threshold. New tasks start from the reference
    setting; pass `--target-kl 0` to reproduce the peg line's setting instead.
  * `--w-kp` defaults to 0, i.e. the task's OFFICIAL reward untouched. Anything
    else is no longer comparable to the published baseline.

WHAT IS NOT HERE, DELIBERATELY: no curriculum and no domain randomisation.
Dex4D's three stages exist for 3,200 objects; these tasks have one each. Adding
either would make a head-to-head difference impossible to attribute.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from msppo.ppe import (FlatActorCritic, PairedKeypointActorCritic,
                       PlainActorCritic)
from msppo.ppo import PPOConfig, train

# task -> (make_env, proprio qpos/qvel width, default horizon, default steps)
TASKS = {
    "liftpeg": dict(env_id="LiftPegUpright-v1", qdim=9, horizon=50,
                    steps=25_000_000, robot="panda"),
    "plugcharger": dict(env_id="PlugCharger-v1", qdim=9, horizon=100,
                        steps=25_000_000, robot="panda_wristcam"),
    "pusht": dict(env_id="PushT-v1", qdim=7, horizon=100,
                  steps=25_000_000, robot="panda_stick"),
    # StackCube already has results through the OTHER stack (`kp_env.py` +
    # `train_rl.py`), but that one has no camera. It is registered here so it can
    # reach the shared perception path and the --goal-delta interface; the
    # existing numbers come from the other line and are untouched.
    #
    # `panda_wristcam`, NOT `panda`, and this one flag is worth 0.61 of success.
    # StackCube-v1 declares `robot_uids="panda_wristcam"` (stack_cube.py:39) and
    # its init qpos puts joint 7 at -pi/4; forcing `panda` starts the wrist at
    # +pi/4, a quarter turn away, with every other dim of the 48-D state
    # identical. Measured on one frozen checkpoint
    # (`diag_kpenc_priv_s0`) through THIS env, N=256, seed 999:
    #     robot="panda"           0.289
    #     robot="panda_wristcam"  0.898      (its old-harness score: 0.895)
    # so the shared env is otherwise equivalent to `kp_env.py` and the whole
    # StackCube collapse on this line (sc_paired 0.008, sc_fix 0.027) sits on
    # this flag. It is also what ManiSkill's official PPO baseline uses, which
    # keeps column (1) comparable.
    #
    # peg keeps `panda` deliberately (peg_kp_env.py:340) and reaches 0.98 there,
    # so a non-default robot is not fatal per se -- StackCube is the task that
    # cares.
    "stack": dict(env_id="StackCube-v1", qdim=9, horizon=50,
                  steps=25_000_000, robot="panda_wristcam"),
    # See <private-repo>/docs/SPEC_PRETRAINED_EXECUTOR_20260816.md: the cheapest place
    # to tune the unified family reward's new free parameters, and the only task
    # in the benchmark with a verified G1 pass (hidden goal marker, independently
    # randomised goal xy), so a number here is attributable to the reward.
    "pickcube": dict(env_id="PickCube-v1", qdim=9, horizon=50,
                     steps=10_000_000, robot="panda"),
    # PegInsertionSide via the thin shared-contract subclass in
    # msppo/peginsert_kp_env.py. `msppo/peg_kp_env.py` is untouched, so every
    # existing peg number stands. Official PPO is 250M steps here -- 10-50x the
    # others -- but for a mixed STUDENT we distil the frozen 0.98 reference
    # teacher, so that budget is not on this path.
    "peginsert": dict(env_id="PegInsertionSide-v1", qdim=9, horizon=100,
                      steps=30_000_000, robot="panda"),
    # Non-prehensile family member: official state-PPO exists (50M/4096); our keypoint version converges
    # far earlier. goal_radius 0.1 makes it the loosest task in the set.
    "pushcube": dict(env_id="PushCube-v1", qdim=9, horizon=50,
                     steps=8_000_000, robot="panda"),
    # Zero-shot probe of the pick-and-place family: sphere into a shallow bin, xy tolerance 5 mm.
    "placesphere": dict(env_id="PlaceSphere-v1", qdim=9, horizon=50,
                        steps=8_000_000, robot="panda"),
}


def make_env(task, **kw):
    from msppo.patch_basepose import maybe_patch as _basepose   # MSPPO_BASE_POSE="x,y,z" (real lab geometry); no-op when unset
    _basepose()
    if task == "liftpeg":
        from msppo.liftpeg_kp_env import make_liftpeg_kp_env
        return make_liftpeg_kp_env(**kw)
    if task == "stack":
        from msppo.stack_kp_env import make_stack_kp_env
        return make_stack_kp_env(**kw)
    if task == "pickcube":
        from msppo.pickcube_kp_env import make_pickcube_kp_env
        return make_pickcube_kp_env(**kw)
    if task == "pushcube":
        from msppo.pushcube_kp_env import make_pushcube_kp_env
        return make_pushcube_kp_env(**kw)
    if task == "placesphere":
        from msppo.placesphere_kp_env import make_placesphere_kp_env
        return make_placesphere_kp_env(**kw)
    if task == "plugcharger":
        from msppo.plugcharger_kp_env import make_plugcharger_kp_env
        return make_plugcharger_kp_env(**kw)
    if task == "peginsert":
        from msppo.peginsert_kp_env import make_peginsert_kp_env
        return make_peginsert_kp_env(**kw)
    from msppo.pusht_kp_env import make_pusht_kp_env
    return make_pusht_kp_env(**kw)


def branch_slices(env, last_action: bool, qdim: int):
    """{name: slice} over the teacher observation, derived from the env.

    Generalises `peg_teacher_ppe.branch_slices`, which hardcodes qpos 0:9 /
    qvel 9:18 -- true for the Panda tasks and FALSE for PushT, whose panda_stick
    has 7 joints and no gripper. `StrictPairedActorCritic` asserts these cover
    every dim outside `kp_slice`, so a wrong width fails loudly here rather than
    silently mis-routing a branch.
    """
    act = env.single_action_space.shape[0]
    kp_start = env.kp_slice.start
    b = {"qpos": slice(0, qdim), "qvel": slice(qdim, 2 * qdim)}
    if last_action:
        a0 = kp_start - act
        b["action"] = slice(a0, a0 + act)
        b["state"] = slice(2 * qdim, a0)
    else:
        b["state"] = slice(2 * qdim, kp_start)
    return b


def build_agent(head, env, obs_dim, act_dim, num_kp, last_action, qdim, args):
    if head == "strict_paired":
        from msppo.ppe_strict import StrictPairedActorCritic
        br = branch_slices(env, last_action, qdim)
        agent = StrictPairedActorCritic(
            obs_dim, act_dim, env.kp_slice, num_kp, br,
            token_dim=args.token_dim,
            pn_hidden=tuple(int(x) for x in args.pn_hidden.split(",")),
            hidden=tuple(int(x) for x in args.hidden.split(","))).to("cuda")
        return agent, {k: [v.start, v.stop] for k, v in br.items()}
    Cls = {"plain": PlainActorCritic, "flat": FlatActorCritic,
           "paired": PairedKeypointActorCritic}[head]
    agent = Cls(obs_dim, act_dim, env.kp_slice, num_kp)
    if getattr(args, "priv_critic", 0) and getattr(env, "priv_slice", None) is not None:
        from msppo.ppe import PrivCritic
        agent = PrivCritic(agent, env.priv_slice)
    return agent.to("cuda"), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--head", default="paired",
                    choices=["plain", "flat", "paired", "strict_paired"])
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--num-steps", type=int, default=None)
    ap.add_argument("--max-episode-steps", type=int, default=None)
    ap.add_argument("--control", default="pd_joint_delta_pos")
    ap.add_argument("--reward", default="stock",
                    help="`family` = the unified pick-and-place reward "
                         "(msppo/fam_reward.py, spec §3). It DEMOTES the task's "
                         "own dense reward and success to eval-only, so a family "
                         "run and a native run can never be mixed by accident.")
    ap.add_argument("--fam-eps", type=float, default=None,
                    help="family reward placement threshold, metres. Dex4D uses "
                         "goal_obj_dist <= 0.05; this is a NEW FREE PARAMETER "
                         "and tuning it is what gate T-G1 is for.")
    ap.add_argument("--fam-weights", default=None,
                    help="w_reach,w_grasp,w_transport,w_settle,bonus")
    ap.add_argument("--w-kp", type=float, default=0.0,
                    help="0 = the official reward untouched")
    ap.add_argument("--num-kp", type=int, default=64)
    # TEACHER OBSERVATION NOISE. Magnitudes are the MEASURED
    # perception error of the pose the student actually consumes
    # (<private-repo>/experiments/20260824_objerr/o3_chamfer.py, gated Chamfer, N=256 seed 999):
    #   peginsert  translation 22.2 mm  rotation 13.1 mm of cloud displacement
    #   stack      translation 21.9 mm  rotation 11.2 mm
    # and for the goal, the planner error the student meets at deployment.
    # Rotation is given as a DISPLACEMENT because the objects are symmetric, so
    # an angle has no unique meaning while a cloud displacement does.
    ap.add_argument("--obj-noise-mm", type=float, default=0.0)
    ap.add_argument("--obj-rot-mm", type=float, default=0.0)
    ap.add_argument("--goal-noise-mm", type=float, default=0.0)
    ap.add_argument("--goal-rot-mm", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--gae-lambda", type=float, default=0.9)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--minibatches", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--target-kl", type=float, default=0.1)
    # TEACHER v2 (<private-repo>/docs/SPEC_TEACHER_V2_20260826.md).
    # --noise-v2 replaces the isotropic zero-mean v1 draw with the measured
    # anisotropic model (mu + per-axis sd in the approach frame, heavy tail) and
    # a per-step OU object error; --sig-obs adds the 4-dim noise-scale channel.
    ap.add_argument("--noise-v2", type=float, default=0.0,
                    help="goal severity scale; 0 disables. 1.0 = the measured model")
    ap.add_argument("--obj-v2", type=float, default=1.0,
                    help="object severity scale for --noise-v2")
    ap.add_argument("--sig-obs", type=int, default=0)
    ap.add_argument("--sig-mode", default="true", choices=("true", "proxy"),
                    help="true = arm T-a (the policy is told the exact severity, "
                         "an upper bound); proxy = arm T-b (a readout corrupted "
                         "to the MEASURED deployment proxy quality, deployable)")
    ap.add_argument("--noise-curriculum", type=float, default=0.0,
                    help="fraction of training over which the severity UPPER "
                         "BOUND rises a0 -> 1; 0 disables")
    ap.add_argument("--noise-a0", type=float, default=0.1)
    ap.add_argument("--noise-zmode", default="reflect", choices=("reflect","clamp"),
                    help="with --noise-zreflect: reflect dz to |dz|, or clamp it to >=0 (on the surface)")
    ap.add_argument("--noise-goal-rot", type=float, default=1.0,
                    help="scale on the goal ROTATION noise; 0 removes it")
    ap.add_argument("--noise-zreflect", type=int, default=0,
                    help="reflect the goal z error to |dz| (fixture-derived goals cannot sit below the surface)")
    ap.add_argument("--noise-tail", type=float, default=1.0,
                    help="multiplier on the measured heavy-tail rate; 0 removes the tail")
    ap.add_argument("--init-from", default=None,
                    help="warm-start: load this agent.pt (strict) before PPO; layout must match")
    ap.add_argument("--priv-critic", type=int, default=0,
                    help="asymmetric critic: true obj/goal positions in the obs, actor sees them zeroed")
    ap.add_argument("--shaping-true-goal", type=int, default=0,
                    help="w_kp shaping toward the TRUE goal keypoints (stack)")
    ap.add_argument("--contact-obs", type=int, default=0,
                    help="StackCube: add the cubeA-cubeB contact force to the observation")
    ap.add_argument("--bump-mode", default="delta", choices=("cum", "delta"))
    ap.add_argument("--w-bump", type=float, default=0.0,
                    help="StackCube only: penalty on cubeB displacement")
    ap.add_argument("--ap2ap-fields", type=int, default=1)
    ap.add_argument("--ignore-terminations", type=int, default=1)
    ap.add_argument("--last-action", type=int, default=0)
    ap.add_argument("--token-dim", type=int, default=128)
    ap.add_argument("--hidden", default="1024,1024,512,512")
    ap.add_argument("--pn-hidden", default="128,128")
    args = ap.parse_args()

    spec = TASKS[args.task]
    horizon = args.max_episode_steps or spec["horizon"]
    total = args.total_steps or spec["steps"]
    nsteps = args.num_steps or horizon

    out = f"runs_rl/{args.tag}"
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(args.seed)

    fam = None
    if args.reward == "family":
        from msppo.fam_reward import FamilyRewardCfg
        fam = FamilyRewardCfg()
        if args.fam_eps is not None:
            fam.eps = args.fam_eps
        if args.fam_weights:
            w = [float(x) for x in args.fam_weights.split(",")]
            (fam.w_reach, fam.w_grasp, fam.w_transport,
             fam.w_settle, fam.bonus) = w
    extra_kw = {"fam_cfg": fam} if args.reward == "family" else {}
    env = make_env(args.task, num_envs=args.num_envs, num_kp=args.num_kp,
                   reward=args.reward, w_kp=args.w_kp, seed=args.seed,
                   **extra_kw,
                   control=args.control, max_episode_steps=horizon,
                   ap2ap_fields=bool(args.ap2ap_fields),
                   ignore_terminations=bool(args.ignore_terminations),
                   robot=spec["robot"], last_action=bool(args.last_action),
                   **({"sig_obs": bool(args.sig_obs)} if args.sig_obs else {}),
                   **({"priv_obs": True} if args.priv_critic else {}),
                   **({"contact_obs": True} if (args.contact_obs and args.task == "stack") else {}),
                   **({"shaping_true_goal": True} if (args.shaping_true_goal and args.task == "stack") else {}),
                   **({"bump_mode": args.bump_mode} if args.w_bump and args.task == "stack" else {}),
                   **({"w_bump": args.w_bump}
                      if (args.w_bump and args.task == "stack") else {}))
    if args.w_bump and args.task != "stack":
        raise SystemExit("--w-bump is StackCube-only (it penalises cubeB displacement)")
    # OBSERVATION NOISE. Slices differ per task but the CONTRACT does not: in
    # both envs every pose-derived field (stock block, AP2AP block, keypoints)
    # is computed from base_obs, so rewriting these two pose pairs makes all
    # three consistent. r_rms is the canonical cloud radius, used to turn a
    # displacement into the angle that produces it.
    NOISE_SL = {
        "peginsert": (dict(obj_p=slice(25, 28), obj_q=slice(28, 32),
                           goal_p=slice(35, 38), goal_q=slice(38, 42)), 57.4),
        "stack":     (dict(obj_p=slice(25, 28), obj_q=slice(28, 32),
                           goal_p=slice(32, 35), goal_q=slice(35, 39)), 22.3),
        # PickCube: the goal POSITION is a stock slice, the goal ORIENTATION is
        # committed at reset and reached through `set_goal_offset`.
        "pickcube":  (dict(obj_p=slice(29, 32), obj_q=slice(32, 36),
                           goal_p=slice(26, 29)), 20.0),
        # LiftPegUpright: NO goal anywhere in the stock observation, so there is
        # no goal slice at all -- the whole goal goes through `set_goal_offset`.
        "liftpeg":   (dict(obj_p=slice(25, 28), obj_q=slice(28, 32)), 120.0),
        # PushCube: goal POSITION is a stock slice (25:28); no goal orientation anywhere (goal_region is flat).
        "pushcube":  (dict(obj_p=slice(28, 31), obj_q=slice(31, 35), goal_p=slice(25, 28)), 22.3),
        "placesphere": (dict(obj_p=slice(29, 32), obj_q=slice(32, 36), goal_p=slice(26, 29)), 20.0),
    }
    if args.noise_v2:
        if args.task not in NOISE_SL:
            raise SystemExit(f"--noise-v2 has no slice map for task {args.task!r}; "
                             f"known: {sorted(NOISE_SL)}")
        if args.obj_noise_mm or args.goal_noise_mm:
            raise SystemExit("--noise-v2 and the v1 --*-noise-mm flags are exclusive")
        sl, r_rms = NOISE_SL[args.task]
        from msppo.obs_noise2 import PoseNoiseV2
        env.obs_noise = PoseNoiseV2(
            args.num_envs, sl, args.task,
            getattr(env, "_goal_from_obs_clean", env._goal_from_obs), horizon,
            device="cuda", seed=args.seed, goal_scale=args.noise_v2,
            obj_scale=args.obj_v2, r_rms_mm=r_rms, sig_mode=args.sig_mode,
            goal_attr_fn=getattr(env, "set_goal_offset", None),
            curriculum=args.noise_curriculum, a0=args.noise_a0,
            total_ticks=total // args.num_envs, tail_scale=args.noise_tail,
            z_reflect=bool(args.noise_zreflect), goal_rot_scale=args.noise_goal_rot,
            z_mode=args.noise_zmode)
        env.noise_horizon = horizon
        gm, om = env.obs_noise.gm, env.obs_noise.om
        print(f"[noise-v2] {args.task} goal mu_u {gm['mu_u']}mm "
              f"sd {gm['sd']} tail {gm['tail']:.4f} rot {gm['rot']}deg x{args.noise_v2}"
              f"  |  obj {om['s0']}->{om['sT']}mm rho {om['rho']} x{args.obj_v2}"
              f"  |  sig_obs={bool(args.sig_obs)} mode={args.sig_mode} "
              f"eta={env.obs_noise.eta} sign={env.obs_noise.sign}"
              f"  curriculum={args.noise_curriculum} a0={args.noise_a0}"
              f"  redraw every {horizon} steps")
    elif args.obj_noise_mm or args.goal_noise_mm or args.obj_rot_mm or args.goal_rot_mm:
        if args.task not in NOISE_SL:
            raise SystemExit(f"--*-noise-* has no slice map for task {args.task!r}; "
                             f"known: {sorted(NOISE_SL)}")
        sl, r_rms = NOISE_SL[args.task]
        from msppo.obs_noise import PoseNoise
        env.obs_noise = PoseNoise(args.num_envs, sl, device="cuda", seed=args.seed,
                                  obj_mm=args.obj_noise_mm, obj_rot_mm=args.obj_rot_mm,
                                  goal_mm=args.goal_noise_mm, goal_rot_mm=args.goal_rot_mm,
                                  r_rms_mm=r_rms)
        env.noise_horizon = horizon
        print(f"[noise] obj {args.obj_noise_mm}mm/{args.obj_rot_mm}mm-rot  "
              f"goal {args.goal_noise_mm}mm/{args.goal_rot_mm}mm-rot  "
              f"-> obj {env.obs_noise.obj_deg:.1f}deg  goal {env.obs_noise.goal_deg:.1f}deg  "
              f"redraw every {horizon} steps")

    obs_dim = env.single_observation_space.shape[0]
    act_dim = env.single_action_space.shape[0]
    agent, branches = build_agent(args.head, env, obs_dim, act_dim, args.num_kp,
                                  bool(args.last_action), spec["qdim"], args)
    if args.init_from:
        sd = torch.load(args.init_from, map_location="cuda")
        # WARM START ACROSS AN OBSERVATION-LAYOUT CHANGE. Both heads keep
        # obs[:, :kp_start] verbatim in their MLP input, so any block the source
        # checkpoint predates can be spliced in as ZERO WEIGHT COLUMNS at that
        # block's own offset. Zero columns make the new inputs multiply by 0, so
        # the network computes exactly the source function at step 0 regardless
        # of what those inputs contain, and then learns to use them.
        #
        # The offset matters. The first version assumed the missing block always
        # ENDS at kp_start, which is true for sig/priv/contact (they are the last
        # prefix blocks) but FALSE for ap2ap, which sits immediately after the
        # stock vector with sig/contact after it. Inserting ap2ap at kp_start-20
        # would silently shift sig and contact into the wrong columns -- a warm
        # start that loads without error and means nothing. The offsets are
        # therefore rebuilt from the SOURCE run's own run.json, never from flags.
        target = agent.base if hasattr(agent, "base") else agent      # PrivCritic wraps
        own = target.state_dict()

        def _widths(cfg, ap_w, act_w):
            """[(name, width)] of the prefix blocks AFTER stock, in obs order.
            Mirrors `_augment_core`: stock | ap2ap | last_action | sig | contact
            | priv | keypoints."""
            return [("ap2ap", ap_w if cfg.get("ap2ap_fields") else 0),
                    ("last_action", act_w if cfg.get("last_action") else 0),
                    ("sig", 4 if cfg.get("sig_obs") else 0),
                    ("contact", 4 if cfg.get("contact_obs") else 0),
                    ("priv", 6 if cfg.get("priv_critic") else 0)]

        AP_W = 20            # identical for all four tasks; asserted below
        src_json = os.path.join(os.path.dirname(args.init_from), "run.json")
        inserts = None       # [(offset_in_target, width)], ascending
        if os.path.exists(src_json):
            scfg = json.load(open(src_json))
            tcfg = dict(ap2ap_fields=args.ap2ap_fields, last_action=args.last_action,
                        sig_obs=getattr(env, "sig_obs", False),
                        contact_obs=getattr(env, "contact_obs", False),
                        priv_critic=getattr(env, "priv_slice", None) is not None)
            sw, tw = _widths(scfg, AP_W, act_dim), _widths(tcfg, AP_W, act_dim)
            s_kp = scfg["obs_dim"] - 6 * scfg["num_kp"]
            t_kp = env.kp_slice.start
            s_stock = s_kp - sum(w for _, w in sw)
            t_stock = t_kp - sum(w for _, w in tw)
            if s_stock != t_stock or s_stock <= 0:
                raise SystemExit(
                    f"--init-from: stock block differs ({s_stock} vs {t_stock}); "
                    f"the source is not the same task/observation family")
            inserts, off = [], t_stock
            for (nm, a), (_, b) in zip(sw, tw):
                if a == 0 and b > 0:
                    inserts.append((off, b, nm))
                elif a != b:
                    raise SystemExit(
                        f"--init-from: cannot drop or resize block {nm!r} "
                        f"({a} -> {b}); only additions are supported")
                off += b
        n_exp = 0
        for k, v in list(sd.items()):
            if k in own and own[k].shape != v.shape:
                d = own[k].shape[1] - v.shape[1] if v.dim() == 2 else -1
                ok = d > 0 and own[k].shape[0] == v.shape[0]
                if ok and inserts is not None and d == sum(w for _, w, _ in inserts):
                    row = v
                    for off, w, _ in inserts:          # ascending: offsets already
                        z = torch.zeros((row.shape[0], w), device=row.device,
                                        dtype=row.dtype)   # account for prior ones
                        row = torch.cat([row[:, :off], z, row[:, off:]], dim=1)
                    sd[k] = row
                    n_exp += 1
                elif ok and inserts is None and d <= (
                        (4 if getattr(env, "sig_obs", False) else 0)
                        + (6 if getattr(env, "priv_slice", None) is not None else 0)):
                    ins = env.kp_slice.start - d      # legacy path: no source run.json
                    z = torch.zeros((v.shape[0], d), device=v.device, dtype=v.dtype)
                    sd[k] = torch.cat([v[:, :ins], z, v[:, ins:]], dim=1)
                    n_exp += 1
                else:
                    raise SystemExit(f"--init-from shape mismatch on {k}: {tuple(v.shape)} vs {tuple(own[k].shape)}")
        target.load_state_dict(sd, strict=True)
        where = ", ".join(f"{nm}@{off}+{w}" for off, w, nm in (inserts or [])) or "none"
        print(f"[init-from] loaded {args.init_from} (strict, {len(sd)} tensors, "
              f"{n_exp} expanded; inserted {where})")

    logf = open(f"{out}/train.log", "w", buffering=1)

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")

    nparam = sum(p.numel() for p in agent.parameters())
    log(f"tag={args.tag} task={args.task} env={spec['env_id']} head={args.head} "
        f"obs={obs_dim} act={act_dim} qdim={spec['qdim']} "
        f"kp_slice=[{env.kp_slice.start},{env.kp_slice.stop}] num_kp={args.num_kp} "
        f"has_scene={env.has_scene} params={nparam/1e6:.3f}M "
        f"reward={args.reward} w_kp={args.w_kp} envs={args.num_envs} "
        f"ep_len={horizon} nsteps={nsteps} gamma={args.gamma} "
        f"lam={args.gae_lambda} epochs={args.epochs} mb={args.minibatches} "
        f"lr={args.lr} target_kl={args.target_kl} steps={total} "
        f"last_action={args.last_action} seed={args.seed}")

    hist = train(env, agent, PPOConfig(total_steps=total, num_steps=nsteps,
                                       gamma=args.gamma,
                                       gae_lambda=args.gae_lambda,
                                       epochs=args.epochs,
                                       minibatches=args.minibatches,
                                       target_kl=args.target_kl, lr=args.lr),
                 logger=log, save_best=f"{out}/agent_best.pt")

    torch.save(agent.state_dict(), f"{out}/agent.pt")
    peak = max((h["success"] for h in hist if h["success"] == h["success"]),
               default=None)
    json.dump(dict(tag=args.tag, task=args.task, env=spec["env_id"],
                   noise_v2=args.noise_v2, obj_v2=args.obj_v2,
                   noise_curriculum=args.noise_curriculum, noise_a0=args.noise_a0,
                   noise_tail=args.noise_tail, noise_zreflect=bool(args.noise_zreflect), noise_zmode=args.noise_zmode,
                   noise_goal_rot=args.noise_goal_rot,
                   sig_obs=bool(args.sig_obs), sig_mode=args.sig_mode,
                   priv_critic=bool(args.priv_critic), shaping_true_goal=bool(args.shaping_true_goal),
                   init_from=args.init_from,
                   w_bump=args.w_bump, bump_mode=args.bump_mode, contact_obs=bool(args.contact_obs),
                   obj_noise_mm=args.obj_noise_mm, obj_rot_mm=args.obj_rot_mm,
                   goal_noise_mm=args.goal_noise_mm, goal_rot_mm=args.goal_rot_mm,
                   obs=f"{args.task}_kp", arm=args.task, head=args.head,
                   reward=args.reward, seed=args.seed, num_kp=args.num_kp,
                   fam_cfg=(vars(fam) if fam else None),
                   w_kp=args.w_kp, qdim=spec["qdim"], has_scene=env.has_scene,
                   ap2ap_fields=args.ap2ap_fields, last_action=args.last_action,
                   robot=spec["robot"],
                   ignore_terminations=args.ignore_terminations,
                   reconfig_freq=0, control=args.control,
                   num_envs=args.num_envs, max_episode_steps=horizon,
                   num_steps=nsteps, gamma=args.gamma,
                   gae_lambda=args.gae_lambda, epochs=args.epochs,
                   minibatches=args.minibatches, lr=args.lr,
                   target_kl=args.target_kl, token_dim=args.token_dim,
                   hidden=args.hidden, pn_hidden=args.pn_hidden,
                   params=nparam, obs_dim=obs_dim, branches=branches,
                   total_steps=total, peak_success=peak,
                   final_success=hist[-1]["success"] if hist else None,
                   history=hist), open(f"{out}/run.json", "w"), indent=2)
    log(f"saved -> {out}")
    env.close()


if __name__ == "__main__":
    main()
