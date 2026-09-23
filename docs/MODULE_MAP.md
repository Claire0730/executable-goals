# Module map of the release

This document classifies every Python module shipped under `msgen/`, `msppo/` and `tools/` by whether the
released pipeline (`scripts/*.sh`) executes it, and names the component of the paper *Predicted Futures Are Not
Enough: Learning Executable Goals for Robot Manipulation* that each module implements (paper terms are mapped to
repository identifiers in `docs/GLOSSARY.md`). The classification was derived from the release tree (the root of
this repository) by (i) reading every `scripts/*.sh`, (ii) collecting the modules each script starts with
`python -m`, `runpy.run_module`, a here-document import, or a direct file invocation, and (iii) following the
`import` statements of those modules transitively, including imports placed inside functions. The release contains
exactly this import closure: 35 modules under `msgen/` (plus `__init__.py`), 38 modules under `msppo/` (plus
`__init__.py` and `gpu_guard.sh`) and 7 scripts under `tools/`. A module is **paper chain** when at least one
released script reaches it in this way; every other shipped module is **imported but inactive** (reached by an
import, but returning without effect because its enabling variable is unset) or **retained for record** (an
environment wrapper or network variant that `kp_teacher.make_env` / `build_agent` can select but no released script
selects). Modules of the private research repository outside this closure are not shipped; among them are the
CoTracker3 online-tracking line (`tracked_perception`, `track_server`, `track_rpc`), the Oracle Goal perturbation
patch that produced the Fig. 7 records (`patch_goal_perturb`), the stock-clearance patch (`patch_clearance`) and the
offline SAM 2 peg-mask tool (`peg_sam2_masks`, mentioned in a comment of `msppo/peg_tgbank.py`).

Script abbreviations used in the tables:

| Abbreviation | Script | Interpreter |
|---|---|---|
| CFG | `scripts/config.sh` (sourced by every script) | bash |
| S00 | `scripts/00_link_checkpoints.sh` | bash |
| S10 | `scripts/10_render_banks.sh` | `$PM` (maniskill) |
| S20 | `scripts/20_predict.sh` | `$PG` (trace_gen) |
| S30 | `scripts/30_goals.sh` | `$PG` for `msgen.ksample`; `$PM` for the rest |
| S31 | `scripts/31_goals_sam2marker.sh` (optional; SAM 2 marker localiser, PickCube `sam2mk6` bank) | `$PV` (a separate environment with `sam2`, `imageio`, `scipy`) |
| S40 | `scripts/40_eval_row4.sh` (closed loop with the composed planner goal; Entity-Level Goal Readout row) | `$PM` |
| S41 | `scripts/41_eval_row3.sh` (closed loop with the Oracle Goal) | `$PM` |
| S50 | `scripts/50_collect_data.sh` | `$PM` |
| S51 | `scripts/51_label_t2k.sh` | `$PM` |
| S52 | `scripts/52_train_planner.sh` | `$PG` |
| S60 | `scripts/60_train_teachers.sh` | `$PM` |
| S70 | `scripts/70_distill.sh` (`INJECT=iid` default, or `frozen`) | `$PM` |
| VM | `scripts/verify_main_table.sh` (`ROW=final\|k1\|k4\|k4pick\|oracle`; runs `msppo.multi_eval` directly with the executor and the frozen banks of that Table II row) | bash, `$PM` |

"Used by" lists the scripts in whose process the module is imported. "(via X)" names the importing module
when the script does not name the module itself. "(conditional)" marks an import that sits inside a function
or branch and is executed only on the released code path stated in the role column.

## 1. Entry points of the released scripts

| Script | Modules started directly |
|---|---|
| CFG | sources `msppo/gpu_guard.sh` (`gpu_wait`) |
| S00 | none (symlinks for the three executors and five teachers, `sha256sum`) |
| S10 | `msgen.patch_wall.maybe_patch` (asserted), `msppo.task_tgbank.render`, `msppo.rebank_obj` (`--n-obj 16 --seed 0`) |
| S20 | `msgen.predict` |
| S30 | `msgen.ksample`, `msppo.task_tgbank solve --ransac` (Rigid Readout), `tools/t2k_decode.py`, `tools/mk_fitgoals.py`, `tools/build_pc_src.py` (pickcube), `tools/build_gmapdcc.py` (stack), `tools/psi_bank.py` and `tools/psinat.py` (all tasks except pushcube) |
| S31 | `tools/build_sam2marker.py` (`msppo.goal_depth_check.op_marker_sam2`, which imports SAM 2 lazily) |
| S40, S41, VM | `msppo.multi_eval` |
| S50 | `msgen.patch_wall.maybe_patch` (asserted), `msgen.replay` (via `runpy`), `msgen.labels` |
| S51 | `msgen.labels_t2k` |
| S52 | `msgen.run_train_logged` |
| S60 | `msppo.patch_frame` or `msppo.patch_stack_frame` (asserted), `msppo.kp_teacher` (via `runpy`); the process receives the `MSPPO_*` environment recorded in `configs/teacher/<tag>/patches.json` |
| S70 | `msppo.patch_iid_inject` (`INJECT=iid`) or `msppo.patch_frozen_inject` (`INJECT=frozen`) (asserted), `msppo.multi_distill` (via `runpy`) |

## 2. `msgen/` (3D Trace Planner side; runs in the `trace_gen` environment except where noted)

### 2.1 Paper chain

| Module | Used by | Role |
|---|---|---|
| `msgen/__init__.py` | all | Package marker. |
| `msgen/ds_audit.py` | S52 (via `run_train`, unless `MSGEN_DS_AUDIT=off`) | Compares the clip directories the TraceGen loader will enumerate with what `dataset.json` claims; warns by default (`MSGEN_DS_AUDIT=warn`), aborts under `strict`. |
| `msgen/ksample.py` | S30 | Averages K prediction files of the same bank into one file with the `msgen.predict` schema (`nanmean` of absolute traces, mask = logical AND): the K = 4 trace averaging before the Rigid Readout fit; refuses bit-identical samples and mismatched `episode_id`. |
| `msgen/labels.py` | S50; S10, S30, S51, S40, S41, S70, VM (via `labels_t2k`, `task_tgbank`, `peg_tgbank`, `rebank_obj`, `objmetrics`, `tools/t2k_decode.py`, `peg_perception`) | Synthesises exact simulator ground-truth 3-D traces for the 400 query pixels (`grid_pixels`, `object_aware_pixels`, `unproject`, `project`, `quat_to_R`) and writes TraceGen episode datasets. |
| `msgen/labels_t2k.py` | S51 | Derives the entity-level labels of the Entity-Level Goal Readout (entity branch: keyframes, twists, contact, structure-relative terminal pose, entity membership; map branch: goal map) offline from the raw replays (`samples/<stem>_t2k.npz`). |
| `msgen/objmetrics.py` | S50 (via `labels.target_id_for`, conditional on `--n-obj > 0`) | Resolves the target body id per clip; also holds the object-endpoint error metric (not run by the release). |
| `msgen/patch_all.py` | S20, S52 (via `predict`, `run_train`) | Single entry point that applies every environment-flagged planner patch and installs a load guard that turns a silently dropped patch parameter into an error. See Section 5 for which patches are active. |
| `msgen/patch_ckpt.py` | S52 (via `run_train`) | Active without a variable: suppresses the epoch-0 and best-model checkpoint writes; `MSGEN_KEEP_ALL_CKPT=1` (set by S52 for the map-branch stage) opts out. |
| `msgen/patch_seed.py` | S20 (`MSGEN_SEED`), S52 (imported, inactive) | Makes `predict_trajectory` deterministic (per-batch generator seeded `MSGEN_SEED + k`) and seeds the global RNG used for the per-sample instruction choice. |
| `msgen/patch_steps.py` | S20 (`MSGEN_STEPS=20 MSGEN_DT=fix` for the entity-branch, map-branch and PushCube K = 4 predictions; the four-task `mix4` K = 4 source runs the native sampler), S52 (imported, inactive) | Corrects the Euler step size so that `num_inference_steps` integrates the full t = 1 to 0 span (dt = 1/N). |
| `msgen/patch_t2k.py` | S20 (entity-branch and map-branch predictions), S52 (both readout stages) | The Entity-Level Goal Readout ("T2K head" in the code): entity token by masked attention pooling, cross attention over the 576 geometry tokens, terminal rotation and structure-relative pose, auxiliary keyframe / twist / contact outputs, the map branch (`MSGEN_T2K_GMAP`, `Linear(768, 1)` over the 24 x 24 tokens), the losses `L_ent` and `L_map`, and the inference-time rule mask. |
| `msgen/patch_wall.py` | S10, S50 (asserted); S40, S41, S70, VM (via `multi_eval.env_patches`); imported by `patch_all` in S20 and S52 but inactive there (see Section 5, note W) | Wraps `_load_scene` of the five ManiSkill task classes with a static, collision-free four-sided enclosure (radius 2.0 m, six 0.70 m bands starting at z = -1.05 m) so that depth is bounded; consumes no RNG. |
| `msgen/paths.py` | S20, S52 | Locations (`TRACEGEN_DIR`, `TRACEGEN_GENERALIST`) and the shims that drive the pristine TraceGen checkout. |
| `msgen/predict.py` | S20 | Builds the trainer in-process (the upstream trainer loads the checkpoint with `load_state_dict(strict=False)`, guarded by `patch_all`, so the released parameter-only planner files load and the frozen encoders come from the Hub), runs `predict_trajectory` on a bank, converts deltas to absolute (u px, v px, depth m), saves `results/preds/<tag>.npz` with `prov_*` keys and the `t2k_*` readout outputs. |
| `msgen/replay.py` | S50 (via `runpy`); S10 (via `task_tgbank.render`, conditional on `show_goal`, i.e. PickCube) | Replays official ManiSkill demonstrations by driving `env_states`, records RGB, metric depth, segmentation and every rigid-body pose; `unhide_goal` renders the PickCube goal marker. |
| `msgen/run_eval.py` | S20, S52 (via `predict`, `run_train`; only `patch_split_all_to_val` and `Tee` are used) | Loader shims; its own benchmark entry point is not run by the release. |
| `msgen/run_train.py` | S52 (via `run_train_logged`) | Warm-start fine-tuning of the Generalist through TraceGen's `train.py` under `runpy`; routes every episode to the training split. |
| `msgen/run_train_logged.py` | S52 | Same CLI as `run_train`, plus a provenance sidecar `runs/<tag>/patches.json` (active patches, `MSGEN_*`/`MSPPO_*` variables, git HEAD, argv) mirrored into `run.json`. |
| `msgen/tasks.py` | every Python process of S10 to S70 and VM | Task registry (env ids, instructions, registry cameras), `IMAGE_SIZE = 384`, `GRID = 20`, `NUM_KPS = 400`, `TRAJ_STEPS = 33`, demo roots (`MS_DEMO_DIR`), and the `MSGEN_CAM_{EYE,TARGET,FOV}_<TASK>` override applied at import. |
| `msgen/trace_seg.py` | S20 and S52 (via `patch_t2k`, rule mask), S30 (via `tools/psi_bank.py`) | Rule-based motion segmentation of a trace (moving points > 8 px, onset > 4 px, arm/object split by onset), grasp/release steps and the world-frame approach and carry-height profiles (the source of the psi token); reads no segmentation. |

### 2.2 Imported but inactive

The modules below are imported into every S20 and S52 process by `patch_all.apply_all` (or by `run_train`), but
each returns without effect because its enabling variable is unset in the released scripts. They are exploratory
ablations of the planner (depth handling, sampler statistics, auxiliary heads, classifier-free guidance, coupling
of the readout to the flow decoder) and are shipped because the import closure reaches them. None of them changes a
released number.

| Module | Used by | Role |
|---|---|---|
| `msgen/contact_sheet.py` | not used by the released pipeline (exercised by `tests/test_contact_sheet.py`) | Tiles the first RGB frame of every clip with object/goal outlines for visual inspection. |
| `msgen/patch_bestckpt.py` | S52 (imported, inactive) | Opt-in task-balanced validation split and best-model writes (`MSGEN_BEST_CKPT=1`). |
| `msgen/patch_cfg.py` | S20, S52 (imported, inactive) | Learned null condition for classifier-free guidance (`MSGEN_CFG`). |
| `msgen/patch_coloraug.py` | S20, S52 (imported, inactive) | Training-time photometric jitter (`MSGEN_COLOR_AUG`). |
| `msgen/patch_depthaug.py` | S20, S52 (imported, inactive) | Training-time depth degradation model (`MSGEN_DEPTH_AUG`). |
| `msgen/patch_depthnorm.py` | S20, S52 (imported, inactive) | Log/affine depth input normalisation (`MSGEN_DEPTHNORM`). |
| `msgen/patch_depthscale.py` | S20, S52 (imported; `scale()` returns 1.0 when `MSGEN_DEPTH_SCALE` is unset) | Rescales the depth channel of the learning target. |
| `msgen/patch_depthtower.py` | S20, S52 (imported, inactive) | Unfreezes the last N blocks of the depth tower (`MSGEN_DEPTH_UNFREEZE`). |
| `msgen/patch_divergence.py` | S20, S52 (imported, inactive) | Divergence of the velocity field as a confidence signal (`MSGEN_DIVSTATS`). |
| `msgen/patch_lrgroups.py` | S20, S52 (imported, inactive) | Makes `lr_backbone` reach the encoder parameter groups (`MSGEN_LRGROUPS`). |
| `msgen/patch_movehead.py` | S20, S52 (imported, inactive) | Per-query "does this point move" auxiliary head (`MSGEN_MOVEHEAD`). |
| `msgen/patch_odestats.py` | S20, S52 (imported, inactive) | Records the sampler's integration-path statistics (`MSGEN_ODESTATS`). |
| `msgen/patch_qfeat.py` | S20, S52 (imported, inactive) | Gives each trace query the visual feature at its own pixel (`MSGEN_QFEAT`). |
| `msgen/patch_rotloss.py` | S20, S52 (imported, inactive) | Centred-residual auxiliary loss on the object's motion shape (`MSGEN_ROTLOSS`). |
| `msgen/patch_scenegate.py` | S40, S41, S70, VM (imported by `multi_eval.env_patches`, inactive) | Rejection sampling of scenes whose task bodies fall outside the usable frame (`MSGEN_SCENE_GATE`); changes the task distribution and is off in every released number. |
| `msgen/patch_t2k_couple.py` | S20, S52 (imported, inactive) | Couplings of the Entity-Level Goal Readout to the flow decoder (`MSGEN_T2K_DEC`, `MSGEN_T2K_KC`). |
| `msgen/patch_vispos.py` | S20, S52 (imported, inactive) | Positional embedding for the visual conditioning tokens (`MSGEN_VISPOS`). |

## 3. `msppo/` (simulator, privileged teachers, Pose-Native Executor; `maniskill` environment)

### 3.1 Paper chain

| Module | Used by | Role |
|---|---|---|
| `msppo/__init__.py` | all | Package marker. |
| `msppo/gpu_guard.sh` | CFG (sourced), hence every script | `gpu_wait <MiB>`: waits for free VRAM before a launch. |
| `msppo/fam_reward.py` | S10, S40, S41, S60, S70, VM (via `pickcube_kp_env`, top-level import) | Unified grasp-and-place family reward and success predicate; only the `pickcube_kp_env` import is on the released path. |
| `msppo/frame_patches.py` | S40, S41, S70, VM (via `multi_eval`, `multi_distill`, when the executor carries the psi block) | Applies the psi (k1, h) environment patches for a set of tasks in one process: `patch_stack_frame` for stack, `patch_frame` for the others. |
| `msppo/goal_depth_check.py` | S30 (via `tools/mk_fitgoals.py` for stack, `tools/build_pc_src.py`, `tools/build_gmapdcc.py`); S31 (via `tools/build_sam2marker.py`, `op_marker_sam2`) | Depth cross-check (DCC): entity-branch-gated plateau snap of a terminal goal against the t = 0 depth image; `unproject_full`; the SAM 2 marker localiser `op_marker_sam2` (prompt ensemble at the map-branch peak, mask gates, centroid + median depth along the ray), which imports SAM 2 lazily and is reached by S31 only. |
| `msppo/kabsch.py` | S10, S40, S41, S70, VM (via `task_student_env`, `peg_perception`) | Weighted paired-point SE(3) fit with residual. |
| `msppo/kp.py` | S10, S40, S41, S60, S70, VM (via `stack_kp_env`, `kabsch`) | Canonical object keypoints, `to_world`, `CUBE_HALF`. |
| `msppo/kp_teacher.py` | S60 (entry via `runpy`); S10 (via `task_tgbank.render`), S40, S41, S70, VM (via `multi_eval`, `multi_distill`, `task_student_env`) | Privileged PPO teacher driver: `TASKS` table (env id, `qdim`, horizon 50 or 100 for peginsert, robot variant), `make_env`, `build_agent`, noise-v2 injector wiring. |
| `msppo/liftpeg_kp_env.py`, `msppo/pickcube_kp_env.py`, `msppo/peginsert_kp_env.py`, `msppo/pushcube_kp_env.py`, `msppo/stack_kp_env.py` | S10, S40, S41, S60, S70, VM (via `kp_teacher.make_env`, conditional on the task) | Keypoint environment wrappers of the five released tasks under the shared contract (`RAW_DIM`, `has_scene`, `_canon_b`, `raw_from_sim`, `teacher_observation`, `_goal_from_obs`). StackCube uses `panda_wristcam`; `peginsert_kp_env` sets the clearance 0.01 (`PegInsertionSideEnv._clearance`). |
| `msppo/masking.py` | S10, S40, S41, S70, VM (conditional imports inside `task_student_env`, `peg_student_env`) | Dex4D keypoint occlusion model (one-side and random-height masks) for the synthetic-occlusion state path. |
| `msppo/multi_distill.py` | S70 | Multi-teacher DAgger into one Pose-Native Executor (beta = 0, L1 behaviour cloning plus next-state auxiliary loss), relbank goal-perturbation injection, `run.json` and `train.log`. |
| `msppo/multi_eval.py` | S40, S41, VM; S70 (via `multi_distill`, `env_patches`) | Per-task closed-loop evaluation of an executor rebuilt from its `run.json`; `--goal-delta` bank consumption (terminal step), `--psi-bank`, `--ckpt final`, `env_patches` (wall, scene gate, camera, base pose) recorded in the output JSON; without `--goal-delta` the Oracle Goal. |
| `msppo/obs_noise.py` | S60, S70 (via `obs_noise2`, `teacher_perc`, top-level `_qmul`); the v1 injector branch in `kp_teacher` is not taken | Pose-level observation noise v1 (quaternion helpers used by v2). |
| `msppo/obs_noise2.py` | S60 (via `kp_teacher`, `--noise-v2` set in every released teacher `run.json`), S70 (via `teacher_perc`) | Teacher observation noise v2: anisotropic goal error in the approach frame, time-correlated object error, sig channel. |
| `msppo/patch_basepose.py` | S10, S40, S41, S60, S70, VM (via `kp_teacher.make_env`; returns False, `MSPPO_BASE_POSE` unset) | Eval-side probe that moves the robot base; no effect in the release. |
| `msppo/patch_frame.py` | S60 (`MSPPO_FRAME_TASK`, pickcube/liftpeg/peginsert/pushcube); S40, S41, S70, VM (via `frame_patches`) | The privileged guidance variable psi = (k1 approach direction, h carry height) and the dependence-making rewards for the non-stack teacher envs; per-task cone axis and h range. |
| `msppo/patch_frozen_inject.py` | S70 (`INJECT=frozen`, `MSPPO_FROZEN_INJECT=1`) | Goal perturbation drawn once per episode and held (deployment semantics) instead of re-expressed from the live object every step; the recipe of `mt5_rcfz_gmpc_s0` and `mt5_rcfz_t2k_s0`. |
| `msppo/patch_iid_inject.py` | S70 (`INJECT=iid`, `MSPPO_IID_INJECT=1`; default) | Goal perturbation redrawn i.i.d. at every control step from a pool of 64 draws expanded once per episode (same per-episode marginal as the frozen variant); the recipe of `mt5_rciid_gmpc_s0`, the executor of the Entity-Level Goal Readout row. Refuses to run together with `MSPPO_FROZEN_INJECT`. |
| `msppo/patch_stack_frame.py` | S60 (`MSPPO_STACK_FRAME=1`, stack); S40, S41, S70, VM (via `frame_patches`) | psi interface and rewards for the StackCube teacher env. |
| `msppo/peg_kp_env.py` | S10, S40, S41, S60, S70, VM (top-level import of every `*_kp_env`, `peg_student_env`, `peg_teacher_ref`) | Base PegInsertionSide keypoint wrapper and shared helpers (`to_world`, `quat_to_R`, `unit_box_keypoints`, peg/hole constants). |
| `msppo/peg_perception.py` | S10, S40, S41, S70, VM (via `task_student_env`, `perception=True`) | RGB-D perception of the executor: t = 0 query points on the object mask, per-step depth re-measurement, weighted Kabsch to the t = 0 PCA canonical frame (the simulator's per-step object correspondence carries the points). |
| `msppo/peg_relbank.py` | S70 (via `multi_distill.load`, `task_student_env.expand`/`scale_error`); `export` is the documented way to regenerate `banks/*_relbank_*.npz` | The measured planner error distribution (frac, lat_v, lat_w, dq) used as goal perturbations: export, load, expansion in the current scene's axis, severity scaling. |
| `msppo/peg_student_env.py` | S10, S40, S41, S70, VM (via `task_student_env`, top-level `_R_to_quat`) | Peg student environment of an earlier single-task line; the release imports its helpers. |
| `msppo/peg_teacher_ref.py` | S10, S40, S41, S70, VM (via `peg_student_env`, top-level) | Loader of the frozen reference peg teacher (`ref_kp64`, `ref_obs`); the released peg teacher is `pi_v9_frame4_s0`, so only the helpers are on the path. |
| `msppo/peg_tgbank.py` | S10, S30 (via `task_tgbank`) | `kabsch_np`, `kabsch_wdisp`, `ransac_kabsch` (3 cm tolerance, 64 iterations, seed 0) and `_R_to_quat` used by the Rigid Readout. |
| `msppo/ppe.py` | S60 (via `kp_teacher`); S40, S41, S70, VM (imported with `kp_teacher`) | Paired point encoding actor-critic heads (`PairedKeypointActorCritic`, released head `paired`; `FlatActorCritic`, `PlainActorCritic`, `PrivCritic`). |
| `msppo/ppo.py` | S60 (via `kp_teacher`); S40, S41, S70, VM (imported with `kp_teacher`) | PPO for ManiSkill vectorised environments (`PPOConfig`, `train`). |
| `msppo/rebank_obj.py` | S10 | Re-places 16 of the 400 planner queries on the object using the bank's simulator segmentation (`--n-obj 16 --seed 0`), keeping images, depth and `configs.json` unchanged. |
| `msppo/student.py` | S40, S41, S70, VM | The Pose-Native Executor: `StudentTransformer` (six modality tokens incl. the psi token, d_model 128, 4 layers, 4 heads), `goal_slices`, `goal_extra_slices`, `student_from_cfg`. |
| `msppo/task_student_env.py` | S10 (via `task_tgbank.render`), S40, S41, S70, VM | Executor environment: 60-D observation packer, perception path, goal composition once per episode from the bank's (dR, dt) (T_g relative to the initial object frame), relbank injection, psi slot. |
| `msppo/task_tgbank.py` | S10 (`render`), S30 (`solve --ransac`) | Renders the evaluation scene banks and solves per-step rigid transforms (dR, dt) from the traced object points: the Rigid Readout (Kabsch inside RANSAC). |
| `msppo/teacher_perc.py` | S70 (via `multi_distill`) | Gives the teacher the executor's relative goal error during distillation (`teacher_obs_from_perception`). |
| `msppo/wb.py` | S70 (via `multi_distill.maybe_init`) | Weights & Biases monitoring wrapper (monitoring only, never a result source). It attempts an online `wandb.init` with a 30 s timeout unless `MSPPO_WANDB=0`; CFG exports `MSPPO_WANDB=0` by default, so S70 opens no online session unless the user overrides the variable. |

### 3.2 Retained for record

| Module | Used by | Role |
|---|---|---|
| `msppo/placesphere_kp_env.py`, `msppo/plugcharger_kp_env.py`, `msppo/pusht_kp_env.py` | not used by the released pipeline (imported by `kp_teacher.make_env` only when that task is requested) | Keypoint wrappers of tasks outside the five released ones. |
| `msppo/ppe_strict.py` | not used by the released pipeline (`build_agent` branch for `--head strict_paired`; the released teachers use `paired`) | Strict port of the Dex4D teacher network. |

## 4. `tools/`

| Module | Used by | Role |
|---|---|---|
| `tools/t2k_decode.py` | S30 (all five tasks) | Converts the entity-branch outputs (camera frame) to a world-frame goal, `goal = ext^-1 (struct_T_cam rel_T)`, written into the terminal step of the production goal file (`*_t2k.npz`); other steps keep the Rigid Readout. |
| `tools/mk_fitgoals.py` | S30 (pickcube, stack, liftpeg, pushcube; not peginsert) | Splices the entity-branch position into the Rigid Readout rotation (`*_t2kpos.npz`; `dt = goal_p - dR p_obj` with the bank's `configs.json` object position); for stack additionally the DCC snap (`*_dcc.npz`). Reads `PLANNER_TAG`, `ROT_TAG`. |
| `tools/build_pc_src.py` | S30 (pickcube) | `gmappeakNC`: map-branch peak patch, median patch depth, unprojection, 0.12 m gate against the last valid keyframe, abstain to that keyframe (the PickCube bank of Table I); also writes an `islandNC` variant that the release does not consume. |
| `tools/build_gmapdcc.py` | S30 (stack) | `gmapdcc`: map-branch-weighted depth centroid in xy with z from the last valid keyframe, 0.12 m gate, fallback to the `_dcc` goal. |
| `tools/build_sam2marker.py` | S31 (pickcube) | `sam2mk6`: SAM 2 marker localiser prompted at the map-branch peak (`goal_depth_check.op_marker_sam2`) replacing the terminal position of the `gmappeakNC` base bank where the gates accept a mask (the PickCube bank of the Entity-Level Goal Readout row of Table II); reads `SAM2MK_TAG`, `SAM2MK_BASE`, `SAM2MK_BASE_PATH`, `SAM2MK_TASK`. |
| `tools/psi_bank.py` | S30 (pickcube, liftpeg, peginsert, stack) | psi = (k1, h) per scene from the K-mean trace through `msgen.trace_seg` (25-degree trust cone, per-task h range), `*_psi_*.npz`. |
| `tools/psinat.py` | S30 (same four tasks) | Natural-axis psi bank: k1 replaced by the task prior axis, h kept from `psi_bank`; writes `*_psinat_*.npz`, the psi token bank S40, S41 and VM consume. |

`tests/` is not started by any script. Six tests are shipped: `test_ksample.py` (`msgen.ksample`), `test_relbank_k.py`
(`msppo.peg_relbank`), `test_student_form.py` (`msppo.student`), `test_contact_sheet.py` (`msgen.contact_sheet`),
`test_sam2marker.py` (`msppo.goal_depth_check`, pytest; mask selection, gating and `op_marker_sam2` with a stub predictor, no SAM 2 weights) and
`test_depth_patches.py` (`msgen.predict`, GPU; it refers to a checkpoint and a dataset that are not shipped). Where a
test carries a docstring it names its interpreter (`$PM` or `$PG`).

## 5. Environment variables set by the released scripts

| Variable (value) | Set by | Consumer | Effect |
|---|---|---|---|
| `MSGEN_WALL=1` | CFG (`$RC`), exported into S10, S40, S41, S70, VM; S50 sets it directly for every replay; not passed to S20 or S52 (note W) | `msgen/patch_wall.maybe_patch` | Wraps `_load_scene` of `PickCubeEnv`, `StackCubeEnv`, `PegInsertionSideEnv`, `LiftPegUprightEnv`, `PushCubeEnv` with the static four-sided enclosure. S10 and S50 assert that it applied; `multi_eval.env_patches` (S40, S41, S70, VM) aborts if it did not and records `env_patches.wall` in the JSON. See note W. |
| `MSGEN_CAM_EYE_<T>`, `MSGEN_CAM_TARGET_<T>`, `MSGEN_CAM_FOV_<T>` for `T` in PICKCUBE, STACK, PEG, LIFTPEG, PUSHCUBE | CFG (`$RC`): eye `0.574,-0.051,0.378`, target `-0.4751,0.0562,0.0200`, fov `0.754`, exported into S10, S40, S41, S70, VM; S50 sets per-view jittered values for the task being replayed; not passed to S20 or S52 | `msgen/tasks.py` at import | Overrides the registry camera of the task; consumed by `msgen.replay.make_env`, `task_tgbank.render` and the executor's perception camera (`task_student_env`, `tracegen_camera`). Recorded in every evaluation JSON and in `configs/student/*/run.json` (`env_patches.camera`). |
| `MSGEN_STEPS=20`, `MSGEN_DT=fix` | S20 (`$PRED_ENV`) for the entity-branch, map-branch and PushCube K = 4 predictions; not applied to the four-task `mix4` K = 4 source, which runs the native sampler (production logs `[patch_all] active: seed`) | `msgen/patch_steps` | 20 Euler steps with dt = 1/20 over the full t = 1 to 0 span. |
| `MSGEN_SEED=<s>` with s in 1234, 1235, 1236, 1237 (Rigid Readout source) and 1234 (entity branch, map branch) | S20 | `msgen/patch_seed` (`patch_seed`, `seed_from_env`) | Seeds the initial latent per batch (`s + k`) and the global RNG that selects one of the three instructions per sample. Recorded as `prov_seed_env`. |
| `MSGEN_T2K=1` | S20 (entity-branch and map-branch predictions), S52 (both readout stages) | `msgen/patch_t2k` | Installs the Entity-Level Goal Readout; at inference runs it on the rule mask of the predicted trace and stores `t2k_*` outputs. |
| `MSGEN_T2K_GMAP=1` | S20 (pickcube and stack map-branch predictions), S52 (map-branch stage) | `msgen/patch_t2k` | Adds the map branch (`Linear(768, 1)` over the 576 patch tokens) and its loss `L_map`. |
| `MSGEN_T2K_W=0.3` | S52 (both readout stages) | `msgen/patch_t2k` | Weight of the readout loss in the total loss. |
| `MSGEN_KEEP_ALL_CKPT=1` | S52 (map-branch stage only) | `msgen/patch_ckpt` | Opts out of the checkpoint-write suppression (writes every checkpoint). |
| `MSPPO_IID_INJECT=1` | S70 (`INJECT=iid`, default) | `msppo/patch_iid_inject` | Per-step i.i.d. redraw of the injected goal perturbation (recipe of `mt5_rciid_gmpc_s0`). |
| `MSPPO_FROZEN_INJECT=1` | S70 (`INJECT=frozen`) | `msppo/patch_frozen_inject` | Freezes the injected goal perturbation for the whole episode (recipe of `mt5_rcfz_gmpc_s0`). |
| `MSPPO_FRAME_TASK=<task>` | S60 for pickcube, liftpeg, peginsert, pushcube | `msppo/patch_frame` | Adds the 4-D psi block and the k1/h-dependent rewards to that task's teacher env. |
| `MSPPO_STACK_FRAME=1` | S60 for stack; set programmatically by `frame_patches.apply` in S40, S41, S70, VM | `msppo/patch_stack_frame` | Same for the StackCube teacher env. |
| `MSPPO_FRAME_WPUSH`, `MSPPO_FRAME_WH`, `MSPPO_FRAME_K1_BASE` | S60, exported from the `env` block of `configs/teacher/<tag>/patches.json`: `pc_v9_nz_s0` `MSPPO_FRAME_WPUSH=0.25`; `pi_v9_frame4_s0` `MSPPO_FRAME_WH=0 MSPPO_FRAME_WPUSH=0 MSPPO_FRAME_K1_BASE=-0.28,-0.56,-0.78`; `lp_v9_nz_s0` and `sc_v9_nz03b_s0` record no reward-weight override; `push_v9_nz_s0` has no `patches.json` | `msppo/patch_frame` | Push-reward weight (default 0.5), carry-height reward weight (default per-task `wh`) and cone axis (default per-task `k1_base`) of the psi rewards. S60 exports every key of the recorded block, including `MSPPO_EVAL_TASK=peginsert` for `pi_v9_frame4_s0`, which no shipped module reads. |
| `MSPPO_WANDB=0` | CFG (default; overridable from the environment) | `msppo/wb.py` | Disables the online Weights & Biases session that `multi_distill` (S70) would otherwise attempt. |
| `SAM2_DIR`, `SAM2_CKPT` (defaults `third_party/sam2`, `<SAM2_DIR>/checkpoints/sam2.1_hiera_small.pt`) | S31 (exported) | `msppo/goal_depth_check._sam2_predictor` | Location of the SAM 2 repository and checkpoint (not redistributed). |
| `SAM2MK_TAG=sam2mk6`, `SAM2MK_BASE=gmappeakNC` (`SAM2MK_BASE_PATH`, `SAM2MK_TASK` optional) | S31 | `tools/build_sam2marker.py` | Output bank suffix and base bank of the SAM 2 marker localiser. |
| `PICKCUBE_ROUTE` (default `sam2mk6`; alternative `gmappeakNC`) | CFG (overridable) | CFG `ROW4_GOAL`, hence S40 and VM `ROW=final` | Selects the PickCube goal bank of the Entity-Level Goal Readout row (Table II cell: `sam2mk6`; Table I bank: `gmappeakNC`). |
| `STUDENT_TAG` (default `mt5_rciid_gmpc_s0`; `STUDENT_TAGS` lists the three released executors), `STUDENT_RUN` | CFG (overridable) | S40, S41 | Executor evaluated by S40 and S41; VM selects the executor per row (`ROW`). |
| `ROW` (`final`, `k1`, `k4`, `k4pick`, `oracle`) | user | VM | Table II row to reproduce (executor and banks of that row). |
| `INJECT` (`iid`, `frozen`), `TAG` | user | S70 | Temporal process of the goal perturbation and run name of the distilled executor. |
| `PV` | user | S31 | Interpreter of the SAM 2 environment. |
| `TAG_MIX4`, `TAG_HEAD`, `TAG_GMAP` | CFG (exported; defaults `mix4_realcam_n2400`, `mix5_t2k_n3000`, `mix5_t2k_gmap`) | `tools/build_pc_src.py`, `tools/build_gmapdcc.py`, `tools/build_sam2marker.py`, `tools/psi_bank.py`, `tools/psinat.py`; the scripts name every prediction and goal file by them | Checkpoint tags. A different planner is evaluated by overriding `CK_*` and the matching `TAG_*` together. |
| `GPU_WAIT=0` | user (unset by default) | `msppo/gpu_guard.sh` | Skips the free-VRAM wait before each launch; otherwise `nvidia-smi` must be on `PATH`. |
| `PLANNER_TAG`, `ROT_TAG` | S30 | `tools/mk_fitgoals.py` | Entity-branch checkpoint tag and rotation-source tag (`mix4_realcam_n2400`, or `mix5_t2k_n3000` for pushcube). |
| `EG_REPO`, `TRACEGEN_DIR`, `TRACEGEN_GENERALIST`, `CKPT_DIR`, `PYTHONPATH`, `PYTHONSAFEPATH=1`, `PG`, `PM`, `MS_DEMO_DIR`, `SEED`, `N_EP`, `N`, `WARM`, `DRY` | CFG and the individual scripts | `msgen/paths.py`, `msgen/tasks.py` (`MS_DEMO_DIR`), `tools/*.py` (`EG_REPO`), shell | Paths, interpreters and run names; no patch is attached to them. |

Patches that are imported into a released process and remain inactive because their variable is unset:
`patch_cfg`, `patch_coloraug`, `patch_depthaug`, `patch_depthnorm`, `patch_depthscale`, `patch_depthtower`,
`patch_divergence`, `patch_lrgroups`, `patch_movehead`, `patch_odestats`, `patch_qfeat`, `patch_rotloss`,
`patch_t2k_couple`, `patch_vispos`, `patch_bestckpt` (planner side); `patch_scenegate`, `patch_basepose`
(simulator side). Patches or checks that act without a variable: `patch_ckpt` (S52, except the map-branch stage),
the `patch_all` load guard (S20, S52), `ds_audit` in `warn` mode (S52).

**Note W (planner processes).** S20 and S52 do not pass `MSGEN_WALL` (or the camera variables) into the
`trace_gen` interpreter; the wall and camera variables belong to the rendering stages S10 and S50 (and to the
executor stages S40, S41, S70, VM, whose environments render the executor's camera). `patch_all.apply_all` would otherwise call `patch_wall.maybe_patch`, which imports `sapien`
and `mani_skill`; these are not in `requirements/trace_gen_freeze.txt`, and the call raises `ModuleNotFoundError`
(checked with `MSGEN_WALL=1 $PG -c "from msgen.patch_wall import maybe_patch; maybe_patch()"`).
This matches the production runs: the prediction logs read `[patch_all] active: t2k, seed, steps`
(`<private-repo>/experiments/20260911_row4_997998/pred_*.log`) and the recorded planner-run provenance lists only
`MSGEN_T2K`, `MSGEN_T2K_W`, `MSGEN_T2K_GMAP`, `MSGEN_KEEP_ALL_CKPT` and records `wall: false`
(`configs/planner/mix5_t2k_n3000/patches.json`, `configs/planner/mix5_t2k_gmap/patches.json`); `mix4_realcam_n2400`
has no `patches.json` and its `run.json` records no `MSGEN_*` variable at all.
In the planner process the variable would have no function beyond the `prov_wall` key that `msgen.predict` writes;
the wall is a property of the rendered images, produced in the `maniskill` processes (S10, S50). The planner smoke
test of the release build (16 scenes, `mix5_t2k_gmap`, seed 1234, with the released parameter-only checkpoint)
reproduced the production prediction bitwise.
