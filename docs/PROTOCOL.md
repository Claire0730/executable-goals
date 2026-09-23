# Evaluation and data protocol of the release, as implemented

This document states the protocol that the released scripts execute and, where the shipped artefacts differ
from what those scripts would regenerate, the historical deviations. Component names follow the paper *Predicted
Futures Are Not Enough: Learning Executable Goals for Robot Manipulation* (3D Trace Planner, Rigid Readout,
Entity-Level Goal Readout with entity branch and map branch / Spatial Goal Map, Pose-Native Executor, Oracle Goal);
the code identifiers are given in backticks next to them and are mapped in `docs/GLOSSARY.md`. Every value is taken
from one of the following sources, cited in brackets: the release scripts (`scripts/config.sh` and `scripts/*.sh`,
abbreviated CFG and S10 to S70 as in `MODULE_MAP.md`), the shipped configuration, bank and result files (`configs/`,
`banks/`, `paper_results/` with `paper_results/PROVENANCE.json`), the internal evidence audit of 2026-09-13
(abbreviated E00 to E03 by its section files; its prose is not part of this release, whereas the data it was built
from are shipped in `evidence/`: `NUMBERS_PROVENANCE.csv`, `01_checkpoint_registry.csv`, `04_goal_summary.json`,
`07_param_recount.json`, `06_verify_perturbation.out.json`, `goal_table.json`, `fig6_position_errors.json` and the
gzipped per-scene / per-episode CSVs), Sections 2 and 3 of an internal report that is not released (abbreviated R2,
R3; translated), and `docs/KNOWN_ISSUES.md` (cited by item number). Values that no record supports are marked "not
recorded".

## 1. Simulator, tasks and success predicates

ManiSkill 3.0.1 with SAPIEN 3.0.3 (`requirements/maniskill_freeze.txt`; E01 Section 2). Five tasks are evaluated
[CFG `TASKS5`; `msppo/kp_teacher.py` `TASKS`]:

| Task key | Environment | Robot | Horizon (control steps) | Success predicate (ManiSkill native) [E03 A.3] |
|---|---|---|---|---|
| pickcube | PickCube-v1 | panda | 50 | cube within 0.025 m of `goal_site` and robot static |
| liftpeg | LiftPegUpright-v1 | panda | 50 | peg upright within 0.08 rad and z within 0.005 m of 0.12 m |
| peginsert | PegInsertionSide-v1 | panda | 100 | peg head inside the hole within the clearance; the release uses clearance 0.01 (ManiSkill stock: 0.003) [R3; `msppo/peginsert_kp_env.py` `clearance=0.01`, `PegInsertionSideEnv._clearance`] |
| stack | StackCube-v1 | panda_wristcam | 50 | cube A on cube B (xy within half-size + 0.005 m, z offset 0.04 +/- 0.005 m), static, not grasped |
| pushcube | PushCube-v1 | panda | 50 | cube xy within 0.1 m of the goal region and on the table |

StackCube must be built with `panda_wristcam`; forcing `panda` changes joint 7 by pi/2 and lowered a reference
checkpoint from 0.898 to 0.289 [`msppo/kp_teacher.py` comment; R3]. Success is recorded as `success_once`, i.e.
whether the predicate held at any step of the episode; an environment is frozen once it has succeeded
[`msppo/multi_eval.py`; E01 Section 2]. Control runs at 20 Hz (`sim_freq` 100, `control_freq` 20, ManiSkill
defaults, no override in `msppo/`) [E02 Section 9].

Task goals as used for the Oracle Goal (the simulator's designated correct goal; `true goal` in the records; "row 3"
in older file names) [E03 A.3]: PickCube, the `goal_site` position with the cube yaw committed at reset;
LiftPegUpright, the peg standing at its start xy with z = 0.12 m; PegInsertionSide, the hole pose composed with the
peg-head offset; StackCube, cube B plus (0, 0, 0.04) with cube A's yaw; PushCube, the goal-region xy at the cube's
rest height 0.02 m with the cube's yaw. The PickCube target is specified to the planner only by the rendered green
goal sphere (radius 0.025 m, the success tolerance), which S10 and S50 un-hide with `--show-goal`; the instruction
carries no coordinates, and at seed 999 the marker has zero pixels in 15 of 256 scenes [E03 A.3, A.4]. The PushCube
instruction text says "red cube" while the simulated cube is blue [E03 A.3]; the text is left as trained.

## 2. Visual protocol

One camera for all five tasks, installed through the `MSGEN_CAM_{EYE,TARGET,FOV}_<TASK>` variables that CFG
exports as `$RC` [CFG; `msgen/tasks.py`]:

| Quantity | Value |
|---|---|
| Eye (world, m) | (0.574, -0.051, 0.378) |
| Look-at target (world, m) | (-0.4751, 0.0562, 0.0200) |
| Vertical field of view | 0.754 rad |
| Resolution | 384 x 384 RGB-D (`IMAGE_SIZE = 384`, `msgen/tasks.py`) |
| Intrinsics recovered from the banks | f_x = f_y = 484.924 px, c_x = c_y = 192 px (`K_PROD`, `msgen/patch_t2k.py`; E03 A.1) |
| Depth | rendered as int16 millimetres, stored in the banks as float32 metres [E03 A.1] |
| Backdrop | `MSGEN_WALL=1`: a static, collision-free four-sided enclosure at +/- 2.0 m around the workspace, six 0.70 m colour bands from z = -1.05 m, no RNG consumed [`msgen/patch_wall.py`] |

The same camera and wall are used for the planner's training clips (S50), the evaluation scene banks (S10) and
the executor's perception (S40, S41, S70 and `verify_main_table.sh` through `multi_eval.env_patches` and
`make_task_student_env`) [CFG comment; E03 A.1]. The planner processes (S20, S52) receive neither variable: they
consume rendered images only (Section 4). The world frame is the ManiSkill scene frame: table top at z = 0, robot base
at (-0.615, 0, 0), metres and radians, quaternions in (w, x, y, z) order [E03 B].

## 3. Evaluation scene banks (S10)

| Item | Value | Source |
|---|---|---|
| Scenes per task per seed | N = 256 (`N_EP=256`), rendered with `num_envs = 256`; `num_envs` is part of scene identity | CFG; `msppo/task_tgbank.py` |
| Seeds | 999 canonical; 997 and 998 repeat seeds; the paper pools the three seeds (768 episodes per task) | CFG; E01 Section 2 |
| Scene-episode pairing | episode j of the evaluation is scene `cfg_{j:05d}` of the bank (same constructor, same `num_envs` and seed) | E01 Section 2 |
| Bank contents per scene | RGB frame, depth, segmentation (`seg.npz`), `configs.json` with object pose, goal pose, K, `extrinsic_cv`, `obj_seg_id`; three instructions | `msppo/task_tgbank.py`; E03 A.4 |
| Query pixels | 400 (20 x 20 lattice) of which 16 are re-placed on the object with `msppo.rebank_obj --n-obj 16 --seed 0`, using the bank's simulator segmentation; index order is re-sorted so that lattice neighbours stay image neighbours | S10; `msgen/labels.py` `object_aware_pixels`; E02 Section 5 |
| Goal marker | PickCube banks render `goal_site` (`show_goal=1`); other tasks do not | S10 |
| Instruction per scene | one of three drawn at random per sample; the draw is pinned by `MSGEN_SEED` at prediction time; which instruction a scene received is **not recorded** | E03 A.2 |

The 16 object-aware queries are placed with the simulator segmentation both in the training data
(`--n-obj 16`, S50) and in the evaluation banks; the paper's statement that no segmentation model is needed at
inference refers to the entity branch's mask, which is derived from the predicted trace [E02 Section 5, C10].

## 4. 3D Trace Planner inference (S20)

Checkpoints [CFG]: `mix4_realcam_n2400` (Rigid Readout source and rotation source), `mix5_t2k_n3000` (entity
branch of the Entity-Level Goal Readout), `mix5_t2k_gmap` (entity branch plus map branch / Spatial Goal Map).
Architecture [E02 Sections 1 and 10; R2.2]: frozen DINOv3 ViT-L/16 (`timm vit_large_patch16_dinov3.lvd1689m`),
frozen SigLIP-B/16-384 for RGB and a second frozen SigLIP-B/16-384 with a 1-to-3 stem adapter for depth, frozen
T5-base text encoder, trainable fusion `Linear(1024+768+768 -> 768)` producing the 24 x 24 = 576 geometry tokens, a
CogVideoX-type flow decoder (6 layers, 12 heads, head dimension 64) predicting 400 traces x 32 steps of (u, v, depth)
increments, and, on the two `mix5` checkpoints, the Entity-Level Goal Readout (`msgen/patch_t2k.py`, "T2K head";
3,049,572 parameters) with, on `mix5_t2k_gmap`, the map branch `Linear(768, 1)` over the 576 tokens (769
parameters). Parameter counts: 674,546,334 unique parameters for `mix4`, 677,596,675 for `mix5_t2k_gmap` (the
paper's 677.6M); trainable 75,486,238 and 78,536,579 [E02 Section 10; `evidence/NUMBERS_PROVENANCE.csv`]. The
released `planner/*.pth` files contain the trained parameters only (about 0.30 GB for `mix4`, 0.31 GB for each `mix5`
file): the frozen encoder tensors, bitwise identical to the Hub weights, and the optimizer and scaler state were
removed; the upstream trainer loads the state dict with `strict=False` and re-creates the encoders from the Hugging
Face Hub at model construction, so S20 and S52 need an authenticated Hub session with the DINOv3 licence accepted
[KNOWN_ISSUES item 1; S20].

| Setting | Value | Source |
|---|---|---|
| Sampler | differs per prediction family and is reproduced as recorded. Four-task `mix4` K = 4 Rigid Readout source: the native TraceGen sampler, no `MSGEN_STEPS` (production logs `[patch_all] active: seed`; private launchers `<private-repo>/experiments/20260827_confhead/f5_test999.sh`, `f4_labels.sh`). Entity-branch, map-branch and PushCube K = 4 predictions: 20 Euler steps with dt = 1/20 (`MSGEN_STEPS=20 MSGEN_DT=fix`; production logs `[patch_all] active: t2k, seed, steps`). `guidance_scale = 1.0` throughout | S20; CFG `PRED_ENV`; KNOWN_ISSUES item 8; E02 Section 2.1 |
| Rigid Readout source (K stochastic trace predictions) | K = 4 independent samples of `mix4_realcam_n2400` with `MSGEN_SEED` = 1234, 1235, 1236, 1237 (native sampler); K = 1 uses the seed-1234 sample alone; for PushCube the same K = 4 from `mix5_t2k_n3000` with `MSGEN_T2K=1` and 20 fixed steps, because no `mix4` PushCube prediction exists; the PushCube seed-1234 sample file is also its entity-branch file | CFG `KSEEDS`, `ROT_CK`; S20; E02 C11 |
| Entity-branch readout | `mix5_t2k_n3000`, seed 1234, `MSGEN_T2K=1` | S20 |
| Map-branch readout | `mix5_t2k_gmap`, seed 1234, `MSGEN_T2K=1 MSGEN_T2K_GMAP=1`; PickCube and StackCube only | S20 |
| Batch | 8, 4 loader workers | S20 |
| Saved prediction | absolute (u px, v px, depth m) per query and step, `[N, 400, 33, 3]`; `t2k_*` readout outputs; `prov_ckpt`, `prov_dataset`, `prov_seed_env`, `prov_wall` | `msgen/predict.py`; E03 B |
| Entity mask at inference | rule-based motion segmentation of the predicted trace (`msgen/trace_seg.py`); fallback to the 16 largest-displacement queries; no simulator segmentation | E02 Section 4 |
| Planner input | one RGB frame, one depth map, one instruction, the 400 query pixels and K; no extrinsics, no goal coordinates | E03 A.2 |
| Latency | 0.253 s per scene (`mix4`, measured at 20 fixed steps; the production `mix4` predictions use the native 100-step sampler, whose latency is not recorded), 0.277 to 0.278 s (Entity-Level Goal Readout checkpoints), batch 1, RTX 5090; K = 4 timing **not recorded**. The paper's Table III (1.27 s, 5.60 GB) is not backed by a record in this release [KNOWN_ISSUES item 19] | R2.9; E02 Section 8.1; `evidence/NUMBERS_PROVENANCE.csv` |

Note: S20 and S52 do not export `MSGEN_WALL` or the camera variables into the planner interpreter (the wall patch
would try to import `mani_skill` there); the production predictions and trainings were likewise made without them:
the prediction logs read `[patch_all] active: t2k, seed, steps`, the `mix5` planner `patches.json` files record
`wall: false`, and the `mix4` run recorded no `MSGEN_*` variable (`MODULE_MAP.md`, Note W). The wall and camera
variables belong to rendering (S10, S50).

## 5. Goal composition (S30, S31)

### 5.1 Rigid Readout (`k1ransac`, `kmean_ransac` banks)

1. `msgen.ksample` averages the K = 4 prediction files point-wise (`nanmean` of absolute traces; mask = AND of the
   four masks) into `*_kmean.npz`; K = 4 therefore averages traces before the solve, and no candidate selection
   takes place [`msgen/ksample.py`; E02 Section 8.1]. K = 1 skips this step and solves the seed-1234 sample.
2. `msppo.task_tgbank solve --ransac` selects the object's query points as those whose bank segmentation equals
   `obj_seg_id` (at least 3 points), lifts step 0 and every step s to world coordinates, runs `ransac_kabsch`
   (minimal sample 3, inlier residual < 0.03 m, 64 iterations, seed 0, refit on the consensus set, fallback to all
   points), fixes the inlier set at the endpoint, and stores per-step (dR_s, dt_s) with plain (unweighted)
   Kabsch; `goal_p = dR p_obj + dt` and `goal_q` are diagnostics [E02 Section 8.1; E02 C2]. Output
   `*_kmean_ransac.npz` (33 steps); the shipped K = 1 banks (`*_k1ransac.npz`) were produced the same way from the
   single sample [`banks/*_k1ransac.npz` `prov_pred_npz`]. The released S30 writes the K = 4 bank; the K = 1 banks of
   the three seeds are shipped.

### 5.2 Entity-branch decode and splices (`t2k`, `t2kpos`, `dcc` banks)

- `tools/t2k_decode.py`: world goal `G = ext^-1 (T_struct,cam T_rel)` from the seed-1234 entity-branch outputs;
  translations are averaged over the given prediction files (one file in the release), rotations from the first
  file; only the terminal step of the production goal file is replaced (`*_t2k.npz`) [E02 Section 8.1; E03 C.4].
- `tools/mk_fitgoals.py`: `*_t2kpos.npz` = entity-branch terminal position with the Rigid Readout rotation;
  `dt = goal_p - dR p_obj` with `p_obj` the simulator object position in `configs.json` (the executor later applies
  (dR, dt) to its perceived points, so the planner's absolute position is reproduced up to the perception error)
  [E03 C.5]. For StackCube it also writes `*_dcc.npz`, the depth cross-check snap (`msppo/goal_depth_check.py`:
  gate on entity-branch contact mode grasp, terminal descent >= 0.02 m, support height >= 0.015 m; plateau band
  +/- 0.012 m within 0.06 m, >= 20 points, exclusion radius 0.035 m around the object start) [E02 Section 8.2].
- PegInsertionSide keeps the full entity-branch pose: S30 copies `*_t2k.npz` to `*_t2kpos.npz`.

### 5.3 Map-branch (Spatial Goal Map) readouts

- PickCube `gmappeakNC` (`tools/build_pc_src.py`): softmax over the 576 map-branch logits (24 x 24 patches of 16 px),
  argmax patch, median valid depth of that patch (>= 5 pixels with depth in (0.05, 3.0) m), unprojection at the
  patch centre; gate ||candidate - g0|| <= 0.12 m with g0 the entity branch's last valid keyframe translation in
  world; otherwise g0; with no valid keyframe the `_t2kpos` goal [E02 Section 8.2; script]. This bank carries the
  PickCube goal error of Table I.
- StackCube `gmapdcc` (`tools/build_gmapdcc.py`): map-branch-weighted xy centroid over all depth-valid pixels, z from
  g0, the same 0.12 m gate; on rejection or with no valid keyframe the `_dcc` goal [E02 Section 8.2; script].

### 5.4 SAM 2 marker localiser (S31; PickCube `sam2mk6` bank of Table II)

`tools/build_sam2marker.py` takes the `gmappeakNC` bank as base (rotation, non-terminal steps and the fallback goal)
and replaces the terminal position by `msppo/goal_depth_check.py::op_marker_sam2` where it applies: SAM 2
(`sam2.1_hiera_small`) is prompted at the map-branch peak with a prompt ensemble (the centre and nearest-depth pair,
then single points at the centre and at +/- 6 and +/- 12 px), the candidate masks pass the acceptance gates
(constants `SAM2_*` and `MARKER_RADIUS_M` in `goal_depth_check.py`: mask area 20 to 4000 px, mask score >= 0.8, mask
median depth within 0.04 m of the prompt depth, at most 0.03 m behind the surrounding ring, bounding-box aspect
<= 2.0), and the accepted mask's pixel centroid with its median depth plus 0.0185 m along the ray is unprojected to
the world position; no colour rule is used. Scenes with no accepted mask keep the `gmappeakNC` goal (`applied` and
`reason` fields of the bank): 227 of 256 scenes applied at seed 999, 216 at 997, 218 at 998 [`banks/pickcube_goals_mix5_t2k_gmap_<seed>_sam2mk6.npz`].
S31 needs SAM 2 in a separate environment (`PV`, `SAM2_DIR`, `SAM2_CKPT`; `third_party/README.md`); the frozen
`sam2mk6` banks are shipped.

### 5.5 Deployed goal bank per task [CFG `ROW4_GOAL`, `PICKCUBE_ROUTE`, `ROT_TAG`; E02 Section 8.2; `paper_results/PROVENANCE.json`]

| Task | Bank suffix (Table II, Entity-Level Goal Readout row) | Position source | Rotation source |
|---|---|---|---|
| pickcube | `pickcube_goals_mix5_t2k_gmap_<seed>_sam2mk6` (`PICKCUBE_ROUTE=sam2mk6`, default); `gmappeakNC` is the SAM 2-free route and the bank of Table I | SAM 2 marker localiser at the map-branch peak, fallback map-branch peak + ray, fallback entity branch (`mix5_t2k_gmap`, seed 1234) | `mix4` K = 4 Rigid Readout |
| stack | `stack_goals_mix5_t2k_gmap_<seed>_gmapdcc` | map-branch weighted depth centroid (`mix5_t2k_gmap`, seed 1234) | `mix4` K = 4 Rigid Readout |
| peginsert | `peginsert_goals_mix5_t2k_n3000_<seed>_t2kpos` | full entity-branch pose (`mix5_t2k_n3000`, seed 1234) | entity branch (see Section 10, deviation c) |
| liftpeg | `liftpeg_goals_mix5_t2k_n3000_<seed>_t2kpos` | entity-branch position | `mix4` K = 4 Rigid Readout |
| pushcube | `pushcube_goals_mix5_t2k_n3000_<seed>_t2kpos` | entity-branch position | `mix5_t2k_n3000` K = 4 Rigid Readout |

Rigid Readout rows of Table II: `<task>_goals_mix4_realcam_n2400_<seed>_{k1ransac,kmean_ransac}` for the four tasks
and `pushcube_goals_mix5_t2k_n3000_<seed>_{k1ransac,kmean_ransac}` (CFG `RIGID_GOAL`); the printed PickCube K = 4
cell used `pickcube_goals_mix5_t2k_n3000_999_k4ransac` (Section 7). Unsolved scenes stay in the bank as NaN and count
as failures; the executor maps a non-finite row to the identity transform [E01 Section 3; E03 C.6].

### 5.6 psi bank (four tasks, not PushCube)

`tools/psi_bank.py` segments the `mix4` K-mean trace with `msgen.trace_seg` and extracts k1 (unit displacement of the
arm points over the four steps before the grasp step, world frame) and h (carry height = maximum object-centroid
rise), clipped to a per-task range (pickcube and liftpeg [0.06, 0.14] m, peginsert [0.06, 0.12] m, stack
[0.06, 0.13] m); k1 is pulled back onto a 25-degree cone around the task prior axis; failed scenes fall back to the
prior. `tools/psinat.py` then writes the bank S40, S41 and `verify_main_table.sh` consume (`*_psinat_*.npz`): k1 = the
task prior axis (pickcube, liftpeg, stack (0, 0, -1); peginsert (-0.28, -0.56, -0.78) normalised) and h from
`psi_bank` [scripts; shipped `banks/*_psinat_*.npz` `prov`]. PushCube has no psi bank: `multi_eval` applies the
`patch_frame` psi sampler for it and the environment draws psi from that task's prior [`msppo/multi_eval.py`;
`msppo/patch_frame.py` `TASKS`]. The 4-dimensional (k1, h) vector is the psi token of the executor (Section 6).

## 6. Pose-Native Executor interface

Three executors are released, `mt5_rciid_gmpc_s0`, `mt5_rcfz_gmpc_s0` and `mt5_rcfz_t2k_s0`
[`configs/student/<tag>/run.json`; E02 Section 9; R2.7]; their observation, network and action are identical
(Section 9.3 for how they were distilled):

| Item | Value |
|---|---|
| Observation (60-D) | qpos 9 \| qvel 9 \| tcp pose 7 \| previous action 8 \| object pose 7 \| goal pose 7 \| rel 9 (obj - tcp, goal - tcp, goal - obj) \| psi 4 |
| Tokens | qpos, qvel, tcp, action, obj+goal (`Linear(14 -> 128)`), psi (`Linear(4 -> 128)`, `psi: true`, `psi_token: true`); no token reads `rel` (`scene_kp = False`, `rel_token = False`), so d(action)/d(rel) = 0 |
| Network | `StudentTransformer`, d_model 128, 4 layers, 4 heads, 804,002 parameters; action mean read from the previous-action token; no temporal axis |
| Action (8-D) | `pd_joint_delta_pos`: a in [-1, 1]^8; seven arm joints `q_target = q + 0.1 a` (rad); gripper a_8 mapped to a mimicked finger-joint position target in [-0.01, 0.04] m, i.e. 0.015 + 0.025 a_8 [E02 C12; R2.7] |
| Control rate | 20 Hz |
| Perception (Table II records) | 64 query pixels chosen once at t = 0 on the object's simulator segmentation; each step the points are carried by the simulator object pose, projected, re-measured in depth (tolerance 0.01 m), unprojected and fitted by weighted Kabsch to the t = 0 PCA canonical frame (the simulator's per-step object correspondence); the executor receives no image, instruction or camera parameters as inputs [E03 A.5, A.6, B; KNOWN_ISSUES item 18] |
| Goal composition | once per episode at t = 0: goal points = dR cur + dt on the perceived t = 0 object points, then Kabsch to the canonical set; goal7 = absolute world position plus rotation in the same PCA convention, i.e. T_g relative to the initial object frame; frozen for the episode; the K rows of a bank are averaged (`goal_form = mean`, `goal_k = 4`); a non-finite row gives the identity [E03 C.6; `msppo/task_student_env.py`] |
| psi token | the 4-dimensional (k1, h) of Section 5.6: task-constant approach axis and carry height from the predicted trace; all three executors were distilled with `--psi --psi-token` and consume it [`run.json`; KNOWN_ISSUES item 17] |
| Language | none (`lang_dim = 0`); one set of weights for the five tasks |

## 7. Closed-loop evaluation (S40, S41, VM)

| Item | Value | Source |
|---|---|---|
| Harness | `msppo.multi_eval --run runs_rl/<executor> --episodes 256 --seed <SD> --ckpt final --only pickcube,liftpeg,peginsert,stack,pushcube --psi-bank <...> [--goal-delta <...>]` under `$RC` | S40, S41, VM |
| Checkpoint | `--ckpt final` loads `student.pt`; `student_best.pt` is never scored | E01 Section 1 |
| Planner goal (Table II rows) | `--goal-delta` with the banks of Section 5.5 (terminal step, `goal_step = None`) | S40, VM |
| Oracle Goal | no `--goal-delta`: the simulator goal through the same perception; reference of Fig. 5a and 0 mm point of Fig. 7 | S41, VM `ROW=oracle` |
| Metric | per-task `success_once` over 256 scenes, one rollout per scene; the five-task mean is printed as `success_once` in the JSON | `msppo/multi_eval.py`; `paper_results/table2/*.json` |
| Recorded in the JSON | `run`, `ckpt`, `episodes`, `eval_seed`, `train_seed`, `per_task`, `goal_source`, `psi_bank`, `env_patches` (wall, scene gate, camera, base pose), `per_env` (per-episode success, grasped, bumped, final error, steps) | `paper_results/table2/final_rciid_999.json` |
| Verification | `ROW=final\|k1\|k4\|k4pick\|oracle bash scripts/verify_main_table.sh <seed>` links the frozen banks of `banks/` into `results/`, runs `msppo.multi_eval` with the executor and banks of that row, and accepts each task within +/- 0.04 of the shipped record (N = 256 sampling noise; PegInsertionSide is not bitwise reproducible across runs) | VM; E00 Section D |
| Noise floor | repeated evaluation of one checkpoint: sd 0.018, differences below 0.036 (9 episodes) not distinguishable; training-seed sd about 0.04 | R3 |

Shipped records and the paper items they support [`paper_results/PROVENANCE.json`]:

| Paper item | Executor | Goal source | Records |
|---|---|---|---|
| Table II, Rigid Readout K = 1 | `mt5_rcfz_gmpc_s0` | `k1ransac` banks | `table2/rigid_k1_gmpc_{999,997,998}.json` |
| Table II, Rigid Readout K = 4 (LiftPegUpright, PegInsertionSide, StackCube, PushCube) | `mt5_rcfz_gmpc_s0` | `kmean_ransac` banks | `table2/rigid_k4_gmpc_{999,997,998}.json` (the PickCube entries, pooled 33.07, are not the printed cell) |
| Table II, Rigid Readout K = 4, PickCube cell (46.48) | `mt5_rcfz_t2k_s0` | `pickcube_goals_mix5_t2k_n3000_999_k4ransac.npz` | `table2/rigid_k4_pickcube_t2k_s0_999.json` (seed 999 only) |
| Table II, Entity-Level Goal Readout (LiftPegUpright, PegInsertionSide, StackCube, PushCube) | `mt5_rciid_gmpc_s0` | Section 5.5 banks | `table2/final_rciid_{999,997,998}.json` (their PickCube entries use `gmappeakNC`: 78.12 / 72.27 / 71.88) |
| Table II, Entity-Level Goal Readout, PickCube | `mt5_rciid_gmpc_s0` | `sam2mk6` bank | `table2/final_pickcube_sam2_rciid_{999,997,998}.json` |
| Oracle Goal references (Fig. 5a) | `mt5_rcfz_gmpc_s0` | simulator goal | `table2/oracle_gmpc_{999,997,998}.json` |
| Fig. 7 (Oracle Goal translated by 0 to 120 mm) | `mt5_rcfz_gmpc_s0` | simulator goal + translation | `fig7_perturbation/trans_m{0,5,10,20,30,50,80,120}_{999,997}.json`; verification `evidence/06_verify_perturbation.out.json`; the perturbation patch is not among the released modules |
| Not in the paper: final routing with the episode-fixed executor | `mt5_rcfz_gmpc_s0` | Section 5.5 banks with PickCube `gmappeakNC` | `supplementary/final_routing_gmpc_{999,997,998}.json` (named `row4_mt5_rcfz_gmpc_s0_nc_<seed>.json` in `evidence/NUMBERS_PROVENANCE.csv`) |

Pooled (three seeds, 768 episodes) and per-seed values of every row are tabulated in the README ("Reproducing
Table II"); printed and recorded values differ by at most 0.08 percentage points except for the PickCube K = 4 cell
[KNOWN_ISSUES items 15, 20].

Row definitions used across the evidence files [R3]: row 1 official PPO; row 2 teacher; row 3 the Oracle Goal; row 4
the planner goal with the simulator's per-step correspondence (the Table II rows); row 5 de-privileged goal chain;
row 6 row 4 with CoTracker3-online tracking (not part of this release, KNOWN_ISSUES item 11). The release reproduces
rows 3 and 4.

## 8. Training data (S50, S51)

| Item | Value | Source |
|---|---|---|
| Demonstrations | 200 official ManiSkill demonstrations per task (`N=200`), replayed by driving `env_states`; PickCube and LiftPegUpright from the `rl` packages, the others from `motionplanning`; all 200 succeed at the end | S50; `msgen/tasks.py`; E01 Section 9.1 |
| Views | three per demonstration: v0 nominal; v1 eye + (0.02, -0.02, 0.01), target + (0.04, 0.03, 0); v2 eye + (-0.02, 0.02, -0.01), target + (-0.04, -0.03, 0) (metres), i.e. eye offsets +/- (0.02, 0.02, 0.01) m and target offsets +/- (0.04, 0.03, 0) m, about +/- 2 to 3 degrees of re-aim | S50 `VIEWS` and header |
| Backdrop and marker | `MSGEN_WALL=1` for all five tasks (asserted); `--show-goal` for PickCube only | S50 |
| Frames per clip | up to 64 (`MAX_FRAMES`) | `msgen/tasks.py` |
| Trace labels | `msgen.labels --stride 2 --min-future 8 --time-mode arclen --n-obj 16`: 33-step traces (index 0 + 32 future) for 400 queries, 16 on the target from the simulator segmentation; static pixels keep their point, unowned pixels are masked | S50; `msgen/labels.py` |
| Index datasets | `data/ds/realcam_n2400` (pickcube, stack, peg, liftpeg; 2,400 clips) and `data/ds/realcam_t2k_n3000` (+ pushcube; 3,000 clips), symlink farms | S50; E01 Section 8 |
| Entity labels | `msgen.labels_t2k --overwrite` per view: keyframes (onset > 5 mm, turns > 35 degrees, K <= 4), screw-decomposed twists, contact descriptor at onset, structure frame (`STRUCT` = cubeB / box_with_hole_0 / goal_site / table / goal_region), relative terminal pose, object membership of the 400 queries (simulator segmentation), goal map | S51; E02 Section 2.3 |
| Goal-map label | the released `mix5_t2k_gmap` was trained on v1 labels (fraction of t = 0 depth pixels within 0.06 m of the terminal object position, per 16 x 16 patch); the shipped `msgen/labels_t2k.py` writes v2 (Gaussian, sigma 0.02 m, cut at 0.06 m); a retrain is therefore not bitwise comparable, and the v1 label files are not preserved | S52 header; E02 C3, Section 2.4 |
| Leakage | demonstration seeds are disjoint from 990 to 999; no exact initial-pose duplicate between any training clip and any bank scene; the planners have no held-out split other than the banks | E01 Section 9 |

## 9. Training runs

### 9.1 Planner fine-tuning (S52) [S52; `configs/planner/*/run.json`, `patches.json`; E01 Section 8; E02 Sections 1 to 3]

| Run | Warm start | Dataset | Epochs | Environment flags | Trainable parameters |
|---|---|---|---|---|---|
| `mix4_realcam_n2400` | upstream Generalist checkpoint | `realcam_n2400` (48,516 samples per epoch) | 15 | none recorded | 75,486,238 |
| `mix5_t2k_n3000` | `mix4` final | `realcam_t2k_n3000` (63,306 samples per epoch) | 8 | `MSGEN_T2K=1 MSGEN_T2K_W=0.3` | 78,535,810 |
| `mix5_t2k_gmap` | `mix4` final (not `mix5_t2k_n3000`) | `realcam_t2k_n3000` | 8 | `MSGEN_T2K=1 MSGEN_T2K_GMAP=1 MSGEN_T2K_W=0.3 MSGEN_KEEP_ALL_CKPT=1` | 78,536,579 |

Common settings: batch 8, one AdamW group at lr 1.5e-4 (`lr_backbone` is never used), weight decay 0.05, no
scheduler, gradient clipping at norm 1.0, mixed precision, `torch.compile`; every episode goes to the training
split and the logged validation loss is on one training clip; the configuration seed is 1337 (`eval.yaml`, not
logged per run). Flow loss: targets normalised per axis by (-0.05, -0.05, -0.04) / (0.05, 0.05, 0.04), interpolant
x_t = (1 - t) x_0 + t eps, MSE on the masked velocity. Total loss `L = L_flow + 0.3 (sum_k W_k l_k + 0.3 L_map)`,
i.e. the paper's `L = L_flow + lambda_ent L_ent + lambda_map L_map` with the per-part weights of `msgen/patch_t2k.py`;
the entity-branch and map-branch losses update the readout, the fusion layer and the per-encoder layer norms and
depth stem, not the flow decoder [E02 Sections 1.3, 2, 3]. Git commit at training: `ba7de4d` (private repository)
for both `mix5` runs; **not recorded** for `mix4`. Wall-clock estimates in S52 (about 10 h, 6 h, 6 h on one RTX
5090) are script comments, not logged measurements. S52 passes no `MSGEN_WALL` or camera variable to the planner
interpreter, in agreement with the record: the `mix5` `patches.json` files list `wall: false` and the `mix4` run
recorded no `MSGEN_*` variable [S52; `configs/planner/*/patches.json`]. S52 builds the planner from the Hub encoders
(Section 4).

### 9.2 Privileged teachers (S60) [S60; `configs/teacher/*/run.json`; `msppo/kp_teacher.py`]

| Tag | Task | Head | Noise-v2 goal scale | Total steps | Seed | Robot |
|---|---|---|---|---|---|---|
| `pc_v9_nz_s0` | pickcube | paired | 0.4 | 12,000,000 | 0 | panda |
| `lp_v9_nz_s0` | liftpeg | paired | 1.0 | 12,000,000 | 0 | panda |
| `pi_v9_frame4_s0` | peginsert | paired | 1.0 | 8,000,000 | 0 | panda |
| `sc_v9_nz03b_s0` | stack | paired | 0.3 | 12,000,000 | 0 | panda_wristcam |
| `push_v9_nz_s0` | pushcube | paired | 1.0 | 12,000,000 | 0 | panda |

Each teacher is a privileged keypoint PPO policy (`msppo/ppo.py`, `msppo/ppe.py`) with the observation-noise
injector v2 (`sig_obs` true) and the privileged guidance variable psi from `patch_frame` (`MSPPO_FRAME_TASK`) or
`patch_stack_frame` (`MSPPO_STACK_FRAME=1`); every flag is regenerated from the shipped `run.json`, and the `MSPPO_*`
environment recorded in `configs/teacher/<tag>/patches.json` is exported as well: `pc_v9_nz_s0` `MSPPO_FRAME_WPUSH=0.25`;
`pi_v9_frame4_s0` `MSPPO_FRAME_WH=0 MSPPO_FRAME_WPUSH=0 MSPPO_FRAME_K1_BASE=-0.28,-0.56,-0.78` (and
`MSPPO_EVAL_TASK=peginsert`, which no shipped module reads); `lp_v9_nz_s0` only the frame-task selector that S60
sets in any case; `sc_v9_nz03b_s0` an empty environment block; `push_v9_nz_s0` has no `patches.json` (its launcher
set `MSPPO_FRAME_TASK=pushcube` only) [S60; `configs/teacher/*/patches.json`; KNOWN_ISSUES item 3]. The released
weights were warm-started from an earlier lineage (`init_from`) that is not shipped; S60 trains from scratch with
the same recipe and step budget, so the same range of success rates is expected, not the same weights [S60 header].

### 9.3 Distillation of the Pose-Native Executor (S70) [S70; `configs/student/*/run.json`; E01 Sections 5, 7, 9.2; E02 Section 9; `evidence/01_checkpoint_registry.csv`]

| Item | Value |
|---|---|
| Algorithm | DAgger with beta = 0 (the executor acts, the frozen teacher labels every visited state); loss `L1(a_student, a_teacher) + 0.1 (L1 qpos' + L1 qvel')`; Adam lr 3e-4; 2 epochs x 4 minibatches per update |
| Budget | 40,000 iterations, 320 environments (64 per task), seed 0; identical for the three released executors |
| Flags | `--tasks pickcube,liftpeg,peginsert,stack,pushcube --lang none --no-scene --psi --psi-token --goal-err-rel <relbanks> --goal-err-scale uniform` |
| Teacher observation | true state; the goal is shifted by the executor's perceived relative error (`msppo/teacher_perc.py`) |
| Goal perturbations | drawn from the relbanks (the measured planner error distribution of the goal source each task is deployed with), expanded in the current scene's object-to-goal axis (`peg_relbank.expand`), each draw additionally scaled by a ~ U(0, 1) per episode (`goal_err_scale = uniform`: frac -> 1 + a (frac - 1), lateral -> a lateral, dq -> slerp(I, dq, a)). Temporal process (`INJECT`): `iid`, one draw redrawn i.i.d. at every control step from a pool of 64 expanded draws with the same per-episode marginal (`msppo/patch_iid_inject.py`, `MSPPO_IID_INJECT=1`); `frozen`, one draw per episode drawn at reset and held until the environment resets (`msppo/patch_frozen_inject.py`, `MSPPO_FROZEN_INJECT=1`) |
| Released executors | `mt5_rciid_gmpc_s0` (2026-09-13): `INJECT=iid`, relbanks stack `gmapdcc`, pickcube `gmappeak`, others `t2kdcc`; the executor of the Entity-Level Goal Readout row. `mt5_rcfz_gmpc_s0` (2026-09-07): `INJECT=frozen`, the same relbanks; the executor of the Rigid Readout rows, the Oracle Goal references and Fig. 7. `mt5_rcfz_t2k_s0` (2026-09-01): `INJECT=frozen`, `run.json` records `t2kdcc` relbanks for all five tasks (`pickcube_relbank_realcam_t2kdcc.npz` and `stack_relbank_realcam_t2kdcc.npz` are not shipped); the executor of the printed PickCube K = 4 cell only |
| Relbank per task (shipped) | stack `stack_relbank_realcam_gmapdcc.npz` (1,767 rows, fit seeds 990 to 996); pickcube `pickcube_relbank_realcam_gmappeak.npz` (1,792 rows, 990 to 996); liftpeg `liftpeg_relbank_realcam_t2kdcc.npz` (2,048 rows, 990 to 997); peginsert `peginsert_relbank_realcam_t2kdcc.npz` (2,042 rows, 990 to 997); pushcube `pushcube_relbank_realcam_t2kdcc.npz` (1,024 rows, 990 to 993) [E01 Section 9.2; shipped `banks/`] |
| Relbank content | scene-relative error components (frac along the object-to-goal axis, lat_v, lat_w, dq); no scene identity or absolute goal enters distillation; seed 997 therefore overlaps the fit set of the liftpeg and peginsert error families as a distribution, not as scenes |

The fit seeds are therefore 990 to 996 for StackCube and PickCube, 990 to 997 for LiftPegUpright and
PegInsertionSide, and 990 to 993 for PushCube [E01 Section 9.2; S70 header; KNOWN_ISSUES item 10]. The relbanks are
shipped; regenerating them requires running the deployed goal chain on fit-seed banks and `msppo.peg_relbank
export`, which no released script automates. S70 reproduces the recipe of `mt5_rciid_gmpc_s0` (default) or
`mt5_rcfz_gmpc_s0`; the recipe of `mt5_rcfz_t2k_s0` cannot be re-run from the shipped relbanks.

## 10. Historical deviations between the shipped artefacts and the released scripts

(a) **PushCube training clips of the released planners were rendered without the wall.** `mix5_t2k_n3000` and
`mix5_t2k_gmap` were trained on `data/ds/realcam_t2k_n3000`, whose PushCube clips (`realcam_pushcube_v{0,1,2}`,
collected 2026-08-31) predate the extension of `patch_wall` to `PushCubeEnv` (wrap added 2026-09-01, committed
`fce09ba` of the private repository on 2026-09-08); the walled `realcam_pushcube_v*w` variants that exist in the private workspace were not
used by these checkpoints [E01 Sections 0 and 8; `msgen/patch_wall.py` comment; `msppo/pushcube_kp_env.py`
docstring]. The public S50 renders all five tasks with the wall, so a re-collected dataset differs from the
released checkpoints' training data on PushCube, and a retrained `mix5` is not expected to match them on that task.

(b) **The PushCube evaluation banks used by the paper were rendered without the wall.**
`realcam_pushcube_{999,997,998}_obj` were rendered on 2026-08-31 with `MSGEN_WALL=1` set, but before the patch
covered `PushCubeEnv`; the render logs contain no `[wall] PushCubeEnv` line, and the seed-999 frames show a median
of 9,562 no-hit pixels against 0 for PickCube. The planner therefore saw wall-less PushCube images, while the
executor evaluations (2026-09-07 onward) ran with the wall [E00 Section E; E01 Section 2; E03 A.1]. The shipped
`banks/pushcube_goals_mix5_t2k_n3000_<seed>_{t2kpos,k1ransac,kmean_ransac}.npz` derive from those renders, so
`verify_main_table.sh` reproduces the paper cells; the public S10 renders PushCube with the wall, so a regenerated
PushCube bank and its goals are a different planner input.

(c) **PegInsertionSide at seed 998 consumed the Rigid Readout rotation.** The shipped
`banks/peginsert_goals_mix5_t2k_n3000_998_t2kpos.npz` carries `prov_t2kpos = "head position, solver rotation (fit
seed)"` and its rotation equals the `mix4` K = 4 RANSAC solve, whereas the 999 and 997 files are the full entity-branch
pose (`prov_t2k` only). The released S30 always copies the full entity-branch pose for PegInsertionSide. Both the
Entity-Level Goal Readout cell (`final_rciid_998.json`) and the supplementary final-routing cell at 998 used the
solve-rotation bank [E02 C6; shipped `banks/`]. More generally, the 998 banks of every task were produced by the
fit-seed builder (`prov_t2kpos` "fit seed") and the 999/997 banks by the production builder; the audit found this to
be a label difference and did not test for a numeric one [E01 Section 3].

(d) **Rigid Readout source for PushCube.** The paper text describes the four-task planner sampled with K = 4;
the PushCube rotation source and its K = 1 / K = 4 banks come from the `mix5_t2k_n3000` samples, which is what CFG
`ROT_TAG` encodes [E02 C11].

(e) **PickCube bank of Table I versus Table II.** The PickCube goal error of Table I is measured on the `gmappeakNC`
bank, the PickCube success of the Entity-Level Goal Readout row on the `sam2mk6` bank (Section 5.4); the released
S30 writes `gmappeakNC`, S31 writes `sam2mk6` [KNOWN_ISSUES item 16].

## 11. Not recorded

- Which of the three instructions each evaluation scene received at prediction time [E03 A.2].
- Git commit of the `mix4_realcam_n2400` training run; the source text of `patch_t2k.py` and `labels_t2k.py` at the
  time of the `mix5` trainings (first committed 2026-09-10); the v1 goal-map label files [E01 Section 8; E02 C3].
- Exact code version of the executors' distillation runs (no git field in any `run.json`; `mt5_rcfz_gmpc_s0` trained on
  2026-09-07 on an uncommitted working tree before private commit `a2ccaad`) and of the seed-999 final-routing
  evaluation; per-evaluation weight hashes [E01 Sections 0, 1; KNOWN_ISSUES item 4].
- `prov_*` keys of the `mix4` K = 4 predictions at seeds 999 and 997 [E01 Section 3]. Their launchers were located after the audit (`<private-repo>/experiments/20260827_confhead/f5_test999.sh` and `f4_labels.sh`, logs `[patch_all] active: seed`), which fixes the sampler of that family as the native one (Section 4).
- Per-episode scene seeds of distillation and teacher training [E01 Section 9.3].
- Wall-clock time of a K = 4 planner call and of the native-sampler `mix4` prediction; latency of the released executors
  (1.4 ms per step is from an older executor of the same size); any record behind the 1.27 s and 5.60 GB of the paper's
  Table III [E00 cost table; KNOWN_ISSUES item 19].
