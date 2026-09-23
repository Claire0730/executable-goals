# Reproducing the reported numbers

Long-form companion to the README: which record, bank and executor produced every cell of the paper, and how to
re-run each stage. Task order throughout: PickCube / LiftPegUpright / PegInsertionSide / StackCube / PushCube.
See also [GLOSSARY.md](GLOSSARY.md) (paper terms to repository identifiers), [PROTOCOL.md](PROTOCOL.md)
(the evaluation and data protocol as implemented) and [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## Reproducing Table II

`verify_main_table.sh` re-runs the closed-loop evaluation of one row of the shared-executor block of Table II, for one evaluation seed, with the executor that produced that row and the frozen goal banks in `banks/` (no planner run, no planner checkpoint, no TraceGen checkout, no Hub session and no SAM 2 needed; only `checkpoints/student/`, the `maniskill` environment and a GPU). It compares each task with the shipped record in `paper_results/table2/` within a +-0.04 band (N = 256 sampling noise; PegInsertionSide is not bitwise reproducible across runs). One row and seed takes about 10 minutes on one GPU (5 tasks x 256 episodes).

```bash
ROW=final  bash scripts/verify_main_table.sh 999   # Entity-Level Goal Readout row; also 997, 998
ROW=k1     bash scripts/verify_main_table.sh 999   # Rigid Readout K=1
ROW=k4     bash scripts/verify_main_table.sh 999   # Rigid Readout K=4 (mt5_rcfz_gmpc_s0, all five tasks)
ROW=k4pick bash scripts/verify_main_table.sh 999   # the PickCube cell of the K=4 row as printed (mt5_rcfz_t2k_s0; seed 999 only)
ROW=oracle bash scripts/verify_main_table.sh 999   # Oracle Goal references (Fig. 5a)
```

Task order in every table below: PickCube / LiftPegUpright / PegInsertionSide / StackCube / PushCube; values are success rates in percent. "Printed" is the value in the paper; "recorded" is the mean over the three evaluation seeds 999, 997, 998 of the `per_task` field of the shipped records (256 scenes per seed, 768 episodes per task). Where the printed and the recorded values differ by at most 0.08 percentage points, the difference is rounding of the printed values.

| Table II row | Printed | Recorded (pooled) | Executor | Goal banks (`banks/`) | Records (`paper_results/table2/`) | Verify |
|---|---|---|---|---|---|---|
| Rigid Readout K = 1 | 29.80 / 70.18 / 20.05 / 52.47 / 99.74 (mean 54.45) | 29.82 / 70.18 / 20.05 / 52.47 / 99.74 (mean 54.45) | `mt5_rcfz_gmpc_s0` | `<task>_goals_mix4_realcam_n2400_<seed>_k1ransac.npz`; PushCube `pushcube_goals_mix5_t2k_n3000_<seed>_k1ransac.npz` | `rigid_k1_gmpc_<seed>.json` | `ROW=k1` |
| Rigid Readout K = 4 | 46.48 / 76.95 / 21.88 / 54.82 / 99.22 (mean 59.87) | 33.07 / 76.95 / 21.88 / 54.82 / 99.22 (mean 57.19) with `mt5_rcfz_gmpc_s0`; the printed PickCube cell 46.48 is the record `rigid_k4_pickcube_t2k_s0_999.json` | `mt5_rcfz_gmpc_s0`; PickCube cell: `mt5_rcfz_t2k_s0`, seed 999 only | `<task>_goals_mix4_realcam_n2400_<seed>_kmean_ransac.npz`; PushCube `pushcube_goals_mix5_t2k_n3000_<seed>_kmean_ransac.npz`; PickCube cell: `pickcube_goals_mix5_t2k_n3000_999_k4ransac.npz` | `rigid_k4_gmpc_<seed>.json`, `rigid_k4_pickcube_t2k_s0_999.json` | `ROW=k4`, `ROW=k4pick` |
| Entity-Level Goal Readout | 81.50 / 98.35 / 32.84 / 86.54 / 99.20 (mean 79.69) | 81.51 / 98.31 / 32.81 / 86.46 / 99.22 (mean 79.66) | `mt5_rciid_gmpc_s0` | PickCube `pickcube_goals_mix5_t2k_gmap_<seed>_sam2mk6.npz`; StackCube `stack_goals_mix5_t2k_gmap_<seed>_gmapdcc.npz`; others `<task>_goals_mix5_t2k_n3000_<seed>_t2kpos.npz` | `final_pickcube_sam2_rciid_<seed>.json` (PickCube), `final_rciid_<seed>.json` (other four tasks) | `ROW=final` |
| Oracle Goal (Fig. 5a) | 94.66 / 97.27 / 36.72 / 87.11 / 99.74 | 94.66 / 97.27 / 36.72 / 87.11 / 99.74 | `mt5_rcfz_gmpc_s0` | simulator goal (`true goal` in the records) | `oracle_gmpc_<seed>.json` | `ROW=oracle` |

Per-seed values of every row (`per_task` of the records, in percent; every value is a multiple of 100/256):

| Row | Seed 999 | Seed 997 | Seed 998 |
|---|---|---|---|
| Rigid Readout K = 1 | 30.86 / 68.36 / 17.58 / 52.34 / 100.00 | 32.42 / 71.88 / 18.75 / 51.56 / 99.22 | 26.17 / 70.31 / 23.83 / 53.52 / 100.00 |
| Rigid Readout K = 4 (`mt5_rcfz_gmpc_s0`) | 30.47 / 78.52 / 18.36 / 53.91 / 100.00 | 39.06 / 76.95 / 23.83 / 53.91 / 99.22 | 29.69 / 75.39 / 23.44 / 56.64 / 98.44 |
| Rigid Readout K = 4, PickCube cell (`mt5_rcfz_t2k_s0`) | 46.48 (119 of 256) | not evaluated | not evaluated |
| Entity-Level Goal Readout | 84.38 / 98.05 / 36.72 / 88.28 / 99.61 | 80.86 / 98.44 / 29.30 / 86.33 / 99.22 | 79.30 / 98.44 / 32.42 / 84.77 / 98.83 |
| Oracle Goal | 93.75 / 96.88 / 37.11 / 88.28 / 100.00 | 94.53 / 98.44 / 32.81 / 86.72 / 99.61 | 95.70 / 96.48 / 40.23 / 86.33 / 99.61 |

The `final_rciid_<seed>.json` records also contain a PickCube entry evaluated with the `gmappeakNC` bank (78.12 / 72.27 / 71.88; pooled 74.09), which is not the printed cell; `ROW=final` compares PickCube with `final_pickcube_sam2_rciid_<seed>.json`. `paper_results/supplementary/final_routing_gmpc_<seed>.json` holds the same routing (PickCube `gmappeakNC`) evaluated with `mt5_rcfz_gmpc_s0`: 71.48 / 96.88 / 31.64 / 81.25 / 100.00 (999), 71.09 / 97.27 / 32.42 / 86.72 / 100.00 (997), 68.36 / 94.92 / 26.95 / 80.86 / 100.00 (998), pooled 70.31 / 96.35 / 30.34 / 82.94 / 100.00; these values are not in the paper. `evidence/NUMBERS_PROVENANCE.csv` refers to them under their former names `row4_mt5_rcfz_gmpc_s0_nc_<seed>.json`, and to `oracle_gmpc_999.json` as `row3_mt5_rcfz_gmpc_s0_999.json`.

### Which executor produced which row

Three Pose-Native Executors are released under `checkpoints/student/`. All three were distilled with `msppo.multi_distill` from the same five teachers with the same recipe (five tasks, 804,002 parameters, `d_model` 128, 4 layers, 4 heads, no scene channel, psi token, seed 0, 40,000 iterations, 320 environments, `--goal-err-rel` relbanks with `--goal-err-scale uniform`). `mt5_rciid_gmpc_s0` (2026-09-13) produced every cell of the Entity-Level Goal Readout row; it differs from `mt5_rcfz_gmpc_s0` only in the temporal process of the injected goal error during distillation: the error is redrawn i.i.d. at every control step (`msppo/patch_iid_inject.py`, `MSPPO_IID_INJECT=1`) instead of being drawn once per episode and held (`msppo/patch_frozen_inject.py`, `MSPPO_FROZEN_INJECT=1`); both use the same relbanks (StackCube `gmapdcc`, PickCube `gmappeak`, the other tasks `t2kdcc`). `mt5_rcfz_gmpc_s0` (2026-09-07) produced the Rigid Readout K = 1 row, the K = 4 row for LiftPegUpright, PegInsertionSide, StackCube and PushCube, the Oracle Goal references and the Fig. 7 curves. `mt5_rcfz_t2k_s0` (2026-09-01) is an earlier executor with episode-fixed injection whose `run.json` records `t2kdcc` relbanks for all five tasks (its PickCube and StackCube relbanks are not shipped); it produced the PickCube cell of the K = 4 row only, from the K = 4 RANSAC bank of the `mix5_t2k_n3000` checkpoint at seed 999 (`rigid_k4_pickcube_t2k_s0_999.json`, whose LiftPegUpright, PegInsertionSide and StackCube entries with the same banks, 88.67 / 26.95 / 66.02, are not used in the paper). `scripts/config.sh` defaults `STUDENT_TAG` to `mt5_rciid_gmpc_s0`; `verify_main_table.sh` selects the executor per row.

## Reproducing Table I and Fig. 6

Table I (terminal goal position error, mm, mean +- sd) and Fig. 6 (median / 90th percentile) are goal-bank statistics at evaluation seed 999 (256 scenes; the four unsolved StackCube scenes of the Rigid banks are excluded, `n_solved` 252). No script recomputes them; the statistics are shipped in `evidence/`:

| Table I row | Printed (PickCube / PegInsertionSide / StackCube) | `evidence/04_goal_summary.json` keys (`mean_all_finite` +- `sd_all_finite`) | `evidence/goal_table.json` keys (Fig. 6 `med` / `p90`) |
|---|---|---|---|
| Rigid Readout K = 1 | 41.1 +- 21.7 / 42.6 +- 31.6 / 28.9 +- 38.7 | `rigid_K1_ransac\|pickcube\|999` (41.066 +- 21.680), `rigid_K1_ransac\|peginsert\|999` (42.574 +- 31.578), `rigid_K1_ransac\|stack\|999` (28.859 +- 38.693) | `999\|<task>\|solve_k1`: PegInsertionSide 33.80 / 87.37, StackCube 21.95 / 44.85 |
| Rigid Readout K = 4 | 37.2 +- 19.3 / 40.8 +- 33.0 / 25.4 +- 38.8 | `rigid_K4_ransac\|pickcube\|999` (37.164 +- 19.299), `rigid_K4_ransac\|peginsert\|999` (40.833 +- 32.979), `rigid_K4_ransac\|stack\|999` (25.392 +- 38.796) | `999\|<task>\|solve_kmean`: PegInsertionSide 30.39 / 83.85, StackCube 18.40 / 41.77 |
| Entity-Level Goal Readout | 31.5 +- 20.1 / 29.4 +- 16.4 / 12.8 +- 34.0 | `final_pick_gmappeakNC\|pickcube\|999` (31.532 +- 20.112), `t2k_head_full\|peginsert\|999` = `t2k_head_pos\|peginsert\|999` (29.445 +- 16.405), `final_stack_gmapdcc\|stack\|999` (12.820 +- 34.045) | `999\|pickcube\|gmappeakNC` 24.51 / 51.59; `999\|peginsert\|t2k_full` = `999\|peginsert\|t2kpos` 26.83 / 47.63; `999\|stack\|gmapdcc` 6.48 / 17.09 |

The "No geo." curve of Fig. 6 (PegInsertionSide 148.9 / 198.2, StackCube 128.1 / 224.1) is `abl_nodepth_999|<task>|999` in `04_goal_summary.json` (`median_all` / `p90_all`) and variant `abl_nodepth_999` in `04_goal_per_scene.csv.gz`; the goal bank of that ablation is not shipped. The PickCube error of Table I is measured on the `gmappeakNC` bank, whereas the PickCube success of Table II is measured on the `sam2mk6` bank; both banks are shipped for all three seeds. `evidence/fig6_position_errors.json` holds the per-scene position errors of the K = 4 Rigid Readout (`solve`), the entity branch (`t2k_head`) and the final goal (`final`) per task, pooled over the three seeds (768 scenes) with the per-task tolerance `tol_mm`; its pooled quantiles differ from the seed-999 values printed in Fig. 6 (for example PegInsertionSide K = 4 pooled median 32.3 mm). The per-scene rows behind every statistic (bank, predicted and reference goal, error, tolerance criterion) are in `04_goal_per_scene.csv.gz`.

## Fig. 7

`paper_results/fig7_perturbation/trans_m<mm>_<seed>.json` (mm in 0, 5, 10, 20, 30, 50, 80, 120; seeds 999 and 997) are the records of the Oracle Goal translated by the stated distance, evaluated with `mt5_rcfz_gmpc_s0` (two seeds x 256 = 512 episodes per point; PegInsertionSide with 10 mm clearance). The paper shows StackCube and PegInsertionSide; the records contain all five tasks. The 0 mm records coincide with the Oracle Goal records of the same seeds. The perturbation driver (`patch_goal_perturb`) is not part of the released module set, so these curves are shipped as records and are not regenerated by any script; `evidence/06_verify_perturbation.out.json` is the verification record of their reconstruction. Values in percent (seed 999 / seed 997):

| Translation (mm) | 0 | 5 | 10 | 20 | 30 | 50 | 80 | 120 |
|---|---|---|---|---|---|---|---|---|
| StackCube | 88.28 / 86.72 | 86.33 / 85.94 | 84.38 / 85.55 | 71.09 / 65.62 | 54.30 / 55.86 | 16.41 / 18.75 | 3.12 / 5.08 | 0.39 / 1.95 |
| PegInsertionSide | 36.72 / 32.81 | 35.16 / 35.16 | 34.77 / 35.55 | 30.08 / 26.17 | 25.39 / 24.61 | 17.19 / 12.11 | 4.69 / 3.12 | 2.73 / 0.78 |

## Running the full chain

Requires the installation steps 1-4 (including the TraceGen checkout and an authenticated Hub session for the frozen encoders). Every stage is idempotent per task and seed and skips outputs that already exist; the seed argument defaults to 999.

```bash
bash scripts/10_render_banks.sh 999   # SAPIEN render of N_EP=256 scenes per task under the camera + wall protocol, then 16 object-aware queries; GPU, minutes per task
bash scripts/20_predict.sh 999        # planner predictions: mix4 K=4 samples (seeds 1234-1237) with the native sampler; entity branch, map branch (seed 1234) and the PushCube K=4 samples (mix5_t2k_n3000) with MSGEN_STEPS=20 MSGEN_DT=fix; fetches the encoders from the Hub
bash scripts/30_goals.sh 999          # K-mean -> Rigid Readout (RANSAC) -> entity-branch decode -> per-task goal banks (incl. gmappeakNC, gmapdcc) and psi banks in results/
PV=/path/to/sam2/python bash scripts/31_goals_sam2marker.sh 999   # optional: PickCube sam2mk6 bank (SAM 2 marker localiser prompted at the map peak); otherwise set PICKCUBE_ROUTE=gmappeakNC or link the frozen bank from banks/
bash scripts/40_eval_row4.sh 999      # closed loop with the composed goals (Entity-Level Goal Readout row); executor $STUDENT_TAG (default mt5_rciid_gmpc_s0); ~10 min on one GPU
bash scripts/41_eval_row3.sh 999      # closed loop with the Oracle Goal; the paper's references used STUDENT_TAG=mt5_rcfz_gmpc_s0
```

`40_eval_row4.sh` and `41_eval_row3.sh` write `results/row{4,3}_<STUDENT_TAG>_<seed>.json` with the same schema as `paper_results/`. A different executor is evaluated with `STUDENT_TAG=<tag>` (one of `STUDENT_TAGS` in `scripts/config.sh`) or `STUDENT_RUN=runs_rl/<tag>`. A different planner is evaluated by overriding the checkpoint path and its tag together (`CK_MIX4` with `TAG_MIX4`, `CK_HEAD` with `TAG_HEAD`, `CK_GMAP` with `TAG_GMAP`): the scripts and `tools/*.py` name their inputs and outputs by `TAG_*`.

## Training

Planner data and fine-tuning (`trace_gen` env for `52`; the rest `maniskill`):

```bash
for e in PickCube-v1 StackCube-v1 PegInsertionSide-v1 LiftPegUpright-v1 PushCube-v1; do $PM -m mani_skill.utils.download_demo $e; done
N=200 bash scripts/50_collect_data.sh   # CPU only: replays 200 demos x 3 camera views x 5 tasks = 3,000 clips (~50 GB, physx_cpu ~2.3 s/clip), labels dense traces with 16 object-aware queries, builds data/ds/realcam_n2400 and data/ds/realcam_t2k_n3000
bash scripts/51_label_t2k.sh            # CPU only: entity-level labels of the Entity-Level Goal Readout (entity branch and map branch; samples/<stem>_t2k.npz)
bash scripts/52_train_planner.sh all    # mix4 (15 epochs, from the Generalist, ~10 h) -> mix5_t2k_n3000 (8 epochs, ~6 h) -> mix5_t2k_gmap (8 epochs, ~6 h); one RTX 5090, waits for 16,000 MiB free VRAM
```

Stages of `52` can be run separately (`mix4|head|gmap`); a stage whose final checkpoint exists is skipped, and the entity-branch and map-branch stages warm-start from your own `mix4` if present, else from the released one. Training uses batch 8, decoder learning rate 1.5e-4, and for the two readout checkpoints `MSGEN_T2K=1 MSGEN_T2K_W=0.3` (plus `MSGEN_T2K_GMAP=1`). Neither `52_train_planner.sh` nor `20_predict.sh` sets `MSGEN_WALL` or the camera variables: these affect rendering only (`10_render_banks.sh`, `50_collect_data.sh`), and the production planner processes did not have them either (the `mix5` planner `patches.json` files record `wall: false`; the `mix4` run recorded no `MSGEN_*` variable). `52` builds the planner from the Hub encoders and therefore needs the authenticated Hub session of installation step 3. The released `mix5_t2k_gmap` was trained on goal-map labels v1 (6 cm hard ball); `msgen.labels_t2k` now writes v2 (Gaussian sigma 2 cm cut at 6 cm), so a retrain is not bitwise comparable (KNOWN_ISSUES.md item 2).

Teachers and executor (`maniskill` env):

```bash
bash scripts/60_train_teachers.sh                  # five single-task privileged PPO teachers; every flag is generated from configs/teacher/<tag>/run.json and the recorded MSPPO_* environment is exported from configs/teacher/<tag>/patches.json; DRY=1 prints the commands
INJECT=iid    TAG=my_student bash scripts/70_distill.sh   # DAgger distillation into the Pose-Native Executor with per-step i.i.d. goal-error injection (msppo/patch_iid_inject.py) = the recipe of mt5_rciid_gmpc_s0 (default)
INJECT=frozen TAG=my_student bash scripts/70_distill.sh   # episode-fixed injection (msppo/patch_frozen_inject.py) = the recipe of mt5_rcfz_gmpc_s0
STUDENT_RUN=runs_rl/my_student bash scripts/40_eval_row4.sh 999
```

`70_distill.sh` uses 40,000 iterations, 320 environments, no scene channel, the psi token, and goal perturbations drawn from the relbanks in `banks/` (the measured error distribution of the goal source each task is deployed with: StackCube `gmapdcc`, PickCube `gmappeak`, the other tasks `t2kdcc`; each draw scaled by U(0,1)); about 6 h on one RTX 5090. The relbanks were fitted on seeds 990-996 (StackCube, PickCube), 990-997 (LiftPegUpright, PegInsertionSide) and 990-993 (PushCube) (KNOWN_ISSUES.md item 10) and can be regenerated on your own banks with `msppo.peg_relbank export`. The released teachers were warm-started from an earlier lineage that is not shipped (`init_from` in their `run.json`); `60_train_teachers.sh` trains from scratch with the same recipe and step budget, so the same range of results is expected, not the same weights. Besides the flags generated from `run.json`, it exports the `MSPPO_*` environment recorded in `configs/teacher/<tag>/patches.json`: `MSPPO_FRAME_WPUSH=0.25` for `pc_v9_nz_s0`; `MSPPO_FRAME_WH=0 MSPPO_FRAME_WPUSH=0 MSPPO_FRAME_K1_BASE=-0.28,-0.56,-0.78` for `pi_v9_frame4_s0`; no reward-weight override for `lp_v9_nz_s0` and `sc_v9_nz03b_s0`; `push_v9_nz_s0` has no `patches.json` (its launcher set `MSPPO_FRAME_TASK=pushcube` only). `60` has no recorded wall-clock time (n/a).

## Evaluation protocol

Every cell is the success rate over N = 256 episodes of one task and one evaluation seed; the paper pools the three seeds 999 (canonical), 997 and 998 to 768 episodes per task (997 overlaps the relbank fit seeds of two tasks and is therefore a scene replicate rather than a fully independent repeat). An episode counts as a success if the environment's success flag is raised at any step of the horizon (`success_once`; 50 steps, PegInsertionSide 100), evaluated with the final checkpoint (`--ckpt final` = `student.pt`). Training data and evaluation banks share the same visual protocol: one front camera (`EYE 0.574,-0.051,0.378`, `TARGET -0.4751,0.0562,0.0200`, FOV 0.754 rad) and a back wall (`MSGEN_WALL=1`), applied through `scripts/config.sh` to every rendering and executor stage (`10`, `40`, `41`, `50`, `70`); the planner processes (`20`, `52`) receive neither variable, since they consume rendered images only. The sampler differs per prediction family, as in production: the four-task `mix4` K = 4 Rigid Readout source uses the native TraceGen sampler, while the entity-branch and map-branch readouts and the PushCube K = 4 samples use 20 fixed-step Euler steps (`MSGEN_STEPS=20 MSGEN_DT=fix`) (KNOWN_ISSUES.md item 8). The Rigid Readout averages K = 4 samples drawn with sampler seeds 1234-1237 in trace space (K = 1 uses the seed-1234 sample), while the readouts use the single sample of seed 1234. The Table II records use the simulator's per-step object correspondence for the executor's pose feedback (`msppo.multi_eval`); the CoTracker3 online-tracking loop of the paper is not part of this release. The composed goal places 16 of the 400 planner queries on the object using simulator segmentation (`msppo/rebank_obj.py`). StackCube uses the `panda_wristcam` robot variant; PegInsertionSide is evaluated with clearance 0.01 instead of the native 0.003. Measured noise: repeated evaluation of the same checkpoint has a standard deviation of 0.018, and the standard deviation across training seeds is about 0.04, which is the origin of the +-0.04 acceptance band of `verify_main_table.sh`. See PROTOCOL.md.

