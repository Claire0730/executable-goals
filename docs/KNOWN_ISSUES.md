# Known issues and reproducibility caveats

This file lists the properties of the released code, data and checkpoints that a user must know to interpret a
reproduction (items 1 to 14), followed by the relation between the tables and figures of the paper *Predicted Futures
Are Not Enough: Learning Executable Goals for Robot Manipulation* and the records shipped in `paper_results/` and
`evidence/` (items 15 to 21). Paper terms are mapped to repository identifiers in `docs/GLOSSARY.md`. Every statement
below is taken from the shipped scripts, configuration files, banks and records; nothing here evaluates the paper.

## Checkpoints

1. **Planner checkpoints contain the trained parameters only.** The released `planner/*.pth` files (about 0.30 GB
   each) hold the flow decoder, the vision fusion layer, the Entity-Level Goal Readout (entity branch and, on
   `mix5_t2k_gmap`, the map branch) and the depth mask token. The frozen encoders (DINOv3 ViT-L/16, SigLIP-B/16-384,
   its copy used for depth, T5-base; about 2.5 GB per file) were removed because every one of their tensors was
   bitwise identical to the Hub weights that `timm` / `transformers` download at model construction; the upstream
   trainer built by `msgen.predict` loads the checkpoint with `strict=False` (guarded by `msgen.patch_all`) and the
   encoders come from the Hub. Consequence: planner inference needs the Hub weights (DINOv3 is gated; accept its
   licence and run `hf auth login` once). The release build verified that the reduced `mix5_t2k_gmap` reproduces the
   production prediction bitwise on 16 scenes. Optimizer and gradient-scaler state were dropped as well.
2. **Goal-map labels v1 vs v2.** `mix5_t2k_gmap` (the map branch / Spatial Goal Map) was trained on goal-map labels v1
   (6 cm hard-ball fraction). `msgen/labels_t2k.py` now writes v2 (Gaussian sigma 2 cm, cut at 6 cm); the v1 labeller
   was not under version control at training time. A retrain with `52_train_planner.sh gmap` is therefore not bitwise
   comparable to the released file.
3. **Teachers were warm-started from an unreleased lineage.** `configs/teacher/<tag>/run.json` records `init_from`
   checkpoints that are not shipped. `60_train_teachers.sh` trains the privileged teachers from scratch with the
   recorded flags and the recorded `MSPPO_*` environment (`patches.json`); expect the same range of teacher success,
   not the same weights. `push_v9_nz_s0` has no `patches.json` (its launcher set `MSPPO_FRAME_TASK=pushcube` only).
4. **Code version of the executors.** The Pose-Native Executor `mt5_rcfz_gmpc_s0` and its seed-999 final-routing
   evaluation (`paper_results/supplementary/final_routing_gmpc_999.json`) ran on code that was committed later than
   the run (private commit `a2ccaad`, 2026-09-08); the exact working-tree state is not recorded. No executor
   `run.json` carries a git field. The frozen banks and the released weights are the primary record and reproduce the
   Table II cells within the verification band (item 5; README, "Reproducing Table II").

## Evaluation

5. **PegInsertionSide is not bitwise reproducible.** Identical runs of the same checkpoint and banks differ by a few
   episodes (94 vs 96 of 256 in the audit; 84, 80, 82 and 81 of 256 in four runs of the release build against 81 in
   the supplementary final-routing record at seed 999). The `+-0.04` band of `verify_main_table.sh` covers this; the
   other four tasks reproduced exactly.
6. **Planner queries use simulator segmentation.** The evaluation banks place 16 of the 400 planner query pixels on
   the object using the simulator's segmentation mask (`msppo/rebank_obj.py`, `10_render_banks.sh`), and the training
   data do the same (`--n-obj 16`). In the simulated protocol this is a privilege of the planner input; a real-camera
   deployment needs an external segmenter for the same 16 queries.
7. **PushCube evaluation banks of the paper were rendered without the wall** (the wall patch covered PushCube only
   from 2026-09-01); the shipped frozen banks reproduce the paper, but `10_render_banks.sh` renders every task with the
   wall, so a regenerated PushCube bank is not the paper's bank. The PushCube training clips of `mix5_t2k_n3000` /
   `mix5_t2k_gmap` were likewise rendered without the wall; `50_collect_data.sh` renders all five tasks with it.
8. **Sampler settings differ per prediction family and are reproduced as recorded.** The four-task `mix4` K = 4
   Rigid Readout source was predicted with the native sampler (production logs: only the seed shim active); the
   entity-branch, map-branch and PushCube predictions used 20 fixed Euler steps (`MSGEN_STEPS=20 MSGEN_DT=fix`).
   `20_predict.sh` applies exactly this split; forcing 20 steps on the `mix4` source changes the K = 4 banks and the
   psi banks.
9. **K = 4 sampler inputs of seeds 1235-1237 for the entity-branch checkpoint were overwritten on 2026-09-06**; the
   shipped K = 4 banks are the production files, but the individual sample files behind them cannot all be re-derived.
10. **Relbank fit seeds.** The goal-perturbation families injected during distillation (`banks/*_relbank_*.npz`) were
    fitted on seeds 990-996 (StackCube, PickCube), 990-997 (LiftPegUpright, PegInsertionSide) and 990-993 (PushCube).
    Seed 997 is also an evaluation repeat seed: for the two tasks the overlap concerns the error distribution, not the
    scenes.
11. **Online tracked perception (CoTracker3) is not reproducible from this package.** The tracked-perception loop of
    the paper (CoTracker3-online correspondence between planner frames), its driver modules, the CoTracker3 weights and
    the solve-anchored StackCube bank it used are not included, and no evaluation record of it is shipped. The Table II
    records use the simulator's per-step object correspondence (item 18).

## Environment

12. The `trace_gen` environment freeze was taken on Python 3.10; the `maniskill` freeze on Python 3.11. The three
    `+cu128` wheels need the PyTorch index (`--extra-index-url` is the first line of both freeze files).
13. The upstream data loader caches dataset metadata under `data/cache/` keyed by path; after rebuilding or moving a
    bank directory, delete `data/cache/` or prediction fails with `num_samples=0`.
14. `msppo/gpu_guard.sh` waits for free VRAM before every launch; without `nvidia-smi` the wait is skipped with a
    warning and the stage launches immediately (`GPU_WAIT=0` skips the wait).

## Relation between the paper's tables and the shipped records

15. **Three Pose-Native Executors stand behind Table II.** The shared-executor block of Table II was evaluated with
    three executor checkpoints, all released under `checkpoints/student/` with their `run.json`
    (`paper_results/PROVENANCE.json`, `evidence/01_checkpoint_registry.csv`):
    - `mt5_rciid_gmpc_s0` (sha256 `5ed8f4361eb6a676...`): every cell of the Entity-Level Goal Readout row
      (`table2/final_rciid_<seed>.json` for LiftPegUpright, PegInsertionSide, StackCube, PushCube;
      `table2/final_pickcube_sam2_rciid_<seed>.json` for PickCube). Distilled with per-step i.i.d. goal-error
      redraw (`msppo/patch_iid_inject.py`, `MSPPO_IID_INJECT=1`; `INJECT=iid` in `70_distill.sh`).
    - `mt5_rcfz_gmpc_s0` (sha256 `50467d34909e0d82...`): the Rigid Readout K = 1 row, the Rigid Readout K = 4 row for
      LiftPegUpright, PegInsertionSide, StackCube and PushCube, the Oracle Goal references of Fig. 5a and the Fig. 7
      curves. Distilled with one frozen goal-error draw per episode (`msppo/patch_frozen_inject.py`,
      `MSPPO_FROZEN_INJECT=1`; `INJECT=frozen`). Same relbanks, teachers, architecture, flags and seed as
      `mt5_rciid_gmpc_s0`; the two differ only in the temporal process of the injected error.
    - `mt5_rcfz_t2k_s0` (sha256 `99aaa1f6eb50ac93...`): the PickCube cell of the Rigid Readout K = 4 row only
      (46.48, `table2/rigid_k4_pickcube_t2k_s0_999.json`: seed 999, 119 of 256, goal bank
      `pickcube_goals_mix5_t2k_n3000_999_k4ransac.npz`, the K = 4 RANSAC solve of the `mix5_t2k_n3000` checkpoint).
      An earlier executor (2026-09-01) with episode-fixed injection whose `run.json` records `t2kdcc` relbanks for
      all five tasks; its PickCube and StackCube relbanks are not shipped, so `70_distill.sh` cannot regenerate it.
      The same executor and banks gave 88.67 / 26.95 / 66.02 for LiftPegUpright / PegInsertionSide / StackCube in
      that record; these entries are not used in the paper. With `mt5_rcfz_gmpc_s0` and the `mix4_realcam_n2400`
      K = 4 banks the PickCube K = 4 cell is 33.07 pooled (30.47 / 39.06 / 29.69 per seed).
    `verify_main_table.sh` selects the executor per row (`ROW=final|k1|k4|k4pick|oracle`).
16. **Two PickCube banks of the Entity-Level Goal Readout.** The PickCube goal error of Table I (31.5 +- 20.1 mm) is
    measured on the `gmappeakNC` bank (map-branch peak + ray, abstain to the entity branch;
    `evidence/04_goal_summary.json` key `final_pick_gmappeakNC|pickcube|999`), whereas the PickCube success of the
    Entity-Level Goal Readout row of Table II (81.50) is measured on the `sam2mk6` bank (the SAM 2 marker localiser
    prompted at the map-branch peak replaces the terminal position where its gates accept a mask: 227, 216 and 218 of
    256 scenes at seeds 999, 997, 998; abstained scenes keep the `gmappeakNC` goal). Both banks are shipped for all
    three seeds; `sam2mk6` is regenerated by the optional `31_goals_sam2marker.sh` (SAM 2 in a separate environment),
    `gmappeakNC` by `30_goals.sh`. The `final_rciid_<seed>.json` records contain the `gmappeakNC` PickCube evaluation of
    the same executor (78.12 / 72.27 / 71.88; pooled 74.09), which is not the printed cell.
17. **The executor receives a psi token.** The observation of every released executor is 60-dimensional in the
    code: `qpos 9 | qvel 9 | tcp 7 | prev_a 8 | obj 7 | goal 7 | rel 9 | psi 4` (`msppo/student.py`,
    `configs/student/<tag>/run.json`: `psi: true`, `psi_token: true`). The 4-dimensional psi token is the
    task-constant approach axis k1 (the teacher cone axis) and the carry height h read from the predicted trace
    (`tools/psi_bank.py`, `tools/psinat.py`, `banks/*_psinat_*.npz`); PushCube has no psi bank and `msppo.multi_eval`
    applies the `patch_frame` prior for it. The Oracle Goal, Rigid Readout, Entity-Level Goal Readout and Fig. 7
    records all list the same psi banks per seed (`psi_bank` field).
18. **Simulator correspondence versus the CoTracker3 loop.** The Table II closed-loop records (`msppo.multi_eval`)
    use the simulator's per-step object correspondence for the executor's pose feedback: 64 query pixels chosen once
    at t = 0 on the object's simulator segmentation, carried by the simulator object pose at every step, re-measured
    in depth and fitted by weighted Kabsch (`msppo/peg_perception.py`). The CoTracker3 online-tracking loop is not
    part of this release (item 11).
19. **Table III timing.** The paper's Table III lists 677.6M parameters, 5.60 GB and 1.27 s for the planner. The
    parameter count corresponds to the 677,596,675 unique parameters of the `mix5_t2k_gmap` model
    (`evidence/NUMBERS_PROVENANCE.csv`; `evidence/07_param_recount.json` counts the state dict, 702,270,979, with the
    tied T5 embedding twice). The timing records shipped with this release are batch-1 measurements at 20 fixed
    Euler steps (`MSGEN_STEPS=20 MSGEN_DT=fix`) on an RTX 5090: 0.253 s per scene for `mix4_realcam_n2400` and
    0.277-0.278 s for the two Entity-Level Goal Readout checkpoints (`NUMBERS_PROVENANCE.csv`). The production
    four-task `mix4` K = 4 predictions use the native sampler (100-step schedule), whose latency is not recorded, and
    no timing of the full K = 4 planning call or of the executor is recorded. No record behind the 1.27 s and
    5.60 GB values is part of this release.
20. **Printed versus recorded Table II values.** Pooled over the three seeds (768 episodes per task), the shipped
    records give Rigid Readout K = 1: 29.82 / 70.18 / 20.05 / 52.47 / 99.74; Rigid Readout K = 4
    (`mt5_rcfz_gmpc_s0`): 33.07 / 76.95 / 21.88 / 54.82 / 99.22; Entity-Level Goal Readout: 81.51 / 98.31 / 32.81 /
    86.46 / 99.22; Oracle Goal: 94.66 / 97.27 / 36.72 / 87.11 / 99.74 (PickCube / LiftPegUpright / PegInsertionSide /
    StackCube / PushCube). The paper prints 29.80 / 70.18 / 20.05 / 52.47 / 99.74, 46.48 / 76.95 / 21.88 / 54.82 /
    99.22, 81.50 / 98.35 / 32.84 / 86.54 / 99.20 and 94.66 / 97.27 / 36.72 / 87.11 / 99.74. Apart from the PickCube
    K = 4 cell (item 15), the printed and the recorded values differ by at most 0.08 percentage points, which is
    rounding of the printed values. The per-seed values are in the README ("Reproducing Table II").
21. **Records kept under former names; records without a regenerating script.** `evidence/NUMBERS_PROVENANCE.csv`
    and `evidence/05_closed_loop_per_episode.csv.gz` name the records by the file names of the internal audit:
    `row4_mt5_rcfz_gmpc_s0_nc_<seed>.json` is `paper_results/supplementary/final_routing_gmpc_<seed>.json`,
    `row3_mt5_rcfz_gmpc_s0_999.json` is `paper_results/table2/oracle_gmpc_999.json`, and the `row6_*` records named
    there are not shipped (item 11). The Fig. 7 records were produced with a goal-perturbation patch that is outside
    the released module set; they are shipped as records with their verification (`evidence/06_verify_perturbation.out.json`)
    and are not regenerated by any script. The "No geo." curve of Fig. 6 (`abl_nodepth_999` in `evidence/`) comes
    from an ablation bank that is not shipped.
