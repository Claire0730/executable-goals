# Predicted Futures Are Not Enough: Learning Executable Goals for Robot Manipulation

[Paper](https://arxiv.org/abs/<ARXIV_ID>) · [Project page](https://claire0730.github.io/executable-goals) · [Checkpoints](https://huggingface.co/Claire0730/executable-goals)

A 3D trace world model predicts one future per episode; the **Entity-Level Goal Readout** turns that prediction into a
single executable goal in SE(3), and a shared **Pose-Native Executor** runs it closed loop at 20 Hz. Five ManiSkill3
tasks: PickCube, LiftPegUpright, PegInsertionSide, StackCube, PushCube. This repository holds the code, the frozen
goal banks and the evaluation records the paper's numbers were read from.

## System overview

![Frozen T5, DINOv3 and SigLIP encoders feed a shared world-model representation; the 3D Trace Planner predicts sparse future 3D traces and the Entity-Level Goal Readout predicts one object-relative SE(3) goal, which the Pose-Native Executor tracks in a 20 Hz closed loop](docs/img/overview.jpg)

Frozen T5, DINOv3 and SigLIP encoders supply a shared representation. The **3D Trace Planner** (73.5M trained
parameters) predicts 400 query points over 32 future steps; the **Entity-Level Goal Readout** (3.05M) reads one
object-relative SE(3) goal out of the same representation, fixed for the episode. The **Pose-Native Executor**
(804,002 parameters) closes the loop at 20 Hz on object-pose feedback, emitting seven joint increments and a gripper
target without rerunning the planner. The **Rigid Readout** baseline recovers the same goal by weighted Kabsch inside
RANSAC on the predicted tracks instead of predicting it.

Three planner checkpoints and three executors are released; which one produced which table cell is in
[docs/REPRODUCTION.md](docs/REPRODUCTION.md).

Docs: [GLOSSARY](docs/GLOSSARY.md) (paper terms to code identifiers) · [REPRODUCTION](docs/REPRODUCTION.md)
(every reported number, and how to re-run it) · [PROTOCOL](docs/PROTOCOL.md) · [MODULE_MAP](docs/MODULE_MAP.md) ·
[KNOWN_ISSUES](docs/KNOWN_ISSUES.md)

> The project page and the weights are live; `<ARXIV_ID>` is filled in once the preprint is online.

## Quick start: reproduce one row of Table II

One conda environment, one GPU, 70 MB of downloads. No planner, no DINOv3 licence, no TraceGen checkout -- the
goal banks are frozen, so the planner does not have to run.

```bash
git clone https://github.com/Claire0730/executable-goals && cd executable-goals
conda create -n maniskill python=3.11 -y
conda run -n maniskill pip install -r requirements/maniskill_freeze.txt
export PM=$(conda run -n maniskill which python)

hf download Claire0730/executable-goals --local-dir checkpoints --include "student/*" --include "teacher/*" --include "SHA256SUMS"      # 12 MB of weights
hf download Claire0730/executable-goals --local-dir . --include "banks/*"                    # 58 MB of frozen goals
bash scripts/00_link_checkpoints.sh
ROW=final bash scripts/verify_main_table.sh 999        # ~10 min on one GPU
```

It runs 5 tasks x 256 episodes with the executor that produced that row and compares each task with the shipped
record in `paper_results/table2/` within +-0.04. `ROW` also takes `k1`, `k4`, `k4pick` and `oracle`; the seed also
takes `997` and `998`. The full installation, which adds the planner, is below.

## Repository layout

| Path | Content |
|---|---|
| `scripts/` | `config.sh` (paths, interpreters, pinned tags, camera protocol, per-task goal routing), pipeline stages `00`–`70`, `verify_main_table.sh` |
| `msgen/` | planner side (`trace_gen` env): labelling, training, prediction, runtime patches (`patch_t2k.py` = Entity-Level Goal Readout) |
| `msppo/` | simulator side (`maniskill` env): keypoint environments, privileged teachers, DAgger distillation, closed-loop evaluation, Rigid Readout |
| `tools/` | goal-bank builders used by `30_goals.sh` and `31_goals_sam2marker.sh` |
| `banks/` | psi banks and relbanks (in git); the frozen goal banks are downloaded from the weights repository — see [banks/README.md](banks/README.md) |
| `paper_results/`, `evidence/` | the evaluation records the paper's numbers were read from, and their provenance |
| `configs/` | `run.json` / `patches.json` recorded for every released checkpoint |
| `requirements/`, `third_party/`, `tests/` | the two frozen environments; the pinned upstream TraceGen commit and local patch; six unit tests |

## Pipeline stages

`scripts/` is numbered in dependency order. Each stage reads what the previous one wrote, is idempotent per task and
seed, and skips outputs that already exist; the seed argument defaults to 999. Only `20`, `30` and `52` need the
planner environment (`PG`); everything else runs on the simulator environment (`PM`) alone.

**Setup**

| Stage | In | Out |
|---|---|---|
| `00_link_checkpoints.sh` | the downloaded `checkpoints/` | symlinks under `runs_rl/<tag>`, the layout the evaluation and distillation code expects; checks `SHA256SUMS` |

**Inference chain — from a scene to a success rate**

| Stage | In | Out |
|---|---|---|
| `10_render_banks.sh` | a task and an evaluation seed | 256 rendered scenes per task under the fixed camera and wall protocol: RGB-D, camera pose, object segmentation, and 16 query points placed on the object |
| `20_predict.sh` | those scenes + a planner checkpoint | the planner's raw output per scene: where its 400 query points travel over the next 32 steps (and, for the readout checkpoints, the entity-branch pose and the goal map) |
| `30_goals.sh` | those raw predictions | **one SE(3) goal per scene**, stored as a `.npz` "goal bank" — the file the executor actually consumes. This is where the readouts differ: the Rigid baseline fits a rigid transform to the predicted tracks, the entity branch decodes its predicted pose, the map branch turns the goal map into a position through observed depth. Also writes the small psi banks the executor's psi token reads |
| `31_goals_sam2marker.sh` | the PickCube scenes and goal map | optional: the PickCube goal bank of the final row, with SAM 2 localising the marker at the goal-map peak (needs a separate SAM 2 environment) |
| `40_eval_row4.sh` | a goal bank + an executor | closed-loop rollouts with the planner's goals, 256 episodes per task -> a success-rate JSON |
| `41_eval_row3.sh` | the simulator's true goal + an executor | the same rollouts with the Oracle Goal, i.e. what the executor achieves when the goal is correct by construction |
| `verify_main_table.sh` | the **frozen** goal banks in `banks/` | skips `10`-`31` entirely and re-runs `40`/`41` for one Table II row, then compares with the shipped record in `paper_results/table2/` |

**Training chain — everything from scratch**

| Stage | In | Out |
|---|---|---|
| `50_collect_data.sh` | the official ManiSkill demonstrations | those demos replayed under the production camera and wall, with dense 3D traces labelled on them: the planner's training set (~50 GB) |
| `51_label_t2k.sh` | the same replays | the extra labels the readout is trained against: terminal object pose, keyframes, contact, and the goal map |
| `52_train_planner.sh` | that dataset + the upstream Generalist checkpoint | the three planner checkpoints (`mix4` from the Generalist; both readout checkpoints warm-started from `mix4`) |
| `60_train_teachers.sh` | the simulator | five single-task PPO policies that see privileged state — the teachers |
| `70_distill.sh` | the five teachers + the relbanks in `banks/` | the Pose-Native Executor, distilled with DAgger under goal errors drawn from the planner's measured error distribution |

The two chains meet at the executor: `70` produces it, `40`/`41`/`verify_main_table.sh` evaluate it.

## Full installation (planner and training)

In addition to the quick start. Needs Linux x86_64, an NVIDIA GPU (sm_75+, CUDA 12.8 wheels; the freezes were taken
on an RTX 5090) and a Vulkan-capable driver for SAPIEN off-screen rendering. Every stage waits for free VRAM through
`nvidia-smi`; without it, set `GPU_WAIT=0`.

```bash
# planner environment (the quick start only builds `maniskill`)
conda create -n trace_gen python=3.10 -y && conda run -n trace_gen pip install -r requirements/trace_gen_freeze.txt
export PG=$(conda run -n trace_gen which python)

# upstream TraceGen at the pinned commit, plus the local patch (not vendored here)
git clone https://github.com/jayLEE0301/TraceGen third_party/TraceGen
git -C third_party/TraceGen checkout $(cat third_party/TRACEGEN_COMMIT)
git -C third_party/TraceGen apply ../tracegen_local.patch

hf download Claire0730/executable-goals --local-dir checkpoints     # now the planners too (~0.95 GB)
# banks/ as in the quick start; regenerating them instead is 10 -> 20 -> 30
bash scripts/00_link_checkpoints.sh
```

Both freeze files start with `--extra-index-url https://download.pytorch.org/whl/cu128`, which pip needs for the
`+cu128` wheels. Building a planner downloads the frozen encoders from the Hugging Face Hub, and **DINOv3 is gated**:
accept its licence and run `hf auth login` once. ManiSkill demonstration packages are needed only for
`50_collect_data.sh`.

## Checkpoints

| Group | Files | Size | Role |
|---|---|---|---|
| `planner/` | `mix4_realcam_n2400`, `mix5_t2k_n3000`, `mix5_t2k_gmap` | 0.30–0.31 GB each | Rigid Readout source; entity branch; entity + map branch |
| `student/` | `mt5_rciid_gmpc_s0`, `mt5_rcfz_gmpc_s0`, `mt5_rcfz_t2k_s0` | 3.2 MB each | the three Pose-Native Executors behind the Table II rows |
| `teacher/` | five task policies | ~4 MB each | privileged PPO teachers used for distillation |

The planner files hold the **trained parameters only**. Every frozen-encoder tensor was bitwise identical to the
published Hub weights and was removed, together with the optimizer and gradient-scaler state; the encoders are
re-created from the Hub when the planner is built, and the reduced files reproduce the production predictions
bitwise (checked on 16 scenes). Digests: `checkpoints/SHA256SUMS`.

Each planner file also carries the configuration of the run that trained it. Its `data.dataset_dirs`, `data.cache_dir` and `checkpoint_dir` are released as `[path-to-the-repository-root-here]/...`: the authors' absolute paths were replaced, the relative part kept. These fields are metadata — prediction and evaluation never read them — but **`52_train_planner.sh` resuming from a released checkpoint needs them set to real directories**, so edit them (or pass the equivalent flags) before you resume training.

## The other Table II rows

The quick start runs `ROW=final`. The same script takes the other rows and the other two evaluation seeds:

```bash
ROW=k1     bash scripts/verify_main_table.sh 999   # Rigid Readout K=1
ROW=k4     bash scripts/verify_main_table.sh 999   # Rigid Readout K=4
ROW=k4pick bash scripts/verify_main_table.sh 999   # the PickCube cell of the K=4 row as printed (seed 999 only)
ROW=oracle bash scripts/verify_main_table.sh 999   # Oracle Goal references (Fig. 5a)
```

Success rates in percent, PickCube / LiftPegUpright / PegInsertionSide / StackCube / PushCube. "Printed" is the
paper; "recorded" is the mean over seeds 999, 997 and 998 of the shipped records (768 episodes per task).

| Table II row | Printed | Recorded |
|---|---|---|
| Rigid Readout K = 1 | 29.80 / 70.18 / 20.05 / 52.47 / 99.74 (mean 54.45) | 29.82 / 70.18 / 20.05 / 52.47 / 99.74 (mean 54.45) |
| Rigid Readout K = 4 | 46.48 / 76.95 / 21.88 / 54.82 / 99.22 (mean 59.87) | 33.07 / … (mean 57.19) with one executor; the printed PickCube cell is a second executor at seed 999 |
| Entity-Level Goal Readout | 81.50 / 98.35 / 32.84 / 86.54 / 99.20 (mean 79.69) | 81.51 / 98.31 / 32.81 / 86.46 / 99.22 (mean 79.66) |

Per-seed values, the bank and record behind every cell, Table I, Figs. 6 and 7, the full inference chain and the
training chain: [docs/REPRODUCTION.md](docs/REPRODUCTION.md).

## Third-party models (not redistributed)

Nothing in `checkpoints/` is a third-party weight. Every external model the code touches is listed here with its
source and licence; each is obtained by the user, not by this repository.

| Model | Identifier / source | Licence | Needed for |
|---|---|---|---|
| DINOv3 ViT-L/16 (vision encoder, frozen) | `timm/vit_large_patch16_dinov3.lvd1689m` on the Hugging Face Hub | Meta DINOv3 licence, **gated** | `20_predict.sh`, `52_train_planner.sh` |
| SigLIP-B/16-384 (vision-language encoder, frozen) | `google/siglip-base-patch16-384` on the Hub | Apache-2.0 | as above |
| SigLIP-B/16-384, second copy used as the depth tower | the same identifier, loaded again (`msgen/patch_depthtower.py`) | Apache-2.0 | as above |
| T5-base (text encoder, frozen) | `t5-base` on the Hub | Apache-2.0 | as above |
| TraceGen, upstream code | the commit in `third_party/TRACEGEN_COMMIT` + `third_party/tracegen_local.patch` | Apache-2.0 | every planner stage |
| TraceGen Generalist checkpoint | obtain from the upstream project | upstream terms | `52_train_planner.sh` only |
| SAM 2 (`sam2.1_hiera_small`) | separate environment (`PV`, `SAM2_DIR`, `SAM2_CKPT`) | Apache-2.0 | `31_goals_sam2marker.sh` only; the banks it produces are shipped |
| CoTracker3 | not included | CC BY-NC 4.0 | the online-tracking loop, which is **not part of this release** |
| ManiSkill3 | `mani_skill 3.0.1` and its demo packages | code Apache-2.0; assets CC BY-NC 4.0 | every simulator stage |

The frozen encoders are fetched from the Hub when the planner is built. **DINOv3 is gated**: accept its licence with
the account you will use and run `hf auth login` once, or the planner cannot be built. `verify_main_table.sh` builds
no planner and therefore needs no Hub session, no DINOv3 licence, no TraceGen checkout and no SAM 2.

## Limitations

- Real-robot deployment and the CoTracker3 online-tracking loop are not part of this release; the Table II records
  use the simulator's per-step object correspondence for the executor's pose feedback.
- The goal chain places object queries with simulator segmentation, which at test time is a privilege of the
  simulated setting.
- `verify_main_table.sh` reproduces each row within ±0.04, not bitwise; PegInsertionSide differs between identical
  runs. All results are single-training-seed; the three evaluation seeds vary scenes, not training.
- Retraining `mix5_t2k_gmap` uses goal-map labels v2 and is not bitwise comparable to the released checkpoint.
  Numbered caveats: [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md).

## License and citation

Code and released checkpoints: Apache-2.0 (`LICENSE`, `NOTICE`). The released checkpoints contain only parameters
trained by the authors; no third-party weights are redistributed. Cite the paper via [CITATION.cff](CITATION.cff).
