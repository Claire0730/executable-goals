# banks/

Two kinds of file live here.

**In git (17 files, 0.23 MB).** Small, and needed by stages that do not otherwise touch the planner:

- `<task>_psinat_mix4_realcam_n2400_<seed>.npz` — the psi banks. The executor's observation ends in a
  4-dimensional psi token (a task-constant approach axis and the carry height read from the predicted trace);
  `41_eval_row3.sh` and `verify_main_table.sh` pass them with `--psi-bank`.
- `<task>_relbank_realcam_<route>.npz` — the measured planner goal-error distributions. `70_distill.sh` samples
  from them to expose the executor to imperfect goals during distillation.

**Downloaded (49 files, 58 MB).** The frozen goal banks — one SE(3) goal per scene, for every Table II row, task
and evaluation seed. They are data, not code, so they live beside the weights:

```bash
hf download <HF_REPO> --include "banks/*" --local-dir .
```

This writes them into this directory, where `verify_main_table.sh`, `40_eval_row4.sh` and `41_eval_row3.sh` find
them. `.gitignore` excludes `*_goals_*.npz`, so a download never shows up as a local change. Verify with
`(cd banks && sha256sum -c SHA256SUMS)`.

Regenerating them instead of downloading is `10_render_banks.sh` -> `20_predict.sh` -> `30_goals.sh`, which needs
the planner checkpoints, the upstream TraceGen checkout and an authenticated Hugging Face session (DINOv3 is
gated). The release build checked that a regenerated bank matches the frozen one bitwise on 16 scenes.

## File names

`<task>_goals_<planner tag>_<seed>_<route>.npz`, where the route is how the goal was read out of the prediction:

| Route | Meaning |
|---|---|
| `k1ransac` | Rigid Readout, K = 1: weighted Kabsch inside RANSAC on one sampled trace |
| `kmean_ransac` | Rigid Readout, K = 4: four traces averaged before the fit |
| `t2kpos` | entity-branch position (PegInsertionSide uses the full entity pose) |
| `gmapdcc` | map branch, depth-weighted centroid with a depth cross-check (StackCube) |
| `gmappeakNC` | map branch, peak plus ray, abstaining to the entity branch (PickCube; the Table I error) |
| `sam2mk6` | SAM 2 marker localiser prompted at the map peak (PickCube; the Table II cell) |

`docs/GLOSSARY.md` maps each one to the paper's wording, and `docs/REPRODUCTION.md` says which bank produced
which table cell.
