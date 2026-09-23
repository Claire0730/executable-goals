# third_party

The planner is a fine-tuned [TraceGen](https://github.com/jayLEE0301/TraceGen) (Apache-2.0). Its code is **not
vendored**; clone it here at the pinned commit and apply our local diff:

```bash
git clone https://github.com/jayLEE0301/TraceGen third_party/TraceGen
git -C third_party/TraceGen checkout $(cat third_party/TRACEGEN_COMMIT)
git -C third_party/TraceGen apply ../tracegen_local.patch
```

`tracegen_local.patch` (7 files, 389 insertions and 41 deletions; comments inside the patch cite directories of the private research repository and have no effect on the code) adds optional conditioning / loss knobs that were explored in July 2026
(`attractor` labels, query-position channels, motion-weighted loss, an `act_clamp` switch). Every one of them is
**off by default and inert** for the released checkpoints; the patch is shipped so that the code that produced the
checkpoints is byte-identical, not because a released model depends on it.

Also obtained from upstream sources (not redistributed here):

- the **Generalist** checkpoint `tracegen_model.pth` → `third_party/TraceGen/assets_ckpt/Generalist/`
  (warm start of `mix4_realcam_n2400`; the `zeroshot` family of `evidence/goal_table.json`). Needed only for planner training
  (`scripts/52_train_planner.sh`). Source: the upstream README / Hugging Face collection `furonghuang-lab/tracegen`.
- the frozen encoders `timm/vit_large_patch16_dinov3.lvd1689m` (DINOv3 ViT-L/16), `google/siglip-base-patch16-384`
  (used twice: RGB encoder and depth encoder) and `t5-base`. The released planner checkpoints contain the trained
  parameters only (`docs/KNOWN_ISSUES.md`, item 1); `timm` / `transformers` fetch the encoder weights from the Hugging
  Face Hub whenever the planner is built, for inference (`scripts/20_predict.sh`) as well as training
  (`scripts/52_train_planner.sh`). DINOv3 is gated: accept its licence on Hugging Face and run `hf auth login` once.
  `scripts/verify_main_table.sh` builds no planner and needs neither the encoders nor this checkout.
- **SAM 2** (Meta Platforms, Apache-2.0), optional: `scripts/31_goals_sam2marker.sh` regenerates the PickCube goal bank of the
  Entity-Level Goal Readout row of Table II (`sam2mk6`: the SAM 2 marker localiser prompted at the Spatial Goal Map peak,
  `msppo/goal_depth_check.py::op_marker_sam2`). It expects the SAM 2 repository at `third_party/sam2` (`SAM2_DIR`) with the
  `sam2.1_hiera_small` checkpoint at `third_party/sam2/checkpoints/sam2.1_hiera_small.pt` (`SAM2_CKPT`), and a separate
  Python environment with `sam2`, `imageio` and `scipy` addressed by `PV`. The frozen `sam2mk6` banks for seeds 999 / 997 / 998
  are shipped in `banks/`, so `scripts/verify_main_table.sh` and the closed-loop evaluation on the frozen banks need no SAM 2.
