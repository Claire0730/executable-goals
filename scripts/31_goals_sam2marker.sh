#!/usr/bin/env bash
# 31_goals_sam2marker.sh -- PickCube goal bank of the Entity-Level Goal Readout row: the SAM2 marker localiser prompted at
# the goal-map peak replaces the terminal position of the gmappeakNC bank where its gates accept a mask (tools/build_sam2marker.py,
# msppo/goal_depth_check.py::op_marker_sam2); abstained scenes keep the gmappeakNC goal. Needs SAM 2 (third_party/sam2 with
# the sam2.1_hiera_small checkpoint, not redistributed) in an environment with sam2, imageio and scipy: set PV to its python
# and SAM2_DIR / SAM2_CKPT if they differ from third_party/sam2. Requires 30_goals.sh (gmappeakNC bank) first.
# The frozen banks in banks/ (seeds 999/997/998) are the ones behind the paper; this script regenerates them.
# Usage: PV=/path/to/python bash scripts/31_goals_sam2marker.sh [seed]
set -u; . "$(dirname "$0")/config.sh" || exit 1
SD=${1:-$SEED}; PV=${PV:?set PV to a python with sam2, imageio and scipy}
export SAM2_DIR=${SAM2_DIR:-$REPO/third_party/sam2} SAM2_CKPT=${SAM2_CKPT:-$SAM2_DIR/checkpoints/sam2.1_hiera_small.pt}
[ -s results/pickcube_goals_${TAG_GMAP}_${SD}_gmappeakNC.npz ] || { echo "[31] !! run scripts/30_goals.sh $SD first (gmappeakNC base bank)"; exit 1; }
OUT=results/pickcube_goals_${TAG_GMAP}_${SD}_sam2mk6.npz
[ -s $OUT ] && { echo "[31] exists: $OUT"; exit 0; }
gpu_wait 4000
env SAM2MK_TAG=sam2mk6 SAM2MK_BASE=gmappeakNC $PV tools/build_sam2marker.py $SD || { echo "[31] !! build_sam2marker failed"; exit 1; }
echo "[31] pickcube $SD: $OUT"
