#!/usr/bin/env bash
# 52_train_planner.sh -- the three planner fine-tunes of the paper (mix4 from the Generalist; both mix5 heads from mix4):
#   mix4_realcam_n2400  <- upstream Generalist; 15 epochs, batch 8, lr_decoder 1.5e-4     (rigid-solve source)
#   mix5_t2k_n3000      <- mix4;  8 epochs, MSGEN_T2K=1 MSGEN_T2K_W=0.3                    (entity head)
#   mix5_t2k_gmap       <- mix4;  8 epochs, + MSGEN_T2K_GMAP=1                              (goal-map head)
# Frozen encoders (DINOv3 / SigLIP / T5) are downloaded on first use. ~10 h / ~6 h / ~6 h on one RTX 5090.
# Usage: bash scripts/52_train_planner.sh [mix4|head|gmap|all]   (default all; skips a stage whose final ckpt exists)
# Note: the released mix5_t2k_gmap was trained on goal-map labels v1 (6 cm hard ball); msgen.labels_t2k now writes
# v2 (Gaussian sigma 2 cm cut at 6 cm), so a retrain is not bitwise comparable -- see docs/KNOWN_ISSUES.md.
set -u; NEED_PG=1; . "$(dirname "$0")/config.sh" || exit 1
STAGE=${1:-all}
final(){ ls -t runs/$1/ckpt/*/final_model.pth 2>/dev/null | head -1; }
train(){ local TAG=$1 DS=$2 EPOCHS=$3 CK=$4; shift 4      # rest: env assignments
  [ -n "$(final $TAG)" ] && { echo "[52] $TAG exists: $(final $TAG)"; return 0; }
  [ -d $DS ] || { echo "[52] !! dataset $DS missing (run 50/51)"; exit 1; }
  gpu_wait 16000; echo "[52] train $TAG from $CK ($EPOCHS epochs) env: $*"
  env "$@" nice -n 10 $PG -m msgen.run_train_logged --dataset $DS --tag $TAG --ckpt $CK --epochs $EPOCHS \
    --batch-size 8 --lr-decoder 1.5e-4 --num-workers 4 || { echo "[52] !! train $TAG failed"; exit 2; }
}
# no MSGEN_WALL here: the wall only affects rendering (50_collect_data.sh); the planner interpreter has no simulator,
# and the released planners' patches.json record wall=False for the training process.
case $STAGE in mix4|all) train $TAG_MIX4 data/ds/realcam_n2400 15 $TRACEGEN_GENERALIST MSGEN_KEEP_ALL_CKPT=0;; esac   # no MSGEN_* flags recorded for mix4
WARM=${WARM:-$(final $TAG_MIX4)}; [ -s "$WARM" ] || WARM=$CK_MIX4     # your own mix4 if trained, else the released one
case $STAGE in head|all) train $TAG_HEAD data/ds/realcam_t2k_n3000 8 $WARM MSGEN_T2K=1 MSGEN_T2K_W=0.3;; esac
case $STAGE in gmap|all) train $TAG_GMAP data/ds/realcam_t2k_n3000 8 $WARM MSGEN_T2K=1 MSGEN_T2K_W=0.3 MSGEN_T2K_GMAP=1 MSGEN_KEEP_ALL_CKPT=1;; esac
echo "[52] done: mix4=$(final $TAG_MIX4) head=$(final $TAG_HEAD) gmap=$(final $TAG_GMAP)"
