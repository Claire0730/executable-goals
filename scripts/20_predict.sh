#!/usr/bin/env bash
# 20_predict.sh -- planner predictions on the seed banks (three checkpoints, see config.sh):
#   A. rotation / solve source: K=4 flow samples (s1234..s1237)
#        four tasks: mix4, NATIVE sampler (production record: only the seed shim active, no MSGEN_STEPS)
#        pushcube:   mix5_t2k_n3000 with MSGEN_T2K=1 + 20 fixed steps; its s1234
#                    file is also the entity-head file of case B, so B is skipped for pushcube
#   B. entity head (T2K): s1234 from mix5_t2k_n3000 with MSGEN_T2K=1 + 20 fixed steps -> t2k_* outputs
#   C. goal-map head:     s1234 from mix5_t2k_gmap  with MSGEN_T2K_GMAP=1 + 20 fixed steps -> t2k_gmap (pickcube, stack)
# Idempotent; predictions land in results/preds/.   Usage: bash scripts/20_predict.sh [seed]
# The upstream loader caches dataset metadata under data/cache/ keyed by the dataset path: if a bank directory was
# rebuilt or was incomplete when first read, delete data/cache/ or the run fails with num_samples=0.
set -u; NEED_PG=1; . "$(dirname "$0")/config.sh" || exit 1
SD=${1:-$SEED}
pred(){ local B=$1 CK=$2 TAG=$3 S=$4 EXTRA=$5
  local P=results/preds/${TAG}_${SD}_s${S}.npz; [ -s $P ] && return 0
  gpu_wait 8000
  # no $RC here: the wall / camera variables only affect rendering (10_render_banks.sh); MSGEN_WALL=1 would make
  # msgen.patch_all import the simulator, which is not installed in the planner env.
  env $EXTRA MSGEN_SEED=$S nice -n 15 $PG -m msgen.predict --dataset $B --ckpt $CK \
    --tag ${TAG}_${SD}_s${S} --batch-size 8 --num-workers 4 || { echo "[20] !! predict $TAG s$S failed"; exit 1; }
}
for T in $TASKS5; do
  B=$(BANK $T $SD); [ -s $B/configs.json ] || { echo "[20] !! no bank $B (run 10_render_banks.sh)"; exit 1; }
  if [ "$T" = pushcube ]; then
    for S in $KSEEDS; do pred $B $CK_HEAD ${T}_${TAG_HEAD} $S "$PRED_ENV MSGEN_T2K=1"; done      # A (pushcube) = B
  else
    for S in $KSEEDS; do pred $B $CK_MIX4 ${T}_${TAG_MIX4} $S ""; done                          # A, native sampler
    pred $B $CK_HEAD ${T}_${TAG_HEAD} 1234 "$PRED_ENV MSGEN_T2K=1"                                # B
  fi
  case $T in pickcube|stack) pred $B $CK_GMAP ${T}_${TAG_GMAP} 1234 "$PRED_ENV MSGEN_T2K=1 MSGEN_T2K_GMAP=1";; esac   # C
  echo "[20] $T $SD: predictions ready"
done
