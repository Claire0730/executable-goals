#!/usr/bin/env bash
# 51_label_t2k.sh -- entity-level labels for the T2K head and the goal map, derived offline from the raw replays
# (keyframes, per-segment twists, contact descriptor, structure-relative terminal pose, entity membership of the
# 400 queries, goal-map target). Writes samples/<stem>_t2k.npz next to every trace sample. CPU only.
# Usage: bash scripts/51_label_t2k.sh
set -u; . "$(dirname "$0")/config.sh" || exit 1
LOG=data/collect_logs; mkdir -p $LOG
for T in pickcube stack peg liftpeg pushcube; do
  ( for V in v0 v1 v2; do D=data/ds/realcam_${T}_${V}; [ -s $D/dataset.json ] || continue
      nice -n 19 $PM -m msgen.labels_t2k --ds $D --overwrite > $LOG/t2k_${T}_${V}.log 2>&1 || echo "[51] !! $T $V failed"
      echo "[51] $T $V: $(tail -1 $LOG/t2k_${T}_${V}.log | cut -c1-100)"; done ) &
done; wait
$PM -c "
import numpy as np, glob, random
fs = random.Random(0).sample(glob.glob('data/ds/realcam_*_v*/clip_*/samples/*_t2k.npz'), 200)
ok = sum(1 for f in fs if 'gmap' in np.load(f).files)
print(f'[51] sample check: {ok}/200 t2k label files carry a goal map')"
