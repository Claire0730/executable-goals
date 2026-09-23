#!/usr/bin/env bash
# 30_goals.sh -- turn the predictions of 20_predict.sh into the goal banks the executor consumes.
# Per task:  ksample (K=4 mean) -> task_tgbank solve --ransac        -> *_kmean_ransac.npz   (rotation, per-step dR/dt)
#            tools/t2k_decode.py (entity head, world frame)          -> *_t2k.npz            (head pose in the bank schema)
#            tools/mk_fitgoals.py (head position x solve rotation)   -> *_t2kpos.npz (+ stack *_dcc.npz depth cross-check)
#            pickcube: tools/build_pc_src.py (goal map peak + ray)   -> *_gmappeakNC.npz
#            stack:    tools/build_gmapdcc.py (goal-map centroid)    -> *_gmapdcc.npz
#            psi = (k1, h) from the mean trace (tools/psi_bank.py, psinat.py) -> *_psinat_*.npz (not for pushcube)
# The deployed bank per task is ROW4_GOAL in config.sh.   Usage: bash scripts/30_goals.sh [seed]
set -u; NEED_PG=1; . "$(dirname "$0")/config.sh" || exit 1
SD=${1:-$SEED}
for T in $TASKS5; do
  B=$(BANK $T $SD); PT=$(ROT_TAG $T)
  LIST=""; for S in $KSEEDS; do LIST="$LIST results/preds/${T}_${PT}_${SD}_s${S}.npz"; done
  KM=results/preds/${T}_${PT}_${SD}_kmean.npz
  [ -s $KM ] || $PG -m msgen.ksample --out $KM $LIST || { echo "[30] !! ksample $T"; exit 1; }
  G=$(GOALS $T $PT $SD kmean_ransac)
  [ -s $G ] || nice -n 10 $PM -m msppo.task_tgbank solve --task $T --bank $B --pred $KM --out $G --ransac || { echo "[30] !! solve $T"; exit 1; }
  GT=$(GOALS $T $TAG_HEAD $SD t2k)
  [ -s $GT ] || nice -n 15 $PM tools/t2k_decode.py $T $B results/preds/${T}_${TAG_HEAD}_${SD}_s1234.npz $G $GT || { echo "[30] !! decode $T"; exit 1; }
  TP=$(GOALS $T $TAG_HEAD $SD t2kpos)
  if [ "$T" = peginsert ]; then
    [ -s $TP ] || cp $GT $TP                      # PegInsertion consumes the full head pose (rotation included)
  else
    [ -s $TP ] || env PLANNER_TAG=$TAG_HEAD ROT_TAG=$PT $PM tools/mk_fitgoals.py $T $SD $REPO || { echo "[30] !! mk_fitgoals $T"; exit 1; }
  fi
  case $T in
    pickcube) [ -s $(ROW4_GOAL $T $SD) ] || $PM tools/build_pc_src.py $SD  || { echo "[30] !! build_pc_src"; exit 1; };;
    stack)    [ -s $(ROW4_GOAL $T $SD) ] || $PM tools/build_gmapdcc.py $SD || { echo "[30] !! build_gmapdcc"; exit 1; };;
  esac
  if [ "$T" != pushcube ]; then
    PSIB=$(PSI $T $SD)
    [ -s $PSIB ] || { $PM tools/psi_bank.py $T $SD && $PM tools/psinat.py $T $SD $REPO; } || { echo "[30] !! psi $T"; exit 1; }
  fi
  echo "[30] $T $SD: $(ROW4_GOAL $T $SD)"
done
