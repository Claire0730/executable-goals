#!/usr/bin/env bash
# 40_eval_row4.sh -- closed-loop evaluation with PLANNER goals (Entity-Level Goal Readout row of Table II, "final pipeline"):
# the Pose-Native Executor consumes the final goal bank of every task through --goal-delta, psi through --psi-bank.
# Executor = $STUDENT_RUN (default mt5_rciid_gmpc_s0, the executor of that row); PickCube bank = $PICKCUBE_ROUTE.
# Usage: bash scripts/40_eval_row4.sh [seed] [out.json]      (~10 min on one GPU for 5 x 256 episodes)
set -u; . "$(dirname "$0")/config.sh" || exit 1
SD=${1:-$SEED}; OUT=${2:-results/row4_${STUDENT_TAG}_${SD}.json}
[ -s $STUDENT_RUN/run.json ] || { echo "[40] !! $STUDENT_RUN/run.json missing (run scripts/00_link_checkpoints.sh)"; exit 1; }
GD=""; PB=""
for T in $TASKS5; do
  G=$(ROW4_GOAL $T $SD)
  if [ "$T" = pickcube ] && [ ! -s $G ] && [ "$PICKCUBE_ROUTE" = sam2mk6 ]; then
    echo "[40] !! $G missing: run scripts/31_goals_sam2marker.sh $SD (needs SAM2), download the frozen banks (banks/README.md), or set PICKCUBE_ROUTE=gmappeakNC"; exit 1; fi
  [ -s $G ] || { echo "[40] !! missing goal bank $G"; exit 1; }; GD="$GD,${T}=$G"
  [ "$T" != pushcube ] && { P=$(PSI $T $SD); [ -s $P ] || { echo "[40] !! missing psi bank $P"; exit 1; }; PB="$PB,${T}=$P"; }
done
gpu_wait 5000
env $RC nice -n 15 $PM -m msppo.multi_eval --run $STUDENT_RUN --episodes $N_EP --seed $SD --ckpt final \
  --only ${TASKS5// /,} --goal-delta "${GD#,}" --psi-bank "${PB#,}" --out $OUT || { echo "[40] !! eval failed"; exit 1; }
$PM -c "import json;d=json.load(open('$OUT'));print('[40] row4 seed $SD:',{k:round(v,4) for k,v in d['per_task'].items()},'mean',round(d['success_once'],4))"
