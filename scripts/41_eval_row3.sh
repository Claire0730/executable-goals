#!/usr/bin/env bash
# 41_eval_row3.sh -- closed-loop evaluation with the Oracle Goal (the simulator's designated correct goal): omitting
# --goal-delta makes msppo.multi_eval fall back to it. The paper's Oracle references (Fig. 5a) used STUDENT_TAG=mt5_rcfz_gmpc_s0.
# Usage: bash scripts/41_eval_row3.sh [seed] [out.json]
set -u; . "$(dirname "$0")/config.sh" || exit 1
SD=${1:-$SEED}; OUT=${2:-results/row3_${STUDENT_TAG}_${SD}.json}
[ -s $STUDENT_RUN/run.json ] || { echo "[41] !! $STUDENT_RUN/run.json missing (run scripts/00_link_checkpoints.sh)"; exit 1; }
# psi banks: use the frozen ones shipped in banks/ unless 30_goals.sh produced them for this seed
mkdir -p results; for f in banks/*_psinat_*_${SD}.npz; do [ -e "$f" ] && [ ! -e results/$(basename $f) ] && ln -s ../$f results/$(basename $f); done; true
PB=""; for T in pickcube liftpeg peginsert stack; do P=$(PSI $T $SD); [ -s $P ] || { echo "[41] !! missing psi bank $P"; exit 1; }; PB="$PB,${T}=$P"; done
gpu_wait 5000
env $RC nice -n 15 $PM -m msppo.multi_eval --run $STUDENT_RUN --episodes $N_EP --seed $SD --ckpt final \
  --only ${TASKS5// /,} --psi-bank "${PB#,}" --out $OUT || { echo "[41] !! eval failed"; exit 1; }
$PM -c "import json;d=json.load(open('$OUT'));print('[41] row3 seed $SD:',{k:round(v,4) for k,v in d['per_task'].items()},'mean',round(d['success_once'],4))"
