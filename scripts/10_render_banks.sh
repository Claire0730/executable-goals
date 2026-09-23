#!/usr/bin/env bash
# 10_render_banks.sh -- render the evaluation scene banks (N_EP scenes per task, one seed) under the
# production camera + wall protocol, then place the 16 object-aware planner queries (rebank_obj).
# Idempotent per task. GPU (SAPIEN) ~ minutes per task.   Usage: bash scripts/10_render_banks.sh [seed]
set -u; . "$(dirname "$0")/config.sh" || exit 1
SD=${1:-$SEED}
for T in $TASKS5; do
  RAW=data/bank/realcam_${T}_${SD}; B=$(BANK $T $SD)
  [ -s "$B/configs.json" ] && { echo "[10] $T $SD: bank exists"; continue; }
  gpu_wait 6000
  # task_tgbank's main never applies the wall patch itself; the planners are wall-trained, so
  # assert it here -- a silent wall-less bank poisons every downstream number.
  SHOWG=0; [ "$T" = pickcube ] && SHOWG=1          # PickCube renders its goal marker (the task's target is a point in space)
  env $RC nice -n 10 $PM - "$T" "$RAW" "$SD" "$SHOWG" "$N_EP" <<'PYR' || { echo "[10] !! render $T failed"; exit 1; }
import sys
from msgen.patch_wall import maybe_patch as w
assert w(), "wall patch did not apply (MSGEN_WALL=1 expected)"
import msppo.task_tgbank as tg
t, raw, sd, showg, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1", int(sys.argv[5])
tg.render(t, raw, num_envs=n, seed=sd, show_goal=showg)
PYR
  nice -n 10 $PM -m msppo.rebank_obj --src $RAW --dst $B --n-obj 16 --seed 0 || { echo "[10] !! rebank $T failed"; exit 1; }
  echo "[10] $T $SD: bank ready"
done
