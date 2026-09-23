#!/usr/bin/env bash
# 60_train_teachers.sh -- the five single-task PPO teachers (msppo.kp_teacher) with the recipe recorded in
# configs/teacher/<tag>/run.json (observation-noise injector "noise-v2", true noise-scale block sig_obs, StackCube
# bump penalty). Every flag is generated from that run.json so recipe and record cannot drift.
# The recorded MSPPO_* environment of each run (configs/teacher/<tag>/patches.json: psi-reward weights such as
# MSPPO_FRAME_WPUSH / MSPPO_FRAME_WH / MSPPO_FRAME_K1_BASE) is exported as well; push_v9_nz_s0 has no patches.json
# (its launcher set MSPPO_FRAME_TASK=pushcube only). The RELEASED teachers were warm-started from an earlier lineage
# (`init_from` in run.json) that is not shipped;
# this script trains from scratch with the same recipe and step budget, so expect the same ballpark, not the same
# weights. StackCube uses the wrist-camera robot variant by default (panda_wristcam, see kp_teacher.py TASKS).
# Usage: bash scripts/60_train_teachers.sh [task ...]     (default: all five; DRY=1 prints the commands)
set -u; . "$(dirname "$0")/config.sh" || exit 1
TASKS=${@:-$TASKS5}
tag_of(){ case $1 in pickcube) echo pc_v9_nz_s0;; liftpeg) echo lp_v9_nz_s0;; peginsert) echo pi_v9_frame4_s0;; stack) echo sc_v9_nz03b_s0;; pushcube) echo push_v9_nz_s0;; esac; }
for T in $TASKS; do
  TAG=$(tag_of $T); [ -e runs_rl/$TAG/run.json ] && { echo "[60] $TAG exists"; continue; }
  FLAGS=$($PM - configs/teacher/$TAG/run.json <<'PY'
import json, re, sys
cfg = json.load(open(sys.argv[1]))
known = set(re.findall(r'add_argument\("(--[a-z0-9-]+)"', open("msppo/kp_teacher.py").read()))
skip = {"tag", "task", "env", "init_from", "arm", "obs", "qdim", "has_scene", "params", "obs_dim", "branches",
        "peak_success", "final_success", "history", "fam_cfg", "robot", "reconfig_freq", "wandb_id"}
out = []
for k, v in cfg.items():
    if k in skip or v is None: continue
    f = "--" + k.replace("_", "-")
    if f not in known: continue
    if isinstance(v, bool): v = int(v)
    out += [f, str(v)]
print(" ".join(out))
PY
)
  EV="MSPPO_FRAME_TASK=$T"; [ "$T" = stack ] && EV="MSPPO_STACK_FRAME=1"
  PJ=configs/teacher/$TAG/patches.json
  [ -f $PJ ] && EV="$EV $($PM -c "import json;print(' '.join(f'{k}={v}' for k,v in json.load(open('$PJ')).get('env',{}).items()))")"
  echo "[60] $TAG: env $EV  kp_teacher --task $T --tag $TAG $FLAGS"
  [ "${DRY:-0}" = 1 ] && continue
  gpu_wait 6000
  env $EV nice -n 10 $PM - "$T" "$TAG" $FLAGS <<'PYE' || { echo "[60] !! $TAG failed"; exit 1; }
import os, sys, runpy
# the framework-teacher patch (psi = (k1, h) conditioning) must be active before the env is built
if os.environ.get("MSPPO_STACK_FRAME") == "1":
    from msppo.patch_stack_frame import maybe_patch
else:
    from msppo.patch_frame import maybe_patch
assert maybe_patch(), "frame patch did not apply"
sys.argv = ["kp_teacher", "--task", sys.argv[1], "--tag", sys.argv[2]] + sys.argv[3:]
runpy.run_module("msppo.kp_teacher", run_name="__main__")
PYE
done
