#!/usr/bin/env bash
# 50_collect_data.sh -- planner fine-tuning data: replay the official ManiSkill demos under the production
# camera (three views: nominal, and two jittered views with eye offsets +-(0.02, 0.02, 0.01) m and target offsets
# +-(0.04, 0.03, 0) m, i.e. about +-2-3 degrees of re-aim), wall backdrop, then label dense 3D traces
# with 16 object-aware queries (msgen.labels --n-obj 16). CPU only (physx_cpu replay ~2.3 s/clip).
# 200 demos x 3 views x 5 tasks = 3,000 clips (~50 GB). Ends by building the two index datasets:
#   data/ds/realcam_n2400      four tasks (pickcube stack peg liftpeg) -> mix4_realcam_n2400
#   data/ds/realcam_t2k_n3000  five tasks (+ pushcube)                -> mix5_t2k_n3000, mix5_t2k_gmap
# Demos: `python -m mani_skill.utils.download_demo <Env-v1>` for PickCube-v1 StackCube-v1 PegInsertionSide-v1
# LiftPegUpright-v1 PushCube-v1 (default location ~/.maniskill/demos, or set MS_DEMO_DIR).
# Usage: N=200 bash scripts/50_collect_data.sh
set -u; . "$(dirname "$0")/config.sh" || exit 1
N=${N:-200}; LOG=data/collect_logs; mkdir -p $LOG
add(){ $PM -c "a=[float(x) for x in '$1'.split(',')];b=[float(x) for x in '$2'.split(',')];print(','.join(f'{x+y:.4f}' for x,y in zip(a,b)))"; }
VIEWS=("v0:0,0,0:0,0,0" "v1:0.02,-0.02,0.01:0.04,0.03,0" "v2:-0.02,0.02,-0.01:-0.04,-0.03,0")
one_task(){ local T=$1 REG=$2 SG=$3
  for V in "${VIEWS[@]}"; do
    IFS=: read -r VN DE DT <<< "$V"
    local E2 T2; E2=$(add "$EYE" "$DE"); T2=$(add "$TGT" "$DT")
    local R=data/raw/realcam_${T}_${VN} D=data/ds/realcam_${T}_${VN}
    if [ ! -f "$R/manifest.json" ]; then
      echo "[50] replay $T $VN eye=$E2 target=$T2"
      # msgen.replay does not apply the wall patch itself: run it through a wrapper that asserts it.
      env MSGEN_WALL=1 MSGEN_CAM_EYE_${REG}=$E2 MSGEN_CAM_TARGET_${REG}=$T2 MSGEN_CAM_FOV_${REG}=$FOV \
        nice -n 15 $PM - "$T" "$R" "$N" "$SG" <<'PYR' > "$LOG/replay_${T}_${VN}.log" 2>&1 || { echo "[50] !! replay failed $T $VN"; return 1; }
import sys, runpy
from msgen.patch_wall import maybe_patch
assert maybe_patch(), "MSGEN_WALL set but the wall patch did not apply"
a = ["replay.py", "--task", sys.argv[1], "--out", sys.argv[2], "--start", "0", "--episodes", sys.argv[3]]
if sys.argv[4]: a.append(sys.argv[4])
sys.argv = a; runpy.run_module("msgen.replay", run_name="__main__")
PYR
    fi
    [ -f "$D/dataset.json" ] || nice -n 15 $PM -m msgen.labels --raw "$R" --out "$D" --task "$T" --stride 2 --min-future 8 \
      --time-mode arclen --n-obj 16 > "$LOG/label_${T}_${VN}.log" 2>&1 || { echo "[50] !! label failed $T $VN"; return 1; }
    echo "[50] $T $VN: $(tail -1 $LOG/label_${T}_${VN}.log | cut -c1-80)"
  done
}
pids=()
one_task pickcube PICKCUBE "--show-goal" & pids+=($!)
one_task stack STACK "" & pids+=($!)
one_task peg PEG "" & pids+=($!)
one_task liftpeg LIFTPEG "" & pids+=($!)
one_task pushcube PUSHCUBE "" & pids+=($!)
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail = 0 ] || { echo "[50] !! a replay/label job failed (see $LOG); not building the index datasets"; exit 1; }
# index datasets: symlink farms (task-prefixed clip names, one dataset.json per task)
$PM - "$N" <<'PY'
import os, sys
N = int(sys.argv[1])
def farm(out, tasks):
    os.makedirs(out, exist_ok=True); n = 0
    for T in tasks:
        for V in ("v0", "v1", "v2"):
            D = f"data/ds/realcam_{T}_{V}"
            for c in sorted(os.listdir(D)):
                src = os.path.realpath(os.path.join(D, c))
                dst = os.path.join(out, f"{T}_dataset.json" if c == "dataset.json" else f"{T}_{V}_{c}")
                if c == "dataset.json" and V != "v0": continue
                if not os.path.lexists(dst): os.symlink(src, dst); n += 1
    print(f"[50] {out}: {n} links")
farm(f"data/ds/realcam_n{N*3*4}", ["pickcube", "stack", "peg", "liftpeg"])
farm(f"data/ds/realcam_t2k_n{N*3*5}", ["pickcube", "stack", "peg", "liftpeg", "pushcube"])
PY
echo "[50] done"
