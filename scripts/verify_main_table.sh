#!/usr/bin/env bash
# verify_main_table.sh -- reproduce one row of Table II (shared-executor block) for one seed from the FROZEN goal banks
# shipped in banks/, with the executor that produced that row, and compare with the shipped record in paper_results/table2/
# within a +-0.04 band (N=256 sampling noise; PegInsertionSide is not bitwise reproducible).
#   ROW=final   Entity-Level Goal Readout row: executor mt5_rciid_gmpc_s0; pickcube sam2mk6, stack gmapdcc, others t2kpos
#   ROW=k1      Rigid Readout K=1:            executor mt5_rcfz_gmpc_s0; <task>_..._k1ransac banks
#   ROW=k4      Rigid Readout K=4:            executor mt5_rcfz_gmpc_s0; <task>_..._kmean_ransac banks (the paper's PickCube
#               K=4 cell came from mt5_rcfz_t2k_s0 with the mix5_t2k_n3000 K=4 bank at seed 999; ROW=k4pick reproduces it)
#   ROW=oracle  Oracle Goal references (Fig. 5a): executor mt5_rcfz_gmpc_s0, simulator goal
# Usage: ROW=final bash scripts/verify_main_table.sh [seed]        (each row ~10 min on one GPU)
set -u; . "$(dirname "$0")/config.sh" || exit 1
SD=${1:-$SEED}; ROW=${ROW:-final}; mkdir -p results verify
for f in banks/*_${SD}_*.npz banks/*_${SD}.npz; do [ -e "$f" ] && [ ! -e results/$(basename $f) ] && ln -s ../$f results/$(basename $f); done; true
if ! ls banks/*_goals_*_${SD}_*.npz >/dev/null 2>&1; then
  echo "[verify] !! no goal banks for seed $SD in banks/. They are downloaded, not committed:"
  echo "              hf download <HF_REPO> --include \"banks/*\" --local-dir ."
  echo "           See banks/README.md, or regenerate: 10_render_banks.sh -> 20_predict.sh -> 30_goals.sh"
  exit 1
fi
PB=""; for T in pickcube liftpeg peginsert stack; do PB="$PB,${T}=$(PSI $T $SD)"; done
run(){ local STU=$1 GD=$2 OUT=$3 ONLY=${4:-${TASKS5// /,}}
  [ -s runs_rl/$STU/run.json ] || { echo "[verify] !! runs_rl/$STU missing (00_link_checkpoints.sh)"; exit 1; }
  gpu_wait 5000
  env $RC nice -n 15 $PM -m msppo.multi_eval --run runs_rl/$STU --episodes $N_EP --seed $SD --ckpt final --only $ONLY \
    ${GD:+--goal-delta "$GD"} --psi-bank "${PB#,}" --out $OUT || { echo "[verify] !! eval failed"; exit 1; }
}
case $ROW in
  final)  GD=""; for T in $TASKS5; do GD="$GD,${T}=$(ROW4_GOAL $T $SD)"; done
          run mt5_rciid_gmpc_s0 "${GD#,}" verify/final_rciid_${SD}.json; REF=paper_results/table2/final_rciid_${SD}.json
          # the PickCube cell of that row is recorded in its own file (same executor, sam2mk6 bank)
          $PM - verify/final_rciid_${SD}.json paper_results/table2/final_pickcube_sam2_rciid_${SD}.json <<'PY'
import json, sys; g, r = (json.load(open(p))["per_task"] for p in sys.argv[1:3]); d = g["pickcube"] - r["pickcube"]
print(f"  pickcube   got {g['pickcube']:.4f}  paper {r['pickcube']:.4f}  diff {d:+.4f}  {'ok' if abs(d) <= 0.04 else 'OUT OF BAND'}   (Entity-Level Goal Readout row, sam2mk6 bank)")
PY
          TASKS_CMP="liftpeg peginsert stack pushcube";;
  k1|k4)  GD=""; for T in $TASKS5; do GD="$GD,${T}=$(RIGID_GOAL $T $SD $ROW)"; done
          run mt5_rcfz_gmpc_s0 "${GD#,}" verify/rigid_${ROW}_gmpc_${SD}.json; REF=paper_results/table2/rigid_${ROW}_gmpc_${SD}.json; TASKS_CMP="$TASKS5";;
  k4pick) [ "$SD" = 999 ] || { echo "[verify] k4pick exists for seed 999 only"; exit 1; }
          run mt5_rcfz_t2k_s0 "pickcube=results/pickcube_goals_${TAG_HEAD}_999_k4ransac.npz" verify/rigid_k4_pickcube_t2k_s0_999.json pickcube
          REF=paper_results/table2/rigid_k4_pickcube_t2k_s0_999.json; TASKS_CMP="pickcube";;
  oracle) run mt5_rcfz_gmpc_s0 "" verify/oracle_gmpc_${SD}.json; REF=paper_results/table2/oracle_gmpc_${SD}.json; TASKS_CMP="$TASKS5";;
  *) echo "[verify] ROW must be final|k1|k4|k4pick|oracle"; exit 1;;
esac
GOT=$(ls -t verify/*_${SD}.json | head -1)
$PM - "$GOT" "$REF" $TASKS_CMP <<'PY'
import json, sys
got, ref = (json.load(open(p))["per_task"] for p in sys.argv[1:3]); ok = True
for t in sys.argv[3:]:
    d = got[t] - ref[t]; flag = abs(d) <= 0.04; ok &= flag
    print(f"  {t:10s} got {got[t]:.4f}  paper {ref[t]:.4f}  diff {d:+.4f}  {'ok' if flag else 'OUT OF BAND'}")
print("[verify] PASS" if ok else "[verify] FAIL"); sys.exit(0 if ok else 1)
PY
