#!/usr/bin/env bash
# 70_distill.sh -- distil the five teachers into the Pose-Native Executor (msppo.multi_distill, DAgger) with the
# paper recipe: 40k iterations, 320 envs, no scene channel, psi token, and the final-pipeline goal-error family
# injected during training (relbank per task = the error distribution of the goal source that task is deployed with:
# stack gmapdcc, pickcube gmappeak, others t2kdcc; each draw scaled by U(0,1)). INJECT selects the temporal process:
#   INJECT=iid     one draw redrawn i.i.d. at every control step (msppo.patch_iid_inject) -> mt5_rciid_gmpc_s0, the
#                  executor of the Entity-Level Goal Readout row of Table II (default)
#   INJECT=frozen  one draw per episode, frozen (msppo.patch_frozen_inject) -> mt5_rcfz_gmpc_s0, the executor of the
#                  Rigid Readout rows and the Oracle references
# Relbanks are shipped in banks/ (fit seeds: stack / pickcube 990-996, liftpeg / peginsert 990-997, pushcube 990-993;
# see docs/PROTOCOL.md section 9.3 and docs/KNOWN_ISSUES.md;
# regenerate with msppo.peg_relbank export on your own banks).
# ~6 h on one RTX 5090.   Usage: TAG=my_student bash scripts/70_distill.sh
set -u; . "$(dirname "$0")/config.sh" || exit 1
INJECT=${INJECT:-iid}; TAG=${TAG:-mt5_${INJECT}_gmpc_s0_repro}
case $INJECT in iid) IENV=MSPPO_IID_INJECT=1; IMOD=msppo.patch_iid_inject;; frozen) IENV=MSPPO_FROZEN_INJECT=1; IMOD=msppo.patch_frozen_inject;; *) echo "[70] INJECT must be iid or frozen"; exit 1;; esac
[ -e runs_rl/$TAG/run.json ] && { echo "[70] $TAG exists"; exit 0; }
mkdir -p results; for f in banks/*_relbank_*.npz; do ln -sf ../$f results/$(basename $f); done
for t in ${TEACHERS//,/ }; do [ -s runs_rl/$t/agent.pt ] || { echo "[70] !! teacher runs_rl/$t missing (00_link_checkpoints.sh or 60_train_teachers.sh)"; exit 1; }; done
REL="stack=results/stack_relbank_realcam_gmapdcc.npz,pickcube=results/pickcube_relbank_realcam_gmappeak.npz"
for T in liftpeg peginsert pushcube; do REL="$REL,${T}=results/${T}_relbank_realcam_t2kdcc.npz"; done
gpu_wait 14000
env $RC $IENV nice -n 10 $PM - "$TAG" "$REL" "$TEACHERS" "$IMOD" <<'PYE' || { echo "[70] !! distill failed"; exit 1; }
import sys, runpy, importlib
tag, rel, teachers, imod = sys.argv[1:5]
assert importlib.import_module(imod).maybe_patch(), f"{imod} did not apply"
sys.argv = ["multi_distill", "--tasks", "pickcube,liftpeg,peginsert,stack,pushcube", "--teachers", teachers,
            "--tag", tag, "--lang", "none", "--seed", "0", "--iterations", "40000", "--no-scene",
            "--goal-err-rel", rel, "--goal-err-scale", "uniform", "--psi", "--psi-token", "--num-envs", "320"]
runpy.run_module("msppo.multi_distill", run_name="__main__")
PYE
echo "[70] done: runs_rl/$TAG  (evaluate with STUDENT_RUN=runs_rl/$TAG bash scripts/40_eval_row4.sh)"
