#!/usr/bin/env bash
# config.sh -- the SINGLE source of paths, interpreters and pinned versions. Every script sources it.
# Every value can be overridden from the environment before sourcing; defaults assume the layout in README.md.
#
#   EG_REPO              this repository (default: parent of scripts/)
#   TRACEGEN_DIR          upstream TraceGen checkout, pinned commit + third_party/tracegen_local.patch applied
#   TRACEGEN_GENERALIST   upstream "Generalist" checkpoint (warm start of every planner here; zero-shot row)
#   CKPT_DIR              released weights (planner/ student/ teacher/), see README "Checkpoints"
#   PG / PM               python of the planner env (trace_gen) / the simulator+executor env (maniskill)
#   NEED_PG=1             set by the three stages that build a planner; otherwise only PM is required
#   MS_DEMO_DIR           ManiSkill demo packages (mani_skill.utils.download_demo), used by 50_collect_data.sh
REPO=${EG_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export EG_REPO=$REPO
export TRACEGEN_DIR=${TRACEGEN_DIR:-$REPO/third_party/TraceGen}
export TRACEGEN_GENERALIST=${TRACEGEN_GENERALIST:-$TRACEGEN_DIR/assets_ckpt/Generalist/tracegen_model.pth}
export CKPT_DIR=${CKPT_DIR:-$REPO/checkpoints}
export PYTHONPATH=$REPO:$TRACEGEN_DIR
# python -m prepends the CWD to sys.path ahead of PYTHONPATH; SAFEPATH removes that hole.
export PYTHONSAFEPATH=1
export MSPPO_WANDB=${MSPPO_WANDB:-0}     # msppo.wb: no online wandb session unless you opt in
# tools and modules use repo-relative data paths (data/bank, results/); pin the cwd once here.
cd "$REPO"
PG=${PG:-$HOME/miniconda3/envs/trace_gen/bin/python}
PM=${PM:-$HOME/miniconda3/envs/maniskill/bin/python}
# Only the interpreters a stage actually uses are required. The planner env (PG) is needed by
# 20_predict.sh, 30_goals.sh and 52_train_planner.sh, which set NEED_PG=1 before sourcing this file;
# every other stage, verify_main_table.sh included, runs on the simulator env (PM) alone.
NEED_PG=${NEED_PG:-0}
_req="$PM"; [ "$NEED_PG" = 1 ] && _req="$PG $PM"
for _p in $_req; do [ -x "$_p" ] || { echo "config.sh: interpreter not found: $_p (set PG / PM, see README)"; return 1 2>/dev/null || exit 1; }; done

# ---- pinned production versions (the checkpoints the paper reports) ----
TAG_MIX4=${TAG_MIX4:-mix4_realcam_n2400}   # rigid-solve source: K=4 trace mean + RANSAC Kabsch (rotation of every task; pushcube: TAG_HEAD)
TAG_HEAD=${TAG_HEAD:-mix5_t2k_n3000}       # entity head: PegInsertion full pose, LiftPeg/PushCube position
TAG_GMAP=${TAG_GMAP:-mix5_t2k_gmap}        # goal-map head: PickCube (peak + ray) and StackCube (weighted depth centroid) positions
CK_MIX4=${CK_MIX4:-$CKPT_DIR/planner/$TAG_MIX4.pth}
CK_HEAD=${CK_HEAD:-$CKPT_DIR/planner/$TAG_HEAD.pth}
CK_GMAP=${CK_GMAP:-$CKPT_DIR/planner/$TAG_GMAP.pth}
export TAG_MIX4 TAG_HEAD TAG_GMAP           # tools/*.py name their inputs and outputs by these tags; override all three together with CK_*
# Pose-Native Executors (five tasks, 804,002 params, no scene channel). Table II of the paper was evaluated with:
#   mt5_rciid_gmpc_s0  Entity-Level Goal Readout row (per-step i.i.d. goal-error injection during distillation)
#   mt5_rcfz_gmpc_s0   Rigid Readout K=1 / K=4 rows and the Oracle Goal references (episode-fixed injection)
#   mt5_rcfz_t2k_s0    the Rigid K=4 PickCube cell only (earlier executor, seed 999)
STUDENT_TAG=${STUDENT_TAG:-mt5_rciid_gmpc_s0}
STUDENT_RUN=${STUDENT_RUN:-runs_rl/$STUDENT_TAG}   # 00_link_checkpoints.sh links every $CKPT_DIR/student/* into runs_rl/
STUDENT_TAGS="mt5_rciid_gmpc_s0 mt5_rcfz_gmpc_s0 mt5_rcfz_t2k_s0"
TEACHERS="pc_v9_nz_s0,lp_v9_nz_s0,pi_v9_frame4_s0,sc_v9_nz03b_s0,push_v9_nz_s0"   # order = TASKS5

# ---- production visual protocol (front camera + back wall; identical for training data and evaluation banks) ----
EYE="0.574,-0.051,0.378"; TGT="-0.4751,0.0562,0.0200"; FOV=0.754
RC="MSGEN_WALL=1"
for _R in PICKCUBE STACK PEG LIFTPEG PUSHCUBE; do
  RC="$RC MSGEN_CAM_EYE_${_R}=$EYE MSGEN_CAM_TARGET_${_R}=$TGT MSGEN_CAM_FOV_${_R}=$FOV"
done
# planner sampler for the entity-head / goal-map / PushCube predictions: 20 fixed-step Euler steps (the native sampler
# ties the step size to a 100-step schedule). The four-task mix4 K=4 solve source was produced with the NATIVE sampler
# (production logs: only the seed shim active), so 20_predict.sh does not apply PRED_ENV there.
PRED_ENV="MSGEN_STEPS=20 MSGEN_DT=fix"
KSEEDS="1234 1235 1236 1237"       # the K=4 flow-sampler seeds; head / goal-map readouts use s1234

SEED=${SEED:-999}                  # canonical evaluation seed; 997 / 998 are the repeat seeds
N_EP=${N_EP:-256}
TASKS5="pickcube liftpeg peginsert stack pushcube"

# ---- data locations (relative to $REPO; all git-ignored) ----
BANK()  { echo data/bank/realcam_${1}_${2:-$SEED}_obj; }                  # BANK task [seed]
GOALS() { echo results/${1}_goals_${2}_${3:-$SEED}_${4}.npz; }            # GOALS task planner_tag [seed] suffix
PSI()   { echo results/${1}_psinat_${TAG_MIX4}_${2:-$SEED}.npz; }         # PSI task [seed]
# the final-pipeline goal bank per task (Entity-Level Goal Readout row of Table II). PickCube: the reported cell uses the
# SAM2 marker localiser on the goal-map peak (suffix sam2mk6; needs SAM2 to regenerate, 31_goals_sam2marker.sh);
# PICKCUBE_ROUTE=gmappeakNC selects the SAM2-free goal-map readout whose goal error is reported in Table I.
PICKCUBE_ROUTE=${PICKCUBE_ROUTE:-sam2mk6}
ROW4_GOAL() { case $1 in
  pickcube) echo results/pickcube_goals_${TAG_GMAP}_${2:-$SEED}_${PICKCUBE_ROUTE}.npz;;
  stack)    echo results/stack_goals_${TAG_GMAP}_${2:-$SEED}_gmapdcc.npz;;
  *)        echo results/${1}_goals_${TAG_HEAD}_${2:-$SEED}_t2kpos.npz;;
esac; }
ROT_TAG() { [ "$1" = pushcube ] && echo $TAG_HEAD || echo $TAG_MIX4; }    # rotation / solve source per task
# Rigid Readout banks (Table II rows K=1 / K=4): RIGID_GOAL task seed k1|k4
RIGID_GOAL() { local sfx=k1ransac; [ "$3" = k4 ] && sfx=kmean_ransac; echo results/${1}_goals_$(ROT_TAG $1)_${2}_${sfx}.npz; }
ROT_CK()  { [ "$1" = pushcube ] && echo $CK_HEAD || echo $CK_MIX4; }

. $REPO/msppo/gpu_guard.sh            # gpu_wait <MiB>: wait for free VRAM before launching
