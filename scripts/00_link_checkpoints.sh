#!/usr/bin/env bash
# 00_link_checkpoints.sh -- expose the downloaded weights under the run-directory layout the
# code expects (msppo.multi_eval --run runs_rl/<tag>; msppo.multi_distill --teachers <tag,...>
# reads runs_rl/<tag>/{agent.pt,run.json}). Planners are addressed by path (CK_* in config.sh).
# Usage: bash scripts/00_link_checkpoints.sh      (after `hf download ... --local-dir checkpoints`)
set -u; . "$(dirname "$0")/config.sh" || exit 1
mkdir -p runs_rl
for d in $CKPT_DIR/student/* $CKPT_DIR/teacher/*; do
  [ -d "$d" ] || continue
  t=$(basename $d); [ -e runs_rl/$t ] || ln -s "$d" runs_rl/$t
  echo "[00] runs_rl/$t -> $d"
done
for f in $CK_MIX4 $CK_HEAD $CK_GMAP; do [ -s "$f" ] || echo "[00] !! missing planner checkpoint: $f"; done
for t in $STUDENT_TAGS ${TEACHERS//,/ }; do [ -s runs_rl/$t/run.json ] || echo "[00] !! missing runs_rl/$t/run.json"; done
# --ignore-missing: downloading only student/ + teacher/ (the Table II path) is a supported partial install.
(cd $CKPT_DIR && sha256sum -c SHA256SUMS --quiet --ignore-missing && echo "[00] SHA256SUMS OK (files present)") || echo "[00] !! checksum mismatch (or SHA256SUMS absent)"
