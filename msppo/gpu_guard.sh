# VRAM and concurrency guard for queue scripts. Source it, then call
# `gpu_wait <MiB>` and `jobs_wait <n>` before each launch.

# GPU_WAIT=0 skips the wait (single-user machines, CI); nvidia-smi is required otherwise -- without it the
# old loop printed 'free 0 MiB' forever.
gpu_free() {
  local u t
  command -v nvidia-smi >/dev/null 2>&1 || { echo "gpu_guard: nvidia-smi not found (set GPU_WAIT=0 to skip the VRAM wait)" >&2; echo 0; return; }
  read u t < <(nvidia-smi --query-gpu=memory.used,memory.total \
    --format=csv,noheader,nounits | tr ',' ' ')
  echo $((t - u))
}

gpu_wait() {
  local need=${1:-8000}
  [ "${GPU_WAIT:-1}" = 0 ] && return 0
  command -v nvidia-smi >/dev/null 2>&1 || { echo "gpu_guard: nvidia-smi not found; refusing to wait forever (GPU_WAIT=0 to skip)" >&2; return 1; }
  while [ "$(gpu_free)" -lt "$need" ]; do
    echo "[$(date '+%F %T')] free $(gpu_free) MiB < ${need}; waiting"
    sleep 60
  done
}

# Concurrency helpers for queue scripts; no released script calls them.
jobs_running() {
  pgrep -f 'msppo\.(multi_distill|kp_teacher|distill|peg_ppo|train_rl)' 2>/dev/null | wc -l
}

jobs_wait() {
  local cap=${1:-3}
  while [ "$(jobs_running)" -ge "$cap" ]; do
    sleep 30
  done
}
