#!/bin/bash
# rjob worker: Libero occupancy GT on high-CPU nodes (MuJoCo CPU only).
# Spawns NUM_WORKERS parallel extract processes (task shards) so many cores
# are used. Each process keeps the same GT kernel as the 开发机 job.
set -u

GPFS_ROOT="${GPFS_ROOT:-/mnt/shared-storage-user/guoshengyu}"
PROJECT_DIR="${PROJECT_DIR:-${GPFS_ROOT}/vla_interpretability_handoff}"
ENV_DIR="${ENV_DIR:-${GPFS_ROOT}/envs/vla-interpretability}"
TOS_BUCKET="${TOS_BUCKET:-ailab-pceval}"
TOS_MOUNT="${TOS_MOUNT:-/data/tos}"
TOS_ENDPOINT="${TOS_ENDPOINT:-http://hdd1.h.pjlab.org.cn:8060}"
S3_CREDS="${S3_CREDS:-${GPFS_ROOT}/.pjlab_s3.sh}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_OUT="${RUN_OUT:-${GPFS_ROOT}/vla_rjob_runs/occupancy_gt_rjob/${RUN_STAMP}}"
LOG="${RUN_OUT}/worker.log"
SMOKE="${SMOKE:-1}"
# Empty means all tasks. Do not use :- here (empty would fall back to SCENE3).
if [ "${SMOKE}" = "1" ]; then
  INCLUDE_TASKS="${INCLUDE_TASKS-KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it}"
else
  INCLUDE_TASKS="${INCLUDE_TASKS-}"
fi
EXCLUDE_TASKS="${EXCLUDE_TASKS-}"
SHARD="${SHARD-}"
STATUS_SUFFIX="${STATUS_SUFFIX:-rjob}"
MAX_DEMOS="${MAX_DEMOS-}"
MAX_TASKS="${MAX_TASKS-}"
SUITES="${SUITES:-libero_10,libero_90}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${TOS_MOUNT}/guoshengyu/vla/occupancy_rjob_smoke}"
TOS_LIBERO="${TOS_LIBERO:-${TOS_MOUNT}/guoshengyu/vla/libero}"
COMPARE_REF="${COMPARE_REF-}"
EXTRA_ARGS="${EXTRA_ARGS-}"
# auto | integer. auto = one process per affinity CPU (capped). Smoke defaults to 1.
NUM_WORKERS="${NUM_WORKERS:-auto}"
REQUESTED_CPU="${REQUESTED_CPU:-}"

mkdir -p "${RUN_OUT}" /tmp/vla_rjob_occupancy_gt
exec > >(tee -a "${LOG}") 2>&1

echo "==== Host ===="
date
hostname
id
echo "HOME=${HOME:-}"
echo "---- CPU probe ----"
nproc
nproc --all || true
lscpu | grep -E 'CPU\(s\)|Model name|Thread|Core|Socket|On-line|Off-line' || true
echo "cpuset.cpus=$(cat /sys/fs/cgroup/cpuset.cpus 2>/dev/null || cat /sys/fs/cgroup/cpuset/cpuset.cpus 2>/dev/null || echo n/a)"
echo "cpu.max=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a)"
if [ -f /sys/fs/cgroup/cpu.max ]; then
  read -r cpu_quota cpu_period < /sys/fs/cgroup/cpu.max || true
  if [ "${cpu_quota:-max}" != "max" ] && [ -n "${cpu_period:-}" ] && [ "${cpu_period}" -gt 0 ] 2>/dev/null; then
    echo "cpu.quota_cores=$(awk -v q="$cpu_quota" -v p="$cpu_period" 'BEGIN{printf "%.2f", q/p}')"
  fi
fi
"${ENV_DIR}/bin/python" - <<'PY' 2>/dev/null || true
import os
print("affinity", sorted(os.sched_getaffinity(0)))
print("affinity_count", len(os.sched_getaffinity(0)))
PY
df -h /mnt/shared-storage-user/guoshengyu /tmp "${TOS_MOUNT}" 2>/dev/null || true
nvidia-smi -L 2>/dev/null || echo "[info] no GPU visible (OK for GT)"

echo "==== Mount TOS if needed ===="
export PATH="${GPFS_ROOT}/bin:${PATH}"
if [ -f "${TOS_MOUNT}/guoshengyu/vla/libero/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5" ]; then
  echo "TOS already visible at ${TOS_MOUNT}"
else
  if [ ! -f "${S3_CREDS}" ]; then
    echo "[ERROR] missing ${S3_CREDS}" >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  source "${S3_CREDS}"
  mkdir -p "${TOS_MOUNT}"
  S3MOUNT_BIN="$(command -v s3mount || true)"
  if [ -z "${S3MOUNT_BIN}" ] && [ -x "${GPFS_ROOT}/bin/s3mount" ]; then
    S3MOUNT_BIN="${GPFS_ROOT}/bin/s3mount"
  fi
  if [ -z "${S3MOUNT_BIN}" ]; then
    echo "[ERROR] s3mount not in PATH or ${GPFS_ROOT}/bin/s3mount" >&2
    exit 2
  fi
  echo "using s3mount=${S3MOUNT_BIN}"
  if command -v apt-get >/dev/null 2>&1; then
    (sudo apt-get update -qq && sudo apt-get install -y -qq libfuse2 fuse3 >/dev/null) \
      || echo "[WARN] apt fuse install failed; continuing"
  fi
  if mountpoint -q "${TOS_MOUNT}" 2>/dev/null; then
    echo "unmounting stale ${TOS_MOUNT}"
    fusermount3 -u "${TOS_MOUNT}" || fusermount -u "${TOS_MOUNT}" || true
  fi
  "${S3MOUNT_BIN}" "${TOS_BUCKET}" "${TOS_MOUNT}" \
    --endpoint-url "${TOS_ENDPOINT}" \
    --allow-delete --allow-overwrite --force-path-style \
    || "${S3MOUNT_BIN}" "${TOS_BUCKET}" "${TOS_MOUNT}" \
      --endpoint-url "${TOS_ENDPOINT}" \
      --allow-delete --allow-overwrite --force-path-style --allow-other
  sleep 2
fi
test -f "${TOS_MOUNT}/guoshengyu/vla/libero/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5"
echo "TOS LIBERO hdf5 OK"

echo "==== Python env (CPU MuJoCo, same GT kernel as 开发机) ===="
export PYTHONNOUSERSITE=1
export MUJOCO_GL="${MUJOCO_GL:-disable}"
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY=1
export LIBERO_ROOT="${LIBERO_ROOT:-${TOS_LIBERO}/LIBERO}"
export LIBERO_ASSETS_PATH="${LIBERO_ASSETS_PATH:-${LIBERO_ROOT}/libero/libero/assets}"
export LIBERO_DATASETS_ROOT="${LIBERO_DATASETS_ROOT:-${TOS_LIBERO}}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${HOME:-/root}/.libero}"
export CONDA_PREFIX="${ENV_DIR}"
export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
PYTHON="${ENV_DIR}/bin/python"
if [ ! -x "${PYTHON}" ]; then
  echo "[ERROR] ${PYTHON} not found" >&2
  exit 2
fi

LIBERO_BENCH="${LIBERO_ROOT}/libero/libero"
write_libero_config() {
  local dest="$1"
  mkdir -p "${dest}"
  cat > "${dest}/config.yaml" <<EOF
assets: ${LIBERO_BENCH}/assets
bddl_files: ${LIBERO_BENCH}/bddl_files
benchmark_root: ${LIBERO_BENCH}
datasets: ${LIBERO_DATASETS_ROOT}
init_states: ${LIBERO_BENCH}/init_files
EOF
  echo "wrote ${dest}/config.yaml"
}
write_libero_config "${LIBERO_CONFIG_PATH}"
if [ "${LIBERO_CONFIG_PATH}" != "/root/.libero" ]; then
  write_libero_config /root/.libero
fi
if [ -n "${HOME:-}" ] && [ "${HOME}/.libero" != "${LIBERO_CONFIG_PATH}" ]; then
  write_libero_config "${HOME}/.libero"
fi

SITE_LIBERO="${ENV_DIR}/lib/python3.12/site-packages/libero/libero"
if [ -d "${LIBERO_ASSETS_PATH}" ]; then
  ln -sfn "${LIBERO_ASSETS_PATH}" "${SITE_LIBERO}/assets"
  echo "libero assets -> ${SITE_LIBERO}/assets -> ${LIBERO_ASSETS_PATH}"
else
  echo "[ERROR] missing LIBERO assets ${LIBERO_ASSETS_PATH}" >&2
  exit 2
fi

cd "${PROJECT_DIR}"
"${PYTHON}" -V

# Effective cores: prefer sched affinity (true cpuset). Do NOT trust bare
# `nproc` on these workers — LXCFS/CFS often reports nproc=1 while affinity
# still has the full --cpu grant (seen: nproc=1, affinity=32).
AFFINITY_COUNT="$("${PYTHON}" -c 'import os; print(len(os.sched_getaffinity(0)))')"
NPROC_COUNT="$(nproc)"
EFFECTIVE_CPU="${AFFINITY_COUNT}"
if [ -n "${REQUESTED_CPU}" ] && [ "${REQUESTED_CPU}" -gt 0 ] 2>/dev/null; then
  if [ "${REQUESTED_CPU}" -lt "${EFFECTIVE_CPU}" ]; then
    EFFECTIVE_CPU="${REQUESTED_CPU}"
  fi
fi
if [ "${AFFINITY_COUNT}" -ge 2 ] && [ "${NPROC_COUNT}" -lt 2 ]; then
  echo "[WARN] nproc=${NPROC_COUNT} looks bogus vs affinity=${AFFINITY_COUNT}; ignoring nproc"
elif [ "${NPROC_COUNT}" -gt 0 ] && [ "${NPROC_COUNT}" -lt "${EFFECTIVE_CPU}" ] && [ "${NPROC_COUNT}" -ge 2 ]; then
  echo "[WARN] nproc=${NPROC_COUNT} < affinity=${AFFINITY_COUNT}; capping to nproc"
  EFFECTIVE_CPU="${NPROC_COUNT}"
fi

if [ "${NUM_WORKERS}" = "auto" ]; then
  if [ "${SMOKE}" = "1" ]; then
    NUM_WORKERS=1
  else
    # One MuJoCo extract process per granted CPU; BLAS threads forced to 1 below.
    NUM_WORKERS="${EFFECTIVE_CPU}"
    if [ -n "${REQUESTED_CPU}" ] && [ "${REQUESTED_CPU}" -gt 0 ] 2>/dev/null; then
      if [ "${REQUESTED_CPU}" -lt "${NUM_WORKERS}" ]; then
        NUM_WORKERS="${REQUESTED_CPU}"
      fi
    fi
    if [ "${NUM_WORKERS}" -gt 64 ]; then
      NUM_WORKERS=64
    fi
    if [ "${NUM_WORKERS}" -lt 1 ]; then
      NUM_WORKERS=1
    fi
  fi
fi

# Multi-process: each worker must use 1 BLAS thread to avoid oversubscription.
if [ "${NUM_WORKERS}" -gt 1 ]; then
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
else
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${OMP_NUM_THREADS}}"
fi

echo "MUJOCO_GL=${MUJOCO_GL} OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "affinity=${AFFINITY_COUNT} nproc=${NPROC_COUNT} effective_cpu=${EFFECTIVE_CPU} NUM_WORKERS=${NUM_WORKERS}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "SMOKE=${SMOKE} INCLUDE_TASKS=${INCLUDE_TASKS} SHARD=${SHARD:-none}"
echo "COMPARE_REF=${COMPARE_REF}"

run_extract() {
  local suffix="$1"
  local shard_arg="$2"
  local -a cmd
  cmd=(
    "${PYTHON}" "${PROJECT_DIR}/scripts/occupancy/extract_libero_gt.py"
    --libero-data "${TOS_LIBERO}"
    --libero-root "${LIBERO_ROOT}"
    --output-root "${OUTPUT_ROOT}"
    --suites "${SUITES}"
    --status-suffix "${suffix}"
  )
  if [ -n "${INCLUDE_TASKS}" ]; then
    cmd+=(--include-tasks "${INCLUDE_TASKS}")
  fi
  if [ -n "${EXCLUDE_TASKS}" ]; then
    cmd+=(--exclude-tasks "${EXCLUDE_TASKS}")
  fi
  if [ -n "${shard_arg}" ]; then
    cmd+=(--shard "${shard_arg}")
  fi
  if [ "${SMOKE}" = "1" ]; then
    cmd+=(--smoke)
    if [ -n "${COMPARE_REF}" ]; then
      cmd+=(--compare-ref "${COMPARE_REF}")
    fi
  else
    if [ -n "${MAX_DEMOS}" ]; then
      cmd+=(--max-demos "${MAX_DEMOS}")
    fi
    if [ -n "${MAX_TASKS}" ]; then
      cmd+=(--max-tasks "${MAX_TASKS}")
    fi
  fi
  if [ -n "${EXTRA_ARGS}" ]; then
    # Intentional word-split of extra CLI tokens from the submit env.
    # shellcheck disable=SC2206
    local -a extra=( ${EXTRA_ARGS} )
    cmd+=("${extra[@]}")
  fi
  printf '%q ' "${cmd[@]}"
  echo
  printf "N\nN\nN\nN\nN\n" | "${cmd[@]}"
}

if [ -n "${SHARD}" ] && [ "${NUM_WORKERS}" -gt 1 ]; then
  echo "[ERROR] set either SHARD=i/N or NUM_WORKERS>1, not both" >&2
  exit 2
fi

fail=0
if [ "${NUM_WORKERS}" -le 1 ]; then
  echo "==== extract single process ===="
  if ! run_extract "${STATUS_SUFFIX}" "${SHARD}"; then
    fail=1
  fi
else
  echo "==== extract ${NUM_WORKERS} parallel task-shard processes ===="
  pids=()
  for ((i = 0; i < NUM_WORKERS; i++)); do
    worker_log="${RUN_OUT}/worker_shard_${i}.log"
    suffix="${STATUS_SUFFIX}${i}"
    shard_arg="${i}/${NUM_WORKERS}"
    (
      echo "[shard ${shard_arg}] start $(date -Is)"
      if ! run_extract "${suffix}" "${shard_arg}"; then
        echo "[shard ${shard_arg}] FAILED"
        exit 1
      fi
      echo "[shard ${shard_arg}] done $(date -Is)"
    ) >"${worker_log}" 2>&1 &
    pids+=("$!")
    echo "launched shard ${shard_arg} pid=${pids[-1]} log=${worker_log}"
  done
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      fail=1
    fi
  done
  echo "==== per-shard tails ===="
  for ((i = 0; i < NUM_WORKERS; i++)); do
    echo "---- shard ${i} ----"
    tail -n 25 "${RUN_OUT}/worker_shard_${i}.log" || true
  done
fi

echo "==== extract done fail=${fail} ===="
ls -la "${OUTPUT_ROOT}/gt" 2>/dev/null || true
date
if [ "${fail}" -eq 0 ]; then
  echo "OCCUPANCY_GT_RJOB_OK out=${OUTPUT_ROOT} workers=${NUM_WORKERS} log=${LOG}"
fi
exit "${fail}"
