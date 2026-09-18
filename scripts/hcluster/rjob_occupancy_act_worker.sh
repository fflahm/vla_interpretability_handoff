#!/bin/bash
# rjob worker: PI0.5 occupancy activations. GPU inference, TOS GT images reused.
set -u

GPFS_ROOT="${GPFS_ROOT:-/mnt/shared-storage-user/guoshengyu}"
PROJECT_DIR="${PROJECT_DIR:-${GPFS_ROOT}/vla_interpretability_handoff}"
ENV_DIR="${ENV_DIR:-${GPFS_ROOT}/envs/vla-interpretability}"
TOS_BUCKET="${TOS_BUCKET:-ailab-pceval}"
TOS_MOUNT="${TOS_MOUNT:-/data/tos}"
TOS_ENDPOINT="${TOS_ENDPOINT:-http://hdd1.h.pjlab.org.cn:8060}"
S3_CREDS="${S3_CREDS:-${GPFS_ROOT}/.pjlab_s3.sh}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_OUT="${RUN_OUT:-${GPFS_ROOT}/vla_rjob_runs/occupancy_act_rjob/${RUN_STAMP}}"
LOG="${RUN_OUT}/worker.log"
SMOKE="${SMOKE:-1}"
if [ "${SMOKE}" = "1" ]; then
  INCLUDE_TASKS="${INCLUDE_TASKS-KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-${TOS_MOUNT}/guoshengyu/vla/occupancy_act_rjob_smoke}"
else
  INCLUDE_TASKS="${INCLUDE_TASKS-}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-${TOS_MOUNT}/guoshengyu/vla/occupancy_activations}"
fi
OCCUPANCY_ROOT="${OCCUPANCY_ROOT:-${TOS_MOUNT}/guoshengyu/vla/occupancy}"
PI05_PATH="${PI05_PATH:-${TOS_MOUNT}/guoshengyu/vla/models/pi05_libero}"
SEED="${SEED:-42}"
NUM_TASKS="${NUM_TASKS:-50}"
TOKENS_PER_BIN="${TOKENS_PER_BIN:-10}"
LAYERS="${LAYERS:-all}"
SHARD="${SHARD-}"
STATUS_SUFFIX="${STATUS_SUFFIX:-rjob}"
EXTRA_ARGS="${EXTRA_ARGS-}"

mkdir -p "${RUN_OUT}" /tmp/vla_rjob_occupancy_act
exec > >(tee -a "${LOG}") 2>&1

echo "==== Host & GPU ===="
date
hostname
id
nvidia-smi || echo "[WARN] nvidia-smi failed"
df -h /mnt/shared-storage-user/guoshengyu /tmp "${TOS_MOUNT}" 2>/dev/null || true

echo "==== Mount TOS if needed ===="
export PATH="${GPFS_ROOT}/bin:${PATH}"
if [ -f "${TOS_MOUNT}/guoshengyu/vla/models/pi05_libero/config.json" ]; then
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
test -f "${PI05_PATH}/config.json"
test -d "${OCCUPANCY_ROOT}/images"
echo "TOS model and occupancy images OK"

echo "==== Python env ===="
export PYTHONNOUSERSITE=1
export MUJOCO_GL="${MUJOCO_GL:-disable}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY=1
export PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-${GPFS_ROOT}/models/paligemma-3b-pt-224-tokenizer}"
export HF_HOME="${HF_HOME:-${TOS_MOUNT}/guoshengyu/vla/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${TOS_MOUNT}/guoshengyu/vla/cache/torch}"
export LIBERO_ROOT="${LIBERO_ROOT:-${TOS_MOUNT}/guoshengyu/vla/libero/LIBERO}"
export LIBERO_ASSETS_PATH="${LIBERO_ASSETS_PATH:-${LIBERO_ROOT}/libero/libero/assets}"
export LIBERO_DATASETS_ROOT="${LIBERO_DATASETS_ROOT:-${TOS_MOUNT}/guoshengyu/vla/libero}"
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

SITE_LIBERO="${ENV_DIR}/lib/python3.12/site-packages/libero/libero"
if [ -d "${LIBERO_ASSETS_PATH}" ]; then
  ln -sfn "${LIBERO_ASSETS_PATH}" "${SITE_LIBERO}/assets"
fi

cd "${PROJECT_DIR}"
"${PYTHON}" -V
test -f "${PALIGEMMA_TOKENIZER}/tokenizer.model"
echo "SMOKE=${SMOKE} OUTPUT_ROOT=${OUTPUT_ROOT} INCLUDE_TASKS=${INCLUDE_TASKS}"
echo "tokens_per_bin=${TOKENS_PER_BIN} seed=${SEED} layers=${LAYERS}"

cmd=(
  "${PYTHON}" "${PROJECT_DIR}/scripts/occupancy/extract_libero_activations.py"
  --occupancy-root "${OCCUPANCY_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --pi05-path "${PI05_PATH}"
  --seed "${SEED}"
  --num-tasks "${NUM_TASKS}"
  --tokens-per-bin "${TOKENS_PER_BIN}"
  --layers "${LAYERS}"
  --status-suffix "${STATUS_SUFFIX}"
  --device auto
)
if [ -n "${INCLUDE_TASKS}" ]; then
  cmd+=(--include-tasks "${INCLUDE_TASKS}")
fi
if [ -n "${SHARD}" ]; then
  cmd+=(--shard "${SHARD}")
fi
if [ "${SMOKE}" = "1" ]; then
  cmd+=(--smoke)
fi
if [ -n "${EXTRA_ARGS}" ]; then
  # shellcheck disable=SC2206
  extra=( ${EXTRA_ARGS} )
  cmd+=("${extra[@]}")
fi
printf '%q ' "${cmd[@]}"
echo
printf "N\nN\nN\nN\nN\n" | "${cmd[@]}"
rc=$?
echo "==== extract rc=${rc} ===="
ls -la "${OUTPUT_ROOT}" || true
if [ -f "${OUTPUT_ROOT}/split.json" ]; then
  echo "---- split.json ----"
  "${PYTHON}" - <<PY
import json
p="${OUTPUT_ROOT}/split.json"
d=json.load(open(p))
print("seed", d.get("seed"), "tasks", len(d.get("tasks", [])), "smoke", d.get("smoke"))
if d.get("tasks"):
    t=d["tasks"][0]
    print("first_task", t["task"], "train", t["train"], "test", t["test"], "ablation", t["ablation"])
PY
fi
date
if [ "${rc}" -ne 0 ]; then
  echo "OCCUPANCY_ACT_RJOB_FAIL log=${LOG}"
  exit "${rc}"
fi
echo "OCCUPANCY_ACT_RJOB_OK out=${OUTPUT_ROOT} log=${LOG}"
exit 0
