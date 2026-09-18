#!/bin/bash
# rjob worker: occupancy decoder train on GPFS, TOS read for activations/GT, sync at end.
set -u

GPFS_ROOT="${GPFS_ROOT:-/mnt/shared-storage-user/guoshengyu}"
PROJECT_DIR="${PROJECT_DIR:-${GPFS_ROOT}/vla_interpretability_handoff}"
ENV_DIR="${ENV_DIR:-${GPFS_ROOT}/envs/vla-interpretability}"
TOS_BUCKET="${TOS_BUCKET:-ailab-pceval}"
TOS_MOUNT="${TOS_MOUNT:-/data/tos}"
TOS_ENDPOINT="${TOS_ENDPOINT:-http://hdd1.h.pjlab.org.cn:8060}"
S3_CREDS="${S3_CREDS:-${GPFS_ROOT}/.pjlab_s3.sh}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${GPFS_ROOT}/vla_rjob_runs/occupancy_decoder_rjob/${RUN_STAMP}}"
LOG="${LOG_DIR}/worker.log"
SMOKE="${SMOKE:-1}"
TOWERS="${TOWERS:-paligemma}"
LAYERS="${LAYERS:-0}"
BIN_INDICES="${BIN_INDICES:-0}"
LOSS="${LOSS:-bce_dice}"
EPOCHS="${EPOCHS:-}"
GPFS_OUTPUT="${GPFS_OUTPUT:-${GPFS_ROOT}/vla_rjob_runs/occupancy_decoders/${RUN_STAMP}}"
TOS_OUTPUT="${TOS_OUTPUT:-${TOS_MOUNT}/guoshengyu/vla/occupancy_decoders/${RUN_STAMP}}"
ACTIVATION_ROOT="${ACTIVATION_ROOT:-${TOS_MOUNT}/guoshengyu/vla/occupancy_activations}"
OCCUPANCY_ROOT="${OCCUPANCY_ROOT:-${TOS_MOUNT}/guoshengyu/vla/occupancy}"
EXTRA_ARGS="${EXTRA_ARGS-}"

mkdir -p "${LOG_DIR}" "${GPFS_OUTPUT}" /tmp/vla_rjob_occupancy_dec
exec > >(tee -a "${LOG}") 2>&1

echo "==== Host & GPU ===="
date
hostname
id
nvidia-smi || echo "[WARN] nvidia-smi failed"
df -h /mnt/shared-storage-user/guoshengyu /tmp "${TOS_MOUNT}" 2>/dev/null || true

echo "==== Mount TOS if needed ===="
export PATH="${GPFS_ROOT}/bin:${PATH}"
if [ -f "${ACTIVATION_ROOT}/split.json" ]; then
  echo "TOS activations visible at ${ACTIVATION_ROOT}"
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
test -f "${ACTIVATION_ROOT}/split.json"
test -d "${OCCUPANCY_ROOT}/gt"
echo "TOS split.json and occupancy GT OK"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TMPDIR=/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="${ENV_DIR}/bin:${PATH}"
PYTHON="${ENV_DIR}/bin/python"
if [ ! -x "${PYTHON}" ]; then
  echo "[ERROR] ${PYTHON} not found" >&2
  exit 2
fi

cmd=(
  "${PYTHON}" "${PROJECT_DIR}/scripts/occupancy/train_libero_decoders.py"
  --activation-root "${ACTIVATION_ROOT}"
  --occupancy-root "${OCCUPANCY_ROOT}"
  --output-dir "${GPFS_OUTPUT}"
  --tos-output-dir "${TOS_OUTPUT}"
  --towers "${TOWERS}"
  --layers "${LAYERS}"
  --bin-indices "${BIN_INDICES}"
  --loss "${LOSS}"
  --device auto
  --sync-tos
  --no-plot-curves
  --drop-layer-cache
)
if [ "${SMOKE}" = "1" ]; then
  cmd+=(--smoke --plot-curves)
fi
if [ -n "${EPOCHS}" ]; then
  cmd+=(--epochs "${EPOCHS}")
fi
if [ -n "${EXTRA_ARGS}" ]; then
  # shellcheck disable=SC2206
  extra=( ${EXTRA_ARGS} )
  cmd+=("${extra[@]}")
fi
printf '%q ' "${cmd[@]}"
echo
"${cmd[@]}"
rc=$?
echo "==== train rc=${rc} ===="
ls -la "${GPFS_OUTPUT}" | head
date
if [ "${rc}" -ne 0 ]; then
  echo "OCCUPANCY_DEC_RJOB_FAIL log=${LOG}"
  exit "${rc}"
fi
echo "OCCUPANCY_DEC_RJOB_OK gpfs=${GPFS_OUTPUT} tos=${TOS_OUTPUT} log=${LOG}"
exit 0
