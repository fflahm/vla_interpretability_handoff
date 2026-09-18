#!/usr/bin/env bash
# Submit the VLA GPU smoke job.
# Matches the working DriveIQA rjob pattern, plus FUSE so the worker can s3mount TOS.
set -euo pipefail

GPFS_ROOT="${GPFS_ROOT:-/mnt/shared-storage-user/guoshengyu}"
PROJECT_DIR="${PROJECT_DIR:-${GPFS_ROOT}/vla_interpretability_handoff}"
ENV_DIR="${ENV_DIR:-${GPFS_ROOT}/envs/vla-interpretability}"
WORKER="${WORKER:-${PROJECT_DIR}/scripts/hcluster/rjob_gpu_smoke_worker.sh}"
NAME="${NAME:-vla-gpu-smoke}"

GPU="${GPU:-1}"
CPU="${CPU:-24}"
MEMORY="${MEMORY:-98304}"
CHARGED_GROUP="${CHARGED_GROUP:-pceval_gpu}"
PRIVATE_MACHINE="${PRIVATE_MACHINE:-group}"
IMAGE="${IMAGE:-registry.h.pjlab.org.cn/ailab-pceval-pceval_gpu/pcgroup:ubuntu22.04-cuda12.2.2-pjlab-testv1}"
NUM_EPISODES="${NUM_EPISODES:-1}"
MAX_STEPS="${MAX_STEPS:-80}"
TASK="${TASK:-libero_spatial}"
TASK_ID="${TASK_ID:-1}"

if [ ! -x "${ENV_DIR}/bin/python" ]; then
  echo "[ERROR] GPFS conda env missing: ${ENV_DIR}/bin/python" >&2
  echo "Clone it first, e.g. conda create --prefix ${ENV_DIR} --clone /home/guoshengyu/.conda/envs/vla-interpretability" >&2
  exit 2
fi
if [ ! -f "${WORKER}" ]; then
  echo "[ERROR] missing worker ${WORKER}" >&2
  exit 2
fi

echo "submit ${NAME}"
echo "  PROJECT_DIR=${PROJECT_DIR}"
echo "  ENV_DIR=${ENV_DIR}"
echo "  IMAGE=${IMAGE}"
echo "  ${GPU} GPU / ${CPU} CPU / ${MEMORY} MiB"

rjob submit --name="${NAME}" \
  --gpu="${GPU}" \
  --cpu="${CPU}" \
  --memory="${MEMORY}" \
  --charged-group="${CHARGED_GROUP}" \
  --private-machine="${PRIVATE_MACHINE}" \
  --mount=gpfs://gpfs1/guoshengyu:/mnt/shared-storage-user/guoshengyu \
  --custom-resources brainpp.cn/fuse=1 \
  --image="${IMAGE}" \
  -P 1 \
  --delete \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PROJECT_DIR="${PROJECT_DIR}" \
  -e ENV_DIR="${ENV_DIR}" \
  -e GPFS_ROOT="${GPFS_ROOT}" \
  -e NUM_EPISODES="${NUM_EPISODES}" \
  -e MAX_STEPS="${MAX_STEPS}" \
  -e TASK="${TASK}" \
  -e TASK_ID="${TASK_ID}" \
  -- bash -exc "bash ${WORKER}"
