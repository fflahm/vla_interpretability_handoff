#!/usr/bin/env bash
# Submit scripts/occupancy/run_demo.py on a GPU worker.
# Same proven rjob pattern as the VLA GPU smoke job.
set -euo pipefail

GPFS_ROOT="${GPFS_ROOT:-/mnt/shared-storage-user/guoshengyu}"
PROJECT_DIR="${PROJECT_DIR:-${GPFS_ROOT}/vla_interpretability_handoff}"
ENV_DIR="${ENV_DIR:-${GPFS_ROOT}/envs/vla-interpretability}"
WORKER="${WORKER:-${PROJECT_DIR}/scripts/hcluster/rjob_occupancy_demo_worker.sh}"
NAME="${NAME:-vla-occupancy-demo}"

GPU="${GPU:-1}"
CPU="${CPU:-16}"
MEMORY="${MEMORY:-65536}"
CHARGED_GROUP="${CHARGED_GROUP:-pceval_gpu}"
PRIVATE_MACHINE="${PRIVATE_MACHINE:-group}"
IMAGE="${IMAGE:-registry.h.pjlab.org.cn/ailab-pceval-pceval_gpu/pcgroup:ubuntu22.04-cuda12.2.2-pjlab-testv1}"
NUM_DEMOS="${NUM_DEMOS:-6}"
FRAMES_PER_DEMO="${FRAMES_PER_DEMO:-4}"
EPOCHS="${EPOCHS:-160}"

if [ ! -x "${ENV_DIR}/bin/python" ]; then
  echo "[ERROR] GPFS conda env missing: ${ENV_DIR}/bin/python" >&2
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
echo "  demos=${NUM_DEMOS} frames=${FRAMES_PER_DEMO} epochs=${EPOCHS}"

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
  -e NUM_DEMOS="${NUM_DEMOS}" \
  -e FRAMES_PER_DEMO="${FRAMES_PER_DEMO}" \
  -e EPOCHS="${EPOCHS}" \
  -- bash -exc "bash ${WORKER}"
