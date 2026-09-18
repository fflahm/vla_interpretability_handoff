#!/usr/bin/env bash
# Submit occupancy decoder training. Writes GPFS, then inplace-copies to TOS.
#
# Smoke (CPU-ok, but still use rjob if you want GPU):
#   bash scripts/hcluster/rjob_submit_occupancy_decoder.sh
#
# Full (1 H200, ~96-160G, all layers/bins, sync TOS at end):
#   NAME=vla-occ-dec-full SMOKE=0 GPU=1 CPU=16 MEMORY=160000 \
#   TOWERS=all LAYERS=all BIN_INDICES=all \
#   bash scripts/hcluster/rjob_submit_occupancy_decoder.sh
set -euo pipefail

GPFS_ROOT="${GPFS_ROOT:-/mnt/shared-storage-user/guoshengyu}"
PROJECT_DIR="${PROJECT_DIR:-${GPFS_ROOT}/vla_interpretability_handoff}"
ENV_DIR="${ENV_DIR:-${GPFS_ROOT}/envs/vla-interpretability}"
WORKER="${WORKER:-${PROJECT_DIR}/scripts/hcluster/rjob_occupancy_decoder_worker.sh}"
NAME="${NAME:-vla-occ-dec-smoke}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

GPU="${GPU:-1}"
CPU="${CPU:-16}"
MEMORY="${MEMORY:-98304}"
CHARGED_GROUP="${CHARGED_GROUP:-pceval_gpu}"
PRIVATE_MACHINE="${PRIVATE_MACHINE:-group}"
PREEMPTIBLE="${PREEMPTIBLE:-no}"
IMAGE="${IMAGE:-registry.h.pjlab.org.cn/ailab-pceval-pceval_gpu/pcgroup:ubuntu22.04-cuda12.2.2-pjlab-testv1}"

SMOKE="${SMOKE:-1}"
TOWERS="${TOWERS:-paligemma}"
LAYERS="${LAYERS:-0}"
BIN_INDICES="${BIN_INDICES:-0}"
LOSS="${LOSS:-bce_dice}"
EPOCHS="${EPOCHS:-}"
GPFS_OUTPUT="${GPFS_OUTPUT:-${GPFS_ROOT}/vla_rjob_runs/occupancy_decoders/${RUN_STAMP}}"
TOS_OUTPUT="${TOS_OUTPUT:-/data/tos/guoshengyu/vla/occupancy_decoders/${RUN_STAMP}}"
ACTIVATION_ROOT="${ACTIVATION_ROOT:-/data/tos/guoshengyu/vla/occupancy_activations}"
OCCUPANCY_ROOT="${OCCUPANCY_ROOT:-/data/tos/guoshengyu/vla/occupancy}"
EXTRA_ARGS="${EXTRA_ARGS-}"

if [ ! -x "${ENV_DIR}/bin/python" ]; then
  echo "[ERROR] GPFS conda env missing: ${ENV_DIR}/bin/python" >&2
  exit 2
fi
if [ ! -f "${WORKER}" ]; then
  echo "[ERROR] missing worker ${WORKER}" >&2
  exit 2
fi

echo "submit ${NAME}"
echo "  SMOKE=${SMOKE} GPU=${GPU} CPU=${CPU} MEMORY=${MEMORY} PREEMPTIBLE=${PREEMPTIBLE}"
echo "  TOWERS=${TOWERS} LAYERS=${LAYERS} BIN_INDICES=${BIN_INDICES} LOSS=${LOSS}"
echo "  GPFS_OUTPUT=${GPFS_OUTPUT}"
echo "  TOS_OUTPUT=${TOS_OUTPUT}"
echo "  RUN_STAMP=${RUN_STAMP}"

SUBMIT_ARGS=(
  --name="${NAME}"
  --gpu="${GPU}"
  --cpu="${CPU}"
  --memory="${MEMORY}"
  --charged-group="${CHARGED_GROUP}"
  --private-machine="${PRIVATE_MACHINE}"
  --preemptible="${PREEMPTIBLE}"
  --mount=gpfs://gpfs1/guoshengyu:/mnt/shared-storage-user/guoshengyu
  --custom-resources brainpp.cn/fuse=1
  --image="${IMAGE}"
  -P 1
  --delete
  -e NVIDIA_DRIVER_CAPABILITIES=all
  -e PROJECT_DIR="${PROJECT_DIR}"
  -e ENV_DIR="${ENV_DIR}"
  -e GPFS_ROOT="${GPFS_ROOT}"
  -e RUN_STAMP="${RUN_STAMP}"
  -e SMOKE="${SMOKE}"
  -e TOWERS="${TOWERS}"
  -e LAYERS="${LAYERS}"
  -e BIN_INDICES="${BIN_INDICES}"
  -e LOSS="${LOSS}"
  -e EPOCHS="${EPOCHS}"
  -e GPFS_OUTPUT="${GPFS_OUTPUT}"
  -e TOS_OUTPUT="${TOS_OUTPUT}"
  -e ACTIVATION_ROOT="${ACTIVATION_ROOT}"
  -e OCCUPANCY_ROOT="${OCCUPANCY_ROOT}"
  -e EXTRA_ARGS="${EXTRA_ARGS}"
)

rjob submit "${SUBMIT_ARGS[@]}" -- bash -exc "bash ${WORKER}"
