#!/usr/bin/env bash
# Submit PI0.5 occupancy activation extract (GPU). Reuses TOS GT images.
#
# Smoke (1 complete libero-90 task, 1 demo, 4 frames, isolated TOS prefix):
#   bash scripts/hcluster/rjob_submit_occupancy_act.sh
#
# Full / resume (50 tasks x 40 train/test demos x 20 frames; skips complete npy):
#   NAME=vla-occ-act-full SMOKE=0 GPU=1 CPU=16 MEMORY=98304 \
#   OUTPUT_ROOT=/data/tos/guoshengyu/vla/occupancy_activations \
#   bash scripts/hcluster/rjob_submit_occupancy_act.sh
#
# A previous full job was SIGTERM'd after ~3.3h (Stopped, not a Python crash).
# Resume reuses split.json and skips intact demos. --delete only removes the
# rjob metadata with the same --name; it does not delete TOS npy.
set -euo pipefail

GPFS_ROOT="${GPFS_ROOT:-/mnt/shared-storage-user/guoshengyu}"
PROJECT_DIR="${PROJECT_DIR:-${GPFS_ROOT}/vla_interpretability_handoff}"
ENV_DIR="${ENV_DIR:-${GPFS_ROOT}/envs/vla-interpretability}"
WORKER="${WORKER:-${PROJECT_DIR}/scripts/hcluster/rjob_occupancy_act_worker.sh}"
NAME="${NAME:-vla-occ-act-smoke}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

GPU="${GPU:-1}"
CPU="${CPU:-16}"
MEMORY="${MEMORY:-98304}"
CHARGED_GROUP="${CHARGED_GROUP:-pceval_gpu}"
PRIVATE_MACHINE="${PRIVATE_MACHINE:-group}"
PREEMPTIBLE="${PREEMPTIBLE:-no}"
IMAGE="${IMAGE:-registry.h.pjlab.org.cn/ailab-pceval-pceval_gpu/pcgroup:ubuntu22.04-cuda12.2.2-pjlab-testv1}"

SMOKE="${SMOKE:-1}"
if [ "${SMOKE}" = "1" ]; then
  INCLUDE_TASKS="${INCLUDE_TASKS-KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-/data/tos/guoshengyu/vla/occupancy_act_rjob_smoke}"
else
  INCLUDE_TASKS="${INCLUDE_TASKS-}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-/data/tos/guoshengyu/vla/occupancy_activations}"
fi
OCCUPANCY_ROOT="${OCCUPANCY_ROOT:-/data/tos/guoshengyu/vla/occupancy}"
PI05_PATH="${PI05_PATH:-/data/tos/guoshengyu/vla/models/pi05_libero}"
SEED="${SEED:-42}"
NUM_TASKS="${NUM_TASKS:-50}"
TOKENS_PER_BIN="${TOKENS_PER_BIN:-10}"
LAYERS="${LAYERS:-all}"
SHARD="${SHARD-}"
STATUS_SUFFIX="${STATUS_SUFFIX:-rjob}"
EXTRA_ARGS="${EXTRA_ARGS-}"

if [ ! -x "${ENV_DIR}/bin/python" ]; then
  echo "[ERROR] GPFS conda env missing: ${ENV_DIR}/bin/python" >&2
  exit 2
fi
if [ ! -f "${WORKER}" ]; then
  echo "[ERROR] missing worker ${WORKER}" >&2
  exit 2
fi
if [ "${SMOKE}" = "1" ] && [ "${OUTPUT_ROOT}" = "/data/tos/guoshengyu/vla/occupancy_activations" ]; then
  echo "[ERROR] smoke must not write the live activation tree" >&2
  exit 2
fi

echo "submit ${NAME}"
echo "  SMOKE=${SMOKE} GPU=${GPU} CPU=${CPU} MEMORY=${MEMORY} PREEMPTIBLE=${PREEMPTIBLE}"
echo "  INCLUDE_TASKS=${INCLUDE_TASKS}"
echo "  OUTPUT_ROOT=${OUTPUT_ROOT}"
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
  -e INCLUDE_TASKS="${INCLUDE_TASKS}"
  -e OCCUPANCY_ROOT="${OCCUPANCY_ROOT}"
  -e OUTPUT_ROOT="${OUTPUT_ROOT}"
  -e PI05_PATH="${PI05_PATH}"
  -e SEED="${SEED}"
  -e NUM_TASKS="${NUM_TASKS}"
  -e TOKENS_PER_BIN="${TOKENS_PER_BIN}"
  -e LAYERS="${LAYERS}"
  -e SHARD="${SHARD}"
  -e STATUS_SUFFIX="${STATUS_SUFFIX}"
  -e EXTRA_ARGS="${EXTRA_ARGS}"
)

rjob submit "${SUBMIT_ARGS[@]}" -- bash -exc "bash ${WORKER}"
