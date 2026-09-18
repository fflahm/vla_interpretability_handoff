#!/usr/bin/env bash
# Submit occupancy GT extract on an rjob worker with many CPU cores.
# Default: --gpu=0 (CPU-only GT; still charged to pceval_gpu). Prefer no GPU.
#
# Smoke (1 demo, compare vs 开发机 GT, separate TOS prefix):
#   bash scripts/hcluster/rjob_submit_occupancy_gt.sh
#
# Full / resume (skips demos whose occupancy.npy already has 20 frames):
# IMPORTANT: for all tasks leave INCLUDE_TASKS unset or set INCLUDE_TASKS=""
# after this fix; SMOKE=0 no longer falls back to KITCHEN_SCENE3.
#   NAME=vla-occ-gt-full SMOKE=0 GPU=0 CPU=32 MEMORY=160000 NUM_WORKERS=16 \
#   OUTPUT_ROOT=/data/tos/guoshengyu/vla/occupancy INCLUDE_TASKS= \
#   STATUS_SUFFIX=rjob \
#   bash scripts/hcluster/rjob_submit_occupancy_gt.sh
set -euo pipefail

GPFS_ROOT="${GPFS_ROOT:-/mnt/shared-storage-user/guoshengyu}"
PROJECT_DIR="${PROJECT_DIR:-${GPFS_ROOT}/vla_interpretability_handoff}"
ENV_DIR="${ENV_DIR:-${GPFS_ROOT}/envs/vla-interpretability}"
WORKER="${WORKER:-${PROJECT_DIR}/scripts/hcluster/rjob_occupancy_gt_worker.sh}"
NAME="${NAME:-vla-occ-gt-smoke}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

# Prefer no GPU; GT is pure CPU MuJoCo. gpu=0 is schedulable on pceval_gpu.
GPU="${GPU:-0}"
CPU="${CPU:-32}"
MEMORY="${MEMORY:-98304}"
CHARGED_GROUP="${CHARGED_GROUP:-pceval_gpu}"
PRIVATE_MACHINE="${PRIVATE_MACHINE:-group}"
IMAGE="${IMAGE:-registry.h.pjlab.org.cn/ailab-pceval-pceval_gpu/pcgroup:ubuntu22.04-cuda12.2.2-pjlab-testv1}"

SMOKE="${SMOKE:-1}"
# Empty INCLUDE_TASKS means all tasks. Use ${VAR-default} (no colon) so
# INCLUDE_TASKS= on the command line is not overwritten by the smoke default.
if [ "${SMOKE}" = "1" ]; then
  INCLUDE_TASKS="${INCLUDE_TASKS-KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it}"
else
  INCLUDE_TASKS="${INCLUDE_TASKS-}"
fi
EXCLUDE_TASKS="${EXCLUDE_TASKS-}"
SHARD="${SHARD-}"
STATUS_SUFFIX="${STATUS_SUFFIX:-rjob}"
SUITES="${SUITES:-libero_10,libero_90}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/tos/guoshengyu/vla/occupancy_rjob_smoke}"
COMPARE_REF="${COMPARE_REF:-/data/tos/guoshengyu/vla/occupancy/gt/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it/demo_0/occupancy.npy}"
# auto => worker picks 1 (smoke) or one process per affinity CPU (full).
NUM_WORKERS="${NUM_WORKERS:-auto}"
# Per-process BLAS threads; worker forces 1 when NUM_WORKERS>1.
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MUJOCO_GL="${MUJOCO_GL:-disable}"
MAX_DEMOS="${MAX_DEMOS-}"
MAX_TASKS="${MAX_TASKS-}"
EXTRA_ARGS="${EXTRA_ARGS-}"

if [ ! -x "${ENV_DIR}/bin/python" ]; then
  echo "[ERROR] GPFS conda env missing: ${ENV_DIR}/bin/python" >&2
  exit 2
fi
if [ ! -f "${WORKER}" ]; then
  echo "[ERROR] missing worker ${WORKER}" >&2
  exit 2
fi
if [ "${SMOKE}" = "1" ] && [ "${OUTPUT_ROOT}" = "/data/tos/guoshengyu/vla/occupancy" ]; then
  echo "[ERROR] smoke must not write the live occupancy tree" >&2
  exit 2
fi

echo "submit ${NAME}"
echo "  PROJECT_DIR=${PROJECT_DIR}"
echo "  ${GPU} GPU / ${CPU} CPU / ${MEMORY} MiB"
echo "  SMOKE=${SMOKE} NUM_WORKERS=${NUM_WORKERS} INCLUDE_TASKS=${INCLUDE_TASKS}"
echo "  OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "  RUN_STAMP=${RUN_STAMP}"
echo "  log dir: ${GPFS_ROOT}/vla_rjob_runs/occupancy_gt_rjob/${RUN_STAMP}"

SUBMIT_ARGS=(
  --name="${NAME}"
  --cpu="${CPU}"
  --memory="${MEMORY}"
  --charged-group="${CHARGED_GROUP}"
  --private-machine="${PRIVATE_MACHINE}"
  --mount=gpfs://gpfs1/guoshengyu:/mnt/shared-storage-user/guoshengyu
  --custom-resources brainpp.cn/fuse=1
  --image="${IMAGE}"
  -P 1
  --delete
  -e PROJECT_DIR="${PROJECT_DIR}"
  -e ENV_DIR="${ENV_DIR}"
  -e GPFS_ROOT="${GPFS_ROOT}"
  -e RUN_STAMP="${RUN_STAMP}"
  -e SMOKE="${SMOKE}"
  -e INCLUDE_TASKS="${INCLUDE_TASKS}"
  -e EXCLUDE_TASKS="${EXCLUDE_TASKS}"
  -e SHARD="${SHARD}"
  -e STATUS_SUFFIX="${STATUS_SUFFIX}"
  -e SUITES="${SUITES}"
  -e OUTPUT_ROOT="${OUTPUT_ROOT}"
  -e COMPARE_REF="${COMPARE_REF}"
  -e OMP_NUM_THREADS="${OMP_NUM_THREADS}"
  -e MUJOCO_GL="${MUJOCO_GL}"
  -e MAX_DEMOS="${MAX_DEMOS}"
  -e MAX_TASKS="${MAX_TASKS}"
  -e EXTRA_ARGS="${EXTRA_ARGS}"
  -e NUM_WORKERS="${NUM_WORKERS}"
  -e REQUESTED_CPU="${CPU}"
)
# Only request GPU if >0; avoid tying a card for CPU GT.
if [ "${GPU}" != "0" ]; then
  SUBMIT_ARGS+=(--gpu="${GPU}" -e NVIDIA_DRIVER_CAPABILITIES=all)
else
  SUBMIT_ARGS+=(--gpu=0)
fi

rjob submit "${SUBMIT_ARGS[@]}" -- bash -exc "bash ${WORKER}"
