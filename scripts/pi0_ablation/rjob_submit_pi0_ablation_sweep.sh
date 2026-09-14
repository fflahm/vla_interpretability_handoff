#!/bin/bash
set -euo pipefail

# Submit one baseline job plus N shard jobs for PI0 activation-ablation sweeping.
# Override any variable below from the shell, e.g.:
#   NUM_SHARDS=12 BIN_STRIDE=2 NUM_EPISODES=5 ./scripts/pi0_ablation/rjob_submit_pi0_ablation_sweep.sh

NUM_SHARDS="${NUM_SHARDS:-8}"
JOB_PREFIX="${JOB_PREFIX:-pi0-ablate-spatial-t1}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
WORKER="${WORKER:-${PROJECT_DIR}/scripts/pi0_ablation/rjob_pi0_ablation_worker.sh}"
ENV_DIR="${ENV_DIR:-}"

GPU="${GPU:-1}"
CPU="${CPU:-32}"
MEMORY="${MEMORY:-160000}"
CHARGED_GROUP="${CHARGED_GROUP:-pceval_gpu}"
PRIVATE_MACHINE="${PRIVATE_MACHINE:-group}"
IMAGE="${IMAGE:-registry.h.pjlab.org.cn/ailab-pceval-pceval_gpu/pcgroup:ubuntu22.04-cuda12.2.2-pjlab-testv1}"
HOST_NETWORK="${HOST_NETWORK:-false}"
RJOB_MOUNT_1="${RJOB_MOUNT_1:-}"
RJOB_MOUNT_2="${RJOB_MOUNT_2:-}"

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/ablation/pi0_ablation_spatial_task1_full_sweep}"
PI0_PATH="${PI0_PATH:-}"
TASK="${TASK:-libero_spatial}"
TASK_ID="${TASK_ID:-1}"
INSTRUCTION="${INSTRUCTION:-pick up the black bowl from table center and place it on the plate}"
NUM_EPISODES="${NUM_EPISODES:-2}"
MAX_STEPS="${MAX_STEPS:-250}"
LAYERS="${LAYERS:-all}"
TOKEN_BINS="${TOKEN_BINS:-96}"
BIN_INDICES="${BIN_INDICES:-all}"
BIN_STRIDE="${BIN_STRIDE:-4}"
MODE_ABLATION="${MODE_ABLATION:-zero}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"
SAVE_ACTIVATIONS="${SAVE_ACTIVATIONS:-0}"
BASELINE_WAIT_SECONDS="${BASELINE_WAIT_SECONDS:-7200}"
AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-}"
AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-}"

if [ -z "${PI0_PATH}" ]; then
  echo "[ERROR] PI0_PATH is not set. Example: PI0_PATH=/path/to/pi0_checkpoint $0" >&2
  exit 1
fi

MOUNT_ARGS=()
if [ -n "${RJOB_MOUNT_1}" ]; then
  MOUNT_ARGS+=(--mount "${RJOB_MOUNT_1}")
fi
if [ -n "${RJOB_MOUNT_2}" ]; then
  MOUNT_ARGS+=(--mount "${RJOB_MOUNT_2}")
fi

submit_one() {
  local name="$1"
  local mode="$2"
  local shard_index="$3"
  rjob submit --name="${name}" \
    --gpu "${GPU}" \
    --cpu "${CPU}" \
    --memory "${MEMORY}" \
    --charged-group "${CHARGED_GROUP}" \
    --private-machine="${PRIVATE_MACHINE}" \
    "${MOUNT_ARGS[@]}" \
    --image "${IMAGE}" \
    --custom-resources brainpp.cn/fuse=1 \
    -P 1 \
    --host-network="${HOST_NETWORK}" \
    -e DISTRIBUTED_JOB=true \
    -e MODE="${mode}" \
    -e PROJECT_DIR="${PROJECT_DIR}" \
    -e ENV_DIR="${ENV_DIR}" \
    -e NUM_SHARDS="${NUM_SHARDS}" \
    -e SHARD_INDEX="${shard_index}" \
    -e OUTPUT_DIR="${OUTPUT_DIR}" \
    -e PI0_PATH="${PI0_PATH}" \
    -e TASK="${TASK}" \
    -e TASK_ID="${TASK_ID}" \
    -e INSTRUCTION="${INSTRUCTION}" \
    -e NUM_EPISODES="${NUM_EPISODES}" \
    -e MAX_STEPS="${MAX_STEPS}" \
    -e LAYERS="${LAYERS}" \
    -e TOKEN_BINS="${TOKEN_BINS}" \
    -e BIN_INDICES="${BIN_INDICES}" \
    -e BIN_STRIDE="${BIN_STRIDE}" \
    -e MODE_ABLATION="${MODE_ABLATION}" \
    -e SAVE_VIDEO="${SAVE_VIDEO}" \
    -e SAVE_ACTIVATIONS="${SAVE_ACTIVATIONS}" \
    -e BASELINE_WAIT_SECONDS="${BASELINE_WAIT_SECONDS}" \
    -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
    -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
    -- bash -exc "bash ${WORKER}"
}

echo "==== Submit baseline ===="
submit_one "${JOB_PREFIX}-baseline" baseline 0

echo "==== Submit ${NUM_SHARDS} shards ===="
for shard_index in $(seq 0 $((NUM_SHARDS - 1))); do
  submit_one "${JOB_PREFIX}-s${shard_index}" shard "${shard_index}"
done

cat <<EOF
Submitted baseline + ${NUM_SHARDS} shards.

After all shards finish, merge:

MODE=merge OUTPUT_DIR="${OUTPUT_DIR}" bash -exc "${WORKER}"

or submit a merge rjob manually:

rjob submit --name=${JOB_PREFIX}-merge ... -e MODE=merge -e OUTPUT_DIR="${OUTPUT_DIR}" -- bash -exc "${WORKER}"
EOF
