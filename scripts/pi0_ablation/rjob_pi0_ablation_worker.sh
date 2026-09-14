#!/bin/bash
set -euo pipefail

echo "==== Host & GPU ===="
hostname
nvidia-smi || true

# This worker intentionally follows the simple working rjob pattern used by
# LeRobot eval: assume the container/mounts are provided by rjob, set cache
# variables, then call the target executable from the env directly.

MODE="${MODE:-shard}"  # baseline | shard | merge
NUM_SHARDS="${NUM_SHARDS:-8}"
SHARD_INDEX="${SHARD_INDEX:-0}"
BASELINE_WAIT_SECONDS="${BASELINE_WAIT_SECONDS:-7200}"

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
ENV_DIR="${ENV_DIR:-}"
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

echo "==== Runtime config ===="
echo "MODE=${MODE}"
echo "NUM_SHARDS=${NUM_SHARDS}"
echo "SHARD_INDEX=${SHARD_INDEX}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TASK=${TASK}"
echo "TASK_ID=${TASK_ID}"
echo "NUM_EPISODES=${NUM_EPISODES}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "LAYERS=${LAYERS}"
echo "TOKEN_BINS=${TOKEN_BINS}"
echo "BIN_INDICES=${BIN_INDICES}"
echo "BIN_STRIDE=${BIN_STRIDE}"

echo "==== Cache/env setup ===="
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_DIR}/outputs/.cache}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${XDG_CACHE_HOME}/torch}"
export LIBERO_ASSETS_PATH="${LIBERO_ASSETS_PATH:-${XDG_CACHE_HOME}/libero/assets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${PROJECT_DIR}/outputs/lerobot_home}"
env | grep -E "XDG_CACHE_HOME|HF_HOME|TORCH_HOME|LIBERO_ASSETS_PATH|HF_HUB_OFFLINE|MUJOCO_GL|HF_LEROBOT_HOME" || true

if [ -z "${PI0_PATH}" ]; then
  echo "[ERROR] PI0_PATH is not set. Export PI0_PATH or pass -e PI0_PATH=... in the rjob submitter." >&2
  exit 1
fi

echo "==== EGL/OpenGL deps ===="
if command -v apt >/dev/null 2>&1; then
  (sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak && \
   sudo sed -i 's/focal/jammy/g' /etc/apt/sources.list && \
   grep -R "focal" /etc/apt/sources.list || true && \
   sudo apt clean && \
   sudo apt update && \
   sudo apt --fix-broken install -y && \
   sudo apt install -y libegl1 libopengl0 libgl1 mesa-utils && \
   sudo ldconfig) || echo "[WARN] apt/sudo step failed; continuing."
  ldconfig -p | grep -E "libEGL|libOpenGL" || true
fi

echo "==== Local cache copy ===="
mkdir -p /root/.cache
if [ -n "${LOCAL_CACHE_SOURCE:-}" ] && [ -d "${LOCAL_CACHE_SOURCE}" ]; then
  cp -r "${LOCAL_CACHE_SOURCE}/." /root/.cache/ || true
fi

echo "==== Project & Python env ===="
cd "${PROJECT_DIR}"
if [ -n "${ENV_DIR}" ]; then
  PYTHON="${ENV_DIR}/bin/python"
else
  PYTHON="${PYTHON:-$(command -v python)}"
fi
if [ -z "${PYTHON}" ] || [ ! -x "${PYTHON}" ]; then
  echo "[ERROR] Python executable not found. Set ENV_DIR or PYTHON." >&2
  exit 1
fi
"${PYTHON}" -V

BASELINE_DIR="${OUTPUT_DIR}/baseline"

run_python_noninteractive() {
  # Some LeRobot/LIBERO versions ask:
  # "Do you want to specify a custom path for the dataset folder? (Y/N):"
  # Keep this close to the known-good lerobot-eval style.
  printf "N\nN\nN\nN\nN\n" | "${PYTHON}" "$@"
}

VIDEO_FLAG="--no-save-video"
if [ "${SAVE_VIDEO}" = "1" ]; then
  VIDEO_FLAG="--save-video"
fi
ACT_FLAG="--no-save-activations"
if [ "${SAVE_ACTIVATIONS}" = "1" ]; then
  ACT_FLAG="--save-activations"
fi

COMMON_ARGS=(
  --config configs/demo.yaml
  --pi0-path "${PI0_PATH}"
  --task "${TASK}"
  --task-id "${TASK_ID}"
  --instruction "${INSTRUCTION}"
  --output-dir "${OUTPUT_DIR}"
  --num-episodes "${NUM_EPISODES}"
  --max-steps "${MAX_STEPS}"
  --layers "${LAYERS}"
  --token-bins "${TOKEN_BINS}"
  --bin-indices "${BIN_INDICES}"
  --bin-stride "${BIN_STRIDE}"
  --mode "${MODE_ABLATION}"
  ${VIDEO_FLAG}
  ${ACT_FLAG}
)

if [ "${MODE}" = "baseline" ]; then
  echo "==== Run baseline ===="
  run_python_noninteractive scripts/pi0_ablation/sweep.py "${COMMON_ARGS[@]}" --baseline-only
elif [ "${MODE}" = "shard" ]; then
  echo "==== Wait for baseline ===="
  waited=0
  until [ -f "${BASELINE_DIR}/summary.json" ]; do
    if [ "${waited}" -ge "${BASELINE_WAIT_SECONDS}" ]; then
      echo "[ERROR] Baseline not found after ${BASELINE_WAIT_SECONDS}s: ${BASELINE_DIR}/summary.json" >&2
      exit 1
    fi
    sleep 30
    waited=$((waited + 30))
    echo "waiting baseline... ${waited}s"
  done

  echo "==== Run shard ${SHARD_INDEX}/${NUM_SHARDS} ===="
  run_python_noninteractive scripts/pi0_ablation/sweep.py "${COMMON_ARGS[@]}" \
    --baseline-dir "${BASELINE_DIR}" \
    --skip-baseline \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${SHARD_INDEX}"
elif [ "${MODE}" = "merge" ]; then
  echo "==== Merge shards ===="
  run_python_noninteractive scripts/pi0_ablation/merge_shards.py --input-dir "${OUTPUT_DIR}"
else
  echo "[ERROR] Unknown MODE=${MODE}. Use baseline, shard, or merge." >&2
  exit 1
fi

echo "==== Done ===="
