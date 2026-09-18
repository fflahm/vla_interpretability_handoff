#!/bin/bash
set -euo pipefail

# Optional PJLab-cluster helper.
#
# Source this file before running the demos only if you are using the same
# cluster-style cache/checkpoint layout. Otherwise, ignore this file and set
# PI0_PATH / PI05_PATH / MUJOCO_GL yourself.
#   source scripts/setup_cluster_env.sh

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VLA_TOS="${VLA_TOS:-/data/tos/guoshengyu/vla}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${VLA_TOS}/cache}"
export HF_HOME="${HF_HOME:-${VLA_TOS}/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${VLA_TOS}/cache/torch}"
export LIBERO_ROOT="${LIBERO_ROOT:-${VLA_TOS}/libero/LIBERO}"
export LIBERO_ASSETS_PATH="${LIBERO_ASSETS_PATH:-${VLA_TOS}/libero/LIBERO/libero/libero/assets}"
export LIBERO_DATASET_DIR="${LIBERO_DATASET_DIR:-${VLA_TOS}/libero/libero_spatial}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${VLA_TOS}/cache/lerobot}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TMPDIR="${TMPDIR:-/tmp}"
unset HF_ENDPOINT

export PI05_PATH="${PI05_PATH:-${VLA_TOS}/models/pi05_libero}"
export PI0_PATH="${PI0_PATH:-${VLA_TOS}/models/pi0_libero_finetuned_v044}"

cd "${PROJECT_ROOT}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "XDG_CACHE_HOME=${XDG_CACHE_HOME}"
echo "HF_HOME=${HF_HOME}"
echo "TORCH_HOME=${TORCH_HOME}"
echo "LIBERO_ROOT=${LIBERO_ROOT}"
echo "LIBERO_ASSETS_PATH=${LIBERO_ASSETS_PATH}"
echo "LIBERO_DATASET_DIR=${LIBERO_DATASET_DIR}"
echo "HF_LEROBOT_HOME=${HF_LEROBOT_HOME}"
echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
echo "MUJOCO_GL=${MUJOCO_GL}"
echo "PI05_PATH=${PI05_PATH}"
echo "PI0_PATH=${PI0_PATH}"
