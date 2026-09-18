#!/usr/bin/env bash
# Full LIBERO hdf5 download onto TOS. Run this inside tmux so SSH disconnect
# does not stop it. Resume-safe: already-copied files are skipped.
set -euo pipefail
source /home/guoshengyu/.pjlab_proxy.sh
proxy_on
unset HF_ENDPOINT HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTHONUNBUFFERED=1
STATUS_DIR=/mnt/shared-storage-user/guoshengyu/vla_rjob_runs/libero_download
LOG="${STATUS_DIR}/download.log"
PYTHON=/mnt/shared-storage-user/guoshengyu/envs/vla-interpretability/bin/python
SCRIPT=/mnt/shared-storage-user/guoshengyu/vla_interpretability_handoff/scripts/hcluster/download_libero_to_tos.py
mkdir -p "${STATUS_DIR}"
echo "==== $(date -Is) start pid=$$ ====" >> "${LOG}"
"${PYTHON}" "${SCRIPT}" "$@" 2>&1 | tee -a "${LOG}"
