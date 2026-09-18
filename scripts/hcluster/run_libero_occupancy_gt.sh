#!/usr/bin/env bash
# Extract Libero-100 occupancy GT + reusable RGB onto TOS.
# CPU/MuJoCo only. Run inside tmux so SSH disconnect does not stop it.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/mnt/shared-storage-user/guoshengyu/vla_interpretability_handoff}"
ENV_DIR="${ENV_DIR:-/mnt/shared-storage-user/guoshengyu/envs/vla-interpretability}"
TOS_LIBERO="${TOS_LIBERO:-/data/tos/guoshengyu/vla/libero}"
TOS_OCC="${TOS_OCC:-/data/tos/guoshengyu/vla/occupancy}"
PYTHON="${ENV_DIR}/bin/python"
SCRIPT="${PROJECT_DIR}/scripts/occupancy/extract_libero_gt.py"
LOG_DIR="${LOG_DIR:-/mnt/shared-storage-user/guoshengyu/vla_rjob_runs/libero_occupancy_gt}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL="${MUJOCO_GL:-disable}"
export LIBERO_ROOT="${LIBERO_ROOT:-${TOS_LIBERO}/LIBERO}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${HOME}/.libero}"
mkdir -p "${LIBERO_CONFIG_PATH}" "${LOG_DIR}"
cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
assets: ${LIBERO_ROOT}/libero/libero/assets
bddl_files: ${LIBERO_ROOT}/libero/libero/bddl_files
benchmark_root: ${LIBERO_ROOT}/libero/libero
datasets: ${TOS_LIBERO}
init_states: ${LIBERO_ROOT}/libero/libero/init_files
EOF
SITE_LIBERO="${ENV_DIR}/lib/python3.12/site-packages/libero/libero"
if [ -d "${LIBERO_ROOT}/libero/libero/assets" ]; then
  ln -sfn "${LIBERO_ROOT}/libero/libero/assets" "${SITE_LIBERO}/assets" 2>/dev/null || true
fi
LOG="${LOG_DIR}/extract.log"
echo "==== $(date -Is) start pid=$$ MUJOCO_GL=${MUJOCO_GL} ====" >> "${LOG}"
# LIBERO may still prompt; never block on stdin.
printf "N\nN\nN\nN\nN\n" | "${PYTHON}" "${SCRIPT}" \
  --libero-data "${TOS_LIBERO}" \
  --libero-root "${LIBERO_ROOT}" \
  --output-root "${TOS_OCC}" \
  "$@" 2>&1 | tee -a "${LOG}"
