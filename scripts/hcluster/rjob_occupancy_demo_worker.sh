#!/bin/bash
# rjob worker: current-code occupancy demo (scripts/occupancy/run_demo.py).
# Results go to GPFS, not the repo outputs/ TOS symlink.
set -u

GPFS_ROOT="${GPFS_ROOT:-/mnt/shared-storage-user/guoshengyu}"
PROJECT_DIR="${PROJECT_DIR:-${GPFS_ROOT}/vla_interpretability_handoff}"
ENV_DIR="${ENV_DIR:-${GPFS_ROOT}/envs/vla-interpretability}"
TOS_BUCKET="${TOS_BUCKET:-ailab-pceval}"
TOS_MOUNT="${TOS_MOUNT:-/data/tos}"
TOS_ENDPOINT="${TOS_ENDPOINT:-http://hdd1.h.pjlab.org.cn:8060}"
S3_CREDS="${S3_CREDS:-${GPFS_ROOT}/.pjlab_s3.sh}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_OUT="${RUN_OUT:-${GPFS_ROOT}/vla_rjob_runs/${RUN_STAMP}}"
DEMO_OUT="${DEMO_OUT:-${RUN_OUT}/occupancy_demo}"
LOG="${RUN_OUT}/worker.log"
NUM_DEMOS="${NUM_DEMOS:-6}"
FRAMES_PER_DEMO="${FRAMES_PER_DEMO:-4}"
EPOCHS="${EPOCHS:-160}"
HDF5="${HDF5:-${TOS_MOUNT}/guoshengyu/vla/libero/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5}"

mkdir -p "${DEMO_OUT}" /tmp/vla_rjob_occupancy
exec > >(tee -a "${LOG}") 2>&1

echo "==== Host & GPU ===="
date
hostname
id
echo "HOME=${HOME:-} NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-}"
df -h /mnt/shared-storage-user/guoshengyu /tmp "${TOS_MOUNT}" 2>/dev/null || true
nvidia-smi || echo "[WARN] nvidia-smi failed"

echo "==== Mount TOS if needed ===="
export PATH="${GPFS_ROOT}/bin:${PATH}"
if [ -f "${TOS_MOUNT}/guoshengyu/vla/models/pi05_libero/config.json" ]; then
  echo "TOS already visible at ${TOS_MOUNT}"
else
  if [ ! -f "${S3_CREDS}" ]; then
    echo "[ERROR] missing ${S3_CREDS}" >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  source "${S3_CREDS}"
  mkdir -p "${TOS_MOUNT}"
  S3MOUNT_BIN="$(command -v s3mount || true)"
  if [ -z "${S3MOUNT_BIN}" ] && [ -x "${GPFS_ROOT}/bin/s3mount" ]; then
    S3MOUNT_BIN="${GPFS_ROOT}/bin/s3mount"
  fi
  if [ -z "${S3MOUNT_BIN}" ]; then
    echo "[ERROR] s3mount not in PATH or ${GPFS_ROOT}/bin/s3mount" >&2
    exit 2
  fi
  echo "using s3mount=${S3MOUNT_BIN}"
  if command -v apt-get >/dev/null 2>&1; then
    (sudo apt-get update -qq && sudo apt-get install -y -qq libfuse2 fuse3 >/dev/null) \
      || echo "[WARN] apt fuse install failed; continuing"
  fi
  if mountpoint -q "${TOS_MOUNT}" 2>/dev/null; then
    echo "unmounting stale ${TOS_MOUNT}"
    fusermount3 -u "${TOS_MOUNT}" || fusermount -u "${TOS_MOUNT}" || true
  fi
  "${S3MOUNT_BIN}" "${TOS_BUCKET}" "${TOS_MOUNT}" \
    --endpoint-url "${TOS_ENDPOINT}" \
    --allow-delete --allow-overwrite --force-path-style \
    || "${S3MOUNT_BIN}" "${TOS_BUCKET}" "${TOS_MOUNT}" \
      --endpoint-url "${TOS_ENDPOINT}" \
      --allow-delete --allow-overwrite --force-path-style --allow-other
  sleep 2
fi
test -f "${TOS_MOUNT}/guoshengyu/vla/models/pi05_libero/config.json"
test -f "${HDF5}"
echo "TOS model and hdf5 OK"

echo "==== EGL libs ===="
GLVND_DIR="${GPFS_ROOT}/opt/glvnd"
if [ -d "${GLVND_DIR}/lib" ]; then
  echo "installing GLVND loader from ${GLVND_DIR}"
  SYS_LIB=""
  for candidate in /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu; do
    if [ -d "${candidate}" ]; then
      SYS_LIB="${candidate}"
      break
    fi
  done
  if [ -n "${SYS_LIB}" ]; then
    for so in libEGL.so.1 libEGL.so.1.1.0 libOpenGL.so.0 libOpenGL.so.0.0.0 \
              libGLESv2.so.2 libGLESv2.so.2.1.0 libGLdispatch.so.0 libGLdispatch.so.0.0.0; do
      if [ -e "${GLVND_DIR}/lib/${so}" ] && [ ! -e "${SYS_LIB}/${so}" ]; then
        cp -a "${GLVND_DIR}/lib/${so}" "${SYS_LIB}/${so}"
        echo "installed ${SYS_LIB}/${so}"
      fi
    done
    ldconfig || true
  fi
else
  echo "[WARN] missing ${GLVND_DIR}/lib"
fi
ldconfig -p | grep -E "libEGL|libOpenGL|libGLdispatch" || true
export LD_LIBRARY_PATH="${GLVND_DIR}/lib:${LD_LIBRARY_PATH:-}"

echo "==== Python env ===="
export PYTHONNOUSERSITE=1
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY=1
export PALIGEMMA_TOKENIZER="${PALIGEMMA_TOKENIZER:-${GPFS_ROOT}/models/paligemma-3b-pt-224-tokenizer}"
export PI05_PATH="${PI05_PATH:-${TOS_MOUNT}/guoshengyu/vla/models/pi05_libero}"
export HF_HOME="${HF_HOME:-${TOS_MOUNT}/guoshengyu/vla/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${TOS_MOUNT}/guoshengyu/vla/cache/torch}"
export LIBERO_ROOT="${LIBERO_ROOT:-${TOS_MOUNT}/guoshengyu/vla/libero/LIBERO}"
export LIBERO_ASSETS_PATH="${LIBERO_ASSETS_PATH:-${LIBERO_ROOT}/libero/libero/assets}"
export LIBERO_DATASET_DIR="${LIBERO_DATASET_DIR:-${TOS_MOUNT}/guoshengyu/vla/libero/libero_spatial}"
export LIBERO_DATASETS_ROOT="${LIBERO_DATASETS_ROOT:-${TOS_MOUNT}/guoshengyu/vla/libero}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${TOS_MOUNT}/guoshengyu/vla/cache/lerobot}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${HOME:-/root}/.libero}"
export CONDA_PREFIX="${ENV_DIR}"
export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
PYTHON="${ENV_DIR}/bin/python"
if [ ! -x "${PYTHON}" ]; then
  echo "[ERROR] ${PYTHON} not found" >&2
  exit 2
fi

LIBERO_BENCH="${LIBERO_ROOT}/libero/libero"
write_libero_config() {
  local dest="$1"
  mkdir -p "${dest}"
  cat > "${dest}/config.yaml" <<EOF
assets: ${LIBERO_BENCH}/assets
bddl_files: ${LIBERO_BENCH}/bddl_files
benchmark_root: ${LIBERO_BENCH}
datasets: ${LIBERO_DATASETS_ROOT}
init_states: ${LIBERO_BENCH}/init_files
EOF
  echo "wrote ${dest}/config.yaml"
}
write_libero_config "${LIBERO_CONFIG_PATH}"
if [ "${LIBERO_CONFIG_PATH}" != "/root/.libero" ]; then
  write_libero_config /root/.libero
fi
if [ -n "${HOME:-}" ] && [ "${HOME}/.libero" != "${LIBERO_CONFIG_PATH}" ]; then
  write_libero_config "${HOME}/.libero"
fi

SITE_LIBERO="${ENV_DIR}/lib/python3.12/site-packages/libero/libero"
if [ -d "${LIBERO_ASSETS_PATH}" ]; then
  ln -sfn "${LIBERO_ASSETS_PATH}" "${SITE_LIBERO}/assets"
  echo "libero assets -> ${SITE_LIBERO}/assets -> ${LIBERO_ASSETS_PATH}"
else
  echo "[ERROR] missing LIBERO assets ${LIBERO_ASSETS_PATH}" >&2
  exit 2
fi

cd "${PROJECT_DIR}"
"${PYTHON}" -V
echo "MUJOCO_GL=${MUJOCO_GL} PALIGEMMA_TOKENIZER=${PALIGEMMA_TOKENIZER}"
echo "DEMO_OUT=${DEMO_OUT}"
echo "HDF5=${HDF5}"
test -f "${PALIGEMMA_TOKENIZER}/tokenizer.model"

echo "==== run_demo.py ===="
# LIBERO may prompt if config.yaml is missing; config is written above.
printf "N\nN\nN\nN\nN\n" | "${PYTHON}" scripts/occupancy/run_demo.py \
  --hdf5 "${HDF5}" \
  --pi05-path "${PI05_PATH}" \
  --output-dir "${DEMO_OUT}" \
  --num-demos "${NUM_DEMOS}" \
  --frames-per-demo "${FRAMES_PER_DEMO}" \
  --epochs "${EPOCHS}" \
  --device auto
rc=$?
echo "==== run_demo.py rc=${rc} ===="
ls -la "${DEMO_OUT}" || true
if [ ! -f "${DEMO_OUT}/summary.json" ]; then
  echo "[ERROR] missing ${DEMO_OUT}/summary.json" >&2
  exit 1
fi
date
echo "OCCUPANCY_DEMO_OK out=${DEMO_OUT}"
exit "${rc}"
