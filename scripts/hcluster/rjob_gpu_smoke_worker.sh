#!/bin/bash
# GPU smoke on H-cluster rjob: CUDA check, GPU unit tests, 1-ep PI0/PI0.5 LIBERO.
set -u

GPFS_ROOT="${GPFS_ROOT:-/mnt/shared-storage-user/guoshengyu}"
PROJECT_DIR="${PROJECT_DIR:-${GPFS_ROOT}/vla_interpretability_handoff}"
ENV_DIR="${ENV_DIR:-${GPFS_ROOT}/envs/vla-interpretability}"
TOS_BUCKET="${TOS_BUCKET:-ailab-pceval}"
TOS_MOUNT="${TOS_MOUNT:-/data/tos}"
TOS_ENDPOINT="${TOS_ENDPOINT:-http://hdd1.h.pjlab.org.cn:8060}"
S3_CREDS="${S3_CREDS:-${GPFS_ROOT}/.pjlab_s3.sh}"
NUM_EPISODES="${NUM_EPISODES:-1}"
MAX_STEPS="${MAX_STEPS:-80}"
TASK="${TASK:-libero_spatial}"
TASK_ID="${TASK_ID:-1}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SMOKE_OUT="${SMOKE_OUT:-${GPFS_ROOT}/vla_rjob_runs/${RUN_STAMP}}"
LOG="${SMOKE_OUT}/worker.log"
FAIL=0

mkdir -p "${SMOKE_OUT}" /tmp/vla_rjob_smoke
exec > >(tee -a "${LOG}") 2>&1

echo "==== Host & GPU ===="
date
hostname
id
echo "HOME=${HOME:-} NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-}"
df -h /mnt/shared-storage-user/guoshengyu /tmp "${TOS_MOUNT}" 2>/dev/null || true
nvidia-smi || echo "[WARN] nvidia-smi failed"
ls -l /usr/share/glvnd/egl_vendor.d 2>/dev/null || echo "no glvnd egl_vendor.d"
ldconfig -p 2>/dev/null | grep -E "libEGL|libOpenGL|libOSMesa" || echo "no EGL/OSMesa in ldconfig"

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
test -f "${TOS_MOUNT}/guoshengyu/vla/models/pi0_libero_finetuned_v044/config.json"
echo "TOS models OK"

echo "==== EGL libs ===="
# Image apt is held-broken; NVIDIA injects libEGL_nvidia.so.0 but not the
# GLVND loader libEGL.so.1. Install the jammy loader from GPFS.
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
if command -v apt >/dev/null 2>&1; then
  (sudo apt-get update -qq \
    && sudo apt-get --fix-broken install -y -qq \
    && sudo apt-get install -y -qq libegl1 libopengl0 libgl1 libosmesa6 mesa-utils \
    && sudo ldconfig) \
    || echo "[WARN] apt EGL install failed; continuing"
fi
ldconfig -p | grep -E "libEGL|libOpenGL|libOSMesa|libGLdispatch" || true
ls -l /usr/share/glvnd/egl_vendor.d 2>/dev/null || true
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
export PI0_PATH="${PI0_PATH:-${TOS_MOUNT}/guoshengyu/vla/models/pi0_libero_finetuned_v044}"
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
echo "MUJOCO_GL=${MUJOCO_GL} NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES} LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH}"
echo "PALIGEMMA_TOKENIZER=${PALIGEMMA_TOKENIZER}"
test -f "${PALIGEMMA_TOKENIZER}/tokenizer.model"

stage() {
  local name="$1"
  shift
  echo
  echo "==== STAGE ${name} ===="
  if "$@"; then
    echo "==== STAGE ${name} OK ===="
    echo "${name}=OK" >> "${SMOKE_OUT}/stages.txt"
    return 0
  else
    local rc=$?
    echo "==== STAGE ${name} FAIL rc=${rc} ===="
    echo "${name}=FAIL:${rc}" >> "${SMOKE_OUT}/stages.txt"
    FAIL=1
    return "${rc}"
  fi
}

run_python_noninteractive() {
  # LIBERO prompts for a custom dataset path if config.yaml is missing.
  printf "N\nN\nN\nN\nN\n" | "${PYTHON}" "$@"
}

stage gpu_smoke "${PYTHON}" scripts/hcluster/smoke_gpu.py

# Disable GL so mujoco can import on workers that still lack libEGL.
stage gpu_unittests env MUJOCO_GL=disable PYOPENGL_PLATFORM= PYTHONNOUSERSITE=1 \
  "${PYTHON}" -m unittest discover -s tests -v

stage egl_probe "${PYTHON}" scripts/hcluster/smoke_egl.py

stage libero_integration env RUN_LIBERO_INTEGRATION=1 PYTHONNOUSERSITE=1 \
  "${PYTHON}" -m unittest tests.test_libero_rich_annotations.RichAnnotationsTest.test_simulator_output_schema -v

stage pi0_libero_spatial_t1 run_python_noninteractive scripts/pi0_rollout/collect.py \
  --config configs/demo.yaml \
  --pi0-path "${PI0_PATH}" \
  --task "${TASK}" \
  --task-id "${TASK_ID}" \
  --num-episodes "${NUM_EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --save-video \
  --no-save-activations \
  --video-format mp4 \
  --no-require-mp4 \
  --output-dir "${SMOKE_OUT}/pi0_${TASK}_t${TASK_ID}"

stage pi05_libero_spatial_t1 run_python_noninteractive -m src.online_rollout_cli \
  --config configs/demo.yaml \
  --pi05-path "${PI05_PATH}" \
  --task "${TASK}" \
  --task-id "${TASK_ID}" \
  --num-episodes "${NUM_EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --save-video \
  --video-format mp4 \
  --output-dir "${SMOKE_OUT}/pi05_${TASK}_t${TASK_ID}"

echo
echo "==== Summary ===="
cat "${SMOKE_OUT}/stages.txt" || true
date
if [ "${FAIL}" -ne 0 ]; then
  echo "SMOKE_FAILED"
  exit 1
fi
echo "SMOKE_OK"
exit 0
