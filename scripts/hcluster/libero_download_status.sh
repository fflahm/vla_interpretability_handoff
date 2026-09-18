#!/usr/bin/env bash
# Print LIBERO TOS download status. Uses only bash; safe with /usr/bin/python.
set -euo pipefail
STATUS_DIR=/mnt/shared-storage-user/guoshengyu/vla_rjob_runs/libero_download
TOS_ROOT=/data/tos/guoshengyu/vla/libero
echo "==== STATUS.txt ===="
if [[ -f "${STATUS_DIR}/STATUS.txt" ]]; then
  cat "${STATUS_DIR}/STATUS.txt"
else
  echo "no STATUS.txt yet"
fi
echo "==== live TOS hdf5 ===="
for suite in libero_spatial libero_object libero_goal libero_10 libero_90; do
  n=0
  if [[ -d "${TOS_ROOT}/${suite}" ]]; then
    n=$(find "${TOS_ROOT}/${suite}" -maxdepth 1 -name '*_demo.hdf5' | wc -l)
  fi
  printf '  %s: %s\n' "${suite}" "${n}"
done
echo "==== process ===="
mapfile -t procs < <(ps -u "$USER" -o pid=,etime=,cmd= | grep '/bin/python .*/download_libero_to_tos.py' | grep -v grep || true)
if ((${#procs[@]})); then
  printf '%s\n' "${procs[@]}"
else
  echo "download process not running"
fi
echo "==== tmux ===="
tmux ls 2>/dev/null || echo "no tmux sessions"
