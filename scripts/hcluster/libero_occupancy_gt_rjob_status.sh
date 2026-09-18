#!/usr/bin/env bash
# Status for occupancy GT rjob smoke / shard jobs. Does not touch the 开发机 extract.
set -euo pipefail
GPFS_ROOT="${GPFS_ROOT:-/mnt/shared-storage-user/guoshengyu}"
LOG_ROOT="${LOG_ROOT:-${GPFS_ROOT}/vla_rjob_runs/occupancy_gt_rjob}"
SMOKE_OUT="${SMOKE_OUT:-/data/tos/guoshengyu/vla/occupancy_rjob_smoke}"
NAME_PREFIX="${NAME_PREFIX:-vla-occ-gt}"

echo "==== rjob (${NAME_PREFIX}) ===="
rjob list 2>/dev/null | grep -E "${NAME_PREFIX}|showname" || rjob list 2>/dev/null | tail -n 30

echo "==== latest worker log dir ===="
if [ -d "${LOG_ROOT}" ]; then
  latest="$(ls -1dt "${LOG_ROOT}"/*/ 2>/dev/null | head -n 1 || true)"
  echo "latest=${latest:-none}"
  if [ -n "${latest}" ] && [ -f "${latest}/worker.log" ]; then
    tail -n 40 "${latest}/worker.log"
  fi
else
  echo "no ${LOG_ROOT} yet"
fi

echo "==== smoke output ===="
if [ -f "${SMOKE_OUT}/gt/STATUS.rjob.txt" ]; then
  cat "${SMOKE_OUT}/gt/STATUS.rjob.txt"
elif [ -f "${SMOKE_OUT}/gt/STATUS.txt" ]; then
  cat "${SMOKE_OUT}/gt/STATUS.txt"
else
  echo "no smoke STATUS yet under ${SMOKE_OUT}"
fi
if [ -f "${SMOKE_OUT}/gt/manifest.rjob.json" ]; then
  python3 -c "import json; print(json.dumps(json.load(open('${SMOKE_OUT}/gt/manifest.rjob.json')), indent=2)[:2500])"
elif [ -f "${SMOKE_OUT}/gt/manifest.json" ]; then
  python3 -c "import json; print(json.dumps(json.load(open('${SMOKE_OUT}/gt/manifest.json')), indent=2)[:2500])"
fi
