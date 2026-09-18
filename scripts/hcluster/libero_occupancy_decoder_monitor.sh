#!/usr/bin/env bash
# Fast monitor for occupancy decoder training (GPFS STATUS + rjob + worker.log).
#
#   bash scripts/hcluster/libero_occupancy_decoder_monitor.sh
set -euo pipefail

GPFS_ROOT="${GPFS_ROOT:-/mnt/shared-storage-user/guoshengyu}"
OUT_ROOT="${OUT_ROOT:-${GPFS_ROOT}/vla_rjob_runs/occupancy_decoders}"
LOG_ROOT="${LOG_ROOT:-${GPFS_ROOT}/vla_rjob_runs/occupancy_decoder_rjob}"
NAME_PREFIX="${NAME_PREFIX:-vla-occ-dec}"
TAIL_N="${TAIL_N:-16}"
PYTHON="${PYTHON:-python3}"

echo "==== rjob (${NAME_PREFIX}) ===="
if command -v rjob >/dev/null 2>&1; then
  rjob list 2>/dev/null | grep -E "${NAME_PREFIX}" || echo "no ${NAME_PREFIX} in rjob list"
else
  echo "rjob not on PATH"
fi

echo
echo "==== latest GPFS run ${OUT_ROOT} ===="
if [ -d "${OUT_ROOT}" ]; then
  latest="$(ls -1dt "${OUT_ROOT}"/*/ 2>/dev/null | head -n 1 || true)"
  echo "latest=${latest:-none}"
  if [ -n "${latest}" ]; then
    "${PYTHON}" - "${latest%/}" <<'PY'
import csv, json, sys
from datetime import datetime, timezone
from pathlib import Path
root = Path(sys.argv[1])
status = root / "status.json"
print(f"dir {root}")
if status.exists():
    d = json.loads(status.read_text())
    print(f"  phase     {d.get('phase')}")
    print(f"  progress  {d.get('done_probes')}/{d.get('total_probes')} probes")
    print(f"  condition {d.get('condition')} bin={d.get('bin')}")
    print(f"  best      {d.get('best')} soft_iou={d.get('best_soft_iou')}")
    print(f"  updated   {d.get('updated_at')}")
    updated = d.get("updated_at") or ""
    try:
        dt = datetime.strptime(updated, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        print(f"  age_s     {(datetime.now(timezone.utc)-dt).total_seconds():.0f}")
    except Exception:
        pass
    done = int(d.get("done_probes") or 0)
    total = int(d.get("total_probes") or 0)
    if done > 0 and total > done:
        # rough remaining assuming constant probe rate since split.json mtime
        split = root / "split.json"
        if split.exists():
            elapsed = (datetime.now(timezone.utc) - datetime.fromtimestamp(split.stat().st_mtime, tz=timezone.utc)).total_seconds()
            rate = done / max(elapsed, 1)
            remain = (total - done) / rate
            eta = datetime.now().astimezone()
            from datetime import timedelta
            eta = datetime.now().astimezone() + timedelta(seconds=remain)
            print(f"  eta       ~{eta.strftime('%H:%M')} ({remain/3600:.2f}h left) rate={rate*60:.2f} probes/min")
else:
    print("  status.json missing")
metrics = root / "metrics.csv"
if metrics.exists():
    rows = list(csv.DictReader(metrics.open()))
    print(f"  metrics   {len(rows)} rows")
    if rows:
        last = rows[-1]
        print(f"  last      {last.get('condition')} bin={last.get('bin')} soft={last.get('soft_iou')}")
man = root / "manifest.json"
if man.exists():
    d = json.loads(man.read_text())
    print(f"  manifest  status={d.get('status')} tos_sync={d.get('tos_sync')}")
PY
    echo "---- STATUS.txt ----"
    cat "${latest%/}/STATUS.txt" 2>/dev/null || true
  fi
else
  echo "no ${OUT_ROOT}"
fi

echo
echo "==== latest worker.log ===="
if [ -d "${LOG_ROOT}" ]; then
  latest_log="$(ls -1dt "${LOG_ROOT}"/*/ 2>/dev/null | head -n 1 || true)"
  echo "latest=${latest_log:-none}"
  if [ -n "${latest_log}" ] && [ -f "${latest_log}/worker.log" ]; then
    grep -E 'Layer ready|Finished probes|TOS sync|OCCUPANCY_DEC_RJOB_|probe paligemma|probe expert' "${latest_log}/worker.log" | tail -n 20 || true
    echo "---- tail -n ${TAIL_N} ----"
    tail -n "${TAIL_N}" "${latest_log}/worker.log"
  fi
else
  echo "no ${LOG_ROOT} yet"
fi
