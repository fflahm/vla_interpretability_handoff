#!/usr/bin/env bash
# Fast live view of occupancy activation extract. Reads only small TOS files
# plus GPFS worker.log / rjob list. Does not scan GT or paligemma.npy trees.
#
#   bash scripts/hcluster/libero_occupancy_act_monitor.sh
#   TOS_ACT=/data/tos/guoshengyu/vla/occupancy_act_rjob_smoke bash ...
set -euo pipefail

GPFS_ROOT="${GPFS_ROOT:-/mnt/shared-storage-user/guoshengyu}"
TOS_ACT="${TOS_ACT:-/data/tos/guoshengyu/vla/occupancy_activations}"
LOG_ROOT="${LOG_ROOT:-${GPFS_ROOT}/vla_rjob_runs/occupancy_act_rjob}"
NAME_PREFIX="${NAME_PREFIX:-vla-occ-act}"
TAIL_N="${TAIL_N:-12}"
PYTHON="${PYTHON:-python3}"

echo "==== rjob (${NAME_PREFIX}) ===="
if command -v rjob >/dev/null 2>&1; then
  rjob list 2>/dev/null | grep -E "${NAME_PREFIX}" || echo "no ${NAME_PREFIX} in rjob list"
else
  echo "rjob not on PATH"
fi

echo
echo "==== TOS status (no npy walk) ${TOS_ACT} ===="
"${PYTHON}" - "${TOS_ACT}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
if not root.exists():
    print(f"missing {root}")
    sys.exit(1)

status_files = sorted(root.glob("status*.json"), key=lambda p: p.stat().st_mtime)
split_path = root / "split.json"
cfg_path = root / "capture_config.json"

split = {}
if split_path.exists():
    split = json.loads(split_path.read_text(encoding="utf-8"))
    tasks = split.get("tasks") or []
    planned = sum(len(t.get("train") or []) + len(t.get("test") or []) for t in tasks)
    print(
        f"split.json  seed={split.get('seed')} suite={split.get('suite')} "
        f"tasks={len(tasks)} planned_infer_demos={planned} smoke={split.get('smoke')}"
    )
    note = split.get("split") or {}
    print(
        f"            per-task split train/test/ablation="
        f"{note.get('train')}/{note.get('test')}/{note.get('ablation')} "
        f"(ablation not inferred)"
    )
else:
    print("split.json  missing")
    tasks = []
    planned = None

if cfg_path.exists():
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    print(
        f"capture     tokens_per_bin={cfg.get('tokens_per_bin')} "
        f"pali_bins={cfg.get('paligemma_bins')} expert_bins={cfg.get('expert_bins')} "
        f"flow_times={cfg.get('flow_times')} smoke={cfg.get('smoke')}"
    )
else:
    print("capture_config.json missing")

if not status_files:
    print("status*.json missing")
    sys.exit(0)

payload = json.loads(status_files[-1].read_text(encoding="utf-8"))
if len(status_files) > 1:
    print("status files:", ", ".join(p.name for p in status_files))
print(f"status file {status_files[-1].name}")
print(f"  phase       {payload.get('phase')}")
print(f"  task        {payload.get('task')}")
print(f"  demo        {payload.get('demo')}")
done = int(payload.get("done_demos") or 0)
total = int(payload.get("total_demos") or planned or 0)
frames = payload.get("done_frames")
print(f"  progress    {done}/{total} demos  frames={frames}")
err = payload.get("last_error") or ""
print(f"  last_error  {err if err else '(none)'}")
updated = payload.get("updated_at") or ""
print(f"  updated_at  {updated}")

task_name = payload.get("task") or ""
if tasks and task_name:
    names = [t.get("task") for t in tasks]
    if task_name in names:
        idx = names.index(task_name) + 1
        print(f"  task index  {idx}/{len(names)}")

now = datetime.now(timezone.utc)
try:
    updated_dt = datetime.strptime(updated, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    age_s = (now - updated_dt).total_seconds()
    print(f"  status age  {age_s:.0f}s")
except Exception:
    updated_dt = None
    age_s = None

start_dt = None
if split_path.exists():
    start_dt = datetime.fromtimestamp(split_path.stat().st_mtime, tz=timezone.utc)
end_dt = updated_dt or now
if start_dt and done > 0 and total > done:
    elapsed = (end_dt - start_dt).total_seconds()
    if elapsed > 1:
        rate = done / elapsed
        remain_s = (total - done) / rate
        eta = datetime.now().astimezone()
        from datetime import timedelta
        eta = datetime.now().astimezone() + timedelta(seconds=remain_s)
        print(
            f"  rate        {rate * 60:.2f} demos/min  "
            f"elapsed={elapsed / 3600:.2f}h  ETA~{eta.strftime('%H:%M')} "
            f"({remain_s / 3600:.2f}h left)"
        )
        print("              (clock starts at split.json mtime; includes model load)")

cur_task = payload.get("task")
cur_demo = payload.get("demo")
suite = split.get("suite") or "libero_90"
if cur_task and cur_demo:
    dest = root / "activations" / suite / cur_task / cur_demo
    bits = []
    for name in ("paligemma.npy", "expert.npy", "metadata.json"):
        p = dest / name
        try:
            bits.append(f"{name}={'yes' if p.is_file() and p.stat().st_size else 'no'}")
        except OSError:
            bits.append(f"{name}=err")
    print(f"  in-progress {dest}")
    print(f"              {', '.join(bits)}  (npy written after this demo finishes)")
PY

echo
echo "==== latest GPFS worker.log ===="
if [ -d "${LOG_ROOT}" ]; then
  latest="$(ls -1dt "${LOG_ROOT}"/*/ 2>/dev/null | head -n 1 || true)"
  echo "latest=${latest:-none}"
  if [ -n "${latest}" ] && [ -f "${latest}/worker.log" ]; then
    grep -E 'Activation extract |PI0.5 Euler capture ready|OCCUPANCY_ACT_RJOB_|extract rc=' "${latest}/worker.log" | tail -n 8 || true
    echo "---- tail -n ${TAIL_N} ----"
    tail -n "${TAIL_N}" "${latest}/worker.log"
  fi
else
  echo "no ${LOG_ROOT} yet"
fi
