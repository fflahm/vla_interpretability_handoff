#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/mnt/shared-storage-user/guoshengyu/vla_interpretability_handoff}"
TOS_OCC="${TOS_OCC:-/data/tos/guoshengyu/vla/occupancy}"
PYTHON="${PYTHON:-/mnt/shared-storage-user/guoshengyu/envs/vla-interpretability/bin/python}"
export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON}" "${PROJECT_DIR}/scripts/occupancy/extract_libero_gt.py" --output-root "${TOS_OCC}" --status
echo "==== local process / tmux (开发机; rjob will not show here) ===="
ps -u "$USER" -o pid=,etime=,cmd= | grep '/extract_libero_gt.py' | grep -v grep || echo "no local extract process"
tmux ls 2>/dev/null || echo "no tmux sessions"
echo "==== rjob ===="
rjob list 2>/dev/null | grep -E 'vla-occ-gt' || echo "no vla-occ-gt in rjob list (or rjob unavailable)"
