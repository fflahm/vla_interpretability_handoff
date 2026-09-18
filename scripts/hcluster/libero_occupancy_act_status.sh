#!/usr/bin/env bash
# Compatibility wrapper. The old path verified occupancy GT (np.load every npy)
# and then stat'd all paligemma.npy files — too slow on TOS FUSE.
# Use the fast monitor instead.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${HERE}/libero_occupancy_act_monitor.sh" "$@"
