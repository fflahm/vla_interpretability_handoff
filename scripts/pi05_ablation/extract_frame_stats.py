#!/usr/bin/env python
"""Extract per-frame scalar stats for PI0.5 ablation sampled_frames.csv."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.libero_rich_annotations import LOCAL_LIBERO_ROOT  # noqa: E402
from src.pi05_frame_stats import (  # noqa: E402
    DEFAULT_DELTA_WINDOW,
    extract_sampled_frame_stats,
    load_sampled_frame_table,
    write_frame_stats,
)
from src.utils import log, resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For every row in sampled_frames.csv, extract body-motion, kinematic, "
            "proximity, contact, and action-phase scalars. Self-occlusion is omitted. "
            "HDF5 fields work with --simulator off; proximity/contact need replay."
        )
    )
    parser.add_argument(
        "--sampled-csv",
        default="outputs/ablation/pi05_probe_guided_frames/sampled_frames.csv",
        help="Ablation provenance table (row_index aligned with chunk_l2).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/ablation/pi05_probe_guided_frames/frame_stats",
    )
    parser.add_argument("--delta-window", type=int, default=DEFAULT_DELTA_WINDOW)
    parser.add_argument("--simulator", choices=("auto", "required", "off"), default="auto")
    parser.add_argument(
        "--replay-mode",
        choices=("selected", "full"),
        default="selected",
        help="selected: set_state only at requested frames. full: replay the demo for sim phases.",
    )
    parser.add_argument("--libero-root", type=Path, default=LOCAL_LIBERO_ROOT)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional prefix of sampled_frames.csv for smoke runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.delta_window < 1:
        raise SystemExit("--delta-window must be positive")
    sampled_csv = resolve_path(args.sampled_csv)
    output_dir = resolve_path(args.output_dir)
    sampled = load_sampled_frame_table(sampled_csv)
    n_run = len(sampled) if args.max_frames is None else min(len(sampled), args.max_frames)
    log(
        f"Extracting frame stats n={n_run} simulator={args.simulator} "
        f"replay_mode={args.replay_mode} delta={args.delta_window}"
    )
    table = extract_sampled_frame_stats(
        sampled,
        delta_window=args.delta_window,
        simulator=args.simulator,
        replay_mode=args.replay_mode,
        libero_root=Path(args.libero_root),
        max_frames=args.max_frames,
    )
    summary = write_frame_stats(
        table,
        output_dir,
        extra_summary={
            "sampled_csv": str(sampled_csv),
            "delta_window": int(args.delta_window),
            "simulator": args.simulator,
            "replay_mode": args.replay_mode,
            "max_frames": args.max_frames,
        },
    )
    log(f"Wrote {summary['csv']}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
