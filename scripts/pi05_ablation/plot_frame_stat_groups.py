#!/usr/bin/env python
"""Group ablation frames by impact value and overlay per-dimension histograms."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.pi05_frame_stat_groups import (  # noqa: E402
    attach_impact_groups,
    merge_stats_and_impact,
    write_group_outputs,
)
from src.utils import log, resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split frames into Group 1 (largest impact), Group 2 (closest to 0), "
            "and Group 3 (smallest impact), then overlay each frame-stat dimension."
        )
    )
    parser.add_argument(
        "--stats-csv",
        default="outputs/ablation/pi05_probe_guided_frames/frame_stats/frame_stats.csv",
    )
    parser.add_argument(
        "--impact-csv",
        default="outputs/ablation/pi05_probe_guided_frames/frame_good_minus_bad.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/ablation/pi05_probe_guided_frames/frame_stat_groups",
    )
    parser.add_argument(
        "--tail-frac",
        type=float,
        default=0.2,
        help="Fraction of frames in each of the three groups (default 0.2 → 200/1000).",
    )
    parser.add_argument("--impact-column", default="good_minus_bad")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--bins", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = pd.read_csv(resolve_path(args.stats_csv))
    impact = pd.read_csv(resolve_path(args.impact_csv))
    merged = merge_stats_and_impact(stats, impact, impact_column=args.impact_column)
    table = attach_impact_groups(merged, tail_frac=args.tail_frac)
    output_dir = resolve_path(args.output_dir)
    log(f"Plotting {len(table)} frames tail_frac={args.tail_frac} -> {output_dir}")
    summary = write_group_outputs(
        table,
        output_dir,
        dpi=args.dpi,
        bins=args.bins,
        extra_summary={
            "tail_frac": float(args.tail_frac),
            "stats_csv": str(resolve_path(args.stats_csv)),
            "impact_csv": str(resolve_path(args.impact_csv)),
            "impact_column": args.impact_column,
        },
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
