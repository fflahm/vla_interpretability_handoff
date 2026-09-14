#!/usr/bin/env python
"""Count frames where good-probe ablations hit harder than bad, vs IoU difference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.pi05_probe_ablation import (  # noqa: E402
    frame_good_vs_bad_impacts,
    plot_frame_iou_diff_distribution,
    plot_group_action_delta_diff_hist,
)
from src.utils import ensure_dir, resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ablation-dir",
        default="outputs/ablation/pi05_probe_guided",
        help="Directory with baseline/ and best_*/worst_* rollouts.",
    )
    p.add_argument("--mode", default="zero")
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument(
        "--recompute",
        action="store_true",
        help="Reload rollouts even if frame_group_impacts.csv already exists.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ablation_dir = resolve_path(args.ablation_dir, ROOT)
    figures = ensure_dir(ablation_dir / "figures")
    group_path = ablation_dir / "frame_group_impacts.csv"
    if group_path.exists() and not args.recompute:
        group_frame = pd.read_csv(group_path)
        summary = None
        pair_frame = None
    else:
        selected = pd.read_csv(ablation_dir / "selected_conditions.csv")
        pair_frame, group_frame, summary = frame_good_vs_bad_impacts(
            ablation_dir, selected, mode=args.mode
        )
        pair_frame.to_csv(ablation_dir / "frame_pair_impacts.csv.gz", index=False)
        group_frame.to_csv(group_path, index=False)
        (ablation_dir / "frame_good_vs_bad_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        plot_frame_iou_diff_distribution(pair_frame, figures, dpi=args.dpi)
    hist_path = plot_group_action_delta_diff_hist(group_frame, figures, dpi=args.dpi)
    if summary is not None:
        print(json.dumps(summary, indent=2))
    diffs = group_frame["impact_good_mean"] - group_frame["impact_bad_mean"]
    print(
        f"frames={len(diffs)} good>bad={int((diffs > 0).sum())} "
        f"bad>good={int((diffs < 0).sum())} tie={int((diffs == 0).sum())}"
    )
    print(f"Wrote {hist_path}")


if __name__ == "__main__":
    main()
