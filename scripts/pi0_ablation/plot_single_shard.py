#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLOT_SCRIPT = ROOT / "scripts" / "18_plot_pi0_ablation_heatmaps.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot heatmaps for one unmerged PI0 ablation shard/result.")
    parser.add_argument(
        "path",
        help=(
            "Path to one ablation_sweep_results.csv, or a directory containing "
            "ablation_sweep_results.csv / ablation_sweep_results_merged.csv."
        ),
    )
    parser.add_argument("--output-dir", default=None, help="Defaults to <csv parent>/figures_single.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["mean_policy_action_delta_l2", "mean_env_action_delta_l2", "mean_gripper_position_delta_l2", "success_gain"],
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--annotate-top", type=int, default=20)
    args = parser.parse_args()

    path = Path(args.path).expanduser()
    if path.is_dir():
        csv_path = first_existing(
            [
                path / "ablation_sweep_results.csv",
                path / "ablation_sweep_results_merged.csv",
                path / "ablation_sweep_results_automerge.csv",
            ]
        )
        default_output_dir = path / "figures_single"
    else:
        csv_path = path
        default_output_dir = path.parent / "figures_single"
    if csv_path is None or not csv_path.exists():
        raise FileNotFoundError(f"Could not find an ablation result CSV from `{path}`.")
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else choose_writable_default_output_dir(default_output_dir, csv_path)
    )

    cmd = [
        sys.executable,
        str(PLOT_SCRIPT),
        "--input",
        str(csv_path),
        "--output-dir",
        str(output_dir),
        "--metrics",
        *args.metrics,
        "--top-k",
        str(args.top_k),
        "--annotate-top",
        str(args.annotate_top),
    ]
    subprocess.run(cmd, check=True)


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def choose_writable_default_output_dir(default_output_dir: Path, csv_path: Path) -> Path:
    try:
        default_output_dir.mkdir(parents=True, exist_ok=True)
        probe = default_output_dir / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return default_output_dir
    except PermissionError:
        fallback = Path.cwd() / "outputs" / "ablation_figures" / csv_path.parent.name
        try:
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback
        except PermissionError:
            fallback = Path("/tmp") / "vla_ablation_figures" / csv_path.parent.name
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback


if __name__ == "__main__":
    main()
