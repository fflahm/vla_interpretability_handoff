#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "outputs" / ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import ensure_dir, load_config, resolve_path

PROBE_TARGETS = (
    "offset",
    "target_position",
    "gripper_position",
    "action",
    "action_chunk",
    "gt_action",
    "gt_action_chunk",
)
TARGET_CHOICES = (*PROBE_TARGETS, "all")


def target_results_path(base_path: Path, target: str) -> Path:
    if target == "offset":
        return base_path
    return base_path.with_name(f"{base_path.stem}_{target}{base_path.suffix}")


def target_figure_path(figures_dir: Path, metric: str, target: str) -> Path:
    if target == "offset":
        return figures_dir / f"layerwise_{metric}.png"
    return figures_dir / f"layerwise_{target}_{metric}.png"


def target_label(target: str) -> str:
    return target.replace("_", " ")


def plot_curve(df: pd.DataFrame, y_col: str, ylabel: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["layer"], df[y_col], marker="o", linewidth=2, label="probe")
    ax.set_xlabel("Layer index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_comparison(curves: dict[str, pd.DataFrame], y_col: str, ylabel: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for target, df in curves.items():
        ax.plot(df["layer"], df[y_col], marker="o", linewidth=2, label=target_label(target))
    ax.set_xlabel("Layer index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def load_target_results(cfg: dict, target: str) -> pd.DataFrame:
    return pd.read_csv(target_results_path(resolve_path(cfg["probe"]["results_path"], ROOT), target))


def plot_target(cfg: dict, figures_dir: Path, target: str, *, plot_mse: bool) -> pd.DataFrame:
    df = load_target_results(cfg, target)
    label = target_label(target)
    r2_path = target_figure_path(figures_dir, "r2", target)
    plot_curve(df, "r2_mean", "Mean R2", f"Layerwise linear decodability of {label}", r2_path)
    print(f"Saved {target} R2 figure to {r2_path}")
    if plot_mse:
        mse_path = target_figure_path(figures_dir, "mse", target)
        plot_curve(df, "mse", "MSE", f"Layerwise {label} probe error", mse_path)
        print(f"Saved {target} MSE figure to {mse_path}")
    return df


def summarize_targets(curves: dict[str, pd.DataFrame], path: Path) -> None:
    rows = []
    for target, df in curves.items():
        best = df.sort_values("r2_mean", ascending=False).iloc[0]
        rows.append(
            {
                "target": target,
                "best_layer": int(best["layer"]),
                "best_layer_name": best["layer_name"],
                "best_r2_mean": float(best["r2_mean"]),
                "best_mse": float(best["mse"]),
                "best_cos_sim": float(best["cos_sim"]),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved probe target summary to {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--target", default=None, choices=TARGET_CHOICES)
    parser.add_argument("--individual", action="store_true", help="For --target all, also save one R2 figure per probe target.")
    parser.add_argument("--mse", action="store_true", help="Also save MSE plots. By default only R2 plots are written.")
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config, ROOT))
    target = args.target or cfg["probe"]["target"]
    figures_dir = ensure_dir(resolve_path(cfg["outputs"]["figures_dir"], ROOT))
    targets = PROBE_TARGETS if target == "all" else (target,)
    curves = {}
    for target_name in targets:
        results_path = target_results_path(resolve_path(cfg["probe"]["results_path"], ROOT), target_name)
        if not results_path.exists():
            if target == "all":
                print(f"Skipping {target_name}: missing {results_path}")
                continue
            raise FileNotFoundError(f"Missing probe result file for `{target_name}`: {results_path}")
        if target == "all" and not args.individual:
            curves[target_name] = load_target_results(cfg, target_name)
        else:
            curves[target_name] = plot_target(cfg, figures_dir, target_name, plot_mse=bool(args.mse))
    if not curves:
        raise RuntimeError("No probe result curves were found to plot.")
    if target == "all":
        r2_path = figures_dir / "layerwise_probe_targets_r2_comparison.png"
        plot_comparison(curves, "r2_mean", "Mean R2", "Layerwise linear decodability across probe targets", r2_path)
        print(f"Saved combined R2 figure to {r2_path}")
        if args.mse:
            mse_path = figures_dir / "layerwise_probe_targets_mse_comparison.png"
            plot_comparison(curves, "mse", "MSE", "Layerwise probe error across targets", mse_path)
            print(f"Saved combined MSE figure to {mse_path}")
        summarize_targets(curves, resolve_path(cfg["outputs"]["probes_dir"], ROOT) / "layerwise_probe_target_summary.csv")


if __name__ == "__main__":
    main()
