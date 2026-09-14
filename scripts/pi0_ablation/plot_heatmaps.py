#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils import ensure_dir, resolve_path


DEFAULT_METRICS = (
    "mean_policy_action_delta_l2",
    "mean_env_action_delta_l2",
    "mean_gripper_position_delta_l2",
    "success_gain",
)

NUMERIC_COLUMNS = (
    "layer_index",
    "token_bin_index",
    "scale",
    "baseline_success_rate",
    "success_rate",
    "success_drop",
    "mean_policy_action_delta_l2",
    "mean_env_action_delta_l2",
    "mean_gripper_position_delta_l2",
    "num_compared_steps",
    "causal_impact_score",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PI0 activation-ablation layer x token-bin heatmaps.")
    parser.add_argument("--input", default=None, help="CSV path. Defaults to INPUT_DIR/ablation_sweep_results_merged.csv.")
    parser.add_argument("--input-dir", default=None, help="Directory containing merged or shard ablation CSVs.")
    parser.add_argument("--output-dir", default=None, help="Defaults to input-dir/figures or CSV parent/figures.")
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--cmap", default="magma")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--annotate-top", type=int, default=10, help="Mark top-K cells on each heatmap; use 0 to disable.")
    args = parser.parse_args()

    csv_path = find_input_csv(args.input, args.input_dir)
    output_dir = ensure_dir(
        resolve_path(args.output_dir, ROOT)
        if args.output_dir
        else csv_path.parent / "figures"
    )
    frame = clean_ablation_frame(pd.read_csv(csv_path))
    frame = add_derived_metrics(frame)
    frame.to_csv(output_dir / "ablation_sweep_results_with_derived_metrics.csv", index=False)

    metrics = [metric for metric in args.metrics if metric in frame.columns]
    if not metrics:
        raise ValueError(f"No requested metrics found in {csv_path}. Available columns: {list(frame.columns)}")

    for metric in metrics:
        heat = pivot_metric(frame, metric)
        heat.to_csv(output_dir / f"heatmap_{metric}.csv")
        plot_heatmap(
            heat=heat,
            frame=frame,
            metric=metric,
            output_path=output_dir / f"heatmap_{metric}.png",
            cmap=args.cmap,
            dpi=int(args.dpi),
            annotate_top=int(args.annotate_top),
        )

    primary_metric = "mean_policy_action_delta_l2" if "mean_policy_action_delta_l2" in frame.columns else metrics[0]
    ranked = frame.sort_values(primary_metric, ascending=False)
    ranked.head(int(args.top_k)).to_csv(output_dir / f"top{args.top_k}_by_{primary_metric}.csv", index=False)
    if "success_gain" in frame.columns:
        frame.sort_values("success_gain", ascending=False).head(int(args.top_k)).to_csv(
            output_dir / f"top{args.top_k}_by_success_gain.csv", index=False
        )
    plot_multi_metric_summary(frame, metrics, output_dir / "ablation_metric_summary.png", dpi=int(args.dpi))
    print(f"Saved ablation heatmaps to {output_dir}")


def find_input_csv(input_path: str | None, input_dir: str | None) -> Path:
    if input_path:
        path = resolve_path(input_path, ROOT)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    if not input_dir:
        raise ValueError("Provide --input or --input-dir.")
    directory = resolve_path(input_dir, ROOT)
    candidates = [
        directory / "ablation_sweep_results_merged.csv",
        directory / "ablation_sweep_results.csv",
    ]
    candidates.extend(sorted(directory.glob("shard_*_of_*/ablation_sweep_results.csv")))
    existing = [path for path in candidates if path.exists()]
    if not existing:
        raise FileNotFoundError(f"No ablation result CSV found under {directory}. Run scripts/pi0_ablation/merge_shards.py first.")
    if len(existing) == 1:
        return existing[0]
    if existing[0].name == "ablation_sweep_results_merged.csv":
        return existing[0]
    frames = []
    for path in existing:
        frame = pd.read_csv(path)
        frame["source_csv"] = str(path)
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    if {"layer_index", "token_bin_index", "mode", "scale"}.issubset(merged.columns):
        merged = merged.drop_duplicates(["layer_index", "token_bin_index", "mode", "scale"], keep="last")
    merged_path = directory / "ablation_sweep_results_automerge.csv"
    merged.to_csv(merged_path, index=False)
    return merged_path


def clean_ablation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Make manually concatenated shard CSVs safe to analyze."""
    frame = frame.copy()
    if "condition" in frame.columns:
        frame = frame[frame["condition"].astype(str) != "condition"]
    if "layer_index" in frame.columns:
        frame = frame[frame["layer_index"].astype(str) != "layer_index"]
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    required = [column for column in ("layer_index", "token_bin_index") if column in frame.columns]
    if required:
        frame = frame.dropna(subset=required)
    if "layer_index" in frame.columns:
        frame["layer_index"] = frame["layer_index"].astype(int)
    if "token_bin_index" in frame.columns:
        frame["token_bin_index"] = frame["token_bin_index"].astype(int)
    return frame.reset_index(drop=True)


def add_derived_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if {"success_rate", "baseline_success_rate"}.issubset(frame.columns):
        frame["success_gain"] = frame["success_rate"] - frame["baseline_success_rate"]
        frame["abs_success_change"] = frame["success_gain"].abs()
    if {"mean_policy_action_delta_l2", "success_gain"}.issubset(frame.columns):
        frame["repair_score"] = frame["mean_policy_action_delta_l2"] * frame["success_gain"].clip(lower=0.0)
        frame["destructive_score"] = frame["mean_policy_action_delta_l2"] * (-frame["success_gain"]).clip(lower=0.0)
    return frame


def pivot_metric(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        frame.pivot_table(index="layer_index", columns="token_bin_index", values=metric, aggfunc="mean")
        .sort_index()
        .sort_index(axis=1)
    )


def plot_heatmap(
    heat: pd.DataFrame,
    frame: pd.DataFrame,
    metric: str,
    output_path: Path,
    cmap: str,
    dpi: int,
    annotate_top: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = heat.to_numpy(dtype=float)
    fig_width = max(8.0, min(16.0, 0.28 * max(1, heat.shape[1]) + 4.0))
    fig_height = max(5.0, min(12.0, 0.22 * max(1, heat.shape[0]) + 2.5))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    if metric == "success_gain":
        limit = max(1e-6, float(np.nanmax(np.abs(data)))) if np.isfinite(data).any() else 1.0
        image = ax.imshow(data, aspect="auto", origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit)
    else:
        vmax = float(np.nanpercentile(data[np.isfinite(data)], 98)) if np.isfinite(data).any() else 1.0
        image = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap, vmin=0.0, vmax=max(vmax, 1e-6))
    ax.set_title(metric.replace("_", " "))
    ax.set_xlabel("Token bin index")
    ax.set_ylabel("Layer index")
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels([str(int(x)) for x in heat.columns], rotation=90, fontsize=7)
    y_step = max(1, int(np.ceil(len(heat.index) / 18)))
    y_positions = np.arange(0, len(heat.index), y_step)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([str(int(heat.index[i])) for i in y_positions], fontsize=8)
    if annotate_top > 0 and metric in frame.columns:
        top = frame.dropna(subset=[metric]).sort_values(metric, ascending=False).head(annotate_top)
        col_to_x = {int(value): index for index, value in enumerate(heat.columns)}
        row_to_y = {int(value): index for index, value in enumerate(heat.index)}
        for _, row in top.iterrows():
            layer = int(row["layer_index"])
            token_bin = int(row["token_bin_index"])
            if layer in row_to_y and token_bin in col_to_x:
                ax.scatter(col_to_x[token_bin], row_to_y[layer], marker="*", s=70, c="cyan", edgecolors="black", linewidths=0.5)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label=metric)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_multi_metric_summary(frame: pd.DataFrame, metrics: list[str], output_path: Path, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    available = [metric for metric in metrics if metric in frame.columns]
    fig, axes = plt.subplots(len(available), 1, figsize=(9, max(3, 2.2 * len(available))), dpi=dpi, sharex=True)
    if len(available) == 1:
        axes = [axes]
    x = np.arange(len(frame))
    ordered = frame.sort_values("layer_index").sort_values("token_bin_index")
    labels = [f"L{int(row.layer_index)}:B{int(row.token_bin_index)}" for row in ordered.itertuples()]
    for ax, metric in zip(axes, available):
        ax.plot(x, ordered[metric].to_numpy(dtype=float), linewidth=1.5)
        ax.set_ylabel(metric.replace("_", "\n"), fontsize=8)
        ax.grid(True, alpha=0.25)
    tick_step = max(1, len(labels) // 24)
    axes[-1].set_xticks(x[::tick_step])
    axes[-1].set_xticklabels(labels[::tick_step], rotation=90, fontsize=6)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
