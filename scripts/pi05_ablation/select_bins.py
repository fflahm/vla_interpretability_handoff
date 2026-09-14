#!/usr/bin/env python
"""Select layer-matched good/bad occupancy bins for PI0.5 ablation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.pi05_probe_ablation import select_layer_matched_probe_pairs  # noqa: E402
from src.utils import ensure_dir, resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "From occupancy metrics.csv, pick two high-IoU and two low-IoU bins in "
            "the same layer, spread across depth. Unlike global top-k/bottom-k, "
            "good/bad cells are layer-matched so ablation does not confound IoU "
            "with layer or a single token position."
        )
    )
    p.add_argument(
        "--metrics",
        default="outputs/self_occupancy/pi05_libero_spatial_10k/metrics.csv",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/ablation/pi05_probe_guided_layer_matched",
    )
    p.add_argument(
        "--layers-per-tower",
        "--pairs-per-tower",
        dest="layers_per_tower",
        type=int,
        default=5,
        help="Number of depth-spread layers per tower. --pairs-per-tower is an alias.",
    )
    p.add_argument(
        "--bins-per-group",
        type=int,
        default=2,
        help="Good bins and bad bins to pick in each chosen layer.",
    )
    p.add_argument("--min-iou-gap", type=float, default=0.03)
    p.add_argument("--max-good-bin-repeats", type=int, default=2)
    p.add_argument(
        "--expert-max-bin",
        type=int,
        default=48,
        help="Drop Expert bins above this index (PI0.5 Expert sequences are shorter than 96). "
        "Pass a negative value to keep all bins.",
    )
    p.add_argument("--token-bins", type=int, default=96)
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def plot_selected(selected: pd.DataFrame, cells: pd.DataFrame, path: Path, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=dpi)
    colors = {"best": "#dc2626", "worst": "#6b7280"}
    for ax, tower in zip(axes, ("paligemma", "expert")):
        background = cells[cells["tower"] == tower]
        ax.scatter(
            background["token_bin_index"],
            background["layer_index"],
            c=background["probe_soft_iou"],
            cmap="viridis",
            s=22,
            alpha=0.35,
            linewidths=0,
        )
        subset = selected[selected["tower"] == tower]
        labeled_layers: set[int] = set()
        for pair_id, pair in subset.groupby("pair_id"):
            best = pair[pair["probe_group"] == "best"].iloc[0]
            worst = pair[pair["probe_group"] == "worst"].iloc[0]
            ax.plot(
                [worst["token_bin_index"], best["token_bin_index"]],
                [worst["layer_index"], best["layer_index"]],
                color="#111827",
                lw=1.0,
                zorder=3,
            )
            layer = int(best["layer_index"])
            if layer not in labeled_layers:
                ax.annotate(
                    f"L{layer}",
                    (best["token_bin_index"], best["layer_index"]),
                    textcoords="offset points",
                    xytext=(4, 2),
                    fontsize=7,
                )
                labeled_layers.add(layer)
        for group, marker in (("best", "o"), ("worst", "X")):
            rows = subset[subset["probe_group"] == group]
            ax.scatter(
                rows["token_bin_index"],
                rows["layer_index"],
                s=70,
                c=colors[group],
                marker=marker,
                label=group,
                zorder=4,
                edgecolors="white",
                linewidths=0.6,
            )
        ax.set_title(tower)
        ax.set_xlabel("token bin")
        ax.set_ylabel("layer")
        ax.invert_yaxis()
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, loc="best")
    fig.suptitle("Layer-matched good/bad occupancy bins")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    metrics_path = resolve_path(args.metrics, ROOT)
    output_dir = ensure_dir(resolve_path(args.output_dir, ROOT))
    metrics = pd.read_csv(metrics_path)
    expert_max_bin = None if args.expert_max_bin < 0 else args.expert_max_bin
    selected = select_layer_matched_probe_pairs(
        metrics,
        pairs_per_tower=args.layers_per_tower,
        bins_per_group=args.bins_per_group,
        min_iou_gap=args.min_iou_gap,
        max_good_bin_repeats=args.max_good_bin_repeats,
        expert_max_bin=expert_max_bin,
    )
    selected["global_layer_index"] = selected["layer_index"] + selected["tower"].map(
        {"paligemma": 0, "expert": 18}
    )
    selected["token_bins"] = int(args.token_bins)
    selected_path = output_dir / "selected_conditions.csv"
    selected.to_csv(selected_path, index=False)

    from src.pi05_probe_ablation import _occupancy_probe_cells

    cells = _occupancy_probe_cells(metrics)
    if expert_max_bin is not None:
        expert = cells["tower"] == "expert"
        cells = cells[~(expert & (cells["token_bin_index"] > int(expert_max_bin)))]
    plot_selected(selected, cells, output_dir / "figures" / "00_selected_layer_matched_bins.png", args.dpi)

    summary = {
        "selection_mode": "layer_matched",
        "metrics": str(metrics_path),
        "pairs_per_tower": int(args.layers_per_tower),
        "layers_per_tower": int(args.layers_per_tower),
        "bins_per_group": int(args.bins_per_group),
        "min_iou_gap": float(args.min_iou_gap),
        "max_good_bin_repeats": int(args.max_good_bin_repeats),
        "expert_max_bin": expert_max_bin,
        "num_selected": int(len(selected)),
        "rule": (
            "Average Expert flow-time soft IoU, then in each chosen layer pick the "
            f"{int(args.bins_per_group)} highest-IoU bins (good) and "
            f"{int(args.bins_per_group)} lowest-IoU bins (bad) with "
            "min(good)-max(bad) >= min_iou_gap. Layers are spread across depth. "
            "Expert bins above expert_max_bin are ignored. A token bin may appear "
            "in the good group at most max_good_bin_repeats times."
        ),
        "by_tower": [],
    }
    for tower, subset in selected.groupby("tower", sort=True):
        good_bins = subset.loc[subset["probe_group"] == "best", "token_bin_index"].tolist()
        summary["by_tower"].append(
            {
                "tower": tower,
                "n": int(len(subset)),
                "layers": sorted(int(v) for v in subset["layer_index"].unique()),
                "mean_iou_gap": float(subset["iou_gap"].mean()),
                "min_iou_gap": float(subset["iou_gap"].min()),
                "good_bins": [int(v) for v in good_bins],
            }
        )
    (output_dir / "selection.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(selected.to_string(index=False))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {selected_path}")


if __name__ == "__main__":
    main()
