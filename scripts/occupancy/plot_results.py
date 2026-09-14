#!/usr/bin/env python
"""Plot summaries from a finished PI0.5 self-occupancy full run.

Figures
-------
1. Layer-wise soft IoU (mean over token bins and flow times), PaliGemma / Expert
2. Token-position soft IoU (mean over layers and flow times), PaliGemma / Expert
3. Train/test curves for a random subset of probes (from saved epoch_curves.csv)
4. Max-y occupancy projections: GT vs all measured bins for one condition
5. Multi-layer projection contrast (good vs bad layers; best bin per condition)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.pi05_occupancy_full import (  # noqa: E402
    load_activation_manifest,
    load_condition_activations,
    probe_dir,
    select_bin_columns,
)
from src.utils import ensure_dir, log, set_seed  # noqa: E402

CONDITION_RE = re.compile(
    r"^(?P<tower>paligemma|expert)/layer_(?P<layer>\d+)(?:/(?:static|t=(?P<flow>[0-9.]+)))?$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "outputs/self_occupancy/pi05_libero_spatial_10k",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to <run-dir>/figures.")
    parser.add_argument("--plots", default="1,2,3,4,5", help="Comma list: 1,2,3,4,5.")
    parser.add_argument("--metric", default="soft_iou", choices=("soft_iou", "hard_iou"))
    parser.add_argument("--curve-bins", type=int, default=10, help="How many probes to show in plot 3.")
    parser.add_argument("--num-frames", type=int, default=4, help="Test frames for projection plots.")
    parser.add_argument(
        "--condition",
        default="auto",
        help="Condition key for plot 4, or `auto` for highest mean soft IoU.",
    )
    parser.add_argument(
        "--contrast-conditions",
        default="auto",
        help=(
            "Plot 5: comma-separated condition keys, or `auto` to pick strong/weak "
            "expert and paligemma layers (best bin each)."
        ),
    )
    parser.add_argument(
        "--contrast-count",
        type=int,
        default=6,
        help="When --contrast-conditions=auto, how many layer probes to show (good+bad mix).",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def parse_condition(condition: str) -> dict[str, Any]:
    match = CONDITION_RE.match(condition)
    if not match:
        raise ValueError(f"Unrecognized condition key: {condition!r}")
    flow = match.group("flow")
    return {
        "condition": condition,
        "tower": match.group("tower"),
        "layer": int(match.group("layer")),
        "flow": None if flow is None else float(flow),
    }


def load_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            parsed = parse_condition(raw["condition"])
            rows.append(
                {
                    **raw,
                    **parsed,
                    "bin": int(raw["bin"]),
                    "soft_iou": float(raw["soft_iou"]),
                    "hard_iou": float(raw["hard_iou"]),
                    "train_loss": float(raw["train_loss"]),
                }
            )
    if not rows:
        raise FileNotFoundError(f"No metric rows in {path}")
    return rows


def mean_by(rows: list[dict[str, Any]], keys: tuple[str, ...], metric: str) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        buckets.setdefault(key, []).append(float(row[metric]))
    out = []
    for key, values in sorted(buckets.items()):
        item = {name: value for name, value in zip(keys, key)}
        item[metric] = float(np.mean(values))
        item["n"] = len(values)
        item["std"] = float(np.std(values))
        out.append(item)
    return out


def plot_layerwise(rows: list[dict[str, Any]], metric: str, output_dir: Path, dpi: int) -> Path:
    aggregated = mean_by(rows, ("tower", "layer"), metric)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for axis, tower, color in zip(axes, ("paligemma", "expert"), ("#2563eb", "#dc2626")):
        tower_rows = [row for row in aggregated if row["tower"] == tower]
        xs = [row["layer"] for row in tower_rows]
        ys = [row[metric] for row in tower_rows]
        err = [row["std"] for row in tower_rows]
        axis.plot(xs, ys, marker="o", linewidth=1.8, markersize=4, color=color)
        axis.fill_between(
            xs, np.asarray(ys) - np.asarray(err), np.asarray(ys) + np.asarray(err),
            color=color, alpha=0.15, linewidth=0,
        )
        axis.set_title(f"{tower}: layer-wise (mean over bins & flow)")
        axis.set_xlabel("Layer index")
        axis.grid(alpha=0.25)
        if tower_rows:
            axis.set_xticks(xs)
    axes[0].set_ylabel(metric.replace("_", " "))
    figure.suptitle("Held-out occupancy decoding vs transformer layer")
    figure.tight_layout()
    path = output_dir / f"01_layerwise_{metric}.png"
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def plot_token_position(rows: list[dict[str, Any]], metric: str, output_dir: Path, dpi: int) -> Path:
    aggregated = mean_by(rows, ("tower", "bin"), metric)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for axis, tower, color in zip(axes, ("paligemma", "expert"), ("#2563eb", "#dc2626")):
        tower_rows = [row for row in aggregated if row["tower"] == tower]
        xs = [row["bin"] for row in tower_rows]
        ys = [row[metric] for row in tower_rows]
        err = [row["std"] for row in tower_rows]
        axis.plot(xs, ys, marker="o", linewidth=1.8, markersize=4, color=color)
        axis.fill_between(
            xs, np.asarray(ys) - np.asarray(err), np.asarray(ys) + np.asarray(err),
            color=color, alpha=0.15, linewidth=0,
        )
        axis.set_title(f"{tower}: token-position (mean over layers & flow)")
        axis.set_xlabel("Token-position bin index")
        axis.grid(alpha=0.25)
        if tower_rows:
            axis.set_xticks(xs)
    axes[0].set_ylabel(metric.replace("_", " "))
    figure.suptitle("Held-out occupancy decoding vs token-position bin")
    figure.tight_layout()
    path = output_dir / f"02_token_position_{metric}.png"
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def load_epoch_curves(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "epoch_curves.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Re-run training with the updated script so per-epoch "
            "histories are written under epoch_curves.csv / probes/*/history.csv."
        )
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "condition": raw["condition"],
                    "bin": int(raw["bin"]),
                    "epoch": int(raw["epoch"]),
                    "train_loss": float(raw["train_loss"]),
                    "test_loss": float(raw.get("test_loss", "nan")),
                    "test_soft_iou": float(raw["test_soft_iou"]),
                    "test_hard_iou": float(raw["test_hard_iou"]),
                }
            )
    return rows


def plot_train_test_curves(curve_rows: list[dict[str, Any]], output_dir: Path, sample_n: int, seed: int, dpi: int) -> Path:
    probes = sorted({(row["condition"], int(row["bin"])) for row in curve_rows})
    rng = np.random.default_rng(seed)
    if len(probes) > sample_n:
        chosen = [probes[i] for i in sorted(rng.choice(len(probes), size=sample_n, replace=False).tolist())]
    else:
        chosen = probes
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    cmap = plt.get_cmap("tab10")
    for probe_index, (condition, bin_index) in enumerate(chosen):
        subset = sorted(
            [row for row in curve_rows if row["condition"] == condition and int(row["bin"]) == bin_index],
            key=lambda row: int(row["epoch"]),
        )
        color = cmap(probe_index % 10)
        label = f"{condition}#bin={bin_index}"
        axes[0].plot([row["epoch"] for row in subset], [row["train_loss"] for row in subset], color=color, lw=1.5, label=label)
        axes[1].plot([row["epoch"] for row in subset], [row["test_soft_iou"] for row in subset], color=color, lw=1.5, label=label)
    axes[0].set_title("Train loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE + Dice")
    axes[0].grid(alpha=0.25)
    axes[1].set_title("Held-out soft IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("soft IoU")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=6, frameon=False, loc="best")
    figure.suptitle("Train / test curves from saved epoch histories")
    figure.tight_layout()
    path = output_dir / "03_train_test_curves.png"
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def choose_condition(rows: list[dict[str, Any]], requested: str) -> str:
    if requested != "auto":
        available = {row["condition"] for row in rows}
        if requested not in available:
            raise ValueError(f"Condition {requested!r} not in metrics.csv")
        return requested
    means = mean_by(rows, ("condition",), "soft_iou")
    return str(max(means, key=lambda row: row["soft_iou"])["condition"])


def best_probe_for_condition(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    subset = [row for row in rows if row["condition"] == condition]
    if not subset:
        raise ValueError(f"No metrics for condition {condition!r}")
    return max(subset, key=lambda row: float(row["soft_iou"]))


def choose_contrast_probes(
    rows: list[dict[str, Any]],
    requested: str,
    count: int,
) -> list[dict[str, Any]]:
    """Return probe dicts with condition/bin/soft_iou for multi-layer projection columns."""
    if requested != "auto":
        probes = []
        for condition in [part.strip() for part in requested.split(",") if part.strip()]:
            best = best_probe_for_condition(rows, condition)
            probes.append(
                {
                    "condition": condition,
                    "bin": int(best["bin"]),
                    "soft_iou": float(best["soft_iou"]),
                    "tower": best["tower"],
                    "layer": int(best["layer"]),
                    "label": _contrast_label(best),
                }
            )
        return probes

    # Auto: rank layers by mean soft IoU, then take each layer's best (condition, bin).
    layer_scores: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        layer_scores.setdefault((row["tower"], int(row["layer"])), []).append(float(row["soft_iou"]))
    ranked = {
        tower: sorted(
            ((layer, float(np.mean(values))) for (tw, layer), values in layer_scores.items() if tw == tower),
            key=lambda item: item[1],
            reverse=True,
        )
        for tower in ("expert", "paligemma")
    }

    # Mix strong/weak for both towers: e.g. count=6 -> expert top2+bot2, pali top1+bot1
    expert_n = max(2, (count * 2 + 2) // 3)
    pali_n = max(2, count - expert_n)
    if expert_n % 2:
        expert_n += 1
    if pali_n % 2:
        pali_n = max(2, pali_n - 1)
    while expert_n + pali_n > count and expert_n > 2:
        expert_n -= 2
    while expert_n + pali_n > count and pali_n > 2:
        pali_n -= 2

    selected_layers: list[tuple[str, int, str]] = []
    for tower, take, tag_good, tag_bad in (
        ("expert", expert_n, "good", "bad"),
        ("paligemma", pali_n, "good", "bad"),
    ):
        ranking = ranked[tower]
        n_good = take // 2
        n_bad = take - n_good
        for layer, _score in ranking[:n_good]:
            selected_layers.append((tower, layer, tag_good))
        for layer, _score in ranking[-n_bad:]:
            selected_layers.append((tower, layer, tag_bad))

    probes: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for tower, layer, quality in selected_layers:
        key = (tower, layer)
        if key in seen:
            continue
        seen.add(key)
        subset = [row for row in rows if row["tower"] == tower and int(row["layer"]) == layer]
        best = max(subset, key=lambda row: float(row["soft_iou"]))
        probes.append(
            {
                "condition": best["condition"],
                "bin": int(best["bin"]),
                "soft_iou": float(best["soft_iou"]),
                "tower": tower,
                "layer": layer,
                "quality": quality,
                "label": _contrast_label(best, quality),
            }
        )
    # Order: good experts, bad experts, good pali, bad pali for readable left-to-right contrast.
    order = {"expert": 0, "paligemma": 1}
    qorder = {"good": 0, "bad": 1}
    probes.sort(key=lambda p: (order.get(p["tower"], 9), qorder.get(p.get("quality", ""), 9), -p["soft_iou"]))
    return probes[:count]


def _contrast_label(row: dict[str, Any], quality: str | None = None) -> str:
    flow = row.get("flow")
    flow_txt = "static" if flow is None else f"t={float(flow):.1f}"
    prefix = f"{quality}\n" if quality else ""
    return (
        f"{prefix}{row['tower'][0].upper()}L{int(row['layer'])}/{flow_txt}\n"
        f"bin{int(row['bin'])} IoU={float(row['soft_iou']):.3f}"
    )


def resolve_device(device: str):
    import torch

    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_decoder_and_predict(
    *,
    run_dir: Path,
    condition: str,
    bin_index: int,
    x_column: np.ndarray,
    sample_indices: list[int],
    device,
) -> np.ndarray:
    import torch
    from torch import nn

    ckpt_path = probe_dir(run_dir, condition, bin_index) / "decoder.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing decoder checkpoint {ckpt_path}. Re-run training with the updated script."
        )
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=device)
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    x = (x_column - mean) / std
    decoder = nn.Sequential(
        nn.Linear(int(checkpoint["input_dim"]), 64),
        nn.GELU(),
        nn.Linear(64, int(checkpoint.get("output_dim", 4096))),
    ).to(device)
    decoder.load_state_dict(checkpoint["state_dict"])
    decoder.eval()
    with torch.inference_mode():
        pred = decoder(torch.from_numpy(x[sample_indices]).to(device)).sigmoid().cpu().numpy()
    return pred.reshape((-1, 16, 16, 16))


def plot_projections(
    *,
    targets: np.ndarray,
    predictions_by_bin: dict[int, np.ndarray],
    sample_ids: list[int],
    condition: str,
    output_dir: Path,
    dpi: int,
) -> Path:
    bins = sorted(predictions_by_bin)
    n_frames = len(sample_ids)
    n_cols = 1 + len(bins)
    figure, axes = plt.subplots(
        n_frames, n_cols,
        figsize=(1.7 * n_cols + 1.0, 1.7 * n_frames + 1.2),
        squeeze=False,
    )
    for row_index, sample_id in enumerate(sample_ids):
        panels = [("GT", targets[row_index])]
        for bin_index in bins:
            panels.append((f"bin {bin_index}", predictions_by_bin[bin_index][row_index]))
        for col_index, (title, volume) in enumerate(panels):
            projection = np.asarray(volume, dtype=np.float32).max(axis=1)
            axis = axes[row_index, col_index]
            axis.imshow(projection.T, origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
            if row_index == 0:
                axis.set_title(title, fontsize=8)
            if col_index == 0:
                axis.set_ylabel(f"id {sample_id}", fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(
        f"{condition}: max projection over base-frame y (GT vs saved-decoder bins)",
        fontsize=11,
    )
    figure.tight_layout()
    path = output_dir / "04_occupancy_projections_all_bins.png"
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def plot_layer_contrast_projections(
    *,
    targets: np.ndarray,
    predictions: list[tuple[dict[str, Any], np.ndarray]],
    sample_ids: list[int],
    output_dir: Path,
    dpi: int,
) -> Path:
    """Rows=frames, columns=GT + one best-bin prediction per selected layer/condition."""
    n_frames = len(sample_ids)
    n_cols = 1 + len(predictions)
    figure, axes = plt.subplots(
        n_frames,
        n_cols,
        figsize=(1.85 * n_cols + 1.2, 1.85 * n_frames + 1.6),
        squeeze=False,
    )
    for row_index, sample_id in enumerate(sample_ids):
        panels: list[tuple[str, np.ndarray, str | None]] = [("GT", targets[row_index], None)]
        for probe, volumes in predictions:
            panels.append((probe["label"], volumes[row_index], probe.get("quality")))
        for col_index, (title, volume, quality) in enumerate(panels):
            projection = np.asarray(volume, dtype=np.float32).max(axis=1)
            axis = axes[row_index, col_index]
            axis.imshow(projection.T, origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
            if row_index == 0:
                color = {"good": "#15803d", "bad": "#b91c1c"}.get(quality or "", "black")
                axis.set_title(title, fontsize=7, color=color)
            if col_index == 0:
                axis.set_ylabel(f"id {sample_id}", fontsize=8)
            if quality == "good":
                for spine in axis.spines.values():
                    spine.set_edgecolor("#15803d")
                    spine.set_linewidth(1.5)
            elif quality == "bad":
                for spine in axis.spines.values():
                    spine.set_edgecolor("#b91c1c")
                    spine.set_linewidth(1.5)
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(
        "Layer contrast: max-y soft occupancy (green=strong layers, red=weak; best bin each)",
        fontsize=11,
    )
    figure.tight_layout()
    path = output_dir / "05_occupancy_projections_layer_contrast.png"
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def _select_test_frames(run_dir: Path, num_frames: int, seed: int) -> tuple[list[int], np.ndarray]:
    split = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
    test_idx = np.asarray(split["test_indices"], dtype=int)
    rng = np.random.default_rng(seed + 7)
    n_frames = min(num_frames, len(test_idx))
    local = sorted(rng.choice(len(test_idx), size=n_frames, replace=False).tolist())
    sample_ids = [int(test_idx[i]) for i in local]
    occupancy = np.load(run_dir / "occupancy.npy", mmap_mode="r").astype(np.float32)
    return sample_ids, occupancy[sample_ids]


def _predict_probe(
    *,
    run_dir: Path,
    condition: str,
    bin_index: int,
    sample_ids: list[int],
    device,
    act_manifest: dict[str, Any],
    cache: dict[str, np.ndarray],
) -> np.ndarray:
    if condition not in cache:
        cache[condition] = load_condition_activations(run_dir, condition)
    x_all = cache[condition]
    bin_columns = select_bin_columns(
        condition=condition,
        x_all=x_all,
        requested_bin_indices=[bin_index],
        manifest=act_manifest,
    )
    _, column = bin_columns[0]
    return load_decoder_and_predict(
        run_dir=run_dir,
        condition=condition,
        bin_index=bin_index,
        x_column=x_all[:, column],
        sample_indices=sample_ids,
        device=device,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_dir = args.run_dir.resolve()
    output_dir = ensure_dir((args.output_dir or (run_dir / "figures")).resolve())
    plot_ids = {part.strip() for part in args.plots.split(",") if part.strip()}
    rows = load_metrics(run_dir / "metrics.csv")
    log(f"Loaded {len(rows)} metric rows from {run_dir / 'metrics.csv'}")
    log(f"Writing figures to {output_dir}; plots={sorted(plot_ids)}")

    saved: list[str] = []
    if "1" in plot_ids:
        path = plot_layerwise(rows, args.metric, output_dir, args.dpi)
        log(f"Saved {path}")
        saved.append(str(path))
    if "2" in plot_ids:
        path = plot_token_position(rows, args.metric, output_dir, args.dpi)
        log(f"Saved {path}")
        saved.append(str(path))
    if "3" in plot_ids:
        curves = load_epoch_curves(run_dir)
        path = plot_train_test_curves(curves, output_dir, args.curve_bins, args.seed, args.dpi)
        log(f"Saved {path} from {len(curves)} epoch history rows")
        saved.append(str(path))

    need_frames = ("4" in plot_ids) or ("5" in plot_ids)
    sample_ids: list[int] = []
    targets = np.zeros((0, 16, 16, 16), dtype=np.float32)
    device = None
    act_manifest: dict[str, Any] = {}
    activation_cache: dict[str, np.ndarray] = {}
    if need_frames:
        sample_ids, targets = _select_test_frames(run_dir, args.num_frames, args.seed)
        device = resolve_device(args.device)
        act_manifest = load_activation_manifest(run_dir)
        log(f"Projection frames sample_ids={sample_ids} device={device}")

    if "4" in plot_ids:
        condition = choose_condition(rows, args.condition)
        condition_bins = sorted({int(row["bin"]) for row in rows if row["condition"] == condition})
        log(f"Plot4 condition={condition} bins={condition_bins}")
        predictions_by_bin: dict[int, np.ndarray] = {}
        for bin_index in condition_bins:
            predictions_by_bin[bin_index] = _predict_probe(
                run_dir=run_dir,
                condition=condition,
                bin_index=bin_index,
                sample_ids=sample_ids,
                device=device,
                act_manifest=act_manifest,
                cache=activation_cache,
            )
        path = plot_projections(
            targets=targets,
            predictions_by_bin=predictions_by_bin,
            sample_ids=sample_ids,
            condition=condition,
            output_dir=output_dir,
            dpi=args.dpi,
        )
        meta = {
            "condition": condition,
            "sample_ids": sample_ids,
            "bins": sorted(predictions_by_bin),
            "note": "Predictions come from saved probes/*/decoder.pt (no retraining).",
        }
        (output_dir / "04_frame_sample_ids.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        log(f"Saved {path}")
        saved.append(str(path))

    if "5" in plot_ids:
        probes = choose_contrast_probes(rows, args.contrast_conditions, args.contrast_count)
        log("Plot5 contrast probes: " + "; ".join(f"{p['condition']}#bin={p['bin']}({p['soft_iou']:.3f})" for p in probes))
        predictions: list[tuple[dict[str, Any], np.ndarray]] = []
        for probe in probes:
            volumes = _predict_probe(
                run_dir=run_dir,
                condition=probe["condition"],
                bin_index=int(probe["bin"]),
                sample_ids=sample_ids,
                device=device,
                act_manifest=act_manifest,
                cache=activation_cache,
            )
            predictions.append((probe, volumes))
        path = plot_layer_contrast_projections(
            targets=targets,
            predictions=predictions,
            sample_ids=sample_ids,
            output_dir=output_dir,
            dpi=args.dpi,
        )
        meta = {
            "sample_ids": sample_ids,
            "probes": [
                {
                    "condition": p["condition"],
                    "bin": int(p["bin"]),
                    "soft_iou": float(p["soft_iou"]),
                    "quality": p.get("quality"),
                    "label": p["label"],
                }
                for p, _ in predictions
            ],
            "note": "Each column uses that layer's best bin by soft IoU; green=strong, red=weak.",
        }
        (output_dir / "05_layer_contrast_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        log(f"Saved {path}")
        saved.append(str(path))

    summary = {"run_dir": str(run_dir), "output_dir": str(output_dir), "saved": saved}
    (output_dir / "plot_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()