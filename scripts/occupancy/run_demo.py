#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.libero_self_occupancy import (
    OccupancyGridSpec,
    collect_self_occupancy_samples,
    extract_pi05_layer_activations,
    grouped_demo_split,
    train_layerwise_decoders,
    write_json,
    write_jsonl,
)
from src.utils import ensure_dir, log, set_seed


DEFAULT_HDF5 = Path(
    "/data/tos/guoshengyu/vla/libero/libero_spatial/"
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
)
DEFAULT_MODEL = Path("/data/tos/guoshengyu/vla/models/pi05_libero")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end PI0.5 layerwise decoding demo for LIBERO robot soft 3D occupancy."
    )
    parser.add_argument("--hdf5", type=Path, default=DEFAULT_HDF5)
    parser.add_argument(
        "--pi05-path",
        default=os.environ.get("PI05_PATH") or os.environ.get("VLA_PI05_PATH") or str(DEFAULT_MODEL),
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "self_occupancy" / "pi05_demo")
    parser.add_argument("--num-demos", type=int, default=6)
    parser.add_argument("--frames-per-demo", type=int, default=4)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--bottleneck", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--reuse-gt",
        action="store_true",
        help="Reuse occupancy_targets.npz and samples.jsonl when present.",
    )
    parser.add_argument(
        "--reuse-activations",
        action="store_true",
        help="Reuse activations.npz when present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_started = time.perf_counter()
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir.resolve())
    spec = OccupancyGridSpec(size=args.grid_size, supersample=args.supersample)
    timings: dict[str, object] = {}
    log("=" * 72)
    log(
        f"PI0.5 self-occupancy demo out={output_dir} demos={args.num_demos} "
        f"frames={args.frames_per_demo} epochs={args.epochs} device={args.device}"
    )

    gt_npz = output_dir / "occupancy_targets.npz"
    samples_path = output_dir / "samples.jsonl"
    gt_metadata_path = output_dir / "occupancy_metadata.json"
    if args.reuse_gt and gt_npz.exists() and samples_path.exists() and gt_metadata_path.exists():
        stage_started = time.perf_counter()
        log("Reusing existing GT occupancy targets")
        packed = np.load(gt_npz, allow_pickle=False)
        occupancy = packed["occupancy"].astype(np.float32)
        sample_rows = _read_jsonl(samples_path)
        samples = [_sample_from_row(row, occupancy[index]) for index, row in enumerate(sample_rows)]
        gt_metadata = json.loads(gt_metadata_path.read_text(encoding="utf-8"))
        timings["gt"] = {"reused": True, "seconds": time.perf_counter() - stage_started}
    else:
        log("Stage GT: collecting soft self-occupancy")
        samples, gt_metadata = collect_self_occupancy_samples(
            hdf5_path=args.hdf5.resolve(),
            output_dir=output_dir,
            num_demos=args.num_demos,
            frames_per_demo=args.frames_per_demo,
            spec=spec,
        )
        occupancy = np.stack([sample.occupancy for sample in samples], axis=0)
        np.savez_compressed(gt_npz, occupancy=occupancy)
        write_jsonl(samples_path, [sample.metadata() for sample in samples])
        write_json(gt_metadata_path, gt_metadata)
        timings["gt"] = {"reused": False, **gt_metadata}

    activation_path = output_dir / "activations.npz"
    if args.reuse_activations and activation_path.exists():
        stage_started = time.perf_counter()
        log("Reusing existing activations.npz")
        packed = np.load(activation_path, allow_pickle=True)
        activations = packed["X_layers"].astype(np.float32)
        layer_names = [str(item) for item in packed["layer_names"].tolist()]
        timings["activations"] = {"reused": True, "seconds": time.perf_counter() - stage_started}
    else:
        log("Stage activations: mean-token PI0.5 capture")
        activations, layer_names, activation_timing = extract_pi05_layer_activations(
            samples,
            model_id=args.pi05_path,
            device=args.device,
        )
        np.savez_compressed(
            activation_path,
            X_layers=activations.astype(np.float16),
            layer_names=np.asarray(layer_names),
        )
        timings["activations"] = {"reused": False, **activation_timing}

    train_idx, test_idx = grouped_demo_split(samples, args.test_fraction, args.seed)
    split = {
        "strategy": "grouped_by_demo",
        "seed": args.seed,
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "train_demos": sorted({samples[index].demo_key for index in train_idx}),
        "test_demos": sorted({samples[index].demo_key for index in test_idx}),
    }
    write_json(output_dir / "split.json", split)
    log(f"Split train={len(train_idx)} test={len(test_idx)} demos_train={len(split['train_demos'])} demos_test={len(split['test_demos'])}")

    log("Stage train: layerwise occupancy decoders")
    rows, predictions, decoder_timing = train_layerwise_decoders(
        activations=activations,
        occupancy=occupancy,
        layer_names=layer_names,
        train_idx=train_idx,
        test_idx=test_idx,
        epochs=args.epochs,
        bottleneck=args.bottleneck,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
    )
    timings["decoder"] = decoder_timing
    metrics_path = output_dir / "layerwise_metrics.csv"
    _write_csv(metrics_path, rows)
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        predictions=predictions.astype(np.float16),
        targets=occupancy[test_idx].astype(np.float16),
        test_indices=test_idx,
        layer_names=np.asarray(layer_names),
    )

    plot_started = time.perf_counter()
    figure_path = output_dir / "layerwise_soft_iou.png"
    _plot_layerwise(rows, figure_path)
    occupancy_figure_path = output_dir / "best_layer_occupancy_examples.png"
    _plot_occupancy_examples(rows, predictions, occupancy[test_idx], occupancy_figure_path)
    timings["plotting_seconds"] = time.perf_counter() - plot_started
    timings["total_seconds"] = time.perf_counter() - total_started
    write_json(output_dir / "timings.json", timings)

    best = max(rows, key=lambda row: float(row["soft_iou"]))
    summary = {
        "experiment": "pi05_layerwise_soft_self_occupancy_demo",
        "status": "completed",
        "scope": "small grouped-demo smoke test; no ablation or information-matched baselines",
        "hdf5": str(args.hdf5.resolve()),
        "pi05_path": args.pi05_path,
        "num_samples": len(samples),
        "num_train": len(train_idx),
        "num_test": len(test_idx),
        "grid": spec.to_dict(),
        "decoder": {
            "architecture": f"Linear(D,{args.bottleneck})-GELU-Linear({args.bottleneck},{args.grid_size ** 3})",
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "loss": "pos-weighted BCE + soft Dice",
        },
        "primary_metric": "held-out-demo soft IoU",
        "best_layer": best,
        "outputs": [
            "occupancy_targets.npz",
            "occupancy_metadata.json",
            "samples.jsonl",
            "activations.npz",
            "split.json",
            "layerwise_metrics.csv",
            "layerwise_soft_iou.png",
            "best_layer_occupancy_examples.png",
            "test_predictions.npz",
            "timings.json",
            "report.md",
        ],
    }
    write_json(output_dir / "summary.json", summary)
    _write_report(output_dir / "report.md", summary, rows, timings)
    log(f"Demo completed total_seconds={timings['total_seconds']:.1f} best={best['layer_name']} soft_iou={best['soft_iou']:.4f}")
    log("=" * 72)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def _sample_from_row(row: dict, occupancy: np.ndarray):
    from src.libero_self_occupancy import SelfOccupancySample

    return SelfOccupancySample(
        sample_id=int(row["sample_id"]),
        demo_key=str(row["demo_key"]),
        frame_index=int(row["frame_index"]),
        image_path=str(row["image_path"]),
        wrist_image_path=str(row["wrist_image_path"]),
        instruction=str(row["instruction"]),
        observation_state=[float(value) for value in row["observation_state"]],
        occupancy=occupancy,
    )


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_layerwise(rows: list[dict], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 4.8))
    colors = {"paligemma": "#2563eb", "expert": "#dc2626"}
    for tower in ("paligemma", "expert"):
        tower_rows = [row for row in rows if row["tower"] == tower]
        local_layers = list(range(len(tower_rows)))
        axis.plot(
            local_layers,
            [row["soft_iou"] for row in tower_rows],
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=tower,
            color=colors[tower],
        )
    axis.set_xlabel("Local transformer layer index")
    axis.set_ylabel("Held-out demo soft IoU")
    axis.set_title("PI0.5 layerwise decoding of Panda soft 3D occupancy")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_occupancy_examples(
    rows: list[dict], predictions: np.ndarray, targets: np.ndarray, path: Path
) -> None:
    best_index = int(max(rows, key=lambda row: float(row["soft_iou"]))["layer"])
    count = min(4, len(targets))
    figure, axes = plt.subplots(2, count, figsize=(3.2 * count, 6.0), squeeze=False)
    for index in range(count):
        target_projection = targets[index].max(axis=1)
        prediction_projection = predictions[best_index, index].max(axis=1)
        axes[0, index].imshow(target_projection.T, origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
        axes[1, index].imshow(prediction_projection.T, origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
        axes[0, index].set_title(f"GT test {index}")
        axes[1, index].set_title(f"Prediction test {index}")
        for axis in (axes[0, index], axes[1, index]):
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(f"Best layer {best_index}: max projection over base-frame y axis")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_report(path: Path, summary: dict, rows: list[dict], timings: dict) -> None:
    best_rows = sorted(rows, key=lambda row: float(row["soft_iou"]), reverse=True)[:5]
    lines = [
        "# PI0.5 Soft Self-Occupancy Demo",
        "",
        f"- Samples: {summary['num_samples']} ({summary['num_train']} train / {summary['num_test']} test)",
        f"- Grid: {summary['grid']['shape']} in Panda base frame",
        f"- Primary metric: {summary['primary_metric']}",
        f"- Best layer: `{summary['best_layer']['layer_name']}`",
        f"- Best soft IoU: {float(summary['best_layer']['soft_iou']):.4f}",
        "",
        "## Top layers",
        "",
        "| Layer | Tower | Soft IoU | Hard IoU | Train seconds |",
        "|---|---|---:|---:|---:|",
    ]
    for row in best_rows:
        lines.append(
            f"| {row['layer_name']} | {row['tower']} | {float(row['soft_iou']):.4f} | "
            f"{float(row['hard_iou']):.4f} | {float(row['train_seconds']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Stage timings",
            "",
            "```json",
            json.dumps(timings, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Interpretation limit",
            "",
            "This is a small held-out-demonstration smoke test. Without proprio, pixel, mean-shape, "
            "or shuffled-label baselines, soft IoU measures decodability only and is not sufficient "
            "evidence for an independent causal 3D self-model.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
