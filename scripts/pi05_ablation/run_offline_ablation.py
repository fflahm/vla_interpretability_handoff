#!/usr/bin/env python
"""Ablate selected PI0.5 bins on occupancy frames with a fresh action chunk each time."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.pi05_frame_ablation import (  # noqa: E402
    Pi05OfflineChunkPredictor,
    frame_provenance_table,
    load_frame_deltas,
    load_occupancy_frames,
    load_selected_bins,
    load_task_hdf5_paths,
    per_frame_good_minus_bad,
    plot_chunk_delta_histograms,
    run_offline_frame_ablation,
    sample_frames,
    save_frame_deltas,
    summarize_bin_deltas,
)
from src.utils import ensure_dir, load_config, resolve_path, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample occupancy frames, run a fresh PI0.5 action-chunk forward for the "
            "baseline and for each selected bin, and record chunk L2 deltas."
        )
    )
    parser.add_argument(
        "--probe-run-dir",
        default="outputs/self_occupancy/pi05_libero_spatial_10k",
        help="Occupancy run with samples.jsonl plus agent/wrist images.",
    )
    parser.add_argument(
        "--selected-csv",
        default="outputs/ablation/pi05_probe_guided_layer_matched/selected_conditions.csv",
        help="Candidate bins from scripts/pi05_ablation/select_bins.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/ablation/pi05_probe_guided_frames",
    )
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--pi05-path", default=None)
    parser.add_argument("--num-frames", type=int, default=1000)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--token-bins", type=int, default=96)
    parser.add_argument("--device", default=None)
    parser.add_argument("--mode", choices=("zero", "scale"), default="zero")
    parser.add_argument("--scale", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probe_run_dir = resolve_path(args.probe_run_dir, ROOT)
    output_dir = ensure_dir(resolve_path(args.output_dir, ROOT))
    selected_src = resolve_path(args.selected_csv, ROOT)
    selected = load_selected_bins(selected_src)
    selected.to_csv(output_dir / "selected_conditions.csv", index=False)
    deltas_path = output_dir / "frame_deltas.npz"
    metrics_path = output_dir / "bin_metrics.csv"

    if args.plot_only:
        packed = load_frame_deltas(deltas_path)
        metrics = pd.read_csv(metrics_path) if metrics_path.exists() else summarize_bin_deltas(
            selected, packed["chunk_l2"]
        )
        _finalize(output_dir, selected, packed["chunk_l2"], metrics, dpi=args.dpi)
        return

    cfg = load_config(resolve_path(args.config, ROOT))
    set_seed(int(args.sample_seed))
    checkpoint_path = (
        args.pi05_path
        or os.environ.get("PI05_PATH")
        or cfg.get("model", {}).get("pi05_pretrained_path")
    )
    if not checkpoint_path:
        raise ValueError("Provide --pi05-path, set PI05_PATH, or configure model.pi05_pretrained_path.")
    device = str(args.device or cfg.get("model", {}).get("device", "auto"))

    frames = load_occupancy_frames(probe_run_dir)
    sampled = sample_frames(frames, args.num_frames, np.random.default_rng(int(args.sample_seed)))
    hdf5_by_task = load_task_hdf5_paths(probe_run_dir)
    provenance = frame_provenance_table(sampled, hdf5_by_task)
    provenance.to_csv(output_dir / "sampled_frames.csv", index=False)
    (output_dir / "sampling.json").write_text(
        json.dumps(
            {
                "num_frames": len(sampled),
                "sample_seed": int(args.sample_seed),
                "probe_run_dir": str(probe_run_dir),
                "alignment": (
                    "sampled_frames.csv row_index i matches frame_deltas.npz "
                    "chunk_l2[i]. libero_frame_index is the timestep inside the "
                    "original LIBERO HDF5 demo (data[demo_key])."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    predictor = Pi05OfflineChunkPredictor(str(checkpoint_path), device=device)
    try:
        sample_ids, chunk_l2 = run_offline_frame_ablation(
            sampled,
            selected,
            predictor.predict,
            token_bins=args.token_bins,
            mode=args.mode,
            scale=args.scale,
            checkpoint_path=deltas_path,
            save_every=args.save_every,
        )
    finally:
        del predictor

    save_frame_deltas(
        deltas_path,
        sample_ids=sample_ids,
        chunk_l2=chunk_l2,
        extra={
            "completed_frames": np.asarray([len(sampled)], dtype=np.int32),
            "libero_frame_index": np.asarray([frame.frame_index for frame in sampled], dtype=np.int32),
        },
    )
    metrics = summarize_bin_deltas(selected, chunk_l2)
    metrics.to_csv(metrics_path, index=False)
    summary = _finalize(output_dir, selected, chunk_l2, metrics, dpi=args.dpi)
    summary.update(
        {
            "probe_run_dir": str(probe_run_dir),
            "selected_csv": str(selected_src),
            "pi05_path": str(checkpoint_path),
            "num_frames": int(len(sampled)),
            "num_bins": int(len(selected)),
            "mode": args.mode,
            "scale": float(args.scale),
            "token_bins": int(args.token_bins),
            "sample_seed": int(args.sample_seed),
            "fresh_chunk_each_forward": True,
            "stored": [
                "selected_conditions.csv",
                "sampled_frames.csv",
                "sampling.json",
                "frame_deltas.npz",
                "bin_metrics.csv",
                "summary.json",
                "figures/",
            ],
        }
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved compact frame-ablation results to {output_dir}")


def _finalize(
    output_dir: Path,
    selected: pd.DataFrame,
    chunk_l2,
    metrics: pd.DataFrame,
    dpi: int,
) -> dict:
    plot_chunk_delta_histograms(selected, chunk_l2, ensure_dir(output_dir / "figures"), dpi=dpi)
    diffs = per_frame_good_minus_bad(selected, chunk_l2)
    diffs.to_csv(output_dir / "frame_good_minus_bad.csv", index=False)
    by_group = {}
    for group, subset in metrics.groupby("probe_group"):
        by_group[str(group)] = {
            "n_bins": int(len(subset)),
            "mean_chunk_delta_l2": float(subset["mean_chunk_delta_l2"].mean()),
            "median_of_bin_means": float(subset["mean_chunk_delta_l2"].median()),
        }
    overall = diffs["good_minus_bad"].to_numpy(dtype=float)
    summary = {
        "mean_chunk_delta_l2": float(metrics["mean_chunk_delta_l2"].mean()),
        "by_group": by_group,
        "num_compared_forwards": int(np.asarray(chunk_l2).shape[0] * (1 + np.asarray(chunk_l2).shape[1])),
        "good_minus_bad": {
            "mean": float(np.mean(overall)),
            "median": float(np.median(overall)),
            "frac_good_gt_bad": float(np.mean(overall > 0)),
            "n_frames": int(len(overall)),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
