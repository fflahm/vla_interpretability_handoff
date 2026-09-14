#!/usr/bin/env python
"""Full, resumable LIBERO-Spatial PI0.5 token-position occupancy experiment."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.libero_self_occupancy import (  # noqa: E402
    OccupancyGridSpec, collect_self_occupancy_samples, hard_iou, parse_index_spec,
    soft_iou, stratified_episode_split, write_json, write_jsonl,
)
from src.pi05_occupancy_full import (  # noqa: E402
    Pi05EulerCapture, list_activation_conditions, load_activation_manifest,
    load_condition_activations, probe_dir, read_jsonl_samples, select_bin_columns,
    write_activation_shards,
)
from src.utils import ensure_dir, log, set_seed  # noqa: E402


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("all", "gt", "activations", "train"), default="all")
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--pi05-path", default="/home/eai/mars/hub/models/pi05_libero")
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/self_occupancy/pi05_full")
    p.add_argument("--max-samples", type=int, default=10_000)
    p.add_argument("--max-tasks", type=int, default=10)
    p.add_argument("--max-demos", type=int, default=500)
    p.add_argument("--max-frames", type=int, default=20, help="Frames per demo.")
    p.add_argument("--layers", default="all", help="all, comma list, or inclusive range such as 0-17.")
    p.add_argument(
        "--bins",
        type=int,
        default=96,
        help="Fixed equal-width token partitioning used when capturing activations.",
    )
    p.add_argument(
        "--bin-indices",
        default="all",
        help=(
            "Which bins from the fixed --bins partitioning to store and train. "
            "Examples: all, 0,8,16,24 or 0-95:8. Each selected bin still uses the "
            "full demo-grouped train/test sample split."
        ),
    )
    p.add_argument("--shard-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=2e-3)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--grid-size", type=int, default=16)
    p.add_argument("--supersample", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--curve-sample",
        type=int,
        default=10,
        help="How many probes to include in the aggregate train/test curve figure.",
    )
    return p.parse_args()


def parse_layers(value: str) -> set[int] | None:
    if value == "all":
        return None
    return set(parse_index_spec(value, max_value=10**9))


def collect(args: argparse.Namespace, out: Path) -> None:
    files = sorted(args.dataset_dir.glob("*_demo.hdf5"))[: args.max_tasks]
    if not files:
        raise FileNotFoundError(f"No *_demo.hdf5 files under {args.dataset_dir}")
    demos_per_task = max(1, int(np.ceil(args.max_demos / len(files))))
    log(
        f"Stage GT: tasks={len(files)} demos_per_task<={demos_per_task} "
        f"frames_per_demo={args.max_frames} max_samples={args.max_samples} "
        f"max_demos={args.max_demos} grid={args.grid_size}^3 supersample={args.supersample}"
    )
    all_rows, occupancies, tasks = [], [], []
    next_id = 0
    stage_started = time.perf_counter()
    task_bar = tqdm(files, desc="GT tasks", unit="task")
    for hdf5 in task_bar:
        task_started = time.perf_counter()
        task_out = ensure_dir(out / "gt_tasks" / hdf5.stem)
        samples, metadata = collect_self_occupancy_samples(
            hdf5_path=hdf5, output_dir=task_out, num_demos=demos_per_task,
            frames_per_demo=args.max_frames,
            spec=OccupancyGridSpec(size=args.grid_size, supersample=args.supersample),
        )
        kept = min(len(samples), args.max_samples - next_id)
        for sample in samples[:kept]:
            sample.sample_id = next_id
            sample.task = hdf5.stem.removesuffix("_demo")
            all_rows.append(sample.metadata())
            occupancies.append(sample.occupancy)
            next_id += 1
        tasks.append(metadata)
        episode_count = len({r["episode_id"] for r in all_rows})
        task_bar.set_postfix(
            samples=next_id,
            demos=episode_count,
            task_s=f"{time.perf_counter() - task_started:.0f}",
            mean_occ=f"{float(np.mean([o.mean() for o in occupancies[-kept:]])):.4f}" if kept else "n/a",
        )
        log(
            f"GT task done {hdf5.stem}: kept={kept}/{len(samples)} "
            f"total_samples={next_id} total_demos={episode_count} "
            f"task_s={time.perf_counter() - task_started:.1f}"
        )
        if next_id >= args.max_samples or episode_count >= args.max_demos:
            log("GT early stop: hit max_samples or max_demos")
            break
    np.save(out / "occupancy.npy", np.stack(occupancies).astype(np.float16))
    write_jsonl(out / "samples.jsonl", all_rows)
    write_json(out / "gt_metadata.json", {"tasks": tasks, "samples": len(all_rows)})
    log(
        f"Stage GT finished samples={len(all_rows)} "
        f"occupancy.npy shape={(len(occupancies),) + occupancies[0].shape} "
        f"seconds={time.perf_counter() - stage_started:.1f}"
    )


def _occupancy_loss(logits, yb, pos_weight):
    import torch
    from torch import nn

    prob = logits.sigmoid()
    dice = 1 - ((2 * (prob * yb).sum(1) + 1e-6) / (prob.sum(1) + yb.sum(1) + 1e-6)).mean()
    bce = nn.functional.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
    return bce + dice, float((bce + dice).detach().cpu())


def _write_history_csv(path: Path, history: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def _plot_probe_curve(history: list[dict], path: Path, title: str) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train_loss")
    axes[0].plot(epochs, [row["test_loss"] for row in history], label="test_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE + Dice")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(epochs, [row["test_soft_iou"] for row in history], label="test_soft_iou")
    axes[1].plot(epochs, [row["test_hard_iou"] for row in history], label="test_hard_iou")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_sample_curves(all_history_rows: list[dict], path: Path, sample_n: int, seed: int) -> None:
    probes = sorted({(row["condition"], int(row["bin"])) for row in all_history_rows})
    rng = np.random.default_rng(seed)
    if len(probes) > sample_n:
        chosen = [probes[i] for i in sorted(rng.choice(len(probes), size=sample_n, replace=False).tolist())]
    else:
        chosen = probes
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    cmap = plt.get_cmap("tab10")
    for index, (condition, bin_index) in enumerate(chosen):
        subset = [
            row for row in all_history_rows
            if row["condition"] == condition and int(row["bin"]) == bin_index
        ]
        subset = sorted(subset, key=lambda row: int(row["epoch"]))
        color = cmap(index % 10)
        label = f"{condition}#bin={bin_index}"
        axes[0].plot([r["epoch"] for r in subset], [r["train_loss"] for r in subset], color=color, lw=1.4, label=label)
        axes[1].plot([r["epoch"] for r in subset], [r["test_soft_iou"] for r in subset], color=color, lw=1.4, label=label)
    axes[0].set_title("Train loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE + Dice")
    axes[0].grid(alpha=0.25)
    axes[1].set_title("Test soft IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("soft IoU")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=6, frameon=False, loc="best")
    fig.suptitle(f"Train/test curves for {len(chosen)} sampled probes")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def train(args: argparse.Namespace, out: Path) -> list[dict]:
    import torch
    from torch import nn

    log("Stage train: loading occupancy targets and activation shards")
    occupancy = np.load(out / "occupancy.npy", mmap_mode="r").astype(np.float32)
    samples = read_jsonl_samples(out / "samples.jsonl", occupancy)
    train_idx, test_idx = stratified_episode_split(samples, args.test_fraction, args.seed)
    selected_bins = parse_index_spec(args.bin_indices, max_value=args.bins)
    act_manifest = load_activation_manifest(out)
    split = {
        "strategy": "per-task grouped demo",
        "seed": args.seed,
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "train_episodes": sorted({samples[i].episode_id for i in train_idx}),
        "test_episodes": sorted({samples[i].episode_id for i in test_idx}),
        "requested_bins": int(args.bins),
        "selected_bin_indices": selected_bins,
        "activation_bin_indices": act_manifest.get("bin_indices"),
        "note": (
            "Activation shards store only --bin-indices columns under the fixed --bins "
            "partitioning; training logs per-epoch losses/IoU and saves decoders under probes/."
        ),
    }
    write_json(out / "split.json", split)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    conditions = list_activation_conditions(out)
    log(
        f"Train setup samples={len(samples)} train={len(train_idx)} test={len(test_idx)} "
        f"train_eps={len(split['train_episodes'])} test_eps={len(split['test_episodes'])} "
        f"conditions={len(conditions)} selected_bins={len(selected_bins)} "
        f"epochs={args.epochs} batch={args.batch_size} device={device}"
    )
    rows: list[dict] = []
    all_history_rows: list[dict] = []
    targets = occupancy.reshape(len(occupancy), -1)
    positive_fraction = float(targets[train_idx].mean())
    pos_weight_value = min(30.0, max(1.0, (1.0 - positive_fraction) / max(positive_fraction, 1e-6)))
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    log(f"Loss weighting positive_fraction={positive_fraction:.5f} bce_pos_weight={pos_weight_value:.2f}")
    best_soft = -1.0
    best_key = ""
    train_started = time.perf_counter()
    figures_dir = ensure_dir(out / "figures")
    condition_bar = tqdm(list(enumerate(conditions)), desc="train conditions", unit="cond")
    for condition_index, condition in condition_bar:
        cond_started = time.perf_counter()
        x_all = load_condition_activations(out, condition)
        bin_columns = select_bin_columns(
            condition=condition,
            x_all=x_all,
            requested_bin_indices=selected_bins,
            manifest=act_manifest,
        )
        condition_bar.set_postfix(
            cond=condition[-40:],
            stored_cols=x_all.shape[1],
            train_bins=len(bin_columns),
            shape=str(tuple(x_all.shape)),
        )
        bin_bar = tqdm(bin_columns, desc=f"bins [{condition[:36]}]", unit="bin", leave=False)
        for bin_index, column in bin_bar:
            probe_started = time.perf_counter()
            torch.manual_seed(args.seed + condition_index * 1000 + bin_index)
            x_raw = x_all[:, column]
            mean, std = x_raw[train_idx].mean(0), x_raw[train_idx].std(0)
            std = std.copy()
            std[std < 1e-5] = 1
            x = (x_raw - mean) / std
            decoder = nn.Sequential(nn.Linear(x.shape[1], 64), nn.GELU(), nn.Linear(64, 4096)).to(device)
            opt = torch.optim.AdamW(decoder.parameters(), lr=args.learning_rate)
            history: list[dict] = []
            last_train_loss = float("nan")
            last_test_loss = float("nan")
            soft = float("nan")
            hard = float("nan")
            for epoch in range(args.epochs):
                decoder.train()
                epoch_losses: list[float] = []
                for start in range(0, len(train_idx), args.batch_size):
                    idx = train_idx[start:start + args.batch_size]
                    xb = torch.from_numpy(x[idx]).to(device)
                    yb = torch.from_numpy(targets[idx]).to(device)
                    logits = decoder(xb)
                    loss, loss_value = _occupancy_loss(logits, yb, pos_weight)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    epoch_losses.append(loss_value)
                last_train_loss = float(np.mean(epoch_losses))
                decoder.eval()
                with torch.inference_mode():
                    test_logits = decoder(torch.from_numpy(x[test_idx]).to(device))
                    test_yb = torch.from_numpy(targets[test_idx]).to(device)
                    _, last_test_loss = _occupancy_loss(test_logits, test_yb, pos_weight)
                    pred = test_logits.sigmoid().cpu().numpy().reshape(-1, 16, 16, 16)
                soft = soft_iou(occupancy[test_idx], pred)
                hard = hard_iou(occupancy[test_idx], pred)
                history.append({
                    "epoch": epoch + 1,
                    "train_loss": last_train_loss,
                    "test_loss": last_test_loss,
                    "test_soft_iou": soft,
                    "test_hard_iou": hard,
                })
                if epoch == 0 or epoch == args.epochs - 1 or (epoch + 1) % max(1, args.epochs // 4) == 0:
                    bin_bar.set_postfix(
                        epoch=f"{epoch + 1}/{args.epochs}",
                        train=f"{last_train_loss:.3f}",
                        test=f"{last_test_loss:.3f}",
                        soft=f"{soft:.3f}",
                        best=f"{best_soft:.3f}" if best_soft >= 0 else "n/a",
                    )
            probe_path = ensure_dir(probe_dir(out, condition, bin_index))
            _write_history_csv(probe_path / "history.csv", history)
            _plot_probe_curve(history, probe_path / "curves.png", f"{condition} bin={bin_index}")
            torch.save(
                {
                    "state_dict": decoder.state_dict(),
                    "mean": mean.astype(np.float32),
                    "std": std.astype(np.float32),
                    "condition": condition,
                    "bin": int(bin_index),
                    "column": int(column),
                    "input_dim": int(x.shape[1]),
                    "output_dim": 4096,
                    "epochs": int(args.epochs),
                    "learning_rate": float(args.learning_rate),
                    "batch_size": int(args.batch_size),
                    "seed": int(args.seed + condition_index * 1000 + bin_index),
                    "final_train_loss": last_train_loss,
                    "final_test_loss": last_test_loss,
                    "final_soft_iou": soft,
                    "final_hard_iou": hard,
                },
                probe_path / "decoder.pt",
            )
            for item in history:
                all_history_rows.append({
                    "condition": condition,
                    "bin": bin_index,
                    "column": column,
                    **item,
                })
            rows.append({
                "condition": condition,
                "bin": bin_index,
                "column": column,
                "requested_bins": int(args.bins),
                "stored_columns": int(x_all.shape[1]),
                "selected_bin_indices": ",".join(map(str, selected_bins)),
                "soft_iou": soft,
                "hard_iou": hard,
                "train_loss": last_train_loss,
                "test_loss": last_test_loss,
                "positive_fraction_train": positive_fraction,
                "bce_pos_weight": pos_weight_value,
                "num_train": int(len(train_idx)),
                "num_test": int(len(test_idx)),
                "train_seconds": time.perf_counter() - probe_started,
                "probe_dir": str(probe_path.relative_to(out)),
            })
            if soft > best_soft:
                best_soft = soft
                best_key = f"{condition}#bin={bin_index}"
            bin_bar.set_postfix(
                soft=f"{soft:.3f}", hard=f"{hard:.3f}",
                train=f"{last_train_loss:.3f}", best=f"{best_soft:.3f}",
            )
            del decoder, opt
        log(
            f"Condition done {condition}: bins={len(bin_columns)} "
            f"seconds={time.perf_counter() - cond_started:.1f} "
            f"running_best={best_key} soft_iou={best_soft:.4f} probes_done={len(rows)}"
        )
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_history_csv(out / "epoch_curves.csv", all_history_rows)
    _plot_sample_curves(all_history_rows, figures_dir / "03_train_test_curves.png", args.curve_sample, args.seed)
    top = sorted(rows, key=lambda r: float(r["soft_iou"]), reverse=True)[:5]
    log(
        f"Stage train finished probes={len(rows)} "
        f"seconds={time.perf_counter() - train_started:.1f} "
        f"best={best_key} soft_iou={best_soft:.4f}"
    )
    for rank, row in enumerate(top, start=1):
        log(
            f"  top{rank}: soft_iou={row['soft_iou']:.4f} hard_iou={row['hard_iou']:.4f} "
            f"{row['condition']} bin={row['bin']}"
        )
    return rows


def plot_heatmap(rows: list[dict], out: Path) -> None:
    log(f"Plotting heatmap for {len(rows)} probe results")
    conditions = sorted({r["condition"] for r in rows})
    bin_indices = sorted({int(r["bin"]) for r in rows})
    matrix = np.full((len(conditions), len(bin_indices)), np.nan)
    for r in rows:
        matrix[conditions.index(r["condition"]), bin_indices.index(int(r["bin"]))] = r["soft_iou"]
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(bin_indices) + 4), max(5, len(conditions) * .22)))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(conditions)), conditions, fontsize=6)
    ax.set_xticks(range(len(bin_indices)), [str(index) for index in bin_indices], fontsize=7, rotation=90)
    ax.set_xlabel("Selected token-position bin index")
    fig.colorbar(image, ax=ax, label="held-out soft IoU")
    fig.tight_layout()
    fig.savefig(out / "heatmaps.png", dpi=180)
    fig.savefig(out / "figures" / "00_heatmaps.png", dpi=180)
    plt.close(fig)
    log(f"Saved heatmap to {out / 'heatmaps.png'}")


def main() -> None:
    args = arguments()
    if args.grid_size != 16:
        raise ValueError("The full experiment fixes GT and decoder output to 16^3; use --grid-size 16.")
    if args.bins <= 0:
        raise ValueError(f"`--bins` must be positive, got {args.bins}.")
    selected_bins = parse_index_spec(args.bin_indices, max_value=args.bins)
    set_seed(args.seed)
    out = ensure_dir(args.output_dir.resolve())
    ensure_dir(out / "figures")
    started = time.perf_counter()
    log("=" * 72)
    log(
        f"PI0.5 self-occupancy full run stage={args.stage} out={out} "
        f"bins={args.bins} bin_indices={args.bin_indices} "
        f"({len(selected_bins)} selected) layers={args.layers}"
    )
    log(
        f"dataset={args.dataset_dir} model={args.pi05_path} "
        f"max_samples={args.max_samples} max_demos={args.max_demos} "
        f"max_frames={args.max_frames} shard_size={args.shard_size} seed={args.seed}"
    )
    timings_path = out / "timings.json"
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}
    if args.stage in ("all", "gt"):
        t = time.perf_counter()
        collect(args, out)
        timings["gt_seconds"] = time.perf_counter() - t
        log(f"Timing gt_seconds={timings['gt_seconds']:.1f}")
    occupancy = np.load(out / "occupancy.npy", mmap_mode="r")
    samples = read_jsonl_samples(out / "samples.jsonl", occupancy)
    log(f"Loaded samples.jsonl count={len(samples)} occupancy.shape={occupancy.shape}")
    if args.stage in ("all", "activations"):
        t = time.perf_counter()
        capture = Pi05EulerCapture(
            args.pi05_path,
            args.device,
            args.bins,
            bin_indices=selected_bins,
        )
        timings["activations"] = write_activation_shards(
            samples, out, capture, args.shard_size, parse_layers(args.layers)
        )
        timings["activation_stage_seconds"] = time.perf_counter() - t
        log(f"Timing activation_stage_seconds={timings['activation_stage_seconds']:.1f}")
    rows: list[dict] = []
    if args.stage in ("all", "train"):
        t = time.perf_counter()
        rows = train(args, out)
        plot_heatmap(rows, out)
        timings["train_plot_seconds"] = time.perf_counter() - t
        log(f"Timing train_plot_seconds={timings['train_plot_seconds']:.1f}")
    timings["total_seconds"] = time.perf_counter() - started
    write_json(timings_path, timings)
    manifest = {
        "status": "completed",
        "samples": len(samples),
        "requested_bins": args.bins,
        "selected_bin_indices": selected_bins,
        "flow_times": [1.0, 0.5, 0.1],
        "prefix_condition": "static_once",
        "decoder": "independent Linear(D,64)-GELU-Linear(64,4096) per condition and selected bin",
        "note": (
            "Activation shards store only --bin-indices; probes/ holds history.csv, "
            "curves.png, and decoder.pt for each trained probe."
        ),
        "timings": timings,
        "outputs": [
            "samples.jsonl", "occupancy.npy", "activation_shards/", "split.json",
            "metrics.csv", "epoch_curves.csv", "probes/", "heatmaps.png",
            "figures/", "timings.json", "manifest.json", "report.md",
        ],
    }
    write_json(out / "manifest.json", manifest)
    (out / "report.md").write_text(
        "# Full PI0.5 self-occupancy run\n\n```json\n" + json.dumps(manifest, indent=2) + "\n```\n"
    )
    log(f"Run completed total_seconds={timings['total_seconds']:.1f}")
    log("=" * 72)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
