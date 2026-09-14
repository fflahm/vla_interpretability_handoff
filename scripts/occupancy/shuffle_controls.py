#!/usr/bin/env python
"""Matched random-target controls for PI0.5 self-occupancy probes.

Experimental control
--------------------
Reuse one finished run's activations and ``split.json``. For the same
``(condition, bin)`` probes, retrain decoders under three target pairings with
**identical** hyperparameters / seeds / architecture / split:

1. ``real``            — original activation↔occupancy pairing
2. ``global_timestep`` — shuffle targets inside train and inside test
3. ``within_demo``     — shuffle targets only within each episode, per split

Only the occupancy targets change. Soft/hard IoU are always evaluated under the
same pairing used for training of that arm.

By default only 1/4 of the original selected bins are retrained to save time.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.libero_self_occupancy import hard_iou, parse_index_spec, soft_iou  # noqa: E402
from src.pi05_occupancy_full import (  # noqa: E402
    list_activation_conditions,
    load_activation_manifest,
    load_condition_activations,
    read_jsonl_samples,
    select_bin_columns,
)
from src.utils import ensure_dir, log, set_seed  # noqa: E402

CONDITION_RE = re.compile(
    r"^(?P<tower>paligemma|expert)/layer_(?P<layer>\d+)(?:/(?:static|t=(?P<flow>[0-9.]+)))?$"
)

# Defaults copied from scripts/occupancy/run_full.py
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 2e-3
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "outputs/self_occupancy/pi05_libero_spatial_10k",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <run-dir>/shuffle_controls.",
    )
    p.add_argument(
        "--bin-indices",
        default="auto-1/4",
        help="Bins to retrain. `auto-1/4` keeps ~1/4 of bins present in metrics.csv.",
    )
    p.add_argument(
        "--arms",
        default="real,global_timestep,within_demo",
        help="Comma list of experimental arms.",
    )
    # Matched to script 23 unless the user overrides.
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--shuffle-seed", type=int, default=123)
    p.add_argument("--device", default="auto")
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def parse_condition(condition: str) -> dict[str, Any]:
    match = CONDITION_RE.match(condition)
    if not match:
        raise ValueError(f"Unrecognized condition: {condition}")
    flow = match.group("flow")
    return {
        "condition": condition,
        "tower": match.group("tower"),
        "layer": int(match.group("layer")),
        "flow": None if flow is None else float(flow),
    }


def resolve_device(device: str):
    import torch

    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_metrics_bins(path: Path) -> list[int]:
    bins = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            bins.add(int(raw["bin"]))
    return sorted(bins)


def choose_control_bins(spec: str, real_bins: list[int], requested_bins: int) -> list[int]:
    if not real_bins:
        raise ValueError("metrics.csv has no bins.")
    if spec == "auto-1/4":
        n_keep = max(1, int(round(len(real_bins) / 4.0)))
        positions = np.linspace(0, len(real_bins) - 1, n_keep)
        return sorted({real_bins[int(round(pos))] for pos in positions})
    return parse_index_spec(spec, max_value=requested_bins)


def global_timestep_shuffle(
    occupancy: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> np.ndarray:
    shuffled = np.array(occupancy, copy=True)
    rng = np.random.default_rng(seed)
    for indices in (train_idx, test_idx):
        order = rng.permutation(len(indices))
        shuffled[indices] = occupancy[indices][order]
    return shuffled


def within_demo_shuffle(
    occupancy: np.ndarray,
    episode_ids: Sequence[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
) -> np.ndarray:
    shuffled = np.array(occupancy, copy=True)
    rng = np.random.default_rng(seed)
    for indices in (train_idx, test_idx):
        by_episode: dict[str, list[int]] = defaultdict(list)
        for sample_index in indices.tolist():
            by_episode[str(episode_ids[sample_index])].append(int(sample_index))
        for members in by_episode.values():
            if len(members) < 2:
                continue
            arr = np.asarray(members, dtype=int)
            order = rng.permutation(len(arr))
            shuffled[arr] = occupancy[arr][order]
    return shuffled


def occupancy_loss(logits, yb, pos_weight):
    """Same loss as scripts/occupancy/run_full.py."""
    from torch import nn

    prob = logits.sigmoid()
    dice = 1 - ((2 * (prob * yb).sum(1) + 1e-6) / (prob.sum(1) + yb.sum(1) + 1e-6)).mean()
    bce = nn.functional.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
    return bce + dice, float((bce + dice).detach().cpu())


def train_one_probe(
    *,
    x_raw: np.ndarray,
    occupancy: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device,
    pos_weight,
) -> dict[str, float]:
    """Mirror the per-probe loop in scripts/occupancy/run_full.py::train."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    targets = occupancy.reshape(len(occupancy), -1).astype(np.float32)
    mean = x_raw[train_idx].mean(0)
    std = x_raw[train_idx].std(0).copy()
    std[std < 1e-5] = 1.0
    x = (x_raw - mean) / std
    decoder = nn.Sequential(
        nn.Linear(x.shape[1], 64),
        nn.GELU(),
        nn.Linear(64, 4096),
    ).to(device)
    opt = torch.optim.AdamW(decoder.parameters(), lr=learning_rate)
    last_train_loss = float("nan")
    last_test_loss = float("nan")
    soft = float("nan")
    hard = float("nan")
    for _epoch in range(epochs):
        decoder.train()
        epoch_losses: list[float] = []
        for start in range(0, len(train_idx), batch_size):
            idx = train_idx[start : start + batch_size]
            xb = torch.from_numpy(x[idx]).to(device)
            yb = torch.from_numpy(targets[idx]).to(device)
            logits = decoder(xb)
            loss, loss_value = occupancy_loss(logits, yb, pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_losses.append(loss_value)
        last_train_loss = float(np.mean(epoch_losses))
        decoder.eval()
        with torch.inference_mode():
            test_logits = decoder(torch.from_numpy(x[test_idx]).to(device))
            test_yb = torch.from_numpy(targets[test_idx]).to(device)
            _, last_test_loss = occupancy_loss(test_logits, test_yb, pos_weight)
            pred = test_logits.sigmoid().cpu().numpy().reshape(-1, 16, 16, 16)
        soft = soft_iou(occupancy[test_idx], pred)
        hard = hard_iou(occupancy[test_idx], pred)
    del decoder, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "train_loss": last_train_loss,
        "test_loss": last_test_loss,
        "soft_iou": soft,
        "hard_iou": hard,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_arm(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    soft = np.asarray([float(r["soft_iou"]) for r in rows], dtype=np.float64)
    hard = np.asarray([float(r["hard_iou"]) for r in rows], dtype=np.float64)
    return {
        "arm": arm,
        "n": len(rows),
        "soft_iou_mean": float(soft.mean()),
        "soft_iou_std": float(soft.std()),
        "hard_iou_mean": float(hard.mean()),
        "hard_iou_std": float(hard.std()),
    }


def paired_against_real(arm_rows: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    real_map = {(r["condition"], int(r["bin"])): r for r in arm_rows["real"]}
    pairs = []
    for arm, rows in arm_rows.items():
        if arm == "real":
            continue
        for row in rows:
            key = (row["condition"], int(row["bin"]))
            real = real_map[key]
            pairs.append(
                {
                    "arm": arm,
                    "condition": row["condition"],
                    "tower": row["tower"],
                    "layer": int(row["layer"]),
                    "bin": int(row["bin"]),
                    "real_soft_iou": float(real["soft_iou"]),
                    "arm_soft_iou": float(row["soft_iou"]),
                    "delta_soft_iou": float(real["soft_iou"]) - float(row["soft_iou"]),
                    "real_hard_iou": float(real["hard_iou"]),
                    "arm_hard_iou": float(row["hard_iou"]),
                    "delta_hard_iou": float(real["hard_iou"]) - float(row["hard_iou"]),
                }
            )
    return pairs


def plot_results(
    arm_rows: dict[str, list[dict[str, Any]]],
    pairs: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    matched: dict[str, Any],
    output_dir: Path,
    dpi: int,
) -> None:
    arms = list(arm_rows)
    colors = {
        "real": "#2563eb",
        "global_timestep": "#dc2626",
        "within_demo": "#ca8a04",
    }

    # Boxplot
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    data = [[float(r["soft_iou"]) for r in arm_rows[arm]] for arm in arms]
    try:
        ax.boxplot(data, tick_labels=arms, showmeans=True)
    except TypeError:
        ax.boxplot(data, labels=arms, showmeans=True)
    ax.set_ylabel("held-out soft IoU")
    ax.set_title("Matched training: real vs shuffled targets")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "01_soft_iou_boxplot.png", dpi=dpi)
    plt.close(fig)

    # Mean bars
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    means = [next(s for s in summaries if s["arm"] == arm)["soft_iou_mean"] for arm in arms]
    stds = [next(s for s in summaries if s["arm"] == arm)["soft_iou_std"] for arm in arms]
    xs = np.arange(len(arms))
    ax.bar(xs, means, yerr=stds, capsize=4, color=[colors.get(a, "#666") for a in arms])
    ax.set_xticks(xs, arms, rotation=15)
    ax.set_ylabel("mean soft IoU ± std")
    ax.set_title(
        f"epochs={matched['epochs']} lr={matched['learning_rate']} "
        f"batch={matched['batch_size']} seed={matched['seed']}"
    )
    ax.grid(alpha=0.25, axis="y")
    for x, mean in zip(xs, means):
        ax.text(x, mean + 0.008, f"{mean:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "02_soft_iou_mean_bars.png", dpi=dpi)
    plt.close(fig)

    # Delta hist
    shuffle_arms = [a for a in arms if a != "real"]
    fig, axes = plt.subplots(1, max(len(shuffle_arms), 1), figsize=(5.2 * max(len(shuffle_arms), 1), 4.2), squeeze=False)
    for ax, arm in zip(axes[0], shuffle_arms):
        deltas = [p["delta_soft_iou"] for p in pairs if p["arm"] == arm]
        ax.hist(deltas, bins=20, color="#0f766e", alpha=0.85)
        ax.axvline(0.0, color="black", lw=1.0)
        ax.axvline(float(np.mean(deltas)), color="#dc2626", lw=1.4, label=f"mean={np.mean(deltas):.3f}")
        ax.set_title(f"real − {arm}")
        ax.set_xlabel("Δ soft IoU")
        ax.set_ylabel("probe count")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle("Advantage of real pairing under matched training")
    fig.tight_layout()
    fig.savefig(output_dir / "03_delta_soft_iou_hist.png", dpi=dpi)
    plt.close(fig)

    # Layer-wise
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for ax, tower in zip(axes, ("paligemma", "expert")):
        for arm in arms:
            bucket: dict[int, list[float]] = defaultdict(list)
            for row in arm_rows[arm]:
                if row["tower"] != tower:
                    continue
                bucket[int(row["layer"])].append(float(row["soft_iou"]))
            layers = sorted(bucket)
            if not layers:
                continue
            ax.plot(
                layers,
                [float(np.mean(bucket[layer])) for layer in layers],
                marker="o",
                label=arm,
                color=colors.get(arm),
            )
        ax.set_title(tower)
        ax.set_xlabel("layer")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("mean soft IoU")
    fig.suptitle("Layer-wise soft IoU under matched real/shuffle training")
    fig.tight_layout()
    fig.savefig(output_dir / "04_layerwise_soft_iou.png", dpi=dpi)
    plt.close(fig)

    # Scatter
    fig, axes = plt.subplots(1, max(len(shuffle_arms), 1), figsize=(5.0 * max(len(shuffle_arms), 1), 4.4), squeeze=False)
    for ax, arm in zip(axes[0], shuffle_arms):
        subset = [p for p in pairs if p["arm"] == arm]
        xs = [p["real_soft_iou"] for p in subset]
        ys = [p["arm_soft_iou"] for p in subset]
        ax.scatter(xs, ys, s=18, alpha=0.75, color=colors.get(arm, "#666"))
        lim = max(max(xs + ys), 0.1)
        ax.plot([0, lim], [0, lim], linestyle="--", color="black", lw=1)
        ax.set_xlabel("real soft IoU")
        ax.set_ylabel(f"{arm} soft IoU")
        ax.set_title(arm)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
    fig.suptitle("Paired probes (below diagonal => real pairing wins)")
    fig.tight_layout()
    fig.savefig(output_dir / "05_paired_scatter.png", dpi=dpi)
    plt.close(fig)


def main() -> None:
    import torch

    args = parse_args()
    set_seed(args.seed)
    run_dir = args.run_dir.resolve()
    out = ensure_dir((args.output_dir or (run_dir / "shuffle_controls")).resolve())
    figures = ensure_dir(out / "figures")

    split = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
    train_idx = np.asarray(split["train_indices"], dtype=int)
    test_idx = np.asarray(split["test_indices"], dtype=int)
    requested_bins = int(split.get("requested_bins") or 96)
    real_bins = load_metrics_bins(run_dir / "metrics.csv")
    selected_bins = choose_control_bins(args.bin_indices, real_bins, requested_bins)
    arms = [part.strip() for part in args.arms.split(",") if part.strip()]
    if "real" not in arms:
        raise ValueError("Arms must include `real` as the matched baseline.")

    occupancy = np.load(run_dir / "occupancy.npy").astype(np.float32)
    samples = read_jsonl_samples(run_dir / "samples.jsonl", occupancy)
    episode_ids = [sample.episode_id for sample in samples]
    act_manifest = load_activation_manifest(run_dir)
    conditions = list_activation_conditions(run_dir)
    device = resolve_device(args.device)

    # pos_weight from the real train occupancy — identical across arms because
    # shuffles are permutations of the same train labels.
    targets = occupancy.reshape(len(occupancy), -1)
    positive_fraction = float(targets[train_idx].mean())
    pos_weight_value = min(30.0, max(1.0, (1.0 - positive_fraction) / max(positive_fraction, 1e-6)))
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)

    matched = {
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "seed": int(args.seed),
        "shuffle_seed": int(args.shuffle_seed),
        "decoder": "Linear(D,64)-GELU-Linear(64,4096)",
        "optimizer": "AdamW",
        "loss": "pos-weighted BCE + soft Dice",
        "split": "reused split.json (per-task grouped demo)",
        "normalization": "train-set mean/std of activations",
        "pos_weight": pos_weight_value,
        "positive_fraction_train": positive_fraction,
        "device": str(device),
        "torch": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "selected_bins": selected_bins,
        "n_conditions": len(conditions),
        "probe_seed_formula": "seed + condition_index * 1000 + bin_index (identical across arms)",
        "note": "Only occupancy-target pairing differs across arms.",
    }
    log("=" * 72)
    log(f"Matched shuffle controls out={out}")
    log(json.dumps(matched, indent=2))

    target_by_arm = {
        "real": occupancy,
        "global_timestep": global_timestep_shuffle(
            occupancy, train_idx, test_idx, args.shuffle_seed
        ),
        "within_demo": within_demo_shuffle(
            occupancy, episode_ids, train_idx, test_idx, args.shuffle_seed + 1
        ),
    }
    for arm in arms:
        if arm not in target_by_arm:
            raise ValueError(f"Unknown arm {arm!r}")
        arr = target_by_arm[arm]
        changed = float(
            np.mean(
                np.any(
                    arr.reshape(len(arr), -1) != occupancy.reshape(len(occupancy), -1),
                    axis=1,
                )
            )
        )
        log(f"Arm {arm}: fraction samples with changed target={changed:.3f}")

    np.savez_compressed(
        out / "shuffled_targets.npz",
        real=occupancy.astype(np.float16),
        global_timestep=target_by_arm["global_timestep"].astype(np.float16),
        within_demo=target_by_arm["within_demo"].astype(np.float16),
    )
    write_json = lambda path, obj: path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    write_json(out / "matched_settings.json", matched)

    arm_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms}
    condition_bar = tqdm(list(enumerate(conditions)), desc="conditions", unit="cond")
    for condition_index, condition in condition_bar:
        x_all = load_condition_activations(run_dir, condition)
        try:
            bin_columns = select_bin_columns(
                condition=condition,
                x_all=x_all,
                requested_bin_indices=selected_bins,
                manifest=act_manifest,
            )
        except ValueError as exc:
            log(f"Skip {condition}: {exc}")
            continue
        parsed = parse_condition(condition)
        for bin_index, column in bin_columns:
            probe_seed = args.seed + condition_index * 1000 + bin_index
            x_raw = x_all[:, column]
            for arm in arms:
                started = time.perf_counter()
                metrics = train_one_probe(
                    x_raw=x_raw,
                    occupancy=target_by_arm[arm],
                    train_idx=train_idx,
                    test_idx=test_idx,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    seed=probe_seed,  # identical init across arms
                    device=device,
                    pos_weight=pos_weight,
                )
                arm_rows[arm].append(
                    {
                        "arm": arm,
                        "condition": condition,
                        "tower": parsed["tower"],
                        "layer": parsed["layer"],
                        "flow": parsed["flow"],
                        "bin": int(bin_index),
                        "column": int(column),
                        "probe_seed": int(probe_seed),
                        **metrics,
                        "train_seconds": time.perf_counter() - started,
                    }
                )
            latest_real = arm_rows["real"][-1]["soft_iou"]
            condition_bar.set_postfix(
                bin=bin_index,
                real=f"{latest_real:.3f}",
                done=len(arm_rows["real"]),
            )

    for arm, rows in arm_rows.items():
        write_csv(out / f"metrics_{arm}.csv", rows)

    summaries = [summarize_arm(arm_rows[arm], arm) for arm in arms]
    pairs = paired_against_real(arm_rows)
    write_csv(out / "paired_vs_real.csv", pairs)
    pair_summary = []
    for arm in arms:
        if arm == "real":
            continue
        subset = [p for p in pairs if p["arm"] == arm]
        delta = np.asarray([p["delta_soft_iou"] for p in subset], dtype=np.float64)
        pair_summary.append(
            {
                "arm": arm,
                "n": len(subset),
                "delta_soft_iou_mean": float(delta.mean()),
                "delta_soft_iou_median": float(np.median(delta)),
                "frac_real_higher": float(np.mean(delta > 0)),
                "frac_delta_gt_0_05": float(np.mean(delta > 0.05)),
                "frac_delta_gt_0_10": float(np.mean(delta > 0.10)),
            }
        )
    write_json(
        out / "summary.json",
        {
            "matched_settings": matched,
            "arm_summaries": summaries,
            "pair_summaries": pair_summary,
        },
    )
    plot_results(arm_rows, pairs, summaries, matched, figures, args.dpi)
    log(json.dumps({"arm_summaries": summaries, "pair_summaries": pair_summary}, indent=2))
    log("=" * 72)


if __name__ == "__main__":
    main()
