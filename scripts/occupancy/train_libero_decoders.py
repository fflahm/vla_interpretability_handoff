#!/usr/bin/env python
"""Train occupancy MLPs on the Libero-90 extract split (train/test only).

Writes run artifacts to GPFS. Optionally copies them to TOS at the end with
inplace ``wb`` (no rename). Layer activations are cached on GPFS as float16
so each PaliGemma/expert layer is read from TOS once.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.libero_self_occupancy import parse_index_spec  # noqa: E402
from src.occupancy_decoder import (  # noqa: E402
    PALI_LAYERS,
    bce_pos_weight,
    cache_array_path,
    iter_split_rows,
    list_extract_conditions,
    load_expert_layer,
    load_extract_split,
    load_occupancy_targets,
    load_or_build_layer_cache,
    load_paligemma_layer,
    metrics_from_history_csv,
    probe_is_complete,
    save_torch_inplace,
    split_indices,
    sync_tree_inplace,
    train_one_probe,
    write_train_status,
)
from src.pi05_occupancy_full import probe_dir  # noqa: E402
from src.utils import ensure_dir, log, set_seed  # noqa: E402

DEFAULT_ACT = ROOT / "data" / "occupancy_activations"
DEFAULT_OCC = ROOT / "data" / "occupancy"
DEFAULT_OUT = Path("/mnt/shared-storage-user/guoshengyu/vla_rjob_runs/occupancy_decoders")
DEFAULT_SMOKE_OUT = Path("/mnt/shared-storage-user/guoshengyu/vla_rjob_runs/occupancy_decoder_smoke")
DEFAULT_TOS = Path("/data/tos/guoshengyu/vla/occupancy_decoders")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-root", type=Path, default=DEFAULT_ACT)
    parser.add_argument("--occupancy-root", type=Path, default=DEFAULT_OCC)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tos-output-dir", type=Path, default=None)
    parser.add_argument("--sync-tos", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--towers", default="paligemma", help="paligemma, expert, or all")
    parser.add_argument("--layers", default="0", help="all, comma list, or range such as 0-2")
    parser.add_argument("--bin-indices", default="0")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-train-demos", type=int, default=None)
    parser.add_argument("--max-test-demos", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--include-tasks", default="")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--loss", default="bce_dice", choices=("bce_dice", "bce"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--bottleneck", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--plot-curves", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--drop-layer-cache", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke:
        if args.sync_tos is None:
            args.sync_tos = True
        if args.plot_curves is None:
            args.plot_curves = False
        return
    args.max_tasks = 1 if args.max_tasks is None else args.max_tasks
    args.max_train_demos = 2 if args.max_train_demos is None else args.max_train_demos
    args.max_test_demos = 1 if args.max_test_demos is None else args.max_test_demos
    args.max_frames = 4 if args.max_frames is None else args.max_frames
    if args.epochs == 20:
        args.epochs = 2
    if args.batch_size == 128:
        args.batch_size = 8
    if args.output_dir is None:
        args.output_dir = DEFAULT_SMOKE_OUT
    if args.sync_tos is None:
        args.sync_tos = False
    if args.plot_curves is None:
        args.plot_curves = True


def resolve_device(value: str) -> str:
    import torch

    if value != "auto":
        return value
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_towers(raw: str) -> list[str]:
    text = str(raw or "paligemma").strip().lower()
    if text in {"all", "both"}:
        return ["paligemma", "expert"]
    if text in {"paligemma", "pali", "expert"}:
        return ["paligemma" if text != "expert" else "expert"]
    raise ValueError(f"unknown --towers {raw!r}")


def bins_tag(selected: list[int], n_bins: int) -> str:
    if selected == list(range(n_bins)):
        return "all"
    return "b" + "-".join(str(v) for v in selected)


def write_history_csv(path: Path, history: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_probe_curve(history: list[dict], path: Path, title: str, loss_name: str) -> None:
    epochs = [row["epoch"] for row in history]
    ylabel = "BCE + Dice" if loss_name == "bce_dice" else "BCE"
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train_loss")
    axes[0].plot(epochs, [row["test_loss"] for row in history], label="test_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(ylabel)
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


def plot_heatmap(rows: list[dict], out: Path) -> None:
    conditions = sorted({r["condition"] for r in rows})
    bin_indices = sorted({int(r["bin"]) for r in rows})
    matrix = np.full((len(conditions), len(bin_indices)), np.nan)
    for row in rows:
        matrix[conditions.index(row["condition"]), bin_indices.index(int(row["bin"]))] = row["soft_iou"]
    fig, ax = plt.subplots(
        figsize=(max(8, 0.45 * len(bin_indices) + 4), max(3.5, len(conditions) * 0.28))
    )
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(conditions)), conditions, fontsize=6)
    ax.set_xticks(range(len(bin_indices)), [str(index) for index in bin_indices], fontsize=7, rotation=90)
    ax.set_xlabel("Token-bin index")
    fig.colorbar(image, ax=ax, label="held-out soft IoU")
    fig.tight_layout()
    fig.savefig(out / "heatmaps.png", dpi=160)
    fig.savefig(out / "figures" / "00_heatmaps.png", dpi=160)
    plt.close(fig)


def flush_metrics(out: Path, metrics_rows: list[dict], all_history: list[dict]) -> None:
    if not metrics_rows:
        return
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)
    if all_history:
        write_history_csv(out / "epoch_curves.csv", all_history)


def history_from_csv(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            converted = dict(row)
            converted["epoch"] = int(row["epoch"])
            for key in ("train_loss", "test_loss", "test_soft_iou", "test_hard_iou"):
                converted[key] = float(row[key])
            rows.append(converted)
    return rows


def main() -> int:
    args = parse_args()
    apply_smoke_defaults(args)
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUT
    if args.tos_output_dir is None:
        args.tos_output_dir = DEFAULT_TOS / Path(args.output_dir).name
    set_seed(args.seed)
    out = ensure_dir(args.output_dir.resolve())
    cache_dir = ensure_dir(out / "layer_cache")
    ensure_dir(out / "figures")
    device = resolve_device(args.device)
    split = load_extract_split(args.activation_root)
    capture_cfg = {}
    cfg_path = args.activation_root / "capture_config.json"
    if cfg_path.exists():
        capture_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    frames_available = int(split.get("frames_per_demo") or 20)
    frames = frames_available if args.max_frames is None else min(int(args.max_frames), frames_available)
    include = [item.strip() for item in str(args.include_tasks).split(",") if item.strip()]
    rows = iter_split_rows(
        split,
        frames=frames,
        max_tasks=args.max_tasks,
        max_train_demos=args.max_train_demos,
        max_test_demos=args.max_test_demos,
        include_tasks=include,
    )
    train_idx, test_idx = split_indices(rows)
    paligemma_bins = int(capture_cfg.get("paligemma_bins") or 97)
    expert_bins = int(capture_cfg.get("expert_bins") or 5)
    flow_times = [float(t) for t in (capture_cfg.get("flow_times") or [1.0, 0.8, 0.6, 0.3, 0.1])]
    towers = parse_towers(args.towers)
    layer_indices = (
        list(range(PALI_LAYERS))
        if args.layers == "all"
        else parse_index_spec(args.layers, max_value=PALI_LAYERS)
    )
    conditions = list_extract_conditions(towers=towers, layer_indices=layer_indices, flow_times=flow_times)
    groups: OrderedDict[tuple[str, int], list[tuple[str, int | None]]] = OrderedDict()
    for condition, tower, layer, time_index in conditions:
        groups.setdefault((tower, layer), []).append((condition, time_index))

    total_probes = 0
    for tower, layer in groups:
        n_bins = paligemma_bins if tower == "paligemma" else expert_bins
        selected = parse_index_spec(args.bin_indices, max_value=n_bins) if args.bin_indices != "all" else list(range(n_bins))
        total_probes += len(groups[(tower, layer)]) * len(selected)

    log(
        f"Extract decoder train smoke={args.smoke} gpfs={out} tos={args.tos_output_dir} "
        f"sync_tos={args.sync_tos} device={device} frames={frames} "
        f"rows={len(rows)} train={len(train_idx)} test={len(test_idx)} "
        f"layers={len(groups)} probes={total_probes} loss={args.loss} plot_curves={args.plot_curves}"
    )
    occupancy = load_occupancy_targets(args.occupancy_root, rows)
    pos_weight_value = bce_pos_weight(occupancy.reshape(len(occupancy), -1)[train_idx])
    log(
        f"Loaded occupancy {tuple(occupancy.shape)} pos_frac={float(occupancy[train_idx].mean()):.5f} "
        f"bce_pos_weight={pos_weight_value:.2f}"
    )

    split_payload = {
        "source": str(args.activation_root / "split.json"),
        "seed": split.get("seed"),
        "strategy": "extract split.json train/test demos; ablation unused",
        "smoke": bool(args.smoke),
        "loss": args.loss,
        "frames": frames,
        "n_train_frames": int(len(train_idx)),
        "n_test_frames": int(len(test_idx)),
        "train_demos": sorted({f"{row.task}/{row.demo_key}" for row in rows if row.split == "train"}),
        "test_demos": sorted({f"{row.task}/{row.demo_key}" for row in rows if row.split == "test"}),
        "tasks": sorted({row.task for row in rows}),
        "gpfs_output": str(out),
        "tos_output": str(args.tos_output_dir),
        "note": "Train writes GPFS; --sync-tos copies artifacts to TOS without layer_cache.",
    }
    (out / "split.json").write_text(json.dumps(split_payload, indent=2) + "\n", encoding="utf-8")

    metrics_rows: list[dict] = []
    all_history: list[dict] = []
    best_soft = -1.0
    best_key = ""
    done_probes = 0
    started = time.perf_counter()
    write_train_status(
        out,
        {
            "phase": "train",
            "done_probes": 0,
            "total_probes": total_probes,
            "best_soft_iou": "",
            "condition": "",
            "bin": "",
            "last_error": "",
        },
    )

    condition_index = 0
    for (tower, layer), cond_list in groups.items():
        n_bins = paligemma_bins if tower == "paligemma" else expert_bins
        selected = parse_index_spec(args.bin_indices, max_value=n_bins) if args.bin_indices != "all" else list(range(n_bins))
        layer_probes = [(c, t, b, col) for (c, t) in cond_list for col, b in enumerate(selected)]
        remaining = [
            item
            for item in layer_probes
            if not probe_is_complete(probe_dir(out, item[0], item[2]))
        ]
        if not remaining:
            log(f"Skip TOS load {tower} layer={layer}: all {len(layer_probes)} probes complete")
            for condition, time_index, bin_index, column in layer_probes:
                probe_path = probe_dir(out, condition, bin_index)
                metrics = metrics_from_history_csv(
                    probe_path / "history.csv",
                    condition=condition,
                    bin_index=bin_index,
                    column=column,
                )
                metrics.update(
                    requested_bins=n_bins,
                    selected_bin_indices=",".join(map(str, selected)),
                    train_seconds=0.0,
                    probe_dir=str(probe_path.relative_to(out)),
                    num_train=int(len(train_idx)),
                    num_test=int(len(test_idx)),
                )
                metrics_rows.append(metrics)
                for item in history_from_csv(probe_path / "history.csv"):
                    all_history.append({"condition": condition, "bin": bin_index, "column": column, **item})
                done_probes += 1
                if float(metrics["soft_iou"]) > best_soft:
                    best_soft = float(metrics["soft_iou"])
                    best_key = f"{condition}#bin={bin_index}"
            condition_index += len(cond_list)
            continue

        tag = bins_tag(selected, n_bins)
        load_started = time.perf_counter()
        if tower == "paligemma":
            cache_path = cache_array_path(cache_dir, f"paligemma_L{layer:02d}_{tag}")
            x_layer = load_or_build_layer_cache(
                cache_path,
                lambda: load_paligemma_layer(
                    args.activation_root, rows, layer=layer, bin_indices=selected
                ),
                expected_shape=(len(rows), len(selected), 2048),
            )
        else:
            cache_path = cache_array_path(cache_dir, f"expert_L{layer:02d}_{tag}")
            x_layer = load_or_build_layer_cache(
                cache_path,
                lambda: load_expert_layer(
                    args.activation_root,
                    rows,
                    layer=layer,
                    bin_indices=selected,
                    n_times=len(flow_times),
                ),
                expected_shape=(len(rows), len(flow_times), len(selected), 1024),
            )
        log(
            f"Layer ready {tower} L{layer:02d} x={tuple(x_layer.shape)} "
            f"seconds={time.perf_counter() - load_started:.1f} cache={cache_path.name}"
        )

        for condition, time_index in cond_list:
            for column, bin_index in enumerate(selected):
                probe_path = ensure_dir(probe_dir(out, condition, bin_index))
                if probe_is_complete(probe_path):
                    metrics = metrics_from_history_csv(
                        probe_path / "history.csv",
                        condition=condition,
                        bin_index=bin_index,
                        column=column,
                    )
                    history = history_from_csv(probe_path / "history.csv")
                    log(f"resume skip {condition} bin={bin_index}")
                else:
                    probe_started = time.perf_counter()
                    if tower == "paligemma":
                        x = np.asarray(x_layer[:, column], dtype=np.float32)
                    else:
                        x = np.asarray(x_layer[:, int(time_index), column], dtype=np.float32)
                    payload, history, metrics = train_one_probe(
                        x=x,
                        occupancy=occupancy,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        condition=condition,
                        bin_index=bin_index,
                        column=column,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        bottleneck=args.bottleneck,
                        seed=args.seed + condition_index * 1000 + bin_index,
                        device=device,
                        pos_weight_value=pos_weight_value,
                        loss_kind=args.loss,
                    )
                    write_history_csv(probe_path / "history.csv", history)
                    if args.plot_curves:
                        plot_probe_curve(
                            history, probe_path / "curves.png", f"{condition} bin={bin_index}", args.loss
                        )
                    save_torch_inplace(probe_path / "decoder.pt", payload)
                    metrics["train_seconds"] = time.perf_counter() - probe_started
                    del x, payload
                metrics.update(
                    requested_bins=n_bins,
                    selected_bin_indices=",".join(map(str, selected)),
                    probe_dir=str(probe_path.relative_to(out)),
                    num_train=int(len(train_idx)),
                    num_test=int(len(test_idx)),
                )
                metrics.setdefault("train_seconds", 0.0)
                metrics_rows.append(metrics)
                for item in history:
                    all_history.append({"condition": condition, "bin": bin_index, "column": column, **item})
                done_probes += 1
                if float(metrics["soft_iou"]) > best_soft:
                    best_soft = float(metrics["soft_iou"])
                    best_key = f"{condition}#bin={bin_index}"
                write_train_status(
                    out,
                    {
                        "phase": "train",
                        "condition": condition,
                        "bin": bin_index,
                        "done_probes": done_probes,
                        "total_probes": total_probes,
                        "best_soft_iou": best_soft,
                        "best": best_key,
                        "last_error": "",
                    },
                )
                if done_probes == 1 or done_probes % 10 == 0 or done_probes == total_probes:
                    flush_metrics(out, metrics_rows, all_history)
                log(
                    f"probe {condition} bin={bin_index} "
                    f"{done_probes}/{total_probes} "
                    f"soft_iou={metrics['soft_iou']:.4f} hard_iou={metrics['hard_iou']:.4f} "
                    f"train_loss={metrics['train_loss']:.4f} seconds={metrics.get('train_seconds', 0):.1f}"
                )
            condition_index += 1

        del x_layer
        if args.drop_layer_cache and cache_path.exists():
            cache_path.unlink()
            log(f"Dropped layer cache {cache_path.name}")

    flush_metrics(out, metrics_rows, all_history)
    plot_heatmap(metrics_rows, out)
    best = max(metrics_rows, key=lambda row: float(row["soft_iou"]))
    manifest = {
        "status": "completed",
        "smoke": bool(args.smoke),
        "activation_root": str(args.activation_root),
        "occupancy_root": str(args.occupancy_root),
        "gpfs_output": str(out),
        "tos_output": str(args.tos_output_dir),
        "decoder": "independent Linear(D,64)-GELU-Linear(64,4096) per condition and selected bin",
        "loss": args.loss,
        "split": "extract split.json train/test; ablation unused",
        "n_probes": len(metrics_rows),
        "best": {
            "condition": best["condition"],
            "bin": best["bin"],
            "soft_iou": best["soft_iou"],
            "hard_iou": best["hard_iou"],
        },
        "seconds": time.perf_counter() - started,
        "outputs": [
            "split.json",
            "metrics.csv",
            "epoch_curves.csv",
            "probes/",
            "heatmaps.png",
            "figures/",
            "manifest.json",
            "STATUS.txt",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_train_status(
        out,
        {
            "phase": "done",
            "condition": best["condition"],
            "bin": best["bin"],
            "done_probes": done_probes,
            "total_probes": total_probes,
            "best_soft_iou": best["soft_iou"],
            "best": f"{best['condition']}#bin={best['bin']}",
            "last_error": "",
        },
    )
    log(f"Finished probes={len(metrics_rows)} best={best['condition']}#bin={best['bin']} soft_iou={best['soft_iou']:.4f}")
    if args.sync_tos:
        log(f"Sync GPFS → TOS {args.tos_output_dir} (skip layer_cache)")
        stats = sync_tree_inplace(out, Path(args.tos_output_dir), skip_dir_names=("layer_cache",))
        manifest["tos_sync"] = stats
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        log(f"TOS sync copied={stats['copied']} skipped_same_size={stats['skipped_same_size']}")
        write_train_status(
            out,
            {
                "phase": "synced",
                "condition": best["condition"],
                "bin": best["bin"],
                "done_probes": done_probes,
                "total_probes": total_probes,
                "best_soft_iou": best["soft_iou"],
                "best": f"{best['condition']}#bin={best['bin']}",
                "last_error": "",
                "tos_sync": stats,
            },
        )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
