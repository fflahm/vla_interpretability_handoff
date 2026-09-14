#!/usr/bin/env python
"""Capacity-constrained I_C(H;O|Q) on a finished PI0.5 occupancy run.

Reuses occupancy.npy, samples.jsonl, split.json, and activation_shards/.
Does not recapture GT or PI0.5 hidden states.

Arms
----
q   : decode occupancy from proprioception Q only (trained once)
hq  : decode from concat(H_{layer,bin}, Q) (trained per cell)

H_C is held-out unweighted BCE. I_C-hat = BCE_Q - BCE_HQ.
Training still uses script 23's pos-weighted BCE + Dice recipe.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.libero_self_occupancy import parse_index_spec  # noqa: E402
from src.pi05_occupancy_cmi import (  # noqa: E402
    DECODER_ARCHITECTURES,
    HIDDEN,
    annotate_history,
    assemble_features,
    decoder_csv_metrics,
    decoder_description,
    ic_hat,
    load_q_matrix,
    parse_condition,
    plot_cmi_figures,
    plot_train_test_curves,
    probe_seed,
    resolve_bin_indices,
    summarize_ic,
    train_decoder,
    write_csv,
    zscore_train,
)
from src.pi05_occupancy_full import (  # noqa: E402
    list_activation_conditions,
    load_activation_manifest,
    load_condition_activations,
    probe_dir,
    read_jsonl_samples,
    select_bin_columns,
)
from src.utils import ensure_dir, log, set_seed  # noqa: E402

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
        help="Defaults to <run-dir>/conditional_mi.",
    )
    p.add_argument(
        "--bin-indices",
        default="stored",
        help="stored (activation/metrics bins), auto-1/4, or an index spec such as 0-95:8.",
    )
    p.add_argument("--q-source", choices=("observation_state",), default="observation_state")
    p.add_argument(
        "--matched-params",
        action="store_true",
        help="Train Q as concat(zeros_like(H), Q) so the first layer matches HQ width.",
    )
    p.add_argument(
        "--decoder",
        default="mlp",
        choices=DECODER_ARCHITECTURES,
        help="mlp = Linear-GELU-Linear (script 23); linear = single Linear(in,4096).",
    )
    p.add_argument(
        "--hidden-size",
        type=int,
        default=HIDDEN,
        help="MLP hidden width. Ignored for --decoder linear.",
    )
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Max epochs. Used alone for a fixed budget.")
    p.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early stop patience on train_loss. 0 disables early stop and runs all --epochs.",
    )
    p.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum train_loss improvement to reset patience. Ignored when --patience 0.",
    )
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-conditions", type=int, default=None, help="Smoke: cap number of H conditions.")
    p.add_argument(
        "--curve-sample",
        type=int,
        default=10,
        help="Number of random HQ probes to overlay with Q on the train/test curve figure.",
    )
    p.add_argument("--dpi", type=int, default=180)
    return p.parse_args()


def resolve_device(device: str):
    import torch

    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stored_bins_from_run(run_dir: Path, requested_bins: int) -> list[int]:
    split = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
    selected = split.get("selected_bin_indices") or split.get("activation_bin_indices")
    if selected:
        return [int(v) for v in selected]
    metrics_path = run_dir / "metrics.csv"
    if metrics_path.exists():
        import csv

        bins = set()
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle):
                bins.add(int(raw["bin"]))
        if bins:
            return sorted(bins)
    return parse_index_spec("all", max_value=requested_bins)


def main() -> None:
    import torch

    args = parse_args()
    set_seed(args.seed)
    run_dir = args.run_dir.resolve()
    out = ensure_dir((args.output_dir or (run_dir / "conditional_mi")).resolve())
    figures = ensure_dir(out / "figures")

    split = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
    train_idx = np.asarray(split["train_indices"], dtype=int)
    test_idx = np.asarray(split["test_indices"], dtype=int)
    requested_bins = int(split.get("requested_bins") or 96)
    stored = stored_bins_from_run(run_dir, requested_bins)
    selected_bins = resolve_bin_indices(args.bin_indices, stored, requested_bins)

    occupancy = np.load(run_dir / "occupancy.npy").astype(np.float32)
    samples = read_jsonl_samples(run_dir / "samples.jsonl", occupancy)
    q_raw = load_q_matrix(samples, args.q_source)
    q, q_mean, q_std = zscore_train(q_raw, train_idx)
    act_manifest = load_activation_manifest(run_dir)
    conditions = list_activation_conditions(run_dir)
    if args.max_conditions is not None:
        conditions = conditions[: max(0, int(args.max_conditions))]
    device = resolve_device(args.device)

    targets = occupancy.reshape(len(occupancy), -1)
    positive_fraction = float(targets[train_idx].mean())
    pos_weight_value = min(30.0, max(1.0, (1.0 - positive_fraction) / max(positive_fraction, 1e-6)))
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)

    matched = {
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "min_delta": float(args.min_delta),
        "early_stop_monitor": "train_loss (no extra validation split)",
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "seed": int(args.seed),
        "architecture": args.decoder,
        "hidden_size": None if args.decoder == "linear" else int(args.hidden_size),
        "decoder": decoder_description(args.decoder, args.hidden_size),
        "optimizer": "AdamW",
        "train_loss": "pos-weighted BCE + soft Dice",
        "h_c_estimator": "held-out unweighted BCE (Bernoulli NLL, no Dice)",
        "ic_hat": "test_bce_q - test_bce_hq",
        "split": "reused split.json (per-task grouped demo)",
        "q_source": args.q_source,
        "q_note": (
            "observation_state is EE+gripper and does not uniquely determine "
            "whole-body occupancy voxels."
        ),
        "normalization": "train-set mean/std of Q and of H separately, then concat",
        "matched_params": bool(args.matched_params),
        "pos_weight": pos_weight_value,
        "positive_fraction_train": positive_fraction,
        "device": str(device),
        "torch": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "selected_bins": selected_bins,
        "n_conditions": len(conditions),
        "q_dim": int(q.shape[1]),
        "probe_seed_formula": "seed + condition_index * 1000 + bin_index",
        "no_activation_recapture": True,
    }
    log("=" * 72)
    log(f"Conditional MI out={out}")
    log(json.dumps(matched, indent=2))
    (out / "matched_settings.json").write_text(json.dumps(matched, indent=2) + "\n", encoding="utf-8")

    reference_h = None
    if args.matched_params:
        x0 = load_condition_activations(run_dir, conditions[0])
        bin_columns = select_bin_columns(
            condition=conditions[0],
            x_all=x0,
            requested_bin_indices=selected_bins,
            manifest=act_manifest,
        )
        reference_h, _, _ = zscore_train(x0[:, bin_columns[0][1]], train_idx)

    q_features = assemble_features(h=reference_h, q=q, arm="q", matched_params=bool(args.matched_params))
    train_kw = dict(
        occupancy=occupancy,
        train_idx=train_idx,
        test_idx=test_idx,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        pos_weight=pos_weight,
        patience=args.patience,
        min_delta=args.min_delta,
        architecture=args.decoder,
        hidden_size=args.hidden_size,
    )
    all_history_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    q_result = train_decoder(x=q_features, seed=args.seed, **train_kw)
    q_history = q_result["history"]
    q_row = {
        "arm": "q",
        "q_source": args.q_source,
        "matched_params": bool(args.matched_params),
        "probe_seed": int(args.seed),
        "train_seconds": time.perf_counter() - started,
        **decoder_csv_metrics(q_result),
    }
    write_csv(out / "metrics_q.csv", [q_row])
    q_hist_rows = annotate_history(q_history, arm="q", probe_seed=int(args.seed))
    all_history_rows.extend(q_hist_rows)
    q_probe = ensure_dir(out / "probes" / "q")
    write_csv(q_probe / "history.csv", q_hist_rows)
    write_csv(out / "epoch_curves.csv", all_history_rows)
    log(
        f"Q decoder done test_bce={q_row['test_bce']:.4f} "
        f"soft_iou={q_row['soft_iou']:.3f} input_dim={q_row['input_dim']} "
        f"best_epoch={q_row['best_epoch']}/{q_row['stopped_epoch']} "
        f"early_stop={q_row['early_stopped']} decoder={q_row['decoder']}"
    )

    hq_rows: list[dict[str, Any]] = []
    ic_rows: list[dict[str, Any]] = []
    condition_bar = tqdm(list(enumerate(conditions)), desc="hq conditions", unit="cond")
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
            h, _, _ = zscore_train(x_all[:, column], train_idx)
            if args.matched_params and h.shape[1] != reference_h.shape[1]:
                raise ValueError(
                    f"matched-params needs a single H dim; {condition} has "
                    f"{h.shape[1]} vs reference {reference_h.shape[1]}."
                )
            x_hq = assemble_features(h=h, q=q, arm="hq", matched_params=False)
            seed = probe_seed(args.seed, condition_index, bin_index)
            started = time.perf_counter()
            result = train_decoder(x=x_hq, seed=seed, **train_kw)
            history = result["history"]
            row = {
                "arm": "hq",
                "condition": condition,
                "tower": parsed["tower"],
                "layer": parsed["layer"],
                "flow": parsed["flow"],
                "bin": int(bin_index),
                "column": int(column),
                "probe_seed": int(seed),
                "train_seconds": time.perf_counter() - started,
                **decoder_csv_metrics(result),
            }
            hq_rows.append(row)
            hist_rows = annotate_history(
                history,
                arm="hq",
                condition=condition,
                tower=parsed["tower"],
                layer=parsed["layer"],
                flow=parsed["flow"],
                bin_index=int(bin_index),
                probe_seed=int(seed),
            )
            all_history_rows.extend(hist_rows)
            write_csv(ensure_dir(probe_dir(out, condition, bin_index)) / "history.csv", hist_rows)
            ic_rows.append(
                {
                    "condition": condition,
                    "tower": parsed["tower"],
                    "layer": parsed["layer"],
                    "flow": parsed["flow"],
                    "bin": int(bin_index),
                    "probe_seed": int(seed),
                    "test_bce_q": float(q_row["test_bce"]),
                    "test_bce_hq": float(row["test_bce"]),
                    "ic_hat": ic_hat(q_row["test_bce"], row["test_bce"]),
                    "soft_iou_q": float(q_row["soft_iou"]),
                    "soft_iou_hq": float(row["soft_iou"]),
                    "delta_soft_iou": float(row["soft_iou"]) - float(q_row["soft_iou"]),
                    "q_input_dim": int(q_row["input_dim"]),
                    "hq_input_dim": int(row["input_dim"]),
                    "q_param_count": int(q_row["param_count"]),
                    "hq_param_count": int(row["param_count"]),
                }
            )
            condition_bar.set_postfix(
                bin=bin_index,
                ic=f"{ic_rows[-1]['ic_hat']:.4f}",
                done=len(hq_rows),
                stop=row["stopped_epoch"],
            )
        write_csv(out / "metrics_hq.csv", hq_rows)
        write_csv(out / "ic_hat.csv", ic_rows)
        write_csv(out / "epoch_curves.csv", all_history_rows)

    if not hq_rows:
        raise RuntimeError("No HQ probes were trained. Check --bin-indices against stored activation bins.")

    summary = {
        "matched_settings": matched,
        "q": {
            k: q_row[k]
            for k in (
                "test_bce",
                "soft_iou",
                "hard_iou",
                "input_dim",
                "param_count",
                "best_epoch",
                "stopped_epoch",
                "early_stopped",
            )
        },
        "ic": summarize_ic(ic_rows),
        "history_rows": len(all_history_rows),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_cmi_figures(ic_rows, q_row, figures, args.dpi)
    sampled = plot_train_test_curves(
        all_history_rows,
        figures / "04_train_test_curves.png",
        hq_sample=args.curve_sample,
        seed=args.seed,
        dpi=args.dpi,
    )
    log(json.dumps(summary["ic"], indent=2))
    log(f"Curve sample HQ probes: {sampled}")
    log("=" * 72)


if __name__ == "__main__":
    main()
