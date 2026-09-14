#!/usr/bin/env python
"""Compare how easily the same H decodes Q vs occupancy O via NDS.

Reuses a finished occupancy run (activations, occupancy.npy, split.json).
At bottleneck 64, script-23 H→O checkpoints are re-evaluated with unweighted
BCE; only H→Q is trained. Occupancy metrics.csv is NOT reused (it stores
BCE+Dice). Permutation controls retrain both heads on shuffled H↔target pairs.
"""
from __future__ import annotations

import argparse
import csv
import gzip
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
from src.pi05_nds_q_vs_o import (  # noqa: E402
    OUTPUT_DIM_O,
    bootstrap_d,
    d_o_minus_q,
    eval_occupancy_checkpoint,
    load_occupancy_checkpoint,
    make_q_baseline_per_sample,
    nds,
    occupancy_ckpt_is_reusable,
    occupancy_frequency_baseline,
    per_sample_bernoulli_nll,
    plot_nds_heatmaps,
    shuffle_rows_by_split,
    summarize_nds_rows,
    train_bottleneck_regressor,
    write_json,
)
from src.pi05_occupancy_cmi import (  # noqa: E402
    load_q_matrix,
    parse_condition,
    probe_seed,
    resolve_bin_indices,
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
DEFAULT_BOTTLENECK = 64


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "outputs/self_occupancy/pi05_libero_spatial_10k",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--bin-indices", default="stored")
    p.add_argument(
        "--bottleneck",
        default=str(DEFAULT_BOTTLENECK),
        help="Hidden width d, or comma list such as 16,32,64,128.",
    )
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", default="auto")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--max-conditions", type=int, default=None)
    p.add_argument(
        "--reuse-occupancy-probes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse script-23 decoder.pt for real H→O when bottleneck=64.",
    )
    p.add_argument(
        "--perm-control",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retrain/eval after shuffling H vs target pairing.",
    )
    p.add_argument(
        "--save-paired",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-test-sample losses for the real arm.",
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
    return parse_index_spec("all", max_value=requested_bins)


def parse_bottlenecks(spec: str) -> list[int]:
    values = [int(part.strip()) for part in str(spec).split(",") if part.strip()]
    if not values or any(v < 1 for v in values):
        raise ValueError(f"Invalid --bottleneck {spec!r}")
    return values


def occupancy_pos_weight(occupancy: np.ndarray, train_idx: np.ndarray, device):
    import torch

    targets = occupancy.reshape(len(occupancy), -1)
    positive_fraction = float(targets[train_idx].mean())
    value = min(30.0, max(1.0, (1.0 - positive_fraction) / max(positive_fraction, 1e-6)))
    return torch.tensor(value, dtype=torch.float32, device=device), value, positive_fraction


def run_one_probe(
    *,
    h_raw: np.ndarray,
    q_z: np.ndarray,
    occupancy: np.ndarray,
    o_targets: np.ndarray,
    o_base_prob: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    hidden: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device,
    pos_weight,
    occupancy_ckpt,
    reuse_o: bool,
) -> dict[str, Any]:
    h, _, _ = zscore_train(h_raw, train_idx)
    q_result = train_bottleneck_regressor(
        x=h,
        y=q_z,
        train_idx=train_idx,
        test_idx=test_idx,
        hidden=hidden,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        device=device,
        kind="q",
    )
    mse_h = q_result["per_sample"]
    mse_base = make_q_baseline_per_sample(q_z, test_idx)
    reused = False
    if reuse_o and occupancy_ckpt is not None and occupancy_ckpt_is_reusable(occupancy_ckpt, hidden):
        bce_h = eval_occupancy_checkpoint(
            checkpoint=occupancy_ckpt,
            x_raw=h_raw,
            occupancy=occupancy,
            test_idx=test_idx,
            bottleneck=hidden,
            device=device,
        )
        o_param_count = int(sum(v.numel() for v in occupancy_ckpt["state_dict"].values()))
        o_train_loss = float(occupancy_ckpt.get("final_train_loss", float("nan")))
        reused = True
        o_state = None
    else:
        o_result = train_bottleneck_regressor(
            x=h,
            y=o_targets,
            train_idx=train_idx,
            test_idx=test_idx,
            hidden=hidden,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            device=device,
            kind="o",
            pos_weight=pos_weight,
        )
        bce_h = o_result["per_sample"]
        o_param_count = int(o_result["param_count"])
        o_train_loss = float(o_result["train_loss"])
        o_state = o_result["state_dict"]
    bce_base = per_sample_bernoulli_nll(o_targets[test_idx], np.broadcast_to(o_base_prob, o_targets[test_idx].shape))
    return {
        "mse_h": mse_h,
        "mse_base": mse_base,
        "bce_h": bce_h,
        "bce_base": bce_base,
        "nds_q": nds(float(mse_h.mean()), float(mse_base.mean())),
        "nds_o": nds(float(bce_h.mean()), float(bce_base.mean())),
        "q_param_count": int(q_result["param_count"]),
        "o_param_count": o_param_count,
        "q_train_loss": float(q_result["train_loss"]),
        "o_train_loss": o_train_loss,
        "q_input_dim": int(q_result["input_dim"]),
        "reused_occupancy_probe": reused,
        "q_state": q_result["state_dict"],
        "o_state": o_state,
    }


def main() -> None:
    import torch

    args = parse_args()
    set_seed(args.seed)
    run_dir = args.run_dir.resolve()
    bottlenecks = parse_bottlenecks(args.bottleneck)
    base_out = (args.output_dir or (run_dir / "nds_q_vs_o")).resolve()
    split = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
    train_idx = np.asarray(split["train_indices"], dtype=int)
    test_idx = np.asarray(split["test_indices"], dtype=int)
    requested_bins = int(split.get("requested_bins") or 96)
    stored = stored_bins_from_run(run_dir, requested_bins)
    selected_bins = resolve_bin_indices(args.bin_indices, stored, requested_bins)
    occupancy = np.load(run_dir / "occupancy.npy").astype(np.float32)
    o_targets = occupancy.reshape(len(occupancy), -1).astype(np.float32)
    samples = read_jsonl_samples(run_dir / "samples.jsonl", occupancy)
    q_raw = load_q_matrix(samples, "observation_state")
    q_z, _, _ = zscore_train(q_raw, train_idx)
    o_base_prob = occupancy_frequency_baseline(occupancy, train_idx)
    act_manifest = load_activation_manifest(run_dir)
    conditions = list_activation_conditions(run_dir)
    if args.max_conditions is not None:
        conditions = conditions[: max(0, int(args.max_conditions))]
    device = resolve_device(args.device)
    pos_weight, pos_weight_value, positive_fraction = occupancy_pos_weight(occupancy, train_idx, device)

    for hidden in bottlenecks:
        out = ensure_dir(base_out if len(bottlenecks) == 1 else base_out / f"d{hidden}")
        figures = ensure_dir(out / "figures")
        reuse_o = bool(args.reuse_occupancy_probes) and int(hidden) == DEFAULT_BOTTLENECK
        matched = {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "seed": int(args.seed),
            "bottleneck": int(hidden),
            "q_decoder": f"Linear(D,{hidden})-GELU-Linear({hidden},{q_z.shape[1]})",
            "o_decoder": f"Linear(D,{hidden})-GELU-Linear({hidden},{OUTPUT_DIM_O})",
            "q_loss": "MSE on train-standardized observation_state",
            "o_train_loss": "pos-weighted BCE + soft Dice (script 23)",
            "o_eval": "held-out unweighted BCE / Bernoulli NLL",
            "nds_q": "1 - MSE_HtoQ / MSE_baseQ  (R^2 of standardized Q)",
            "nds_o": "1 - BCE_HtoO / BCE_baseO",
            "d_o_minus_q": "NDS_O - NDS_Q; not mutual information; negatives kept",
            "q_baseline": "predict train-set mean of standardized Q (zeros)",
            "o_baseline": "per-voxel train occupancy frequency p_j",
            "split": "reused split.json",
            "reuse_occupancy_probes": reuse_o,
            "perm_control": bool(args.perm_control),
            "n_bootstrap": int(args.n_bootstrap),
            "pos_weight": pos_weight_value,
            "positive_fraction_train": positive_fraction,
            "device": str(device),
            "selected_bins": selected_bins,
            "n_conditions": len(conditions),
        }
        log("=" * 72)
        log(f"NDS Q vs O out={out}")
        log(json.dumps(matched, indent=2))
        write_json(out / "matched_settings.json", matched)

        metric_rows: list[dict[str, Any]] = []
        paired_handle = None
        paired_writer = None
        if args.save_paired:
            paired_path = out / "paired_test_scores.csv.gz"
            paired_handle = gzip.open(paired_path, "wt", encoding="utf-8", newline="")
            paired_writer = csv.DictWriter(
                paired_handle,
                fieldnames=[
                    "condition",
                    "bin",
                    "sample_index",
                    "mse_h_q",
                    "mse_base_q",
                    "delta_mse_q",
                    "bce_h_o",
                    "bce_base_o",
                    "delta_bce_o",
                ],
            )
            paired_writer.writeheader()
        arms = ("real", "perm") if args.perm_control else ("real",)
        condition_bar = tqdm(list(enumerate(conditions)), desc=f"nds d={hidden}", unit="cond")
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
                h_raw = x_all[:, column]
                ckpt = load_occupancy_checkpoint(run_dir, condition, bin_index, device) if reuse_o else None
                for arm in arms:
                    seed = probe_seed(args.seed, condition_index, bin_index)
                    if arm == "perm":
                        seed = seed + 10_000_000
                        h_use = shuffle_rows_by_split(h_raw, train_idx, test_idx, seed)
                        reuse_this = False
                        ckpt_use = None
                    else:
                        h_use = h_raw
                        reuse_this = reuse_o
                        ckpt_use = ckpt
                    started = time.perf_counter()
                    result = run_one_probe(
                        h_raw=h_use,
                        q_z=q_z,
                        occupancy=occupancy,
                        o_targets=o_targets,
                        o_base_prob=o_base_prob,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        hidden=hidden,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        seed=seed,
                        device=device,
                        pos_weight=pos_weight,
                        occupancy_ckpt=ckpt_use,
                        reuse_o=reuse_this,
                    )
                    boot = bootstrap_d(
                        result["mse_h"],
                        result["mse_base"],
                        result["bce_h"],
                        result["bce_base"],
                        n_bootstrap=args.n_bootstrap,
                        seed=seed,
                    )
                    d_val = d_o_minus_q(result["nds_o"], result["nds_q"])
                    row = {
                        "arm": arm,
                        "condition": condition,
                        "tower": parsed["tower"],
                        "layer": parsed["layer"],
                        "flow": parsed["flow"],
                        "bin": int(bin_index),
                        "column": int(column),
                        "bottleneck": int(hidden),
                        "probe_seed": int(seed),
                        "nds_q": result["nds_q"],
                        "nds_o": result["nds_o"],
                        "d_o_minus_q": d_val,
                        "nds_q_ci_lo": boot["nds_q_ci_lo"],
                        "nds_q_ci_hi": boot["nds_q_ci_hi"],
                        "nds_o_ci_lo": boot["nds_o_ci_lo"],
                        "nds_o_ci_hi": boot["nds_o_ci_hi"],
                        "d_ci_lo": boot["ci_lo"],
                        "d_ci_hi": boot["ci_hi"],
                        "mse_h_to_q": float(result["mse_h"].mean()),
                        "mse_base_q": float(result["mse_base"].mean()),
                        "bce_h_to_o": float(result["bce_h"].mean()),
                        "bce_base_o": float(result["bce_base"].mean()),
                        "q_train_mse": result["q_train_loss"],
                        "o_train_loss": result["o_train_loss"],
                        "q_param_count": result["q_param_count"],
                        "o_param_count": result["o_param_count"],
                        "q_input_dim": result["q_input_dim"],
                        "reused_occupancy_probe": result["reused_occupancy_probe"],
                        "train_seconds": time.perf_counter() - started,
                    }
                    metric_rows.append(row)
                    if arm == "real" and paired_writer is not None:
                        for i, sample_id in enumerate(test_idx.tolist()):
                            paired_writer.writerow(
                                {
                                    "condition": condition,
                                    "bin": int(bin_index),
                                    "sample_index": int(sample_id),
                                    "mse_h_q": float(result["mse_h"][i]),
                                    "mse_base_q": float(result["mse_base"][i]),
                                    "delta_mse_q": float(result["mse_base"][i] - result["mse_h"][i]),
                                    "bce_h_o": float(result["bce_h"][i]),
                                    "bce_base_o": float(result["bce_base"][i]),
                                    "delta_bce_o": float(result["bce_base"][i] - result["bce_h"][i]),
                                }
                            )
                    if arm == "real":
                        q_dir = ensure_dir(probe_dir(out, condition, bin_index))
                        torch.save(
                            {
                                "state_dict": result["q_state"],
                                "condition": condition,
                                "bin": int(bin_index),
                                "input_dim": result["q_input_dim"],
                                "output_dim": int(q_z.shape[1]),
                                "bottleneck": int(hidden),
                                "target": "q",
                            },
                            q_dir / "q_decoder.pt",
                        )
                    condition_bar.set_postfix(
                        bin=bin_index,
                        arm=arm,
                        D=f"{d_val:.3f}",
                        reuse=int(result["reused_occupancy_probe"]),
                    )
            write_csv(out / "metrics.csv", metric_rows)

        real_rows = [row for row in metric_rows if row["arm"] == "real"]
        if paired_handle is not None:
            paired_handle.close()
        if not real_rows:
            raise RuntimeError("No NDS probes were trained.")
        plot_nds_heatmaps(real_rows, figures, args.dpi)
        summary = {
            "matched_settings": matched,
            "real": summarize_nds_rows(real_rows),
            "perm": summarize_nds_rows([row for row in metric_rows if row["arm"] == "perm"])
            if args.perm_control
            else None,
        }
        write_json(out / "summary.json", summary)
        log(json.dumps(summary["real"], indent=2))
        log("=" * 72)


if __name__ == "__main__":
    main()
