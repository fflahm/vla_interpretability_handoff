#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.activations import load_activations
from src.data import load_states, states_to_targets
from src.probes import train_layerwise_ridge
from src.utils import ensure_dir, load_config, resolve_path, set_seed

PROBE_TARGETS = (
    "offset",
    "target_position",
    "gripper_position",
    "action",
    "action_chunk",
    "gt_action",
    "gt_action_chunk",
)
TARGET_CHOICES = (*PROBE_TARGETS, "all")


def target_results_path(base_path: Path, target: str) -> Path:
    if target == "offset":
        return base_path
    return base_path.with_name(f"{base_path.stem}_{target}{base_path.suffix}")


def train_target(
    *,
    target: str,
    act: dict,
    cfg: dict,
    root: Path,
    seed: int,
    probes_dir: Path,
    action_chunk_path: Path | None = None,
    action_chunk_horizon: int = 4,
) -> None:
    target_key = f"y_{target}"
    if target_key not in act:
        num_samples = int(act["X_layers"].shape[1])
        if target == "action_chunk" and action_chunk_path is not None:
            act[target_key] = load_action_chunk_target(action_chunk_path, num_samples, action_chunk_horizon)
            print(f"Filled missing activation target `{target_key}` from {action_chunk_path}.")
        else:
            state_targets = states_to_targets(load_states(resolve_path(cfg["data"]["state_path"], root)))
            if target in state_targets and len(state_targets[target]) >= num_samples:
                act[target_key] = state_targets[target][:num_samples]
                print(f"Filled missing activation target `{target_key}` from {cfg['data']['state_path']}.")
            else:
                available = sorted(key.removeprefix("y_") for key in act if key.startswith("y_"))
                raise KeyError(
                    f"Target `{target}` is not present in the activation file, state file, or provided chunk file. "
                    f"Available activation targets: {available}. For action chunks, pass "
                    "`--action-chunk-jsonl` or `--action-chunk-npz`, or re-run scripts/probe/extract_activations.py "
                    "with a model wrapper that returns `pred_action_chunk`."
                )
    y = act[target_key]

    results = train_layerwise_ridge(
        act["X_layers"],
        y,
        act["layer_names"],
        alpha=float(cfg["probe"]["alpha"]),
        test_size=float(cfg["probe"]["test_size"]),
        seed=seed,
        shuffle_labels=False,
        save_dir=probes_dir,
        prefix=f"{target}_probe",
    )
    results_path = target_results_path(resolve_path(cfg["probe"]["results_path"], root), target)
    results.to_csv(results_path, index=False)
    print(f"Saved {target} probe results to {results_path}")

    if bool(cfg["probe"].get("shuffled_control", True)):
        shuffled = train_layerwise_ridge(
            act["X_layers"],
            y,
            act["layer_names"],
            alpha=float(cfg["probe"]["alpha"]),
            test_size=float(cfg["probe"]["test_size"]),
            seed=seed,
            shuffle_labels=True,
            save_dir=None,
            prefix=f"{target}_probe",
        )
        shuffled_path = target_results_path(resolve_path(cfg["probe"]["shuffled_results_path"], root), target)
        shuffled.to_csv(shuffled_path, index=False)
        print(f"Saved {target} shuffled-label control to {shuffled_path}")


def load_action_chunk_target(path: Path, num_samples: int, horizon: int) -> np.ndarray:
    if horizon <= 0:
        raise ValueError(f"`--action-chunk-horizon` must be positive, got {horizon}.")
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        if "pred_action_chunk" not in data.files:
            raise KeyError(f"`pred_action_chunk` not found in {path}. Available keys: {data.files}")
        chunks = data["pred_action_chunk"]
    else:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if "pred_action_chunk" not in row:
                        raise KeyError(f"Row is missing `pred_action_chunk` in {path}.")
                    rows.append(row["pred_action_chunk"])
        chunks = np.asarray(rows, dtype=object)
    if len(chunks) < num_samples:
        raise ValueError(f"Need at least {num_samples} action chunks, got {len(chunks)} from {path}.")
    if len(chunks) != num_samples:
        print(
            f"Warning: action chunk file has {len(chunks)} rows but activations have {num_samples} samples; "
            "using the first activation-aligned rows. Make sure both files use the same sample order.",
            flush=True,
        )
    flattened = []
    for chunk in chunks[:num_samples]:
        chunk_arr = np.asarray(chunk, dtype=np.float32)
        if chunk_arr.ndim == 3:
            chunk_arr = chunk_arr[0]
        if chunk_arr.ndim != 2:
            raise ValueError(f"Expected each chunk to have shape [horizon, action_dim], got {chunk_arr.shape}.")
        if len(chunk_arr) < horizon:
            raise ValueError(f"Chunk horizon {len(chunk_arr)} is shorter than requested horizon={horizon}.")
        flattened.append(chunk_arr[:horizon].reshape(-1))
    return np.stack(flattened, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--target", default=None, choices=TARGET_CHOICES)
    parser.add_argument("--action-chunk-jsonl", default=None, help="External JSONL containing `pred_action_chunk` labels.")
    parser.add_argument("--action-chunk-npz", default=None, help="External NPZ containing `pred_action_chunk` labels.")
    parser.add_argument("--action-chunk-horizon", type=int, default=4)
    args = parser.parse_args()
    if args.action_chunk_jsonl and args.action_chunk_npz:
        raise ValueError("Use only one of `--action-chunk-jsonl` and `--action-chunk-npz`.")

    cfg = load_config(resolve_path(args.config, ROOT))
    seed = int(cfg["seed"])
    set_seed(seed)
    act = load_activations(resolve_path(cfg["outputs"]["activations_path"], ROOT))
    target = args.target or cfg["probe"]["target"]
    probes_dir = ensure_dir(resolve_path(cfg["outputs"]["probes_dir"], ROOT))
    action_chunk_path = None
    if args.action_chunk_jsonl:
        action_chunk_path = resolve_path(args.action_chunk_jsonl, ROOT)
    if args.action_chunk_npz:
        action_chunk_path = resolve_path(args.action_chunk_npz, ROOT)
    targets = PROBE_TARGETS if target == "all" else (target,)
    for target_name in targets:
        train_target(
            target=target_name,
            act=act,
            cfg=cfg,
            root=ROOT,
            seed=seed,
            probes_dir=probes_dir,
            action_chunk_path=action_chunk_path,
            action_chunk_horizon=int(args.action_chunk_horizon),
        )


if __name__ == "__main__":
    main()
