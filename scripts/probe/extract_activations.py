#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.activations import save_activations, stack_layer_activations
from src.data import load_states, states_to_targets
from src.models import MockVLAWrapper, Pi05Wrapper, load_image_np
from src.utils import load_config, resolve_path, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--model", default=None, choices=["mock", "pi05"])
    parser.add_argument("--pi05-path", default=None, help="Local pi0.5 checkpoint path or Hugging Face repo id.")
    parser.add_argument("--max-samples", type=int, default=None, help="Only extract the first N states for a smoke test.")
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config, ROOT))
    set_seed(int(cfg["seed"]))
    model_name = args.model or cfg["model"]["name"]
    states = load_states(resolve_path(cfg["data"]["state_path"], ROOT))
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError(f"`--max-samples` must be positive, got {args.max_samples}.")
        states = states[: args.max_samples]

    if model_name == "mock":
        model = MockVLAWrapper(
            num_layers=int(cfg["model"]["num_layers"]),
            hidden_dim=int(cfg["model"]["hidden_dim"]),
            action_dim=int(cfg["model"]["action_dim"]),
            noise_std=float(cfg["model"]["noise_std"]),
            seed=int(cfg["seed"]),
        )
    else:
        pi05_path = (
            args.pi05_path
            or os.environ.get("VLA_PI05_PATH")
            or str(cfg["model"].get("pi05_pretrained_path", "lerobot/pi05_libero_finetuned_quantiles"))
        )
        model = Pi05Wrapper(
            model_id=pi05_path,
            pooling=str(cfg["model"].get("pooling", "mean")),
            device=str(cfg["model"].get("device", "auto")),
        )

    hidden_by_sample = []
    pred_actions = []
    pred_action_chunks = []
    for state in tqdm(states, desc="extract activations"):
        image = load_image_np(state["image_path"])
        out = model.forward(image=image, instruction=state["instruction"], metadata=state)
        hidden_by_sample.append(out["hidden_states"])
        pred_actions.append(np.asarray(out["pred_action"], dtype=np.float32))
        if "pred_action_chunk" in out:
            pred_action_chunks.append(np.asarray(out["pred_action_chunk"], dtype=np.float32))

    x_layers, layer_names = stack_layer_activations(hidden_by_sample)
    targets = states_to_targets(states)
    y_action = np.stack(pred_actions, axis=0)
    y_action_chunk = None
    if pred_action_chunks:
        if len(pred_action_chunks) != len(pred_actions):
            raise RuntimeError("Only some samples returned `pred_action_chunk`; cannot save a partially missing chunk target.")
        min_horizon = min(chunk.shape[0] for chunk in pred_action_chunks)
        y_action_chunk = np.stack([chunk[:min_horizon].reshape(-1) for chunk in pred_action_chunks], axis=0)
    save_activations(
        resolve_path(cfg["outputs"]["activations_path"], ROOT),
        x_layers=x_layers,
        layer_names=layer_names,
        targets=targets,
        y_action=y_action,
        y_action_chunk=y_action_chunk,
        metadata={"model": model_name, "num_samples": len(states), "pooling": cfg["model"].get("pooling", "mean")},
    )
    print(f"Saved activations: X_layers={x_layers.shape}")


if __name__ == "__main__":
    main()
