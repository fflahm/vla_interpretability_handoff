#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.online_pi0_rollout import ActivationIntervention, Pi0LiberoFullTokenRolloutTracer
from src.online_pi0_rollout_cli import DEFAULT_RENAME_MAP
from src.utils import ensure_dir, load_config, read_jsonl, resolve_path, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run causal activation ablations for PI0 online LIBERO rollouts. "
            "Each condition zeros/scales one layer and one token-bin during policy forward passes."
        )
    )
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--pi0-path", default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument("--start-seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--pickup-joint-name", default=None)
    parser.add_argument("--place-joint-name", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-dir", default=None, help="Existing baseline rollout dir. If omitted, a baseline is run.")
    parser.add_argument("--skip-baseline", action="store_true", help="Do not run baseline; requires --baseline-dir for deltas.")
    parser.add_argument("--baseline-only", action="store_true", help="Only run the baseline rollout and exit.")
    parser.add_argument("--num-shards", type=int, default=1, help="Split layer/bin conditions across this many independent jobs.")
    parser.add_argument("--shard-index", type=int, default=0, help="Index of this shard in [0, num_shards).")
    parser.add_argument(
        "--shard-output-subdir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For sharded runs, write into output-dir/shard_XX_of_NN to avoid concurrent writes.",
    )
    parser.add_argument("--layers", default="all", help="Layer indices, e.g. `0,1,17`, or `all`.")
    parser.add_argument("--num-layers", type=int, default=36, help="Used when --layers all.")
    parser.add_argument("--token-bins", type=int, default=96)
    parser.add_argument(
        "--bin-indices",
        default="all",
        help="Token-bin indices, e.g. `0,8,16`, or `all`. With `all`, --bin-stride controls coarse sweep.",
    )
    parser.add_argument("--bin-stride", type=int, default=8)
    parser.add_argument("--mode", choices=("zero", "scale"), default="zero")
    parser.add_argument("--scale", type=float, default=0.0)
    parser.add_argument("--force-replan-every-step", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--replan-interval", type=int, default=None)
    parser.add_argument("--activation-dtype", choices=("float16", "float32"), default=None)
    parser.add_argument(
        "--save-activations",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save full-token activations for every ablated rollout. Off by default because sweeps are IO-heavy.",
    )
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-fps", type=int, default=None)
    parser.add_argument("--video-format", choices=("gif", "mp4"), default="mp4")
    parser.add_argument("--require-mp4", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--rename-map",
        default=None,
        help='JSON rename map for policy inputs, e.g. \'{"observation.images.image2":"observation.images.wrist_image"}\'.',
    )
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config, ROOT))
    rollout_cfg = cfg["online_pi0"]
    seed = int(args.start_seed if args.start_seed is not None else rollout_cfg.get("start_seed", cfg["seed"]))
    set_seed(seed)

    if args.num_shards <= 0:
        raise ValueError(f"`--num-shards` must be positive, got {args.num_shards}.")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(f"`--shard-index` must be in [0, {args.num_shards}), got {args.shard_index}.")
    root_output_dir = ensure_dir(resolve_path(args.output_dir, ROOT))
    output_dir = root_output_dir
    if args.num_shards > 1 and args.shard_output_subdir:
        output_dir = ensure_dir(root_output_dir / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}")
    checkpoint_path = args.pi0_path or os.environ.get("VLA_PI0_PATH") or rollout_cfg.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("Provide `--pi0-path`, set VLA_PI0_PATH, or configure online_pi0.checkpoint_path.")
    rename_map = json.loads(args.rename_map) if args.rename_map else dict(rollout_cfg.get("rename_map", DEFAULT_RENAME_MAP))

    common_kwargs = dict(
        checkpoint_path=str(checkpoint_path),
        task=str(args.task or rollout_cfg.get("task", "libero_spatial")),
        task_id=int(args.task_id if args.task_id is not None else rollout_cfg.get("task_id", 1)),
        device=str(args.device or cfg["model"].get("device", "auto")),
        pickup_joint_name=str(args.pickup_joint_name or rollout_cfg.get("pickup_joint_name", "auto")),
        place_joint_name=str(args.place_joint_name or rollout_cfg.get("place_joint_name", "auto")),
        instruction=str(args.instruction or rollout_cfg.get("instruction", "")),
        rename_map=rename_map,
        rotate_images_180=bool(rollout_cfg.get("rotate_images_180", True)),
        force_replan_every_step=bool(
            args.force_replan_every_step
            if args.force_replan_every_step is not None
            else rollout_cfg.get("force_replan_every_step", True)
        ),
        replan_interval=int(args.replan_interval if args.replan_interval is not None else rollout_cfg.get("replan_interval", 1)),
        activation_dtype=str(args.activation_dtype or rollout_cfg.get("activation_dtype", "float16")),
    )
    collect_kwargs = dict(
        num_episodes=int(args.num_episodes),
        start_seed=seed,
        max_steps=args.max_steps if args.max_steps is not None else rollout_cfg.get("max_steps"),
        save_video=bool(args.save_video),
        save_wrist_video=False,
        video_fps=int(args.video_fps if args.video_fps is not None else rollout_cfg.get("video_fps", 10)),
        video_format=str(args.video_format),
        video_flip_180=bool(rollout_cfg.get("video_flip_180", True)),
        require_mp4=bool(args.require_mp4),
        save_activations=bool(args.save_activations),
    )

    rows: list[dict[str, Any]] = []
    tracer = Pi0LiberoFullTokenRolloutTracer(**common_kwargs)
    try:
        baseline_dir = resolve_path(args.baseline_dir, ROOT) if args.baseline_dir else root_output_dir / "baseline"
        if not args.skip_baseline:
            run_condition(output_dir=baseline_dir, tracer=tracer, collect_kwargs=collect_kwargs, intervention=None)
        elif not baseline_dir.exists():
            raise ValueError(f"--skip-baseline was set but baseline dir does not exist: {baseline_dir}")
        if args.baseline_only:
            print(f"Saved baseline rollout to {baseline_dir}")
            return

        baseline = load_rollout_for_comparison(baseline_dir)
        layers = parse_indices(args.layers, max_value=args.num_layers, stride=1)
        bins = parse_indices(args.bin_indices, max_value=args.token_bins, stride=max(1, int(args.bin_stride)))
        conditions = [(layer_index, bin_index) for layer_index in layers for bin_index in bins]
        conditions = [
            condition for condition_index, condition in enumerate(conditions) if condition_index % args.num_shards == args.shard_index
        ]
        print(
            f"Running shard {args.shard_index}/{args.num_shards}: {len(conditions)} conditions -> {output_dir}",
            flush=True,
        )
        for layer_index, bin_index in tqdm(conditions, desc=f"ablation shard {args.shard_index}/{args.num_shards}"):
            intervention = ActivationIntervention(
                layer_index=int(layer_index),
                token_bin_index=int(bin_index),
                token_bins=int(args.token_bins),
                mode=str(args.mode),
                scale=float(args.scale),
            )
            condition_name = f"layer_{layer_index:02d}_bin_{bin_index:03d}_{args.mode}"
            if args.mode == "scale":
                condition_name += f"{args.scale:g}".replace(".", "p")
            condition_dir = output_dir / condition_name
            run_condition(
                output_dir=condition_dir,
                tracer=tracer,
                collect_kwargs=collect_kwargs,
                intervention=intervention,
            )
            observed = load_rollout_for_comparison(condition_dir)
            result = compare_rollouts(
                baseline=baseline,
                observed=observed,
                condition_name=condition_name,
                layer_index=layer_index,
                bin_index=bin_index,
                mode=args.mode,
                scale=args.scale,
            )
            rows.append(result)
            pd.DataFrame(rows).to_csv(output_dir / "ablation_sweep_results.csv", index=False)
    finally:
        tracer.close()

    results = pd.DataFrame(rows)
    results.to_csv(output_dir / "ablation_sweep_results.csv", index=False)
    write_topk(results, output_dir)
    print(f"Saved ablation sweep to {output_dir}")


def run_condition(
    output_dir: Path,
    tracer: Pi0LiberoFullTokenRolloutTracer,
    collect_kwargs: dict[str, Any],
    intervention: ActivationIntervention | None,
) -> None:
    if (output_dir / "summary.json").exists():
        print(f"[skip] {output_dir} already has summary.json")
        return
    tracer.intervention = intervention
    tracer.collect(output_dir=output_dir, **collect_kwargs)


def parse_indices(spec: str, max_value: int, stride: int) -> list[int]:
    if spec == "all":
        return list(range(0, max_value, max(1, stride)))
    values = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))
    return sorted({value for value in values if 0 <= value < max_value})


def load_rollout_for_comparison(rollout_dir: Path) -> dict[str, Any]:
    with open(rollout_dir / "summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    steps = []
    for episode in summary.get("episodes", []):
        path = Path(episode["steps_jsonl"])
        if not path.exists():
            path = rollout_dir / f"episode_{int(episode['episode_index']):03d}" / "steps.jsonl"
        for row in read_jsonl(path):
            steps.append(row)
    return {"summary": summary, "steps": steps}


def compare_rollouts(
    baseline: dict[str, Any],
    observed: dict[str, Any],
    condition_name: str,
    layer_index: int,
    bin_index: int,
    mode: str,
    scale: float,
) -> dict[str, Any]:
    baseline_steps = {
        (int(row["episode_index"]), int(row["step"])): row
        for row in baseline["steps"]
        if row.get("policy_pred_action") is not None
    }
    deltas = []
    env_deltas = []
    gripper_deltas = []
    for row in observed["steps"]:
        key = (int(row["episode_index"]), int(row["step"]))
        base = baseline_steps.get(key)
        if base is None:
            continue
        deltas.append(vector_l2(row.get("policy_pred_action"), base.get("policy_pred_action")))
        env_deltas.append(vector_l2(row.get("action"), base.get("action")))
        gripper_deltas.append(vector_l2(row.get("gripper_position"), base.get("gripper_position")))

    baseline_success = float(baseline["summary"].get("success_rate", 0.0))
    observed_success = float(observed["summary"].get("success_rate", 0.0))
    action_delta = safe_mean(deltas)
    env_action_delta = safe_mean(env_deltas)
    gripper_delta = safe_mean(gripper_deltas)
    success_drop = baseline_success - observed_success
    causal_impact_score = action_delta + env_action_delta + gripper_delta + max(0.0, success_drop) * 10.0
    return {
        "condition": condition_name,
        "layer_index": int(layer_index),
        "token_bin_index": int(bin_index),
        "mode": mode,
        "scale": float(scale),
        "baseline_success_rate": baseline_success,
        "success_rate": observed_success,
        "success_drop": success_drop,
        "mean_policy_action_delta_l2": action_delta,
        "mean_env_action_delta_l2": env_action_delta,
        "mean_gripper_position_delta_l2": gripper_delta,
        "num_compared_steps": int(len(deltas)),
        "causal_impact_score": causal_impact_score,
    }


def vector_l2(left: Any, right: Any) -> float:
    if left is None or right is None:
        return float("nan")
    left_arr = np.asarray(left, dtype=np.float64).reshape(-1)
    right_arr = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_arr.shape != right_arr.shape:
        return float("nan")
    return float(np.linalg.norm(left_arr - right_arr))


def safe_mean(values: list[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return float(finite.mean()) if finite.size else float("nan")


def write_topk(results: pd.DataFrame, output_dir: Path) -> None:
    if results.empty:
        return
    ranked = results.sort_values("causal_impact_score", ascending=False)
    ranked.head(50).to_csv(output_dir / "ablation_top50_by_impact.csv", index=False)
    pivot = results.pivot_table(index="layer_index", columns="token_bin_index", values="causal_impact_score", aggfunc="mean")
    pivot.to_csv(output_dir / "ablation_impact_layer_by_bin.csv")


if __name__ == "__main__":
    main()
