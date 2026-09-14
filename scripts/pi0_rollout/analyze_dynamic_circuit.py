#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.pi0_dynamic_circuit import DEFAULT_DYNAMIC_TARGETS, analyze_pi0_dynamic_circuit
from src.utils import ensure_dir, load_config, resolve_path, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--rollout-dir", default=None)
    parser.add_argument("--rollout-root", default=None, help="Analyze every matching rollout directory under this root.")
    parser.add_argument("--glob", default="pi0_libero_spatial_task1_full_tokens*", help="Directory glob used with --rollout-root.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-root", default=None, help="Root for multi-directory analysis outputs.")
    parser.add_argument("--timestamp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-name", default=None, help="Optional run folder name; defaults to a timestamped name.")
    parser.add_argument("--targets", nargs="+", default=list(DEFAULT_DYNAMIC_TARGETS))
    parser.add_argument("--pooling", choices=("mean", "last", "flatten", "token_norms"), default="mean")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--top-k-layers", type=int, default=6)
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config, ROOT))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    set_seed(seed)
    alpha = float(args.alpha if args.alpha is not None else cfg["probe"].get("alpha", 1.0))
    test_size = float(args.test_size if args.test_size is not None else cfg["probe"].get("test_size", 0.2))
    run_name = args.run_name or make_run_name(args.pooling, seed)
    if args.rollout_root:
        rollout_root = resolve_path(args.rollout_root, ROOT)
        rollout_dirs = discover_rollout_dirs(rollout_root, args.glob)
        if not rollout_dirs:
            raise FileNotFoundError(f"No rollout directories matched `{args.glob}` under {rollout_root}.")
        output_root = resolve_path(args.output_root or str(rollout_root).replace("VLA-Probe", "VLA-Probe-Analysis"), ROOT)
        if args.timestamp:
            output_root = output_root / "runs" / run_name
        rows = []
        for rollout_dir in rollout_dirs:
            output_dir = output_root / rollout_dir.name
            nodes, edges, _ = analyze_one(
                rollout_dir=rollout_dir,
                output_dir=output_dir,
                targets=tuple(args.targets),
                pooling=args.pooling,
                alpha=alpha,
                test_size=test_size,
                seed=seed,
                top_k_layers=int(args.top_k_layers),
            )
            rows.append(
                {
                    "rollout_dir": str(rollout_dir),
                    "output_dir": str(output_dir),
                    "num_nodes": int(len(nodes)),
                    "num_edges": int(len(edges)),
                }
            )
        ensure_dir(output_root)
        write_multi_summary(output_root / "pi0_dynamic_multi_run_summary.csv", rows)
        write_run_info(
            output_root / "pi0_dynamic_run_info.json",
            run_name=run_name,
            timestamp_enabled=bool(args.timestamp),
            pooling=args.pooling,
            seed=seed,
            targets=list(args.targets),
            rollout_root=str(rollout_root),
            glob_pattern=args.glob,
        )
        print(f"Analyzed {len(rows)} rollout directories under {rollout_root}")
        print(f"Saved multi-run summary to {output_root / 'pi0_dynamic_multi_run_summary.csv'}")
    else:
        rollout_dir = resolve_path(args.rollout_dir or cfg["online_pi0"]["output_dir"], ROOT)
        default_output = Path(str(rollout_dir).replace("VLA-Probe", "VLA-Probe-Analysis"))
        output_dir = resolve_path(args.output_dir or default_output, ROOT)
        if args.timestamp:
            output_dir = output_dir / "runs" / run_name
        nodes, edges, _ = analyze_one(
            rollout_dir=rollout_dir,
            output_dir=output_dir,
            targets=tuple(args.targets),
            pooling=args.pooling,
            alpha=alpha,
            test_size=test_size,
            seed=seed,
            top_k_layers=int(args.top_k_layers),
        )
        print(f"Saved PI0 dynamic circuit analysis to {output_dir}")
        print(f"Candidate nodes: {len(nodes)}")
        print(f"Candidate edges: {len(edges)}")
        write_run_info(
            output_dir / "pi0_dynamic_run_info.json",
            run_name=run_name,
            timestamp_enabled=bool(args.timestamp),
            pooling=args.pooling,
            seed=seed,
            targets=list(args.targets),
            rollout_dir=str(rollout_dir),
        )


def analyze_one(
    *,
    rollout_dir: Path,
    output_dir: Path,
    targets: tuple[str, ...],
    pooling: str,
    alpha: float,
    test_size: float,
    seed: int,
    top_k_layers: int,
):
    print(f"Analyzing rollout directory: {rollout_dir}", flush=True)
    return analyze_pi0_dynamic_circuit(
        rollout_dir=rollout_dir,
        output_dir=output_dir,
        targets=targets,
        pooling=pooling,
        alpha=alpha,
        test_size=test_size,
        seed=seed,
        top_k_layers=top_k_layers,
    )


def discover_rollout_dirs(root: Path, pattern: str) -> list[Path]:
    candidates = sorted(path for path in root.glob(pattern) if path.is_dir())
    return [path for path in candidates if (path / "summary.json").exists() or any(path.glob("episode_*/steps.jsonl"))]


def write_multi_summary(path: Path, rows: list[dict[str, object]]) -> None:
    import csv

    ensure_dir(path.parent)
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_run_name(pooling: str, seed: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_pool-{pooling}_seed-{seed}"


def write_run_info(path: Path, **payload: object) -> None:
    import json

    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
