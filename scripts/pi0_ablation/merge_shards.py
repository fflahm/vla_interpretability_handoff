#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils import ensure_dir, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge PI0 activation-ablation shard CSVs.")
    parser.add_argument("--input-dir", required=True, help="Parent directory containing shard_XX_of_NN subdirectories.")
    parser.add_argument("--output-dir", default=None, help="Defaults to --input-dir.")
    args = parser.parse_args()

    input_dir = resolve_path(args.input_dir, ROOT)
    output_dir = ensure_dir(resolve_path(args.output_dir, ROOT) if args.output_dir else input_dir)
    paths = sorted(input_dir.glob("shard_*_of_*/ablation_sweep_results.csv"))
    if not paths:
        direct = input_dir / "ablation_sweep_results.csv"
        if direct.exists():
            paths = [direct]
    if not paths:
        raise FileNotFoundError(f"No ablation_sweep_results.csv files found under {input_dir}.")

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_csv"] = str(path)
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    if {"layer_index", "token_bin_index", "mode", "scale"}.issubset(merged.columns):
        merged = merged.drop_duplicates(["layer_index", "token_bin_index", "mode", "scale"], keep="last")
        merged = merged.sort_values(["layer_index", "token_bin_index", "mode", "scale"])
    merged.to_csv(output_dir / "ablation_sweep_results_merged.csv", index=False)

    if "causal_impact_score" in merged.columns:
        ranked = merged.sort_values("causal_impact_score", ascending=False)
        ranked.head(50).to_csv(output_dir / "ablation_top50_by_impact_merged.csv", index=False)
        pivot = merged.pivot_table(index="layer_index", columns="token_bin_index", values="causal_impact_score", aggfunc="mean")
        pivot.to_csv(output_dir / "ablation_impact_layer_by_bin_merged.csv")
    print(f"Merged {len(paths)} shard CSVs into {output_dir}")


if __name__ == "__main__":
    main()
