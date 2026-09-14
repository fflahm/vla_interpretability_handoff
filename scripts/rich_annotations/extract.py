#!/usr/bin/env python
"""Extract compact, auditable rich annotations from LIBERO demonstrations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.libero_rich_annotations import extract_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Balanced LIBERO demo/frame sampling with RGB export and JSONL annotations."
    )
    parser.add_argument("input", type=Path, help="A suite directory or root containing suite directories.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-demos", type=int, default=100)
    parser.add_argument("--frames-per-demo", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suite", dest="suites", action="append", default=[],
                        help="Suite directory name to include; repeat for multiple suites.")
    parser.add_argument("--simulator", choices=("auto", "required", "off"), default="auto")
    parser.add_argument("--libero-root", type=Path, default=Path("/home/eai/mars/simulator/LIBERO"),
                        help="LIBERO checkout containing bddl_files and assets.")
    args = parser.parse_args()
    if args.frames_per_demo < 1:
        parser.error("--frames-per-demo must be positive")
    summary = extract_dataset(
        args.input, args.output_dir, args.num_demos, args.frames_per_demo,
        args.seed, args.suites, args.simulator, args.libero_root,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
