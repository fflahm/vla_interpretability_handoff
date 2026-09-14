#!/usr/bin/env python
"""Join one LIBERO episode/frame record and render a readable report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.libero_annotation_report import combined_record, render_markdown


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render one rich-annotation frame together with its episode metadata."
    )
    parser.add_argument(
        "annotation_dir",
        type=Path,
        help="Directory containing episodes.jsonl and frames.jsonl.",
    )
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument(
        "--sample-index",
        type=int,
        help="Zero-based row index in frames.jsonl.",
    )
    selector.add_argument(
        "--episode-id",
        help="Exact episode_id; requires --frame.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        help="Original trajectory frame index used with --episode-id.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Readable Markdown report or joined raw JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write to this file instead of stdout.",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Append the complete joined episode/frame JSON to a Markdown report.",
    )
    args = parser.parse_args()
    if args.episode_id is not None and args.frame is None:
        parser.error("--episode-id requires --frame")
    if args.sample_index is not None and args.frame is not None:
        parser.error("--frame is only valid with --episode-id")

    episode, frame = combined_record(
        args.annotation_dir,
        sample_index=args.sample_index,
        episode_id=args.episode_id,
        frame_index=args.frame,
    )
    if args.format == "json":
        rendered = json.dumps(
            {"episode": episode, "frame": frame},
            indent=2,
            ensure_ascii=False,
        ) + "\n"
    else:
        rendered = render_markdown(
            episode,
            frame,
            args.annotation_dir,
            include_raw=args.include_raw,
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Saved {args.format} report to {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
