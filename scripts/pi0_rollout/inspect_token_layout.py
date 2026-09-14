#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils import ensure_dir, load_config, resolve_path


KEYWORDS = (
    "cat",
    "concat",
    "stack",
    "where",
    "masked",
    "scatter",
    "embed",
    "image",
    "language",
    "token",
    "mask",
    "prefix",
    "suffix",
    "paligemma",
    "expert",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--pi0-path", default=None, help="Local LeRobot PI0 checkpoint directory or Hugging Face repo id.")
    parser.add_argument("--output-dir", default="outputs/pi0_token_layout_inspection")
    parser.add_argument("--max-source-chars", type=int, default=40000)
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config, ROOT))
    checkpoint_path = args.pi0_path or os.environ.get("VLA_PI0_PATH") or cfg.get("online_pi0", {}).get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("Provide `--pi0-path`, set VLA_PI0_PATH, or configure online_pi0.checkpoint_path.")

    output_dir = ensure_dir(resolve_path(args.output_dir, ROOT))
    policy = load_pi0_policy(str(checkpoint_path))
    report = inspect_policy(policy=policy, max_source_chars=int(args.max_source_chars))
    with open(output_dir / "pi0_token_layout_inspection.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    write_text_report(report, output_dir / "pi0_token_layout_inspection.txt")
    print(f"Saved PI0 token-layout inspection to {output_dir}")


def load_pi0_policy(checkpoint_path: str) -> Any:
    try:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.pi0 import PI0Policy
    except Exception as exc:
        raise RuntimeError("This script must run in the cluster environment with LeRobot PI0 installed.") from exc
    config = PreTrainedConfig.from_pretrained(checkpoint_path)
    config.compile_model = False
    return PI0Policy.from_pretrained(checkpoint_path, config=config).eval()


def inspect_policy(policy: Any, max_source_chars: int) -> dict[str, Any]:
    rows = []
    module_tree = []
    interesting_names = []
    for name, module in policy.named_modules():
        class_name = f"{type(module).__module__}.{type(module).__qualname__}"
        module_tree.append({"name": name, "class": class_name})
        lname = name.lower()
        cname = class_name.lower()
        if any(keyword in lname or keyword in cname for keyword in ("pi0", "paligemma", "expert", "gemma")):
            interesting_names.append(name)

    candidates = ["", "model", "model.paligemma_with_expert", "paligemma_with_expert"]
    candidates.extend(interesting_names)
    seen: set[int] = set()
    for name in candidates:
        module = get_named_module(policy, name)
        if module is None or id(module) in seen:
            continue
        seen.add(id(module))
        rows.append(source_report_for_module(name or "<policy>", module, max_source_chars=max_source_chars))

    keyword_hits = []
    for row in rows:
        for line_no, line in enumerate(row.get("source", "").splitlines(), start=1):
            lowered = line.lower()
            if any(keyword in lowered for keyword in KEYWORDS):
                keyword_hits.append(
                    {
                        "module": row["name"],
                        "class": row["class"],
                        "line": line_no,
                        "text": line.strip(),
                    }
                )
    return {
        "policy_class": f"{type(policy).__module__}.{type(policy).__qualname__}",
        "policy_file": inspect.getfile(type(policy)),
        "module_tree": module_tree,
        "source_reports": rows,
        "keyword_hits": keyword_hits,
    }


def get_named_module(policy: Any, name: str) -> Any | None:
    if not name:
        return policy
    current = policy
    for part in name.split("."):
        if not part:
            continue
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def source_report_for_module(name: str, module: Any, max_source_chars: int) -> dict[str, Any]:
    cls = type(module)
    report = {
        "name": name,
        "class": f"{cls.__module__}.{cls.__qualname__}",
        "file": safe_getfile(cls),
        "source": "",
        "source_error": None,
    }
    try:
        source = inspect.getsource(cls)
        report["source"] = source[:max_source_chars]
    except Exception as exc:
        report["source_error"] = repr(exc)
    return report


def safe_getfile(cls: type[Any]) -> str | None:
    try:
        return inspect.getfile(cls)
    except Exception:
        return None


def write_text_report(report: dict[str, Any], path: Path) -> None:
    lines = [
        f"Policy class: {report['policy_class']}",
        f"Policy file: {report['policy_file']}",
        "",
        "Keyword hits:",
    ]
    for hit in report["keyword_hits"]:
        lines.append(f"- {hit['module']}:{hit['line']} | {hit['text']}")
    lines.append("")
    lines.append("Source reports:")
    for row in report["source_reports"]:
        lines.append("=" * 100)
        lines.append(f"{row['name']} | {row['class']}")
        lines.append(f"file: {row['file']}")
        if row["source_error"]:
            lines.append(f"source_error: {row['source_error']}")
        else:
            lines.append(row["source"])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
