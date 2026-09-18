#!/usr/bin/env python
"""Extract LIBERO occupancy GT and reusable RGB frames onto TOS.

Default: Libero-100 (libero_10 + libero_90), 20 interior frames per demo
(the same evenly-spaced subsample as scripts/occupancy/run_full.py).

Layout under --output-root (default /data/tos/guoshengyu/vla/occupancy):

  images/<suite>/<task>/<demo>/frame_XXXX_{agentview,wrist}.png
  gt/<suite>/<task>/<demo>/occupancy.npy      # float16 [F, 16, 16, 16]
  gt/<suite>/<task>/<demo>/samples.jsonl
  gt/<suite>/<task>/<demo>/metadata.json
  gt/STATUS.txt
  gt/manifest.json

Images are keyed by original HDF5 frame index so later jobs can reuse them
without occupancy sample_id. Each demo is flushed immediately (TOS-safe,
no rename) and skipped on rerun if occupancy.npy already has the right length.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.libero_rich_annotations import LOCAL_LIBERO_ROOT  # noqa: E402
from src.libero_self_occupancy import (  # noqa: E402
    OccupancyGridSpec,
    collect_self_occupancy_samples,
    evenly_spaced_indices,
    infer_libero_suite,
    save_npy_inplace,
    write_json,
    write_jsonl,
    _demo_sort_key,
)
from src.utils import ensure_dir, log  # noqa: E402

DEFAULT_OUTPUT = Path("/data/tos/guoshengyu/vla/occupancy")
DEFAULT_LIBERO_DATA = Path("/data/tos/guoshengyu/vla/libero")
DEFAULT_SUITES = ("libero_10", "libero_90")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def parse_shard(raw: str) -> tuple[int, int]:
    text = str(raw or "").strip()
    if not text:
        return 0, 1
    if "/" not in text:
        raise ValueError(f"--shard must look like INDEX/COUNT, got {raw!r}")
    index_s, count_s = text.split("/", 1)
    index, count = int(index_s), int(count_s)
    if count < 1 or not (0 <= index < count):
        raise ValueError(f"--shard INDEX must be in 0..COUNT-1, got {raw!r}")
    return index, count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero-data", type=Path, default=DEFAULT_LIBERO_DATA)
    parser.add_argument("--libero-root", type=Path, default=LOCAL_LIBERO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--suites", default=",".join(DEFAULT_SUITES))
    parser.add_argument(
        "--include-tasks",
        default="",
        help="Comma-separated suite, task name, or suite/task substrings. Empty = all.",
    )
    parser.add_argument(
        "--exclude-tasks",
        default="",
        help="Comma-separated suite/task substrings to skip (e.g. a task the 开发机 job is writing).",
    )
    parser.add_argument(
        "--shard",
        default="",
        help="INDEX/COUNT over the filtered task list, e.g. 0/4. Empty = no shard.",
    )
    parser.add_argument(
        "--status-suffix",
        default="",
        help="Write STATUS.<suffix>.txt instead of STATUS.txt so parallel jobs do not clobber.",
    )
    parser.add_argument(
        "--compare-ref",
        type=Path,
        default=None,
        help="After extract, require exactly one occupancy.npy and array_equal this float16 file.",
    )
    parser.add_argument("--frames-per-demo", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--max-tasks", type=int, default=0, help="0 = all tasks in the suites.")
    parser.add_argument("--max-demos", type=int, default=0, help="0 = all demos in each hdf5.")
    parser.add_argument("--smoke", action="store_true", help="1 task, 1 demo, still 20 frames.")
    parser.add_argument("--status", action="store_true")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Scan TOS occupancy.npy vs hdf5 demos: count, shape, dtype, missing tasks.",
    )
    parser.add_argument("--reuse-images", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def image_root(output_root: Path) -> Path:
    return output_root / "images"


def gt_root(output_root: Path) -> Path:
    return output_root / "gt"


def demo_gt_dir(output_root: Path, suite: str, task: str, demo_key: str) -> Path:
    return gt_root(output_root) / suite / task / demo_key


def task_matches(suite: str, hdf5_path: Path, tokens: list[str]) -> bool:
    if not tokens:
        return True
    task = hdf5_path.name.removesuffix("_demo.hdf5")
    haystacks = (suite, task, f"{suite}/{task}")
    return any(token in hay for hay in haystacks for token in tokens)


def list_tasks(
    libero_data: Path,
    suites: list[str],
    max_tasks: int,
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
) -> list[tuple[str, Path]]:
    tasks: list[tuple[str, Path]] = []
    include = include or []
    exclude = exclude or []
    for suite in suites:
        folder = libero_data / suite
        files = sorted(folder.glob("*_demo.hdf5"))
        if not files:
            raise FileNotFoundError(f"No *_demo.hdf5 under {folder}")
        tasks.extend((suite, path) for path in files)
    if include:
        tasks = [(suite, path) for suite, path in tasks if task_matches(suite, path, include)]
        if not tasks:
            raise FileNotFoundError(f"No hdf5 matched --include-tasks {include}")
    if exclude:
        tasks = [(suite, path) for suite, path in tasks if not task_matches(suite, path, exclude)]
    if shard_count > 1:
        tasks = [item for idx, item in enumerate(tasks) if idx % shard_count == shard_index]
    if max_tasks > 0:
        tasks = tasks[:max_tasks]
    return tasks


def status_paths(output_root: Path, suffix: str = "") -> tuple[Path, Path, Path]:
    root = gt_root(output_root)
    if suffix:
        return (
            root / f"STATUS.{suffix}.txt",
            root / f"status.{suffix}.json",
            root / f"manifest.{suffix}.json",
        )
    return root / "STATUS.txt", root / "status.json", root / "manifest.json"


def demo_complete(output_root: Path, suite: str, task: str, demo_key: str, n_frames: int) -> bool:
    occupancy_path = demo_gt_dir(output_root, suite, task, demo_key) / "occupancy.npy"
    samples_path = demo_gt_dir(output_root, suite, task, demo_key) / "samples.jsonl"
    if not occupancy_path.exists() or not samples_path.exists():
        return False
    try:
        packed = np.load(occupancy_path)
    except Exception:
        return False
    return int(packed.shape[0]) == int(n_frames) and packed.shape[1:] == (16, 16, 16)


def write_status(output_root: Path, payload: dict, suffix: str = "") -> None:
    payload = dict(payload)
    payload["updated_at"] = utc_now()
    if suffix:
        payload["status_suffix"] = suffix
    txt_path, json_path, _ = status_paths(output_root, suffix)
    ensure_dir(txt_path.parent)
    lines = [
        f"updated_at: {payload['updated_at']}",
        f"phase:      {payload.get('phase', '')}",
        f"suite:      {payload.get('suite', '')}",
        f"task:       {payload.get('task', '')}",
        f"demo:       {payload.get('demo', '')}",
        f"progress:   {payload.get('done_demos', 0)}/{payload.get('total_demos', 0)} demos",
        f"frames:     {payload.get('done_frames', 0)}",
        f"last_error: {payload.get('last_error', '')}",
        f"suffix:     {suffix}",
        "",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def print_status(output_root: Path, suffix: str = "") -> int:
    root = gt_root(output_root)
    print("---- live occupancy.npy (source of truth) ----")
    total_npy = 0
    if not root.exists():
        print("  (gt root missing)")
    else:
        for suite_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("libero_")):
            npy = list(suite_dir.glob("*/*/occupancy.npy"))
            total_npy += len(npy)
            print(f"  {suite_dir.name}: {len(npy)} demos")
        print(f"  TOTAL: {total_npy} / 5000 demos")
    shard_files = sorted(root.glob("status.rjob*.json")) if root.exists() else []
    if shard_files:
        print("---- rjob shard status (STATUS.rjob*.json) ----")
        extracting = 0
        for path in shard_files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            phase = payload.get("phase", "")
            if phase == "extract":
                extracting += 1
            task = str(payload.get("task") or "")[:48]
            print(
                f"  {path.name}: {phase} "
                f"{payload.get('done_demos', 0)}/{payload.get('total_demos', 0)} "
                f"{payload.get('suite', '')}/{task} {payload.get('demo', '')} "
                f"{payload.get('updated_at', '')}"
            )
        print(f"  shards_extracting: {extracting}/{len(shard_files)}")
    status, _, _ = status_paths(output_root, suffix)
    print("---- STATUS.txt (开发机 snapshot; rjob does not update this) ----")
    if status.exists():
        sys.stdout.write(status.read_text(encoding="utf-8"))
    else:
        print(f"  no {status.name}")
    return 0


def print_verify(output_root: Path, libero_data: Path, frames_per_demo: int, grid_size: int) -> int:
    """Count live TOS GT files and check occupancy.npy integrity against hdf5 demos."""
    expected: list[tuple[str, str, str]] = []
    for suite in DEFAULT_SUITES:
        hdf5_dir = libero_data / suite
        if not hdf5_dir.is_dir():
            print(f"[error] missing hdf5 dir {hdf5_dir}")
            return 2
        for hdf5 in sorted(hdf5_dir.glob("*.hdf5")):
            task = hdf5.name.removesuffix("_demo.hdf5")
            for demo_i in range(50):
                expected.append((suite, task, f"demo_{demo_i}"))
    print(f"---- expected from hdf5 {libero_data} ----")
    print(f"  tasks={len(expected) // 50} planned_demos={len(expected)}")
    ok = 0
    missing = []
    bad = []
    extra_note = 0
    seen = set()
    root = gt_root(output_root)
    print(f"---- scanning {root} ----")
    for suite, task, demo_key in expected:
        dest = demo_gt_dir(output_root, suite, task, demo_key)
        npy = dest / "occupancy.npy"
        samples = dest / "samples.jsonl"
        seen.add((suite, task, demo_key))
        if not npy.exists():
            missing.append(f"{suite}/{task}/{demo_key}")
            continue
        try:
            packed = np.load(npy, mmap_mode="r")
            shape_ok = tuple(packed.shape) == (frames_per_demo, grid_size, grid_size, grid_size)
            dtype_ok = packed.dtype == np.float16
            samples_ok = samples.exists() and samples.stat().st_size > 0
            if shape_ok and dtype_ok and samples_ok:
                ok += 1
            else:
                bad.append(
                    f"{suite}/{task}/{demo_key} shape={tuple(packed.shape)} "
                    f"dtype={packed.dtype} samples.jsonl={samples.exists()}"
                )
        except Exception as exc:
            bad.append(f"{suite}/{task}/{demo_key} load_error={exc}")
    if root.exists():
        for npy in root.glob("libero_*/*/*/occupancy.npy"):
            key = (npy.parts[-4], npy.parts[-3], npy.parts[-2])
            if key not in seen:
                extra_note += 1
    missing_tasks: dict[str, int] = {}
    for item in missing:
        task = "/".join(item.split("/")[:2])
        missing_tasks[task] = missing_tasks.get(task, 0) + 1
    print(f"  intact occupancy.npy: {ok}/{len(expected)}")
    print(f"  missing: {len(missing)}")
    print(f"  corrupt/incomplete: {len(bad)}")
    print(f"  extra npy not in hdf5 list: {extra_note}")
    if missing_tasks:
        print("---- missing by task ----")
        for task, count in sorted(missing_tasks.items()):
            print(f"  {task}: {count}")
    if bad:
        print("---- bad files (first 20) ----")
        for line in bad[:20]:
            print(f"  {line}")
    print("---- verdict ----")
    if ok == len(expected) and not bad:
        print("  ALL_OK")
        return 0
    print("  INCOMPLETE")
    return 1


def relative_under(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def save_demo_gt(
    output_root: Path,
    samples: list,
    episode: dict,
    occupancy_meta: dict,
) -> None:
    first = samples[0]
    dest = demo_gt_dir(output_root, first.suite, first.task, first.demo_key)
    occupancies = np.stack([sample.occupancy for sample in samples], axis=0).astype(np.float16)
    save_npy_inplace(dest / "occupancy.npy", occupancies)
    rows = []
    for sample in samples:
        row = sample.metadata()
        row["image_rel"] = relative_under(Path(sample.image_path), output_root)
        row["wrist_image_rel"] = relative_under(Path(sample.wrist_image_path), output_root)
        rows.append(row)
    write_jsonl(dest / "samples.jsonl", rows)
    write_json(
        dest / "metadata.json",
        {
            "suite": first.suite,
            "task": first.task,
            "demo_key": first.demo_key,
            "frames": episode["frames"],
            "num_samples": len(samples),
            "occupancy_shape": list(occupancies.shape),
            "grid": occupancy_meta,
            "seconds": episode["seconds"],
        },
    )


def count_planned_demos(tasks: list[tuple[str, Path]], max_demos: int) -> int:
    # Official LIBERO hdf5s have 50 demos. Avoid opening every file on TOS at start.
    per_task = 50 if max_demos <= 0 else max_demos
    return per_task * len(tasks)


def compare_occupancy(got_path: Path, ref_path: Path) -> dict:
    got = np.load(got_path)
    ref = np.load(ref_path)
    payload = {
        "got": str(got_path),
        "ref": str(ref_path),
        "got_shape": list(got.shape),
        "ref_shape": list(ref.shape),
        "got_dtype": str(got.dtype),
        "ref_dtype": str(ref.dtype),
        "equal": bool(got.shape == ref.shape and np.array_equal(got, ref)),
    }
    if got.shape == ref.shape:
        delta = np.abs(got.astype(np.float32) - ref.astype(np.float32))
        payload["max_abs_diff"] = float(delta.max()) if delta.size else 0.0
        payload["mean_abs_diff"] = float(delta.mean()) if delta.size else 0.0
    else:
        payload["max_abs_diff"] = None
        payload["mean_abs_diff"] = None
    return payload


def main() -> int:
    args = parse_args()
    suffix = str(args.status_suffix or "").strip()
    if args.status:
        return print_status(args.output_root, suffix)
    if args.verify:
        return print_verify(
            args.output_root, args.libero_data, args.frames_per_demo, args.grid_size
        )
    if args.grid_size != 16:
        raise ValueError("Occupancy GT is fixed at 16^3 to match run_full.py.")
    suites = [item.strip() for item in args.suites.split(",") if item.strip()]
    max_tasks = 1 if args.smoke else args.max_tasks
    max_demos = 1 if args.smoke else args.max_demos
    frames_per_demo = args.frames_per_demo
    output_root = args.output_root
    shard_index, shard_count = parse_shard(args.shard)
    include = parse_csv(args.include_tasks)
    exclude = parse_csv(args.exclude_tasks)
    ensure_dir(image_root(output_root))
    ensure_dir(gt_root(output_root))
    tasks = list_tasks(
        args.libero_data,
        suites,
        max_tasks,
        include=include,
        exclude=exclude,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    spec = OccupancyGridSpec(size=args.grid_size, supersample=args.supersample)
    planned_demos = count_planned_demos(tasks, max_demos)
    done_demos = 0
    done_frames = 0
    skipped_demos = 0
    payload = {
        "phase": "smoke" if args.smoke else "extract",
        "suite": "",
        "task": "",
        "demo": "",
        "done_demos": 0,
        "total_demos": planned_demos,
        "done_frames": 0,
        "last_error": "",
        "output_root": str(output_root),
        "frames_per_demo": frames_per_demo,
        "include_tasks": include,
        "exclude_tasks": exclude,
        "shard": args.shard,
    }
    write_status(output_root, payload, suffix)
    started = time.perf_counter()
    log(
        f"Libero occupancy GT suites={suites} tasks={len(tasks)} "
        f"planned_demos={planned_demos} frames_per_demo={frames_per_demo} "
        f"smoke={args.smoke} shard={args.shard or 'none'} "
        f"include={include or ['*']} exclude={exclude or []} out={output_root}"
    )
    for suite, hdf5_path in tasks:
        task = hdf5_path.name.removesuffix("_demo.hdf5")
        with h5py.File(hdf5_path, "r") as handle:
            demo_keys = sorted(handle["data"].keys(), key=_demo_sort_key)
            if max_demos > 0:
                demo_keys = demo_keys[:max_demos]
            frame_counts = {
                demo_key: len(
                    evenly_spaced_indices(int(handle["data"][demo_key]["states"].shape[0]), frames_per_demo)
                )
                for demo_key in demo_keys
            }
        skip = {
            demo_key
            for demo_key in demo_keys
            if demo_complete(output_root, suite, task, demo_key, frame_counts[demo_key])
        }
        skipped_demos += len(skip)
        done_demos += len(skip)
        for demo_key in skip:
            done_frames += frame_counts[demo_key]
        payload.update(
            suite=suite, task=task, demo="", done_demos=done_demos, done_frames=done_frames
        )
        write_status(output_root, payload, suffix)
        remaining = [demo_key for demo_key in demo_keys if demo_key not in skip]
        if not remaining:
            log(f"skip complete task {suite}/{task} demos={len(skip)}")
            continue

        def on_demo(demo_samples, episode, _suite=suite, _task=task) -> None:
            nonlocal done_demos, done_frames
            save_demo_gt(output_root, demo_samples, episode, episode.get("occupancy") or spec.to_dict())
            done_demos += 1
            done_frames += len(demo_samples)
            payload.update(
                suite=_suite,
                task=_task,
                demo=demo_samples[0].demo_key,
                done_demos=done_demos,
                done_frames=done_frames,
                last_error="",
            )
            write_status(output_root, payload, suffix)

        collect_self_occupancy_samples(
            hdf5_path=hdf5_path,
            output_dir=gt_root(output_root) / suite / task,
            num_demos=len(demo_keys) if max_demos <= 0 else max_demos,
            frames_per_demo=frames_per_demo,
            spec=spec,
            libero_root=args.libero_root,
            suite=infer_libero_suite(hdf5_path, suite),
            image_root=image_root(output_root),
            reuse_images=args.reuse_images,
            skip_demo_keys=skip,
            on_demo=on_demo,
        )

    manifest = {
        "status": "completed",
        "suites": suites,
        "tasks": len(tasks),
        "frames_per_demo": frames_per_demo,
        "grid_size": args.grid_size,
        "supersample": args.supersample,
        "smoke": bool(args.smoke),
        "done_demos": done_demos,
        "skipped_demos": skipped_demos,
        "done_frames": done_frames,
        "seconds": time.perf_counter() - started,
        "images": str(image_root(output_root)),
        "gt": str(gt_root(output_root)),
        "shard": args.shard,
        "include_tasks": include,
        "exclude_tasks": exclude,
        "status_suffix": suffix,
    }
    _, _, manifest_path = status_paths(output_root, suffix)
    write_json(manifest_path, manifest)
    payload.update(phase="done", demo="", last_error="")
    write_status(output_root, payload, suffix)
    log(f"GT extract done demos={done_demos} frames={done_frames} seconds={manifest['seconds']:.1f}")
    if args.compare_ref is not None:
        npys = sorted(gt_root(output_root).glob("*/*/*/occupancy.npy"))
        if len(npys) != 1:
            raise RuntimeError(
                f"--compare-ref expects exactly one occupancy.npy under {gt_root(output_root)}, got {len(npys)}"
            )
        comparison = compare_occupancy(npys[0], args.compare_ref)
        manifest["compare"] = comparison
        write_json(manifest_path, manifest)
        print(json.dumps({"compare": comparison}, indent=2), flush=True)
        if not comparison["equal"]:
            raise RuntimeError(
                f"occupancy mismatch vs ref max_abs_diff={comparison.get('max_abs_diff')}"
            )
        log(f"GT compare OK equal=True vs {args.compare_ref}")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr, flush=True)
        raise
