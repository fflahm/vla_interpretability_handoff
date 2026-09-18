#!/usr/bin/env python
"""Extract PI0.5 token-bin activations for a Libero-90 occupancy split.

Reuses occupancy GT images and the same 20 evenly-spaced frames. Does not copy
occupancy.npy or RGB into the activation tree. Train/Test demos are inferred;
Ablation is recorded in split.json only.

Default token bins: consecutive 10-token means
  paligemma 968 -> 97 bins, expert 50 -> 5 bins (102 token-position bins / layer).
Default expert flow times: 5 uniform Euler times from the 10-step sampler.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.libero_self_occupancy import (  # noqa: E402
    SelfOccupancySample,
    parse_index_spec,
    save_npy_inplace,
    uniform_euler_flow_times,
    write_json,
    write_jsonl,
)
from src.pi05_occupancy_full import Pi05EulerCapture, UNIFORM_FLOW_TIMES  # noqa: E402
from src.utils import ensure_dir, log, set_seed  # noqa: E402

DEFAULT_OCCUPANCY = Path("/data/tos/guoshengyu/vla/occupancy")
DEFAULT_OUTPUT = Path("/data/tos/guoshengyu/vla/occupancy_activations")
DEFAULT_MODEL = Path("/data/tos/guoshengyu/vla/models/pi05_libero")
EXPECTED_PALI_TOKENS = 968
EXPECTED_EXPERT_TOKENS = 50
FRAMES_PER_DEMO = 20
DEMOS_PER_TASK = 50
TRAIN_N, TEST_N, ABLATION_N = 30, 10, 10


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
    parser.add_argument("--occupancy-root", type=Path, default=DEFAULT_OCCUPANCY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pi05-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--suite", default="libero_90")
    parser.add_argument("--num-tasks", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokens-per-bin", type=int, default=10)
    parser.add_argument("--flow-times", default="", help="Comma floats. Empty = 5 uniform Euler times.")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--frames-per-demo", type=int, default=FRAMES_PER_DEMO)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--shard", default="")
    parser.add_argument("--include-tasks", default="")
    parser.add_argument("--status-suffix", default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def gt_demo_dir(occupancy_root: Path, suite: str, task: str, demo_key: str) -> Path:
    return occupancy_root / "gt" / suite / task / demo_key


def act_demo_dir(output_root: Path, suite: str, task: str, demo_key: str) -> Path:
    return output_root / "activations" / suite / task / demo_key


def demo_complete_gt(occupancy_root: Path, suite: str, task: str, demo_key: str, frames: int) -> bool:
    npy = gt_demo_dir(occupancy_root, suite, task, demo_key) / "occupancy.npy"
    samples = gt_demo_dir(occupancy_root, suite, task, demo_key) / "samples.jsonl"
    if not npy.exists() or not samples.exists():
        return False
    try:
        packed = np.load(npy, mmap_mode="r")
        return tuple(packed.shape) == (frames, 16, 16, 16) and packed.dtype == np.float16
    except Exception:
        return False


def list_complete_tasks(
    occupancy_root: Path,
    suite: str,
    frames: int,
    include: list[str] | None = None,
) -> list[str]:
    suite_dir = occupancy_root / "gt" / suite
    if not suite_dir.is_dir():
        return []
    tasks = []
    for task_dir in sorted(p for p in suite_dir.iterdir() if p.is_dir()):
        name = task_dir.name
        if include and not any(token in name or token in f"{suite}/{name}" for token in include):
            continue
        ok = sum(
            1
            for i in range(DEMOS_PER_TASK)
            if demo_complete_gt(occupancy_root, suite, name, f"demo_{i}", frames)
        )
        if ok == DEMOS_PER_TASK:
            tasks.append(name)
    return tasks


def load_demo_rows(occupancy_root: Path, suite: str, task: str, demo_key: str) -> list[dict]:
    path = gt_demo_dir(occupancy_root, suite, task, demo_key) / "samples.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        image = Path(row["image_path"])
        wrist = Path(row["wrist_image_path"])
        if not image.exists() and row.get("image_rel"):
            row["image_path"] = str(occupancy_root / row["image_rel"])
        if not wrist.exists() and row.get("wrist_image_rel"):
            row["wrist_image_path"] = str(occupancy_root / row["wrist_image_rel"])
    return rows


def sample_from_row(row: dict, sample_id: int) -> SelfOccupancySample:
    return SelfOccupancySample(
        sample_id=sample_id,
        demo_key=row["demo_key"],
        frame_index=int(row["frame_index"]),
        image_path=row["image_path"],
        wrist_image_path=row["wrist_image_path"],
        instruction=row["instruction"],
        observation_state=row["observation_state"],
        occupancy=np.zeros((16, 16, 16), dtype=np.float16),
        suite=row.get("suite", ""),
        task=row.get("task", ""),
    )


def activation_paths(dest: Path) -> dict[str, Path]:
    return {
        "paligemma": dest / "paligemma.npy",
        "expert": dest / "expert.npy",
        "sample_ids": dest / "sample_ids.npy",
        "layout": dest / "layout.json",
    }


def pack_demo_arrays(rows_captured: list[dict[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Stack per-frame capture dicts into TOS-safe arrays.

    paligemma: [layer, frame, bin, hidden]
    expert:    [layer, time, frame, bin, hidden]
    """
    first = rows_captured[0]
    pali_keys = sorted(k for k in first if k.startswith("paligemma/"))
    expert_keys = sorted(k for k in first if k.startswith("expert/"))
    times = sorted({k.split("/t=")[1] for k in expert_keys}, key=lambda t: -float(t))
    layers_e = sorted({k.split("/")[1] for k in expert_keys})
    pali = np.stack(
        [np.stack([frame[key] for frame in rows_captured], axis=0) for key in pali_keys],
        axis=0,
    ).astype(np.float16)
    expert = []
    for layer in layers_e:
        per_time = []
        for t in times:
            key = f"expert/{layer}/t={t}"
            per_time.append(np.stack([frame[key] for frame in rows_captured], axis=0))
        expert.append(np.stack(per_time, axis=0))
    expert_arr = np.stack(expert, axis=0).astype(np.float16)
    layout = {
        "format": "paligemma [layer, frame, bin, hidden]; expert [layer, time, frame, bin, hidden]",
        "paligemma_keys": pali_keys,
        "expert_layers": layers_e,
        "flow_times": [float(t) for t in times],
        "dtype": "float16",
    }
    return pali, expert_arr, layout


def npy_min_bytes(shape: tuple[int, ...], itemsize: int = 2) -> int:
    n = 1
    for dim in shape:
        n *= int(dim)
    return n * itemsize + 64


def activation_complete(
    dest: Path,
    frames: int,
    *,
    pali_bins: int,
    expert_bins: int,
    n_flow: int,
    pali_layers: int = 18,
    expert_layers: int = 18,
    pali_hidden: int = 2048,
    expert_hidden: int = 1024,
) -> bool:
    """Cheap TOS-safe completeness check. Avoids mmap on the resume skip path."""
    paths = activation_paths(dest)
    meta = dest / "metadata.json"
    required = [*paths.values(), meta]
    try:
        if not all(p.exists() and p.stat().st_size > 0 for p in required):
            return False
        recorded = int(json.loads(meta.read_text(encoding="utf-8")).get("frames", -1))
        if recorded != frames:
            return False
        pali_ok = paths["paligemma"].stat().st_size >= npy_min_bytes(
            (pali_layers, frames, pali_bins, pali_hidden)
        )
        expert_ok = paths["expert"].stat().st_size >= npy_min_bytes(
            (expert_layers, n_flow, frames, expert_bins, expert_hidden)
        )
        ids_ok = paths["sample_ids"].stat().st_size >= npy_min_bytes((frames,), itemsize=4)
        return pali_ok and expert_ok and ids_ok
    except Exception:
        return False


def status_paths(output_root: Path, suffix: str) -> tuple[Path, Path]:
    name = f"STATUS.{suffix}.txt" if suffix else "STATUS.txt"
    json_name = f"status.{suffix}.json" if suffix else "status.json"
    return output_root / name, output_root / json_name


def write_status(output_root: Path, payload: dict, suffix: str) -> None:
    payload = dict(payload)
    payload["updated_at"] = utc_now()
    txt_path, json_path = status_paths(output_root, suffix)
    ensure_dir(txt_path.parent)
    lines = [
        f"updated_at: {payload['updated_at']}",
        f"phase:      {payload.get('phase', '')}",
        f"task:       {payload.get('task', '')}",
        f"demo:       {payload.get('demo', '')}",
        f"progress:   {payload.get('done_demos', 0)}/{payload.get('total_demos', 0)} demos",
        f"frames:     {payload.get('done_frames', 0)}",
        f"last_error: {payload.get('last_error', '')}",
        "",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_or_load_split(
    output_root: Path,
    occupancy_root: Path,
    *,
    suite: str,
    num_tasks: int,
    seed: int,
    frames: int,
    include: list[str],
    smoke: bool,
) -> dict:
    split_path = output_root / "split.json"
    if split_path.exists():
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        log(f"Reusing split.json seed={payload.get('seed')} tasks={len(payload.get('tasks', []))}")
        return payload
    pool = list_complete_tasks(occupancy_root, suite, frames, include=include or None)
    complete = list(pool)
    if not complete:
        raise FileNotFoundError(
            f"No complete {suite} GT tasks under {occupancy_root / 'gt' / suite} "
            f"(need {DEMOS_PER_TASK} occupancy.npy each)."
        )
    rng = np.random.default_rng(seed)
    if smoke:
        selected = [complete[0]]
    else:
        if len(complete) < num_tasks:
            raise ValueError(f"Need {num_tasks} complete tasks, found {len(complete)}.")
        chosen = rng.choice(len(complete), size=num_tasks, replace=False)
        selected = [complete[int(i)] for i in sorted(chosen.tolist())]
    tasks = []
    for task in selected:
        perm = rng.permutation(DEMOS_PER_TASK)
        train = [f"demo_{int(i)}" for i in perm[:TRAIN_N]]
        test = [f"demo_{int(i)}" for i in perm[TRAIN_N : TRAIN_N + TEST_N]]
        ablation = [f"demo_{int(i)}" for i in perm[TRAIN_N + TEST_N :]]
        if smoke:
            train, test, ablation = [train[0]], [], []
        tasks.append(
            {
                "suite": suite,
                "task": task,
                "train": sorted(train, key=lambda k: int(k.split("_")[1])),
                "test": sorted(test, key=lambda k: int(k.split("_")[1])),
                "ablation": sorted(ablation, key=lambda k: int(k.split("_")[1])),
            }
        )
    payload = {
        "seed": seed,
        "suite": suite,
        "occupancy_root": str(occupancy_root),
        "complete_task_pool": len(pool),
        "num_tasks_requested": 1 if smoke else num_tasks,
        "split": {"train": TRAIN_N, "test": TEST_N, "ablation": ABLATION_N},
        "frames_per_demo": frames,
        "smoke": smoke,
        "note": (
            "Train and Test are inferred. Ablation demos are recorded only. "
            "Images/GT stay in occupancy_root; this tree stores activations."
        ),
        "tasks": tasks,
    }
    write_json(split_path, payload)
    log(f"Wrote split.json seed={seed} tasks={len(tasks)} smoke={smoke}")
    return payload


def infer_jobs(split: dict, shard_index: int, shard_count: int) -> list[dict]:
    jobs = []
    for task in split["tasks"]:
        for split_name in ("train", "test"):
            for demo_key in task[split_name]:
                jobs.append(
                    {
                        "suite": task["suite"],
                        "task": task["task"],
                        "demo_key": demo_key,
                        "split": split_name,
                    }
                )
    return jobs[shard_index::shard_count]


def parse_layers(value: str) -> set[int] | None:
    if value == "all":
        return None
    return set(parse_index_spec(value, max_value=10**9))


def parse_flow_times(raw: str) -> tuple[float, ...]:
    if not str(raw or "").strip():
        return UNIFORM_FLOW_TIMES
    return tuple(float(item.strip()) for item in raw.split(",") if item.strip())


def print_status(output_root: Path) -> int:
    split_path = output_root / "split.json"
    print(f"---- {split_path} ----")
    if not split_path.exists():
        print("  missing split.json")
        return 1
    split = json.loads(split_path.read_text(encoding="utf-8"))
    print(f"  seed={split.get('seed')} suite={split.get('suite')} tasks={len(split.get('tasks', []))}")
    planned = 0
    done = 0
    for task in split["tasks"]:
        for split_name in ("train", "test"):
            for demo_key in task[split_name]:
                planned += 1
                path = act_demo_dir(output_root, task["suite"], task["task"], demo_key)
                if (path / "paligemma.npy").exists() and (path / "paligemma.npy").stat().st_size > 0:
                    done += 1
    print(f"  paligemma.npy demos: {done}/{planned}")
    for path in sorted(output_root.glob("status*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        print(
            f"  {path.name}: {payload.get('phase')} "
            f"{payload.get('done_demos')}/{payload.get('total_demos')} "
            f"{payload.get('task')} {payload.get('demo')}"
        )
    return 0


def print_verify(output_root: Path, tokens_per_bin: int, flow_times: tuple[float, ...], frames: int) -> int:
    split_path = output_root / "split.json"
    if not split_path.exists():
        print("missing split.json")
        return 2
    split = json.loads(split_path.read_text(encoding="utf-8"))
    pali_bins = (EXPECTED_PALI_TOKENS + tokens_per_bin - 1) // tokens_per_bin
    expert_bins = (EXPECTED_EXPERT_TOKENS + tokens_per_bin - 1) // tokens_per_bin
    ok = 0
    missing = []
    bad = []
    planned = 0
    for task in split["tasks"]:
        for split_name in ("train", "test"):
            for demo_key in task[split_name]:
                planned += 1
                dest = act_demo_dir(output_root, task["suite"], task["task"], demo_key)
                paths = activation_paths(dest)
                if not paths["paligemma"].exists():
                    missing.append(f"{task['task']}/{demo_key}")
                    continue
                try:
                    expected_frames = int(json.loads((dest / "metadata.json").read_text(encoding="utf-8")).get("frames", frames))
                    pali = np.load(paths["paligemma"], mmap_mode="r")
                    expert = np.load(paths["expert"], mmap_mode="r")
                    ids = np.load(paths["sample_ids"], mmap_mode="r")
                    layout = json.loads(paths["layout"].read_text(encoding="utf-8"))
                    pali_ok = pali.ndim == 4 and pali.shape[1] == expected_frames == len(ids) and pali.shape[2] == pali_bins
                    expert_ok = (
                        expert.ndim == 5
                        and expert.shape[2] == expected_frames
                        and expert.shape[3] == expert_bins
                        and expert.shape[1] == len(flow_times)
                    )
                    if pali_ok and expert_ok:
                        ok += 1
                    else:
                        bad.append(
                            f"{task['task']}/{demo_key} pali={tuple(pali.shape)} "
                            f"expert={tuple(expert.shape)} expected_frames={expected_frames} "
                            f"times={layout.get('flow_times')}"
                        )
                except Exception as exc:
                    bad.append(f"{task['task']}/{demo_key} {exc}")
    print(f"intact={ok}/{planned} missing={len(missing)} bad={len(bad)}")
    print(f"expected pali_bins={pali_bins} expert_bins={expert_bins} flow_times={flow_times}")
    for line in missing[:12]:
        print(f"  missing {line}")
    for line in bad[:12]:
        print(f"  bad {line}")
    return 0 if ok == planned and not bad else 1


def main() -> int:
    args = parse_args()
    flow_times = parse_flow_times(args.flow_times)
    layers = parse_layers(args.layers)
    suffix = args.status_suffix
    occupancy_root = args.occupancy_root
    output_root = args.output_root
    frames = 2 if args.smoke and args.frames_per_demo == FRAMES_PER_DEMO else args.frames_per_demo
    if args.smoke:
        frames = min(args.frames_per_demo, 4)
    if args.status:
        return print_status(output_root)
    if args.verify:
        return print_verify(output_root, args.tokens_per_bin, flow_times, frames)

    set_seed(args.seed)
    ensure_dir(output_root)
    include = parse_csv(args.include_tasks)
    split = build_or_load_split(
        output_root,
        occupancy_root,
        suite=args.suite,
        num_tasks=args.num_tasks,
        seed=args.seed,
        frames=FRAMES_PER_DEMO,
        include=include,
        smoke=args.smoke,
    )
    shard_index, shard_count = parse_shard(args.shard)
    jobs = infer_jobs(split, shard_index, shard_count)
    write_json(
        output_root / "capture_config.json",
        {
            "pi05_path": args.pi05_path,
            "tokens_per_bin": args.tokens_per_bin,
            "expected_paligemma_tokens": EXPECTED_PALI_TOKENS,
            "expected_expert_tokens": EXPECTED_EXPERT_TOKENS,
            "paligemma_bins": (EXPECTED_PALI_TOKENS + args.tokens_per_bin - 1) // args.tokens_per_bin,
            "expert_bins": (EXPECTED_EXPERT_TOKENS + args.tokens_per_bin - 1) // args.tokens_per_bin,
            "token_bins_per_layer": (
                (EXPECTED_PALI_TOKENS + args.tokens_per_bin - 1) // args.tokens_per_bin
                + (EXPECTED_EXPERT_TOKENS + args.tokens_per_bin - 1) // args.tokens_per_bin
            ),
            "flow_times": list(flow_times),
            "uniform_euler_flow_times": list(uniform_euler_flow_times(10, 5)),
            "num_steps": 10,
            "layers": args.layers,
            "reuse_images_from": str(occupancy_root / "images"),
            "reuse_gt_from": str(occupancy_root / "gt"),
            "store": "per-demo paligemma.npy + expert.npy (sequential npy, TOS-safe; no npz)",
            "smoke": args.smoke,
            "shard": args.shard or "none",
        },
    )
    payload = {
        "phase": "smoke" if args.smoke else "extract",
        "task": "",
        "demo": "",
        "done_demos": 0,
        "total_demos": len(jobs),
        "done_frames": 0,
        "last_error": "",
        "output_root": str(output_root),
    }
    write_status(output_root, payload, suffix)
    log(
        f"Activation extract tasks={len(split['tasks'])} jobs={len(jobs)} "
        f"tokens_per_bin={args.tokens_per_bin} flow_times={flow_times} "
        f"smoke={args.smoke} shard={args.shard or 'none'} out={output_root}"
    )

    pali_bins = (EXPECTED_PALI_TOKENS + args.tokens_per_bin - 1) // args.tokens_per_bin
    expert_bins = (EXPECTED_EXPERT_TOKENS + args.tokens_per_bin - 1) // args.tokens_per_bin
    capture: Pi05EulerCapture | None = None
    skipped = 0
    done = 0
    done_frames = 0
    started = time.perf_counter()
    for job_i, job in enumerate(jobs):
        dest = act_demo_dir(output_root, job["suite"], job["task"], job["demo_key"])
        if activation_complete(
            dest,
            frames,
            pali_bins=pali_bins,
            expert_bins=expert_bins,
            n_flow=len(flow_times),
        ):
            skipped += 1
            done += 1
            done_frames += frames
            if skipped == 1 or skipped % 25 == 0 or job_i + 1 == len(jobs):
                payload.update(
                    phase="resume-skip",
                    task=job["task"],
                    demo=job["demo_key"],
                    done_demos=done,
                    done_frames=done_frames,
                    last_error="",
                )
                write_status(output_root, payload, suffix)
                log(f"resume skip {skipped} done={done}/{len(jobs)} last={job['task']}/{job['demo_key']}")
            continue
        rows = load_demo_rows(occupancy_root, job["suite"], job["task"], job["demo_key"])
        if frames < len(rows):
            rows = rows[:frames]
        missing_images = [
            row["image_path"] for row in rows if not Path(row["image_path"]).exists()
        ] + [row["wrist_image_path"] for row in rows if not Path(row["wrist_image_path"]).exists()]
        if missing_images:
            payload.update(phase="error", last_error=f"missing image {missing_images[0]}", **job)
            write_status(output_root, payload, suffix)
            raise FileNotFoundError(f"Reuse GT image missing: {missing_images[0]}")
        if capture is None:
            capture = Pi05EulerCapture(
                args.pi05_path,
                args.device,
                tokens_per_bin=args.tokens_per_bin,
                flow_times=flow_times,
                bin_indices=None,
            )
        payload.update(phase="extract", task=job["task"], demo=job["demo_key"], last_error="")
        write_status(output_root, payload, suffix)
        captured_rows: list[dict[str, np.ndarray]] = []
        sample_ids = []
        demo_started = time.perf_counter()
        for local_i, row in enumerate(rows):
            sample = sample_from_row(row, sample_id=done_frames + local_i)
            captured = capture.capture(sample, layers)
            if local_i == 0:
                pali_len = capture.last_token_lengths.get("paligemma")
                expert_len = capture.last_token_lengths.get("expert")
                pali_key = next(k for k in captured if k.startswith("paligemma"))
                expert_key = next(k for k in captured if k.startswith("expert"))
                log(
                    f"token lengths paligemma={pali_len} expert={expert_len} "
                    f"pali_bins={captured[pali_key].shape[0]} "
                    f"expert_bins={captured[expert_key].shape[0]}"
                )
                if pali_len != EXPECTED_PALI_TOKENS or expert_len != EXPECTED_EXPERT_TOKENS:
                    raise RuntimeError(
                        f"Unexpected token lengths pali={pali_len} expert={expert_len}; "
                        f"expected {EXPECTED_PALI_TOKENS}/{EXPECTED_EXPERT_TOKENS}."
                    )
            captured_rows.append(captured)
            sample_ids.append(sample.sample_id)
        pali, expert, layout = pack_demo_arrays(captured_rows)
        paths = activation_paths(dest)
        save_npy_inplace(paths["paligemma"], pali)
        save_npy_inplace(paths["expert"], expert)
        save_npy_inplace(paths["sample_ids"], np.asarray(sample_ids, dtype=np.int32))
        write_json(paths["layout"], layout)
        log(
            f"wrote {paths['paligemma'].name} {tuple(pali.shape)} "
            f"{paths['expert'].name} {tuple(expert.shape)}"
        )
        meta_rows = []
        for row, sample_id in zip(rows, sample_ids):
            meta_rows.append(
                {
                    "sample_id": int(sample_id),
                    "split": job["split"],
                    "suite": job["suite"],
                    "task": job["task"],
                    "demo_key": job["demo_key"],
                    "frame_index": int(row["frame_index"]),
                    "image_path": row["image_path"],
                    "wrist_image_path": row["wrist_image_path"],
                    "image_rel": row.get("image_rel"),
                    "occupancy_npy": str(
                        gt_demo_dir(occupancy_root, job["suite"], job["task"], job["demo_key"])
                        / "occupancy.npy"
                    ),
                }
            )
        write_jsonl(dest / "samples.jsonl", meta_rows)
        write_json(
            dest / "metadata.json",
            {
                "suite": job["suite"],
                "task": job["task"],
                "demo_key": job["demo_key"],
                "split": job["split"],
                "frames": len(rows),
                "seconds": time.perf_counter() - demo_started,
                "token_lengths": capture.last_token_lengths,
                "bin_maps_preview": {
                    key: capture.last_bin_maps[key]
                    for key in list(capture.last_bin_maps)[:2]
                },
            },
        )
        done += 1
        done_frames += len(rows)
        payload.update(done_demos=done, done_frames=done_frames)
        write_status(output_root, payload, suffix)
        log(
            f"demo done {job['task']}/{job['demo_key']} split={job['split']} "
            f"frames={len(rows)} skip={skipped} "
            f"seconds={time.perf_counter() - demo_started:.1f}"
        )
    payload.update(phase="done", done_demos=done, done_frames=done_frames)
    write_status(output_root, payload, suffix)
    log(
        f"Activation extract finished demos={done} skipped={skipped} "
        f"frames={done_frames} seconds={time.perf_counter() - started:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
