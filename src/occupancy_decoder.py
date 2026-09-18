"""Occupancy MLP decoder used by run_full.py and the Libero-90 extract trainer."""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .pi05_occupancy_full import probe_dir
from .utils import ensure_dir


PALI_LAYERS = 18
EXPERT_LAYERS = 18
PALI_HIDDEN = 2048
EXPERT_HIDDEN = 1024
GRID = 16
OUTPUT_DIM = GRID ** 3


def occupancy_loss(logits, yb, pos_weight, kind: str = "bce_dice"):
    """Training objective. ``bce_dice`` is the run_full recipe; ``bce`` is pos-weighted BCE only."""
    import torch.nn.functional as F

    name = str(kind or "bce_dice").strip().lower().replace("-", "_")
    bce = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
    if name == "bce":
        return bce, float(bce.detach().cpu())
    if name in {"bce_dice", "bce+dice", "dice"}:
        prob = logits.sigmoid()
        dice = 1.0 - ((2.0 * (prob * yb).sum(1) + 1e-6) / (prob.sum(1) + yb.sum(1) + 1e-6)).mean()
        loss = bce + dice
        return loss, float(loss.detach().cpu())
    raise ValueError(f"unknown occupancy loss {kind!r}; use bce or bce_dice")


def occupancy_bce_dice_loss(logits, yb, pos_weight):
    """Pos-weighted BCE + soft Dice. Same recipe as scripts/occupancy/run_full.py."""
    return occupancy_loss(logits, yb, pos_weight, kind="bce_dice")


def make_occupancy_mlp(input_dim: int, bottleneck: int = 64, output_dim: int = OUTPUT_DIM):
    from torch import nn

    return nn.Sequential(
        nn.Linear(int(input_dim), int(bottleneck)),
        nn.GELU(),
        nn.Linear(int(bottleneck), int(output_dim)),
    )


def save_torch_inplace(path: Path, payload: dict[str, Any]) -> None:
    """torch.save without a filesystem rename (TOS s3mount cannot rename)."""
    import torch

    path = Path(path)
    ensure_dir(path.parent)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    path.write_bytes(buffer.getvalue())


def bce_pos_weight(train_targets: np.ndarray) -> float:
    positive_fraction = float(np.mean(train_targets))
    return min(30.0, max(1.0, (1.0 - positive_fraction) / max(positive_fraction, 1e-6)))


@dataclass(frozen=True)
class FrameRow:
    suite: str
    task: str
    demo_key: str
    frame: int
    split: str


def load_extract_split(activation_root: Path) -> dict[str, Any]:
    path = Path(activation_root) / "split.json"
    if not path.exists():
        raise FileNotFoundError(f"missing extract split.json: {path}")
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def iter_split_rows(
    split: dict[str, Any],
    *,
    frames: int,
    max_tasks: int | None = None,
    max_train_demos: int | None = None,
    max_test_demos: int | None = None,
    include_tasks: Sequence[str] | None = None,
) -> list[FrameRow]:
    """Train/test frames only. Ablation demos are recorded in split.json but not used."""
    include = [str(item) for item in (include_tasks or []) if str(item).strip()]
    rows: list[FrameRow] = []
    tasks = list(split.get("tasks") or [])
    if max_tasks is not None:
        tasks = tasks[: max(0, int(max_tasks))]
    for task in tasks:
        name = str(task["task"])
        if include and not any(token in name or token in f"{task.get('suite')}/{name}" for token in include):
            continue
        suite = str(task.get("suite") or split.get("suite") or "libero_90")
        for split_name, cap in (("train", max_train_demos), ("test", max_test_demos)):
            demos = list(task.get(split_name) or [])
            if cap is not None:
                demos = demos[: max(0, int(cap))]
            for demo_key in demos:
                for frame in range(int(frames)):
                    rows.append(
                        FrameRow(
                            suite=suite,
                            task=name,
                            demo_key=str(demo_key),
                            frame=frame,
                            split=split_name,
                        )
                    )
    if not rows:
        raise ValueError("split produced no train/test frames")
    return rows


def split_indices(rows: Sequence[FrameRow]) -> tuple[np.ndarray, np.ndarray]:
    train_idx = np.asarray([i for i, row in enumerate(rows) if row.split == "train"], dtype=np.int64)
    test_idx = np.asarray([i for i, row in enumerate(rows) if row.split == "test"], dtype=np.int64)
    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError(f"need both train and test frames, got train={train_idx.size} test={test_idx.size}")
    return train_idx, test_idx


def list_extract_conditions(
    *,
    towers: Sequence[str],
    layer_indices: Sequence[int],
    flow_times: Sequence[float],
) -> list[tuple[str, str, int, int | None]]:
    """Return (condition, tower, layer, expert_time_index_or_None)."""
    out: list[tuple[str, str, int, int | None]] = []
    towers_set = {str(item).lower() for item in towers}
    if "paligemma" in towers_set or "all" in towers_set:
        for layer in layer_indices:
            out.append((f"paligemma/layer_{int(layer):02d}/static", "paligemma", int(layer), None))
    if "expert" in towers_set or "all" in towers_set:
        for layer in layer_indices:
            for time_index, flow_time in enumerate(flow_times):
                out.append(
                    (
                        f"expert/layer_{int(layer):02d}/t={float(flow_time):.1f}",
                        "expert",
                        int(layer),
                        int(time_index),
                    )
                )
    if not out:
        raise ValueError(f"no conditions for towers={list(towers)} layers={list(layer_indices)}")
    return out


def _act_demo_dir(activation_root: Path, row: FrameRow) -> Path:
    return Path(activation_root) / "activations" / row.suite / row.task / row.demo_key


def _gt_demo_dir(occupancy_root: Path, row: FrameRow) -> Path:
    return Path(occupancy_root) / "gt" / row.suite / row.task / row.demo_key


def load_occupancy_targets(occupancy_root: Path, rows: Sequence[FrameRow]) -> np.ndarray:
    targets = np.empty((len(rows), GRID, GRID, GRID), dtype=np.float32)
    cache: dict[tuple[str, str, str], np.ndarray] = {}
    for index, row in enumerate(rows):
        key = (row.suite, row.task, row.demo_key)
        if key not in cache:
            path = _gt_demo_dir(occupancy_root, row) / "occupancy.npy"
            packed = np.load(path, mmap_mode="r")
            cache[key] = packed
        packed = cache[key]
        if row.frame >= packed.shape[0]:
            raise IndexError(f"{key} occupancy frames={packed.shape[0]} need frame {row.frame}")
        targets[index] = np.asarray(packed[row.frame], dtype=np.float32)
    return targets


def load_condition_activations(
    activation_root: Path,
    rows: Sequence[FrameRow],
    *,
    tower: str,
    layer: int,
    time_index: int | None,
    bin_indices: Sequence[int],
) -> np.ndarray:
    """Stack selected bins as float32 [N, n_bins, hidden]."""
    bins = [int(v) for v in bin_indices]
    hidden = PALI_HIDDEN if tower == "paligemma" else EXPERT_HIDDEN
    x_all = np.empty((len(rows), len(bins), hidden), dtype=np.float32)
    cache: dict[tuple[str, str, str], np.ndarray] = {}
    filename = "paligemma.npy" if tower == "paligemma" else "expert.npy"
    for index, row in enumerate(rows):
        key = (row.suite, row.task, row.demo_key)
        if key not in cache:
            path = _act_demo_dir(activation_root, row) / filename
            cache[key] = np.load(path, mmap_mode="r")
        packed = cache[key]
        if tower == "paligemma":
            # [layer, frame, bin, hidden]
            x_all[index] = np.asarray(packed[layer, row.frame, bins, :], dtype=np.float32)
        else:
            if time_index is None:
                raise ValueError("expert conditions require a flow-time index")
            # [layer, time, frame, bin, hidden]
            x_all[index] = np.asarray(packed[layer, time_index, row.frame, bins, :], dtype=np.float32)
    return x_all


def _demo_row_groups(rows: Sequence[FrameRow]) -> dict[tuple[str, str, str], list[int]]:
    groups: dict[tuple[str, str, str], list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault((row.suite, row.task, row.demo_key), []).append(index)
    return groups


def load_paligemma_layer(
    activation_root: Path,
    rows: Sequence[FrameRow],
    *,
    layer: int,
    bin_indices: Sequence[int],
) -> np.ndarray:
    """Load one PaliGemma layer as float16 [N, n_bins, 2048]. One TOS open per demo."""
    bins = [int(v) for v in bin_indices]
    x_all = np.empty((len(rows), len(bins), PALI_HIDDEN), dtype=np.float16)
    groups = _demo_row_groups(rows)
    for (suite, task, demo_key), indices in groups.items():
        path = Path(activation_root) / "activations" / suite / task / demo_key / "paligemma.npy"
        packed = np.load(path, mmap_mode="r")
        try:
            for index in indices:
                frame = rows[index].frame
                x_all[index] = packed[layer, frame, bins, :]
        finally:
            del packed
    return x_all


def load_expert_layer(
    activation_root: Path,
    rows: Sequence[FrameRow],
    *,
    layer: int,
    bin_indices: Sequence[int],
    n_times: int,
) -> np.ndarray:
    """Load one expert layer, all Euler times, as float16 [N, n_times, n_bins, 1024]."""
    bins = [int(v) for v in bin_indices]
    x_all = np.empty((len(rows), int(n_times), len(bins), EXPERT_HIDDEN), dtype=np.float16)
    groups = _demo_row_groups(rows)
    for (suite, task, demo_key), indices in groups.items():
        path = Path(activation_root) / "activations" / suite / task / demo_key / "expert.npy"
        packed = np.load(path, mmap_mode="r")
        try:
            for index in indices:
                frame = rows[index].frame
                per_time = packed[layer, : int(n_times), frame]
                x_all[index] = per_time[:, bins, :]
        finally:
            del packed
    return x_all


def cache_array_path(cache_dir: Path, name: str) -> Path:
    return Path(cache_dir) / f"{name}.npy"


def load_or_build_layer_cache(
    cache_path: Path,
    builder,
    *,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    """Return a float16 array. Reuse GPFS cache when the shape matches."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        packed = np.load(cache_path, mmap_mode="r")
        if tuple(packed.shape) == tuple(expected_shape) and packed.dtype == np.float16:
            from .utils import log

            log(f"Reuse layer cache {cache_path} shape={tuple(packed.shape)}")
            return packed
        del packed
    array = builder()
    if tuple(array.shape) != tuple(expected_shape):
        raise ValueError(f"cache builder shape {array.shape} != {expected_shape}")
    from .libero_self_occupancy import save_npy_inplace
    from .utils import log

    save_npy_inplace(cache_path, np.asarray(array, dtype=np.float16))
    log(f"Wrote layer cache {cache_path} shape={tuple(array.shape)} bytes={cache_path.stat().st_size}")
    return np.load(cache_path, mmap_mode="r")


def write_train_status(output_dir: Path, payload: dict[str, Any]) -> None:
    from .utils import ensure_dir

    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    body = dict(payload)
    body["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    txt = output_dir / "STATUS.txt"
    js = output_dir / "status.json"
    lines = [
        f"updated_at: {body['updated_at']}",
        f"phase:      {body.get('phase', '')}",
        f"condition:  {body.get('condition', '')}",
        f"bin:        {body.get('bin', '')}",
        f"progress:   {body.get('done_probes', 0)}/{body.get('total_probes', 0)} probes",
        f"best_iou:   {body.get('best_soft_iou', '')}",
        f"last_error: {body.get('last_error', '')}",
        "",
    ]
    txt.write_text("\n".join(lines), encoding="utf-8")
    js.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")


def probe_is_complete(probe_path: Path) -> bool:
    decoder = Path(probe_path) / "decoder.pt"
    history = Path(probe_path) / "history.csv"
    try:
        return decoder.exists() and decoder.stat().st_size > 10_000 and history.exists() and history.stat().st_size > 0
    except OSError:
        return False


def metrics_from_history_csv(path: Path, *, condition: str, bin_index: int, column: int) -> dict[str, Any]:
    import csv

    rows = list(csv.DictReader(Path(path).open(newline="", encoding="utf-8")))
    if not rows:
        raise ValueError(f"empty history {path}")
    last = rows[-1]
    return {
        "condition": condition,
        "bin": int(bin_index),
        "column": int(column),
        "soft_iou": float(last["test_soft_iou"]),
        "hard_iou": float(last["test_hard_iou"]),
        "train_loss": float(last["train_loss"]),
        "test_loss": float(last["test_loss"]),
        "resumed": True,
    }


def sync_tree_inplace(src: Path, dst: Path, *, skip_dir_names: Iterable[str] = ("layer_cache",)) -> dict[str, int]:
    """Copy GPFS → TOS with wb (no rename). Skip bulky layer caches."""
    src = Path(src)
    dst = Path(dst)
    skip = set(skip_dir_names)
    copied = skipped = 0
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip for part in path.parts):
            continue
        rel = path.relative_to(src)
        dest = dst / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            if dest.exists() and dest.stat().st_size == path.stat().st_size:
                skipped += 1
                continue
        except OSError:
            pass
        with path.open("rb") as handle_in, dest.open("wb") as handle_out:
            while True:
                chunk = handle_in.read(8 << 20)
                if not chunk:
                    break
                handle_out.write(chunk)
        copied += 1
    return {"copied": copied, "skipped_same_size": skipped}


def train_one_probe(
    *,
    x: np.ndarray,
    occupancy: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    condition: str,
    bin_index: int,
    column: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    bottleneck: int,
    seed: int,
    device: str,
    pos_weight_value: float,
    loss_kind: str = "bce_dice",
) -> tuple[dict[str, Any], list[dict[str, Any]], Any]:
    import torch

    from .libero_self_occupancy import hard_iou, soft_iou

    device_obj = torch.device(device)
    torch.manual_seed(int(seed))
    mean = x[train_idx].mean(axis=0)
    std = x[train_idx].std(axis=0).copy()
    std[std < 1e-5] = 1.0
    x_norm = (x - mean) / std
    targets = occupancy.reshape(len(occupancy), -1).astype(np.float32)
    decoder = make_occupancy_mlp(x.shape[1], bottleneck=bottleneck, output_dim=targets.shape[1]).to(device_obj)
    opt = torch.optim.AdamW(decoder.parameters(), lr=learning_rate)
    pos_weight = torch.tensor(float(pos_weight_value), dtype=torch.float32, device=device_obj)
    history: list[dict[str, Any]] = []
    last_train_loss = float("nan")
    last_test_loss = float("nan")
    soft = float("nan")
    hard = float("nan")
    for epoch in range(int(epochs)):
        decoder.train()
        epoch_losses: list[float] = []
        for start in range(0, len(train_idx), int(batch_size)):
            idx = train_idx[start : start + int(batch_size)]
            xb = torch.from_numpy(x_norm[idx]).to(device_obj)
            yb = torch.from_numpy(targets[idx]).to(device_obj)
            logits = decoder(xb)
            loss, loss_value = occupancy_loss(logits, yb, pos_weight, kind=loss_kind)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_losses.append(loss_value)
        last_train_loss = float(np.mean(epoch_losses))
        decoder.eval()
        with torch.inference_mode():
            test_logits = decoder(torch.from_numpy(x_norm[test_idx]).to(device_obj))
            test_yb = torch.from_numpy(targets[test_idx]).to(device_obj)
            _, last_test_loss = occupancy_loss(test_logits, test_yb, pos_weight, kind=loss_kind)
            pred = test_logits.sigmoid().cpu().numpy().reshape((-1, GRID, GRID, GRID))
        soft = soft_iou(occupancy[test_idx], pred)
        hard = hard_iou(occupancy[test_idx], pred)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": last_train_loss,
                "test_loss": last_test_loss,
                "test_soft_iou": soft,
                "test_hard_iou": hard,
            }
        )
    payload = {
        "state_dict": {key: value.detach().cpu() for key, value in decoder.state_dict().items()},
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "condition": condition,
        "bin": int(bin_index),
        "column": int(column),
        "input_dim": int(x.shape[1]),
        "output_dim": int(targets.shape[1]),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "loss": str(loss_kind),
        "final_train_loss": last_train_loss,
        "final_test_loss": last_test_loss,
        "final_soft_iou": soft,
        "final_hard_iou": hard,
    }
    metrics = {
        "condition": condition,
        "bin": int(bin_index),
        "column": int(column),
        "soft_iou": soft,
        "hard_iou": hard,
        "train_loss": last_train_loss,
        "test_loss": last_test_loss,
        "loss": str(loss_kind),
        "positive_fraction_train": float(np.mean(targets[train_idx])),
        "bce_pos_weight": float(pos_weight_value),
        "num_train": int(len(train_idx)),
        "num_test": int(len(test_idx)),
        "probe_dir": str(probe_dir(Path("."), condition, bin_index)),
    }
    del decoder, opt
    return payload, history, metrics
