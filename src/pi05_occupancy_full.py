"""Scalable helpers for the full LIBERO-Spatial PI0.5 occupancy experiment."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm

from .libero_self_occupancy import SelfOccupancySample, token_position_bins
from .models import Pi05Wrapper, _move_tensors_to_device
from .utils import ensure_dir, log

FLOW_TIMES = (1.0, 0.5, 0.1)


def safe_condition_name(condition: str) -> str:
    return condition.replace("/", "__")


def probe_dir(run_dir: Path, condition: str, bin_index: int) -> Path:
    return run_dir / "probes" / safe_condition_name(condition) / f"bin_{int(bin_index):03d}"


def load_activation_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "activation_shards" / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def condition_stored_bins(
    condition: str,
    *,
    n_columns: int,
    manifest: dict[str, Any],
) -> list[int]:
    """Map activation axis-1 columns back to original token-bin indices."""
    by_condition = manifest.get("condition_bin_indices") or {}
    if condition in by_condition:
        stored = [int(v) for v in by_condition[condition]]
        if len(stored) != n_columns:
            raise ValueError(
                f"Manifest lists {len(stored)} bins for {condition}, but array has {n_columns} columns."
            )
        return stored
    # Legacy shards stored every effective bin as columns 0..N-1.
    return list(range(n_columns))


def select_bin_columns(
    *,
    condition: str,
    x_all: np.ndarray,
    requested_bin_indices: Sequence[int],
    manifest: dict[str, Any],
) -> list[tuple[int, int]]:
    """Return (original_bin_index, column_index) pairs available in ``x_all``."""
    stored = condition_stored_bins(condition, n_columns=int(x_all.shape[1]), manifest=manifest)
    column_by_bin = {bin_index: column for column, bin_index in enumerate(stored)}
    selected = []
    for bin_index in requested_bin_indices:
        column = column_by_bin.get(int(bin_index))
        if column is not None:
            selected.append((int(bin_index), int(column)))
    if not selected:
        raise ValueError(
            f"None of requested bins {list(requested_bin_indices)} are stored for {condition}. "
            f"Stored bins={stored}."
        )
    return selected


class Pi05EulerCapture:
    """Capture prefix once and expert states at real 10-step Euler evaluations."""

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        bins: int = 96,
        bin_indices: Sequence[int] | None = None,
    ) -> None:
        self.model_id = model_id
        self.bins = bins
        self.bin_indices = None if bin_indices is None else [int(v) for v in bin_indices]
        self.last_bin_maps: dict[str, list[int]] = {}
        log(
            f"Loading PI0.5 Euler capture model from {model_id} "
            f"bins={bins} store_bins={self.bin_indices if self.bin_indices is not None else 'all'} "
            f"device={device}"
        )
        started = time.perf_counter()
        self.wrapper = Pi05Wrapper(model_id=model_id, pooling="mean", device=device)
        log(f"PI0.5 Euler capture ready in {time.perf_counter() - started:.1f}s device={self.wrapper.device_obj}")

    def _store_bins(self, binned: np.ndarray) -> tuple[np.ndarray, list[int]]:
        """Keep equal-width boundaries from ``self.bins``, optionally drop unselected bins."""
        if self.bin_indices is None:
            stored = list(range(int(binned.shape[0])))
            return binned.astype(np.float16), stored
        usable = [index for index in self.bin_indices if index < int(binned.shape[0])]
        if not usable:
            raise ValueError(
                f"No selected bin_indices={self.bin_indices} fall inside effective_bins={binned.shape[0]}."
            )
        return binned[usable].astype(np.float16), usable

    def capture(self, sample: SelfOccupancySample, layer_indices: set[int] | None = None) -> dict[str, np.ndarray]:
        torch = self.wrapper.torch
        image = np.asarray(Image.open(sample.image_path).convert("RGB"))
        frame = self.wrapper._make_frame(
            image, sample.instruction,
            {"observation_state": sample.observation_state, "wrist_image_path": sample.wrist_image_path},
        )
        batch = _move_tensors_to_device(self.wrapper.preprocess(frame), self.wrapper.device_obj)
        policy, core = self.wrapper.policy, self.wrapper.policy.model
        images, img_masks = policy._preprocess_images(batch)
        tokens = batch["observation.language.tokens"]
        masks = batch["observation.language.attention_mask"]
        captured: dict[str, np.ndarray] = {}
        bin_maps: dict[str, list[int]] = {}
        current_time: list[float | None] = [None]
        handles = []

        def layer_hook(tower: str, index: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                if layer_indices is not None and index not in layer_indices:
                    return
                tensor = output[0] if isinstance(output, tuple) else output
                if tower == "paligemma":
                    key = f"paligemma/layer_{index:02d}/static"
                    if key in captured:
                        return
                else:
                    t = current_time[0]
                    if t is None or min(abs(t - wanted) for wanted in FLOW_TIMES) > 1e-5:
                        return
                    key = f"expert/layer_{index:02d}/t={min(FLOW_TIMES, key=lambda x: abs(x-t)):.1f}"
                array = tensor.detach().float().cpu().numpy()
                stored, usable = self._store_bins(token_position_bins(array, self.bins)[0])
                captured[key] = stored
                bin_maps[key] = usable
            return hook

        def record_time(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            timestep = kwargs.get("timestep", args[3] if len(args) > 3 else None)
            current_time[0] = float(timestep[0].detach().cpu())

        pg = core.paligemma_with_expert.paligemma.language_model.layers
        expert = core.paligemma_with_expert.gemma_expert.model.layers
        for index, layer in enumerate(pg):
            handles.append(layer.register_forward_hook(layer_hook("paligemma", index)))
        for index, layer in enumerate(expert):
            handles.append(layer.register_forward_hook(layer_hook("expert", index)))
        original_denoise_step = core.denoise_step

        def traced_denoise_step(*args: Any, **kwargs: Any):
            record_time(args, kwargs)
            return original_denoise_step(*args, **kwargs)

        core.denoise_step = traced_denoise_step
        try:
            with torch.inference_mode():
                core.sample_actions(images, img_masks, tokens, masks, num_steps=10)
        finally:
            core.denoise_step = original_denoise_step
            for handle in handles:
                handle.remove()
        expected = {
            *(f"paligemma/layer_{i:02d}/static" for i in range(len(pg)) if layer_indices is None or i in layer_indices),
            *(f"expert/layer_{i:02d}/t={t:.1f}" for i in range(len(expert))
              for t in FLOW_TIMES if layer_indices is None or i in layer_indices),
        }
        missing = expected.difference(captured)
        if missing:
            raise RuntimeError(f"PI0.5 capture missed {sorted(missing)[:5]} ({len(missing)} total).")
        self.last_bin_maps = bin_maps
        return captured


def write_activation_shards(
    samples: Sequence[SelfOccupancySample],
    output_dir: Path,
    capture: Pi05EulerCapture,
    shard_size: int,
    layer_indices: set[int] | None = None,
) -> dict[str, Any]:
    """Write resumable compressed shards containing only selected token bins."""
    shard_dir = ensure_dir(output_dir / "activation_shards")
    manifest_path = shard_dir / "manifest.json"
    requested_manifest = {
        "model_id": str(capture.model_id),
        "requested_bins": int(capture.bins),
        "bin_indices": None if capture.bin_indices is None else list(capture.bin_indices),
        "flow_times": list(FLOW_TIMES),
        "layer_indices": None if layer_indices is None else sorted(layer_indices),
        "sample_ids": [int(sample.sample_id) for sample in samples],
        "format": "float16 [sample, stored_bin, hidden]; condition_bin_indices maps columns to original bins",
    }
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_core = {k: existing_manifest.get(k) for k in requested_manifest}
        if existing_core != requested_manifest:
            raise ValueError(
                "Existing activation shards were created with a different model, "
                "bin/layer configuration, flow-time set, sample selection, or stored bin subset. "
                f"Remove {shard_dir} or choose a new output directory."
            )
        log(f"Resuming activation shards under {shard_dir} (manifest matches)")
    else:
        manifest_path.write_text(json.dumps(requested_manifest, indent=2) + "\n", encoding="utf-8")
        log(
            f"Writing new activation shards under {shard_dir} "
            f"store_bins={requested_manifest['bin_indices'] or 'all'}"
        )
    started = time.perf_counter()
    keys: list[str] | None = None
    completed = 0
    skipped_shards = 0
    written_shards = 0
    capture_seconds: list[float] = []
    condition_bin_indices: dict[str, list[int]] | None = None
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    if existing.get("condition_bin_indices"):
        condition_bin_indices = {
            key: [int(v) for v in values] for key, values in existing["condition_bin_indices"].items()
        }
    starts = list(range(0, len(samples), shard_size))
    shard_bar = tqdm(starts, desc="activation shards", unit="shard")
    for start in shard_bar:
        path = shard_dir / f"shard_{start:06d}.npz"
        if path.exists():
            completed += min(shard_size, len(samples) - start)
            skipped_shards += 1
            shard_bar.set_postfix(
                done=completed, skipped=skipped_shards, written=written_shards, status="resume-skip"
            )
            continue
        shard_samples = samples[start : start + shard_size]
        rows = []
        for sample in tqdm(shard_samples, desc=f"shard {start:06d}", unit="frame", leave=False):
            item_started = time.perf_counter()
            rows.append(capture.capture(sample, layer_indices))
            capture_seconds.append(time.perf_counter() - item_started)
        if condition_bin_indices is None:
            condition_bin_indices = {key: list(values) for key, values in capture.last_bin_maps.items()}
            existing["condition_bin_indices"] = condition_bin_indices
            manifest_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
            pali_bins = next((v for k, v in condition_bin_indices.items() if k.startswith("paligemma")), [])
            exp_bins = next((v for k, v in condition_bin_indices.items() if k.startswith("expert")), [])
            log(f"Recorded condition_bin_indices pali={pali_bins} expert={exp_bins}")
        keys = sorted(rows[0])
        payload = {key: np.stack([row[key] for row in rows]) for key in keys}
        payload["sample_ids"] = np.asarray([s.sample_id for s in shard_samples])
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **payload)
        temporary.replace(path)
        completed += len(rows)
        written_shards += 1
        mean_capture = float(np.mean(capture_seconds[-len(rows) :])) if capture_seconds else float("nan")
        elapsed = time.perf_counter() - started
        rate = completed / max(elapsed, 1e-6)
        remaining = (len(samples) - completed) / max(rate, 1e-6)
        shard_bar.set_postfix(
            done=f"{completed}/{len(samples)}",
            written=written_shards,
            skipped=skipped_shards,
            s_frame=f"{mean_capture:.1f}",
            eta_min=f"{remaining / 60:.1f}",
        )
        if written_shards == 1 or written_shards % 5 == 0 or completed >= len(samples):
            log(
                f"Activation progress {completed}/{len(samples)} "
                f"written_shards={written_shards} skipped_shards={skipped_shards} "
                f"mean_s/frame={mean_capture:.2f} elapsed={elapsed:.0f}s "
                f"eta={remaining / 60:.1f}min keys={len(keys or [])}"
            )
    result = {
        "samples": completed,
        "shard_size": shard_size,
        "requested_bins": capture.bins,
        "bin_indices": None if capture.bin_indices is None else list(capture.bin_indices),
        "condition_bin_indices": condition_bin_indices,
        "flow_times": list(FLOW_TIMES),
        "seconds": time.perf_counter() - started,
        "written_shards": written_shards,
        "skipped_shards": skipped_shards,
        "mean_capture_seconds_per_frame": (
            float(np.mean(capture_seconds)) if capture_seconds else 0.0
        ),
        "format": "one [sample, stored_bin, hidden] array per tower/layer/condition",
    }
    log(
        f"Activation stage done samples={completed} written={written_shards} "
        f"skipped={skipped_shards} seconds={result['seconds']:.1f}"
    )
    return result


def load_condition_activations(run_dir: Path, condition: str) -> np.ndarray:
    shards = sorted((run_dir / "activation_shards").glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No activation shards under {run_dir / 'activation_shards'}")
    pieces: list[np.ndarray] = []
    ids: list[int] = []
    for shard in tqdm(shards, desc=f"load {condition}", unit="shard", leave=False):
        with np.load(shard) as packed:
            if condition not in packed.files:
                raise KeyError(f"{condition} missing from {shard}")
            pieces.append(packed[condition].astype(np.float32))
            ids.extend(packed["sample_ids"].tolist())
    x_all = np.concatenate(pieces, axis=0)
    return x_all[np.argsort(np.asarray(ids))]


def list_activation_conditions(run_dir: Path) -> list[str]:
    shards = sorted((run_dir / "activation_shards").glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No activation shards under {run_dir / 'activation_shards'}")
    with np.load(shards[0]) as first:
        return sorted(k for k in first.files if k != "sample_ids")


def iter_activation_shards(path: Path) -> Iterator[tuple[Path, Any]]:
    for shard in sorted(path.glob("shard_*.npz")):
        with np.load(shard, allow_pickle=False) as packed:
            yield shard, packed


def read_jsonl_samples(path: Path, occupancy: np.ndarray) -> list[SelfOccupancySample]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [
        SelfOccupancySample(
            sample_id=int(row["sample_id"]), demo_key=row["demo_key"], frame_index=int(row["frame_index"]),
            image_path=row["image_path"], wrist_image_path=row["wrist_image_path"],
            instruction=row["instruction"], observation_state=row["observation_state"],
            occupancy=occupancy[i], suite=row.get("suite", "libero_spatial"), task=row.get("task", ""),
        )
        for i, row in enumerate(rows)
    ]
