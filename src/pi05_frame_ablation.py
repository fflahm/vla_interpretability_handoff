"""Offline PI0.5 bin ablation on occupancy frames (fresh chunk per forward)."""
from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import pandas as pd

from .models import Pi05Wrapper, _move_tensors_to_device, _normalize_action_chunk, _tensor_to_numpy, load_image_np
from .online_rollout import Pi05ActivationIntervention, _apply_pi05_activation_intervention
from .utils import ensure_dir, log


PredictFn = Callable[["OccupancyFrame", Pi05ActivationIntervention | None, dict[str, np.ndarray]], np.ndarray]


@dataclass(frozen=True)
class OccupancyFrame:
    sample_id: int
    demo_key: str
    frame_index: int
    image_path: str
    wrist_image_path: str
    instruction: str
    observation_state: list[float]
    task: str = ""
    episode_id: str = ""
    suite: str = ""


def load_occupancy_frames(run_dir: Path, *, require_images: bool = True) -> list[OccupancyFrame]:
    """Load occupancy `samples.jsonl` without touching occupancy.npy."""
    path = run_dir / "samples.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing occupancy frames: {path}")
    frames: list[OccupancyFrame] = []
    missing = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            image_path = str(row["image_path"])
            wrist_path = str(row.get("wrist_image_path") or "")
            if require_images and (not Path(image_path).is_file() or (wrist_path and not Path(wrist_path).is_file())):
                missing += 1
                continue
            suite = str(row.get("suite", "") or "")
            task = str(row.get("task", "") or "")
            demo_key = str(row.get("demo_key", "") or "")
            episode_id = str(row.get("episode_id") or "") or "/".join(
                part for part in (suite, task, demo_key) if part
            )
            frames.append(
                OccupancyFrame(
                    sample_id=int(row["sample_id"]),
                    demo_key=demo_key,
                    frame_index=int(row["frame_index"]),
                    image_path=image_path,
                    wrist_image_path=wrist_path,
                    instruction=str(row.get("instruction", "")),
                    observation_state=[float(v) for v in row["observation_state"]],
                    task=task,
                    episode_id=episode_id,
                    suite=suite,
                )
            )
    if not frames:
        raise ValueError(f"No usable frames in {path} (missing images={missing}).")
    if missing:
        log(f"Skipped {missing} occupancy rows with missing images")
    return frames


def sample_frames(
    frames: Sequence[OccupancyFrame],
    num_frames: int,
    rng: np.random.Generator,
) -> list[OccupancyFrame]:
    if num_frames <= 0:
        raise ValueError("`num_frames` must be positive.")
    n = min(int(num_frames), len(frames))
    order = rng.choice(len(frames), size=n, replace=False)
    return [frames[int(index)] for index in order]


def load_task_hdf5_paths(run_dir: Path) -> dict[str, str]:
    """Map occupancy task name -> original LIBERO HDF5 path from gt_metadata.json."""
    path = run_dir / "gt_metadata.json"
    if not path.exists():
        return {}
    meta = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for item in meta.get("tasks", []):
        hdf5 = str(item.get("hdf5_path") or "")
        if not hdf5:
            continue
        task = Path(hdf5).name.removesuffix("_demo.hdf5")
        mapping[task] = hdf5
    return mapping


def frame_provenance_table(
    frames: Sequence[OccupancyFrame],
    hdf5_by_task: dict[str, str] | None = None,
) -> pd.DataFrame:
    """One row per sampled frame, aligned with `chunk_l2[row_index]`."""
    hdf5_by_task = hdf5_by_task or {}
    rows = []
    for row_index, frame in enumerate(frames):
        episode_id = frame.episode_id or "/".join(
            part for part in (frame.suite, frame.task, frame.demo_key) if part
        )
        rows.append(
            {
                "row_index": int(row_index),
                "sample_id": int(frame.sample_id),
                "suite": frame.suite,
                "task": frame.task,
                "demo_key": frame.demo_key,
                "libero_frame_index": int(frame.frame_index),
                "episode_id": episode_id,
                "hdf5_path": hdf5_by_task.get(frame.task, ""),
                "image_path": frame.image_path,
                "wrist_image_path": frame.wrist_image_path,
            }
        )
    return pd.DataFrame(rows)


def chunk_delta_l2(ablated: np.ndarray, baseline: np.ndarray) -> float:
    ablated = np.asarray(ablated, dtype=np.float32)
    baseline = np.asarray(baseline, dtype=np.float32)
    if ablated.shape != baseline.shape:
        raise ValueError(f"Action chunk shape mismatch: ablated {ablated.shape} vs baseline {baseline.shape}.")
    return float(np.linalg.norm((ablated - baseline).reshape(-1)))


def load_selected_bins(path: Path) -> pd.DataFrame:
    selected = pd.read_csv(path)
    required = {"tower", "layer_index", "token_bin_index"}
    missing = required.difference(selected.columns)
    if missing:
        raise ValueError(f"selected-csv is missing columns: {sorted(missing)}")
    if "probe_group" not in selected.columns:
        selected["probe_group"] = "candidate"
    if "probe_rank" not in selected.columns:
        selected["probe_rank"] = np.arange(1, len(selected) + 1)
    if "global_layer_index" not in selected.columns:
        selected["global_layer_index"] = selected["layer_index"] + selected["tower"].map(
            {"paligemma": 0, "expert": 18}
        ).fillna(0)
    return selected.reset_index(drop=True)


def interventions_from_selected(
    selected: pd.DataFrame,
    *,
    token_bins: int,
    mode: str,
    scale: float,
) -> list[Pi05ActivationIntervention]:
    return [
        Pi05ActivationIntervention(
            tower=str(row.tower),
            layer_index=int(row.layer_index),
            token_bin_index=int(row.token_bin_index),
            token_bins=int(token_bins),
            mode=str(mode),
            scale=float(scale),
        )
        for row in selected.itertuples(index=False)
    ]


def load_frame_images(frame: OccupancyFrame) -> dict[str, np.ndarray]:
    image = load_image_np(frame.image_path)
    wrist = load_image_np(frame.wrist_image_path) if frame.wrist_image_path else image
    return {"image": image, "wrist_image": wrist}


def summarize_bin_deltas(selected: pd.DataFrame, chunk_l2: np.ndarray) -> pd.DataFrame:
    if chunk_l2.ndim != 2 or chunk_l2.shape[1] != len(selected):
        raise ValueError(
            f"chunk_l2 shape {chunk_l2.shape} does not match {len(selected)} selected bins."
        )
    rows = []
    values = np.asarray(chunk_l2, dtype=np.float32)
    for index, row in selected.reset_index(drop=True).iterrows():
        col = values[:, int(index)]
        rows.append(
            {
                "tower": row["tower"],
                "layer_index": int(row["layer_index"]),
                "token_bin_index": int(row["token_bin_index"]),
                "global_layer_index": int(row.get("global_layer_index", row["layer_index"])),
                "probe_group": row["probe_group"],
                "probe_rank": int(row.get("probe_rank", index + 1)),
                "probe_soft_iou": float(row["probe_soft_iou"]) if "probe_soft_iou" in selected.columns else float("nan"),
                "pair_id": int(row["pair_id"]) if "pair_id" in selected.columns and pd.notna(row["pair_id"]) else pd.NA,
                "mean_chunk_delta_l2": float(np.mean(col)),
                "median_chunk_delta_l2": float(np.median(col)),
                "std_chunk_delta_l2": float(np.std(col)),
                "p90_chunk_delta_l2": float(np.quantile(col, 0.90)),
                "num_frames": int(len(col)),
            }
        )
    return pd.DataFrame(rows)


def plot_chunk_delta_histograms(
    selected: pd.DataFrame,
    chunk_l2: np.ndarray,
    output_dir: Path,
    dpi: int = 180,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = ensure_dir(output_dir)
    values = np.asarray(chunk_l2, dtype=np.float32)
    groups = selected["probe_group"].astype(str).to_numpy()
    towers = selected["tower"].astype(str).to_numpy()
    colors = {"best": "#dc2626", "worst": "#6b7280", "candidate": "#2563eb"}

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=dpi)
    for group in ("worst", "best", "candidate"):
        mask = groups == group
        if not mask.any():
            continue
        ax.hist(
            values[:, mask].reshape(-1),
            bins=40,
            density=True,
            histtype="stepfilled",
            alpha=0.35,
            color=colors.get(group, "#111827"),
            label=f"{group} (n={int(mask.sum())} bins)",
        )
    ax.set_xlabel("action-chunk L2 vs baseline")
    ax.set_ylabel("density")
    ax.set_title("Per-frame bin ablation deltas")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "01_chunk_delta_hist_good_vs_bad.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), dpi=dpi, sharex=True, sharey=True)
    for ax, tower in zip(axes, ("paligemma", "expert")):
        for group in ("worst", "best", "candidate"):
            mask = (towers == tower) & (groups == group)
            if not mask.any():
                continue
            ax.hist(
                values[:, mask].reshape(-1),
                bins=40,
                density=True,
                histtype="stepfilled",
                alpha=0.35,
                color=colors.get(group, "#111827"),
                label=group,
            )
        ax.set_title(tower)
        ax.set_xlabel("action-chunk L2 vs baseline")
        ax.set_ylabel("density")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("Per-frame bin ablation deltas by tower")
    fig.tight_layout()
    fig.savefig(output_dir / "02_chunk_delta_hist_by_tower.png", bbox_inches="tight")
    plt.close(fig)

    diffs = per_frame_good_minus_bad(selected, values)
    plot_frame_good_minus_bad_hist(diffs, output_dir, dpi=dpi)


def per_frame_good_minus_bad(selected: pd.DataFrame, chunk_l2: np.ndarray) -> pd.DataFrame:
    """One row per frame: mean(good chunk Δ) − mean(bad chunk Δ)."""
    values = np.asarray(chunk_l2, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != len(selected):
        raise ValueError(
            f"chunk_l2 shape {values.shape} does not match {len(selected)} selected bins."
        )
    groups = selected["probe_group"].astype(str).to_numpy()
    towers = selected["tower"].astype(str).to_numpy()
    best = groups == "best"
    worst = groups == "worst"
    if not best.any() or not worst.any():
        raise ValueError("Need both probe_group='best' and 'worst' to form a per-frame difference.")
    frame = pd.DataFrame(
        {
            "frame_row": np.arange(len(values), dtype=np.int32),
            "good_minus_bad": values[:, best].mean(axis=1) - values[:, worst].mean(axis=1),
        }
    )
    for tower in ("paligemma", "expert"):
        tower_best = best & (towers == tower)
        tower_worst = worst & (towers == tower)
        if tower_best.any() and tower_worst.any():
            frame[f"good_minus_bad_{tower}"] = (
                values[:, tower_best].mean(axis=1) - values[:, tower_worst].mean(axis=1)
            )
    return frame


def plot_frame_good_minus_bad_hist(
    diffs: pd.DataFrame,
    output_dir: Path,
    dpi: int = 180,
    bins: int = 40,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = ensure_dir(output_dir)
    overall = diffs["good_minus_bad"].to_numpy(dtype=np.float32)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), dpi=dpi)
    lo, hi = np.nanpercentile(overall, [0.5, 99.5])
    span = max(float(hi - lo), 1e-6)
    edges = np.linspace(lo - 0.05 * span, hi + 0.05 * span, bins + 1)

    ax = axes[0]
    n_gt = int((overall > 0).sum())
    ax.hist(overall, bins=edges, color="#4b5563", alpha=0.75)
    ax.axvline(0.0, color="black", lw=1.2)
    ax.set_xlabel(r"mean $\Delta_{good}$ $-$ mean $\Delta_{bad}$")
    ax.set_ylabel("number of frames")
    ax.set_title(f"all towers  (good>bad in {n_gt}/{len(overall)} frames)")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    colors = {"paligemma": "#2563eb", "expert": "#dc2626"}
    for tower in ("paligemma", "expert"):
        column = f"good_minus_bad_{tower}"
        if column not in diffs.columns:
            continue
        subset = diffs[column].to_numpy(dtype=np.float32)
        n_tower = int((subset > 0).sum())
        ax.hist(
            subset,
            bins=edges,
            alpha=0.5,
            color=colors[tower],
            label=f"{tower}  good>bad in {n_tower}/{len(subset)}",
        )
    ax.axvline(0.0, color="black", lw=1.2)
    ax.set_xlabel(r"mean $\Delta_{good}$ $-$ mean $\Delta_{bad}$")
    ax.set_ylabel("number of frames")
    ax.set_title("by tower")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Per-frame good minus bad action-chunk delta")
    fig.tight_layout()
    path = output_dir / "03_frame_good_minus_bad_hist.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_frame_deltas(
    path: Path,
    *,
    sample_ids: np.ndarray,
    chunk_l2: np.ndarray,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "sample_ids": np.asarray(sample_ids, dtype=np.int32),
        "chunk_l2": np.asarray(chunk_l2, dtype=np.float16),
    }
    if extra:
        payload.update(extra)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(path)


def load_frame_deltas(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as packed:
        return {key: packed[key] for key in packed.files}


def run_offline_frame_ablation(
    frames: Sequence[OccupancyFrame],
    selected: pd.DataFrame,
    predict_fn: PredictFn,
    *,
    token_bins: int = 96,
    mode: str = "zero",
    scale: float = 0.0,
    checkpoint_path: Path | None = None,
    save_every: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """For each frame: one baseline chunk, then one fresh chunk per candidate bin."""
    interventions = interventions_from_selected(
        selected, token_bins=token_bins, mode=mode, scale=scale
    )
    n_frames = len(frames)
    n_bins = len(interventions)
    sample_ids = np.asarray([frame.sample_id for frame in frames], dtype=np.int32)
    libero_frame_index = np.asarray([frame.frame_index for frame in frames], dtype=np.int32)
    start = 0
    chunk_l2 = np.zeros((n_frames, n_bins), dtype=np.float32)
    if checkpoint_path is not None and checkpoint_path.exists():
        saved = load_frame_deltas(checkpoint_path)
        saved_ids = np.asarray(saved["sample_ids"], dtype=np.int32)
        if saved_ids.shape == sample_ids.shape and np.array_equal(saved_ids, sample_ids):
            chunk_l2 = np.asarray(saved["chunk_l2"], dtype=np.float32)
            done = np.asarray(saved.get("completed_frames", [0]))
            start = int(done.reshape(-1)[0])
            log(f"Resuming offline ablation from frame {start}/{n_frames}")
        else:
            log("Ignoring existing frame_deltas.npz because sampled frame ids differ")

    from tqdm import tqdm

    def _checkpoint(completed: int) -> None:
        if checkpoint_path is None:
            return
        save_frame_deltas(
            checkpoint_path,
            sample_ids=sample_ids,
            chunk_l2=chunk_l2,
            extra={
                "completed_frames": np.asarray([completed], dtype=np.int32),
                "libero_frame_index": libero_frame_index,
            },
        )

    for index in tqdm(range(start, n_frames), desc="offline PI0.5 frame ablation", initial=start, total=n_frames):
        frame = frames[index]
        images = load_frame_images(frame)
        baseline = predict_fn(frame, None, images)
        for bin_index, intervention in enumerate(interventions):
            ablated = predict_fn(frame, intervention, images)
            chunk_l2[index, bin_index] = chunk_delta_l2(ablated, baseline)
        if (index + 1) % max(int(save_every), 1) == 0 or index + 1 == n_frames:
            _checkpoint(index + 1)
    return sample_ids, chunk_l2


class Pi05OfflineChunkPredictor:
    """Fresh `predict_action_chunk` on every call; never `select_action`."""

    def __init__(self, model_id: str, device: str = "auto", wrapper: Pi05Wrapper | None = None) -> None:
        self.wrapper = wrapper if wrapper is not None else Pi05Wrapper(model_id=model_id, pooling="mean", device=device)
        self.modules = dict(_pi05_layer_modules(self.wrapper.policy))
        if not hasattr(self.wrapper.policy, "predict_action_chunk"):
            raise RuntimeError("PI0.5 policy has no predict_action_chunk; refusing select_action chunk reuse.")

    def predict(
        self,
        frame: OccupancyFrame,
        intervention: Pi05ActivationIntervention | None,
        images: dict[str, np.ndarray],
    ) -> np.ndarray:
        wrapper = self.wrapper
        batch_frame = wrapper._make_frame(
            images["image"],
            frame.instruction,
            {
                "observation_state": frame.observation_state,
                "wrist_image": images["wrist_image"],
            },
        )
        batch = _move_tensors_to_device(wrapper.preprocess(batch_frame), wrapper.device_obj)
        if hasattr(wrapper.policy, "reset"):
            wrapper.policy.reset()
        with _intervention_hooks(self.modules, intervention):
            with wrapper.torch.inference_mode():
                pred = wrapper.postprocess(wrapper.policy.predict_action_chunk(batch))
        return _normalize_action_chunk(_tensor_to_numpy(pred)).astype(np.float32)


def _pi05_layer_modules(policy: Any) -> list[tuple[str, Any]]:
    core = getattr(policy, "model", policy)
    expert = getattr(core, "paligemma_with_expert", None)
    candidates: list[tuple[str, Any]] = []
    if expert is None:
        raise RuntimeError("Could not locate paligemma_with_expert on the PI0.5 policy.")
    paligemma = getattr(expert, "paligemma", None)
    pg_layers = None
    for language_model in (
        getattr(paligemma, "language_model", None),
        getattr(getattr(paligemma, "model", None), "language_model", None),
    ):
        pg_layers = getattr(language_model, "layers", None)
        if pg_layers is not None:
            break
    if pg_layers is not None:
        candidates.extend((f"paligemma_layer_{index:02d}", layer) for index, layer in enumerate(pg_layers))
    expert_layers = getattr(getattr(getattr(expert, "gemma_expert", None), "model", None), "layers", None)
    if expert_layers is not None:
        candidates.extend((f"expert_layer_{index:02d}", layer) for index, layer in enumerate(expert_layers))
    if not candidates:
        raise RuntimeError("Could not locate PI0.5 transformer layers for frame ablation.")
    return candidates


@contextmanager
def _intervention_hooks(
    modules: dict[str, Any],
    intervention: Pi05ActivationIntervention | None,
) -> Iterator[None]:
    if intervention is None:
        yield
        return
    name = f"{intervention.tower}_layer_{int(intervention.layer_index):02d}"
    module = modules.get(name)
    if module is None:
        raise KeyError(f"No module named {name}. Available: {sorted(modules)[:8]}")

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        tensor = output[0] if isinstance(output, tuple) else output
        modified = _apply_pi05_activation_intervention(tensor, name, intervention)
        if modified is tensor:
            return None
        if isinstance(output, tuple):
            return (modified, *output[1:])
        return modified

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
