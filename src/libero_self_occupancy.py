from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull
from tqdm import tqdm

from .activations import stack_layer_activations
from .libero_rich_annotations import (
    LOCAL_LIBERO_ROOT,
    SimulatorReplay,
    _resolve_bddl,
    native,
    repair_asset_paths,
    robosuite_assets_root,
)
from .models import Pi05Wrapper
from .utils import ensure_dir, log


GEOM_PLANE = 0
GEOM_HFIELD = 1
GEOM_SPHERE = 2
GEOM_CAPSULE = 3
GEOM_ELLIPSOID = 4
GEOM_CYLINDER = 5
GEOM_BOX = 6
GEOM_MESH = 7


@dataclass(frozen=True)
class OccupancyGridSpec:
    """Fixed Panda-base-frame voxel grid shared by every sample."""

    size: int = 16
    lower: tuple[float, float, float] = (-0.8, -0.8, 0.0)
    upper: tuple[float, float, float] = (0.8, 0.8, 1.6)
    supersample: int = 2

    def __post_init__(self) -> None:
        if self.size <= 1:
            raise ValueError(f"Grid size must exceed one, got {self.size}.")
        if self.supersample <= 0:
            raise ValueError(f"Supersample must be positive, got {self.supersample}.")
        if any(hi <= lo for lo, hi in zip(self.lower, self.upper)):
            raise ValueError(f"Invalid bounds: lower={self.lower}, upper={self.upper}.")

    @property
    def voxel_size(self) -> tuple[float, float, float]:
        return tuple((hi - lo) / self.size for lo, hi in zip(self.lower, self.upper))

    def high_resolution_centers(self) -> np.ndarray:
        high_size = self.size * self.supersample
        axes = [
            np.linspace(lo, hi, high_size, endpoint=False, dtype=np.float64)
            + (hi - lo) / (2.0 * high_size)
            for lo, hi in zip(self.lower, self.upper)
        ]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)

    def soften(self, occupied_high_resolution: np.ndarray) -> np.ndarray:
        expected = (self.size * self.supersample,) * 3
        if occupied_high_resolution.shape != expected:
            raise ValueError(
                f"Expected supersampled occupancy {expected}, got {occupied_high_resolution.shape}."
            )
        s = self.supersample
        return occupied_high_resolution.reshape(
            self.size, s, self.size, s, self.size, s
        ).mean(axis=(1, 3, 5), dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinate_frame": "panda_base",
            "shape": [self.size] * 3,
            "lower_m": list(self.lower),
            "upper_m": list(self.upper),
            "voxel_size_m": list(self.voxel_size),
            "supersample": self.supersample,
            "soft_target": "fraction of occupied subvoxels",
        }


@dataclass
class SelfOccupancySample:
    sample_id: int
    demo_key: str
    frame_index: int
    image_path: str
    wrist_image_path: str
    instruction: str
    observation_state: list[float]
    occupancy: np.ndarray
    suite: str = "libero_spatial"
    task: str = ""

    def metadata(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "episode_id": self.episode_id,
            "suite": self.suite,
            "task": self.task,
            "demo_key": self.demo_key,
            "frame_index": self.frame_index,
            "image_path": self.image_path,
            "wrist_image_path": self.wrist_image_path,
            "instruction": self.instruction,
            "observation_state": self.observation_state,
        }

    @property
    def episode_id(self) -> str:
        return f"{self.suite}/{self.task}/{self.demo_key}"


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        raise ValueError(f"Trajectory length must be positive, got {length}.")
    if count <= 0:
        raise ValueError(f"Frame count must be positive, got {count}.")
    if count >= length:
        return list(range(length))
    indices = np.rint(np.linspace(0, length - 1, count + 2)[1:-1]).astype(int)
    return sorted(set(int(index) for index in indices))


class RobotCollisionVoxelizer:
    """Voxelize MuJoCo's convex robot collision geometry in Panda base frame."""

    def __init__(self, replay: SimulatorReplay, spec: OccupancyGridSpec) -> None:
        self.replay = replay
        self.sim = replay.sim
        self.spec = spec
        self.grid = spec.high_resolution_centers()
        self.points = self.grid.reshape(-1, 3)
        self.base_body_id = self._find_base_body_id()
        self.robot_geom_ids = self._find_robot_collision_geoms()
        self._mesh_halfspaces: dict[int, np.ndarray] = {}
        if not self.robot_geom_ids:
            raise RuntimeError("No dynamic Panda collision geoms were found.")

    def _find_base_body_id(self) -> int:
        preferred = ("robot0_base", "robot0_link0")
        for name in preferred:
            try:
                body_id = int(self.sim.model.body_name2id(name))
            except Exception:
                continue
            if body_id >= 0:
                return body_id
        for body_id in range(int(self.sim.model.nbody)):
            name = self.sim.model.body_id2name(body_id) or ""
            if name.startswith("robot0_"):
                return body_id
        raise RuntimeError("Could not locate the Panda base body.")

    def _find_robot_collision_geoms(self) -> list[int]:
        geom_ids: list[int] = []
        for geom_id in range(int(self.sim.model.ngeom)):
            if int(self.sim.model.geom_group[geom_id]) != 0:
                continue
            body_id = int(self.sim.model.geom_bodyid[geom_id])
            body_name = self.sim.model.body_id2name(body_id) or ""
            if body_name.startswith(("robot0_", "gripper0_")):
                geom_ids.append(geom_id)
        return geom_ids

    def voxelize(self) -> np.ndarray:
        base_pos = np.asarray(self.sim.data.body_xpos[self.base_body_id], dtype=np.float64)
        base_rot = np.asarray(self.sim.data.body_xmat[self.base_body_id], dtype=np.float64).reshape(3, 3)
        occupied = np.zeros(len(self.points), dtype=bool)
        for geom_id in self.robot_geom_ids:
            geom_pos_world = np.asarray(self.sim.data.geom_xpos[geom_id], dtype=np.float64)
            geom_rot_world = np.asarray(self.sim.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
            geom_pos_base = base_rot.T @ (geom_pos_world - base_pos)
            geom_rot_base = base_rot.T @ geom_rot_world
            local_points = (self.points - geom_pos_base) @ geom_rot_base
            occupied |= self._inside_geom(geom_id, local_points)
        high_shape = (self.spec.size * self.spec.supersample,) * 3
        return self.spec.soften(occupied.reshape(high_shape))

    def _inside_geom(self, geom_id: int, points: np.ndarray) -> np.ndarray:
        geom_type = int(self.sim.model.geom_type[geom_id])
        size = np.asarray(self.sim.model.geom_size[geom_id], dtype=np.float64)
        if geom_type == GEOM_SPHERE:
            return np.einsum("ij,ij->i", points, points) <= size[0] ** 2
        if geom_type == GEOM_CAPSULE:
            radial_sq = points[:, 0] ** 2 + points[:, 1] ** 2
            axial = np.maximum(np.abs(points[:, 2]) - size[1], 0.0)
            return radial_sq + axial**2 <= size[0] ** 2
        if geom_type == GEOM_ELLIPSOID:
            safe_size = np.maximum(size[:3], 1e-9)
            return np.sum((points / safe_size) ** 2, axis=1) <= 1.0
        if geom_type == GEOM_CYLINDER:
            return (
                points[:, 0] ** 2 + points[:, 1] ** 2 <= size[0] ** 2
            ) & (np.abs(points[:, 2]) <= size[1])
        if geom_type == GEOM_BOX:
            return np.all(np.abs(points) <= size[:3] + 1e-9, axis=1)
        if geom_type == GEOM_MESH:
            equations = self._mesh_equations(geom_id)
            return np.all(points @ equations[:, :3].T + equations[:, 3] <= 1e-8, axis=1)
        return np.zeros(len(points), dtype=bool)

    def _mesh_equations(self, geom_id: int) -> np.ndarray:
        data_id = int(self.sim.model.geom_dataid[geom_id])
        if data_id in self._mesh_halfspaces:
            return self._mesh_halfspaces[data_id]
        start = int(self.sim.model.mesh_vertadr[data_id])
        count = int(self.sim.model.mesh_vertnum[data_id])
        vertices = np.asarray(self.sim.model.mesh_vert[start : start + count], dtype=np.float64)
        if len(vertices) < 4:
            raise RuntimeError(f"Mesh data {data_id} has only {len(vertices)} vertices.")
        equations = np.asarray(ConvexHull(vertices).equations, dtype=np.float64)
        self._mesh_halfspaces[data_id] = equations
        return equations

    def metadata(self) -> dict[str, Any]:
        return {
            **self.spec.to_dict(),
            "geometry": "MuJoCo convex collision geoms",
            "included_body_prefixes": ["robot0_", "gripper0_"],
            "excluded": ["mount0_", "environment", "task_objects", "held_objects"],
            "geom_ids": self.robot_geom_ids,
            "geom_names": [
                self.sim.model.geom_id2name(geom_id) or f"geom_{geom_id}"
                for geom_id in self.robot_geom_ids
            ],
            "base_body": self.sim.model.body_id2name(self.base_body_id),
        }


def collect_self_occupancy_samples(
    *,
    hdf5_path: Path,
    output_dir: Path,
    num_demos: int,
    frames_per_demo: int,
    spec: OccupancyGridSpec,
    libero_root: Path = LOCAL_LIBERO_ROOT,
) -> tuple[list[SelfOccupancySample], dict[str, Any]]:
    """Replay selected HDF5 frames and create images plus soft occupancy GT."""

    output_dir = ensure_dir(output_dir)
    image_dir = ensure_dir(output_dir / "images")
    samples: list[SelfOccupancySample] = []
    episode_timings: list[dict[str, Any]] = []
    occupancy_metadata: dict[str, Any] | None = None
    started = time.perf_counter()
    with h5py.File(hdf5_path, "r") as handle:
        data = handle["data"]
        instruction = _instruction(data, hdf5_path)
        demo_keys = sorted(data.keys(), key=_demo_sort_key)[:num_demos]
        task = hdf5_path.name.removesuffix("_demo.hdf5")
        log(
            f"GT collect start task={task!r} demos={len(demo_keys)} "
            f"frames_per_demo={frames_per_demo} grid={spec.size}^3"
        )
        bddl_path = _resolve_bddl(handle, task, "libero_spatial", libero_root)
        if bddl_path is None:
            raise FileNotFoundError(f"Could not resolve BDDL for {hdf5_path}.")
        assets_root = libero_root / "libero" / "libero" / "assets"
        robosuite_assets = robosuite_assets_root()

        demo_bar = tqdm(demo_keys, desc=f"GT demos [{task[:40]}]", unit="demo", leave=True)
        for demo_key in demo_bar:
            episode_started = time.perf_counter()
            demo = data[demo_key]
            obs = demo["obs"]
            frame_indices = evenly_spaced_indices(len(demo["states"]), frames_per_demo)
            repaired_xml, _ = repair_asset_paths(
                native(demo.attrs.get("model_file", "")), assets_root, robosuite_assets
            )
            replay = SimulatorReplay(bddl_path, repaired_xml)
            try:
                voxelizer = RobotCollisionVoxelizer(replay, spec)
                occupancy_metadata = voxelizer.metadata()
                for frame_index in frame_indices:
                    replay.sim.set_state_from_flattened(
                        np.asarray(demo["states"][frame_index], dtype=np.float64)
                    )
                    replay.sim.forward()
                    occupancy = voxelizer.voxelize()
                    sample_id = len(samples)
                    image_path = image_dir / f"{sample_id:04d}_agentview.png"
                    wrist_path = image_dir / f"{sample_id:04d}_wrist.png"
                    Image.fromarray(np.asarray(obs["agentview_rgb"][frame_index], dtype=np.uint8)).save(image_path)
                    Image.fromarray(np.asarray(obs["eye_in_hand_rgb"][frame_index], dtype=np.uint8)).save(wrist_path)
                    observation_state = np.concatenate(
                        (
                            np.asarray(obs["ee_states"][frame_index], dtype=np.float32),
                            np.asarray(obs["gripper_states"][frame_index], dtype=np.float32),
                        )
                    )
                    samples.append(
                        SelfOccupancySample(
                            sample_id=sample_id,
                            demo_key=demo_key,
                            frame_index=frame_index,
                            image_path=str(image_path),
                            wrist_image_path=str(wrist_path),
                            instruction=instruction,
                            observation_state=observation_state.astype(float).tolist(),
                            occupancy=occupancy.astype(np.float32),
                        )
                    )
            finally:
                replay.close()
            episode_seconds = time.perf_counter() - episode_started
            episode_timings.append(
                {
                    "demo_key": demo_key,
                    "frames": frame_indices,
                    "seconds": episode_seconds,
                }
            )
            demo_bar.set_postfix(
                samples=len(samples),
                last_s=f"{episode_seconds:.1f}",
                occ=f"{float(samples[-1].occupancy.mean()):.4f}" if samples else "n/a",
            )

    if not samples or occupancy_metadata is None:
        raise RuntimeError("No occupancy samples were generated.")
    elapsed = time.perf_counter() - started
    log(
        f"GT collect done task={hdf5_path.name.removesuffix('_demo.hdf5')!r} "
        f"samples={len(samples)} demos={len({s.demo_key for s in samples})} "
        f"seconds={elapsed:.1f} rate={len(samples) / max(elapsed, 1e-6):.2f} frames/s"
    )
    metadata = {
        "hdf5_path": str(hdf5_path),
        "num_demos": len({sample.demo_key for sample in samples}),
        "frames_per_demo": frames_per_demo,
        "num_samples": len(samples),
        "occupancy": occupancy_metadata,
        "episodes": episode_timings,
        "seconds": elapsed,
    }
    return samples, metadata


def extract_pi05_layer_activations(
    samples: Sequence[SelfOccupancySample],
    *,
    model_id: str,
    device: str = "auto",
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    started = time.perf_counter()
    model_started = time.perf_counter()
    log(f"Loading PI0.5 for mean-token activation capture from {model_id}")
    model = Pi05Wrapper(model_id=model_id, pooling="mean", device=device)
    model_load_seconds = time.perf_counter() - model_started
    log(f"PI0.5 loaded in {model_load_seconds:.1f}s device={model.device_obj}")
    hidden_by_sample: list[dict[str, np.ndarray]] = []
    inference_seconds: list[float] = []
    sample_bar = tqdm(samples, desc="PI0.5 activations", unit="frame")
    for sample in sample_bar:
        image = np.asarray(Image.open(sample.image_path).convert("RGB"))
        item_started = time.perf_counter()
        output = model.forward(
            image=image,
            instruction=sample.instruction,
            metadata={
                "observation_state": sample.observation_state,
                "wrist_image_path": sample.wrist_image_path,
            },
        )
        elapsed = time.perf_counter() - item_started
        inference_seconds.append(elapsed)
        hidden_by_sample.append(output["hidden_states"])
        mean_s = float(np.mean(inference_seconds))
        sample_bar.set_postfix(last_s=f"{elapsed:.1f}", mean_s=f"{mean_s:.1f}")
    activations, layer_names = stack_layer_activations(hidden_by_sample)
    timing = {
        "model_load_seconds": model_load_seconds,
        "inference_total_seconds": float(sum(inference_seconds)),
        "inference_mean_seconds_per_frame": float(np.mean(inference_seconds)),
        "inference_seconds_per_frame": inference_seconds,
        "activation_total_seconds": time.perf_counter() - started,
    }
    log(
        f"Activation capture done samples={len(samples)} layers={len(layer_names)} "
        f"shape={tuple(activations.shape)} mean_s/frame={timing['inference_mean_seconds_per_frame']:.2f}"
    )
    return activations, layer_names, timing


def grouped_demo_split(
    samples: Sequence[SelfOccupancySample], test_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    demos = sorted({sample.demo_key for sample in samples})
    if len(demos) < 2:
        raise ValueError("A grouped split needs at least two demonstrations.")
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(demos))
    test_count = max(1, min(len(demos) - 1, int(math.ceil(len(demos) * test_fraction))))
    test_demos = set(shuffled[:test_count])
    test_idx = np.asarray(
        [index for index, sample in enumerate(samples) if sample.demo_key in test_demos], dtype=int
    )
    train_idx = np.asarray(
        [index for index, sample in enumerate(samples) if sample.demo_key not in test_demos], dtype=int
    )
    return train_idx, test_idx


def stratified_episode_split(
    rows: Sequence[SelfOccupancySample | dict[str, Any]], test_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split demos independently inside each task; frames from a demo never cross the split."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between zero and one.")

    def value(row: SelfOccupancySample | dict[str, Any], name: str) -> str:
        if isinstance(row, dict):
            return str(row[name])
        return str(getattr(row, name))

    tasks: dict[str, set[str]] = {}
    for row in rows:
        task = value(row, "task")
        episode = value(row, "episode_id")
        tasks.setdefault(task, set()).add(episode)
    rng = np.random.default_rng(seed)
    test_episodes: set[str] = set()
    for task, episodes_set in sorted(tasks.items()):
        episodes = sorted(episodes_set)
        if len(episodes) < 2:
            raise ValueError(f"Task {task!r} needs at least two demos for a grouped split.")
        count = max(1, min(len(episodes) - 1, int(math.ceil(len(episodes) * test_fraction))))
        test_episodes.update(rng.permutation(episodes)[:count].tolist())
    test_idx = np.asarray(
        [i for i, row in enumerate(rows) if value(row, "episode_id") in test_episodes], dtype=int
    )
    train_idx = np.asarray(
        [i for i, row in enumerate(rows) if value(row, "episode_id") not in test_episodes], dtype=int
    )
    return train_idx, test_idx


def token_position_bins(tokens: np.ndarray, requested_bins: int) -> np.ndarray:
    """Demo3-style equal-width token bins with no empty bins."""
    tokens = np.asarray(tokens)
    if tokens.ndim < 2:
        raise ValueError(f"Expected [..., tokens, hidden], got {tokens.shape}.")
    if requested_bins <= 0:
        raise ValueError("requested_bins must be positive.")
    length = tokens.shape[-2]
    effective = min(requested_bins, length)
    edges = np.linspace(0, length, effective + 1, dtype=int)
    return np.stack(
        [tokens[..., edges[i] : edges[i + 1], :].mean(axis=-2) for i in range(effective)],
        axis=-2,
    )


def parse_index_spec(spec: str, max_value: int, stride: int = 1) -> list[int]:
    """Parse `all`, comma lists, or inclusive ranges into indices in ``[0, max_value)``.

    Examples: ``all``, ``0,8,16``, ``0-7``, ``0-95:8``.
    """
    if max_value <= 0:
        raise ValueError(f"max_value must be positive, got {max_value}.")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}.")
    text = str(spec).strip()
    if not text:
        raise ValueError("Index specification must be non-empty.")
    if text == "all":
        return list(range(0, max_value, stride))
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part and "-" in part:
            range_part, stride_part = part.rsplit(":", 1)
            local_stride = int(stride_part)
            if local_stride <= 0:
                raise ValueError(f"Range stride must be positive, got {local_stride}.")
            start, end = [int(x) for x in range_part.split("-", 1)]
            values.extend(range(start, end + 1, local_stride))
        elif "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))
    selected = sorted({value for value in values if 0 <= value < max_value})
    if not selected:
        raise ValueError(
            f"Index specification `{spec}` selected no indices in [0, {max_value})."
        )
    return selected


def soft_iou(target: np.ndarray, prediction: np.ndarray, eps: float = 1e-8) -> float:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    intersection = np.minimum(target, prediction).reshape(len(target), -1).sum(axis=1)
    union = np.maximum(target, prediction).reshape(len(target), -1).sum(axis=1)
    return float(np.mean((intersection + eps) / (union + eps)))


def hard_iou(target: np.ndarray, prediction: np.ndarray, threshold: float = 0.5) -> float:
    target_binary = np.asarray(target) >= threshold
    prediction_binary = np.asarray(prediction) >= threshold
    intersection = np.logical_and(target_binary, prediction_binary).reshape(len(target_binary), -1).sum(axis=1)
    union = np.logical_or(target_binary, prediction_binary).reshape(len(target_binary), -1).sum(axis=1)
    return float(np.mean((intersection + 1e-8) / (union + 1e-8)))


def train_layerwise_decoders(
    *,
    activations: np.ndarray,
    occupancy: np.ndarray,
    layer_names: Sequence[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    epochs: int,
    bottleneck: int,
    learning_rate: float,
    seed: int,
    device: str = "auto",
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    import torch
    from torch import nn

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    targets = occupancy.reshape(len(occupancy), -1).astype(np.float32)
    y_train = torch.from_numpy(targets[train_idx]).to(device_obj)
    positive_fraction = float(np.mean(targets[train_idx]))
    pos_weight_value = min(30.0, max(1.0, (1.0 - positive_fraction) / max(positive_fraction, 1e-6)))
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device_obj)
    rows: list[dict[str, Any]] = []
    predictions: list[np.ndarray] = []
    total_started = time.perf_counter()
    log(
        f"Training layerwise occupancy decoders layers={len(layer_names)} "
        f"train={len(train_idx)} test={len(test_idx)} epochs={epochs} "
        f"pos_frac={positive_fraction:.4f} pos_weight={pos_weight_value:.2f} device={device_obj}"
    )

    layer_bar = tqdm(list(enumerate(layer_names)), desc="train occupancy layers", unit="layer")
    for layer_index, layer_name in layer_bar:
        layer_started = time.perf_counter()
        torch.manual_seed(seed + layer_index)
        x = activations[layer_index].astype(np.float32)
        mean = x[train_idx].mean(axis=0, keepdims=True)
        std = x[train_idx].std(axis=0, keepdims=True)
        std[std < 1e-5] = 1.0
        x = (x - mean) / std
        x_train = torch.from_numpy(x[train_idx]).to(device_obj)
        x_test = torch.from_numpy(x[test_idx]).to(device_obj)
        decoder = nn.Sequential(
            nn.Linear(x.shape[1], bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, targets.shape[1]),
        ).to(device_obj)
        optimizer = torch.optim.AdamW(decoder.parameters(), lr=learning_rate, weight_decay=1e-4)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        last_loss = float("nan")
        decoder.train()
        for _ in range(epochs):
            logits = decoder(x_train)
            probabilities = torch.sigmoid(logits)
            intersection = (probabilities * y_train).sum(dim=1)
            dice_loss = 1.0 - (
                (2.0 * intersection + 1e-6)
                / (probabilities.sum(dim=1) + y_train.sum(dim=1) + 1e-6)
            ).mean()
            loss = bce(logits, y_train) + dice_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())
        decoder.eval()
        with torch.inference_mode():
            prediction = torch.sigmoid(decoder(x_test)).cpu().numpy()
        prediction_grid = prediction.reshape((len(test_idx),) + tuple(occupancy.shape[1:]))
        predictions.append(prediction_grid.astype(np.float32))
        soft = soft_iou(occupancy[test_idx], prediction_grid)
        hard = hard_iou(occupancy[test_idx], prediction_grid)
        train_seconds = time.perf_counter() - layer_started
        rows.append(
            {
                "layer": layer_index,
                "layer_name": layer_name,
                "tower": "paligemma" if layer_name.startswith("paligemma") else "expert",
                "soft_iou": soft,
                "hard_iou": hard,
                "train_loss": last_loss,
                "train_seconds": train_seconds,
            }
        )
        layer_bar.set_postfix(
            soft_iou=f"{soft:.3f}",
            hard_iou=f"{hard:.3f}",
            loss=f"{last_loss:.3f}",
            s=f"{train_seconds:.1f}",
        )

    timing = {
        "decoder_total_seconds": time.perf_counter() - total_started,
        "decoder_mean_seconds_per_layer": float(np.mean([row["train_seconds"] for row in rows])),
        "epochs": epochs,
        "bottleneck": bottleneck,
        "learning_rate": learning_rate,
        "positive_fraction_train": positive_fraction,
        "bce_pos_weight": pos_weight_value,
        "device": str(device_obj),
    }
    best = max(rows, key=lambda row: float(row["soft_iou"]))
    log(
        f"Layerwise training done in {timing['decoder_total_seconds']:.1f}s "
        f"best={best['layer_name']} soft_iou={best['soft_iou']:.4f}"
    )
    return rows, np.stack(predictions, axis=0), timing


def write_json(path: Path, value: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(native(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(native(row), ensure_ascii=False) + "\n")


def _instruction(data: h5py.Group, hdf5_path: Path) -> str:
    try:
        problem_info = json.loads(str(data.attrs.get("problem_info", "{}")))
        instruction = str(problem_info.get("language_instruction", "")).strip()
    except (TypeError, ValueError, json.JSONDecodeError):
        instruction = ""
    return instruction or hdf5_path.stem.removesuffix("_demo").replace("_", " ")


def _demo_sort_key(name: str) -> tuple[int, str]:
    try:
        return int(name.rsplit("_", 1)[1]), name
    except (IndexError, ValueError):
        return 10**9, name
