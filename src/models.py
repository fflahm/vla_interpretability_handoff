from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
import time
from typing import Any

import numpy as np


class ModelWrapper(ABC):
    @abstractmethod
    def forward(self, image: np.ndarray, instruction: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        raise NotImplementedError


class MockVLAWrapper(ModelWrapper):
    """Synthetic hidden states with a controllable target-directed signal."""

    def __init__(
        self,
        num_layers: int = 12,
        hidden_dim: int = 128,
        action_dim: int = 2,
        noise_std: float = 0.08,
        seed: int = 0,
    ) -> None:
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        self.offset_basis = self.rng.normal(size=(2, hidden_dim)).astype(np.float32)
        self.offset_basis /= np.linalg.norm(self.offset_basis, axis=1, keepdims=True) + 1e-8
        self.bg_basis = self.rng.normal(size=(3, hidden_dim)).astype(np.float32)
        self.bg_basis /= np.linalg.norm(self.bg_basis, axis=1, keepdims=True) + 1e-8
        self.action_noise = float(noise_std) * 0.35

    def forward(self, image: np.ndarray, instruction: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        if metadata is None or "target_offset" not in metadata:
            offset = self._estimate_offset_from_image(image)
            background = image.reshape(-1, 3).mean(axis=0) / 255.0
        else:
            offset = np.asarray(metadata["target_offset"], dtype=np.float32)[:2]
            background = np.asarray(metadata.get("background_color", [224, 224, 224]), dtype=np.float32) / 255.0

        norm = np.linalg.norm(offset) + 1e-8
        pred_action = (offset / norm) * min(norm * 5.0, 1.0)
        pred_action = pred_action + self.rng.normal(0.0, self.action_noise, size=self.action_dim)

        hidden_states: dict[str, np.ndarray] = {}
        for layer in range(self.num_layers):
            signal_strength = layer / max(self.num_layers - 1, 1)
            bg_strength = max(0.0, 0.25 - 0.015 * layer)
            signal = signal_strength * (offset @ self.offset_basis)
            bg_signal = bg_strength * (background @ self.bg_basis)
            noise = self.rng.normal(0.0, self.noise_std, size=self.hidden_dim)
            hidden_states[f"layer_{layer:02d}"] = (signal + bg_signal + noise).astype(np.float32)

        return {"pred_action": pred_action.astype(np.float32), "hidden_states": hidden_states}

    @staticmethod
    def _estimate_offset_from_image(image: np.ndarray) -> np.ndarray:
        red_mask = (image[..., 0] > 150) & (image[..., 1] < 90) & (image[..., 2] < 90)
        dark_mask = image.mean(axis=-1) < 45
        if not red_mask.any() or not dark_mask.any():
            return np.zeros(2, dtype=np.float32)
        tyx = np.argwhere(red_mask).mean(axis=0)
        gyx = np.argwhere(dark_mask).mean(axis=0)
        diff_yx = tyx - gyx
        return np.asarray([diff_yx[1], diff_yx[0]], dtype=np.float32) / float(image.shape[0])


class Pi05Wrapper(ModelWrapper):
    """LeRobot pi0.5 wrapper with best-effort activation hooks."""

    POOLING_MODES = ("mean", "last", "flatten")

    def __init__(
        self,
        model_id: str = "lerobot/pi05_libero_finetuned_quantiles",
        pooling: str = "mean",
        device: str = "auto",
        **_: object,
    ) -> None:
        self.model_id = model_id
        self.pooling = pooling
        self.device = device
        if pooling not in self.POOLING_MODES:
            raise ValueError(f"`pooling` must be one of {self.POOLING_MODES}, got `{pooling}`.")
        try:
            import torch
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.pi05 import PI05Policy
        except Exception as exc:
            raise RuntimeError(
                "Pi05Wrapper needs LeRobot pi0.5 dependencies. Install on a suitable Linux/GPU machine with "
                '`python -m pip install "lerobot[pi]@git+https://github.com/huggingface/lerobot.git"` '
                "or run `--model mock`."
            ) from exc

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.device_obj = torch.device(device)
        self.num_predictions = 0
        config = PreTrainedConfig.from_pretrained(model_id)
        config.compile_model = False
        # LeRobot's PI0.5 port may report the tied language embedding as missing
        # even though its shared checkpoint tensor is loaded. Strict loading
        # aborts the entire load in that case and silently returns random weights.
        self.policy = PI05Policy.from_pretrained(
            model_id, config=config, strict=False
        ).to(self.device_obj).eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            model_id,
            preprocessor_overrides={"device_processor": {"device": str(self.device_obj)}},
        )

    def forward(self, image: np.ndarray, instruction: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        frame = self._make_frame(image, instruction, metadata)
        batch = self.preprocess(frame)
        batch = _move_tensors_to_device(batch, self.device_obj)
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        started = time.perf_counter()
        if self.num_predictions == 0:
            print("Running first PI0.5 activation-capture inference. This may take a while.", flush=True)
        with self._capture_hidden_states() as hidden_store:
            with self.torch.inference_mode():
                if hasattr(self.policy, "predict_action_chunk"):
                    pred_action_chunk = self.postprocess(self.policy.predict_action_chunk(batch))
                    pred_action_chunk_np = _normalize_action_chunk(_tensor_to_numpy(pred_action_chunk))
                    pred_action = pred_action_chunk_np[0]
                else:
                    pred_action = self.postprocess(self.policy.select_action(batch))
                    pred_action = _tensor_to_numpy(pred_action).reshape(-1)
                    pred_action_chunk_np = None
        hidden_states = {name: self._pool_tensor(tensor) for name, tensor in hidden_store.items()}
        elapsed = time.perf_counter() - started
        if self.num_predictions == 0:
            print(f"First PI0.5 activation-capture inference completed in {elapsed:.1f}s.", flush=True)
        self.num_predictions += 1
        output = {"pred_action": np.asarray(pred_action, dtype=np.float32).reshape(-1), "hidden_states": hidden_states}
        if pred_action_chunk_np is not None:
            output["pred_action_chunk"] = pred_action_chunk_np.astype(np.float32)
        return output

    def _make_frame(self, image: np.ndarray, instruction: str, metadata: dict[str, Any] | None) -> dict[str, Any]:
        state = np.zeros(8, dtype=np.float32)
        if metadata is not None and "gripper_position" in metadata:
            state[:3] = np.asarray(metadata["gripper_position"], dtype=np.float32)[:3]
        if metadata is not None and "observation_state" in metadata:
            raw_state = np.asarray(metadata["observation_state"], dtype=np.float32)
            state[: min(len(state), len(raw_state))] = raw_state[: len(state)]

        # Raw LIBERO simulator frames are rotated relative to the dataset convention.
        image_uint8 = np.flip(image.astype(np.uint8), axis=(0, 1)).copy()
        wrist_image = image_uint8
        if metadata is not None and metadata.get("wrist_image") is not None:
            wrist_image = np.flip(np.asarray(metadata["wrist_image"], dtype=np.uint8), axis=(0, 1)).copy()
        elif metadata is not None and metadata.get("wrist_image_path"):
            wrist_image = np.flip(load_image_np(metadata["wrist_image_path"]).astype(np.uint8), axis=(0, 1)).copy()
        return {
            # PI0.5 base preprocessing pads the state before device processing,
            # so policy inputs must already be tensors at this boundary.
            "observation.state": self.torch.from_numpy(state[None, :]),
            "observation.images.image": self.torch.from_numpy(_hwc_to_chw(image_uint8)[None, :]),
            "observation.images.wrist_image": self.torch.from_numpy(_hwc_to_chw(wrist_image)[None, :]),
            "task": [instruction],
        }

    @contextmanager
    def _capture_hidden_states(self) -> Any:
        hidden: dict[str, Any] = {}
        handles = []

        def save_hook(name: str):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                tensor = output[0] if isinstance(output, tuple) else output
                hidden[name] = tensor.detach() if hasattr(tensor, "detach") else tensor

            return hook

        modules = self._hidden_modules()
        if self.num_predictions == 0:
            print(f"Capturing {len(modules)} PI0.5 transformer layer outputs.", flush=True)
        for name, module in modules:
            handles.append(module.register_forward_hook(save_hook(name)))
        try:
            yield hidden
        finally:
            for handle in handles:
                handle.remove()

    def _hidden_modules(self) -> list[tuple[str, Any]]:
        candidates: list[tuple[str, Any]] = []
        core = getattr(self.policy, "model", self.policy)
        expert = getattr(core, "paligemma_with_expert", None)
        if expert is not None:
            pg_layers = getattr(getattr(getattr(expert, "paligemma", None), "model", None), "language_model", None)
            pg_layers = getattr(pg_layers, "layers", None)
            if pg_layers is not None:
                candidates.extend((f"paligemma_layer_{i:02d}", layer) for i, layer in enumerate(pg_layers))
            ex_layers = getattr(getattr(getattr(expert, "gemma_expert", None), "model", None), "layers", None)
            if ex_layers is not None:
                candidates.extend((f"expert_layer_{i:02d}", layer) for i, layer in enumerate(ex_layers))
        if candidates:
            return candidates

        for name, module in self.policy.named_modules():
            lname = name.lower()
            if lname.endswith("layers") or ".layers." not in lname:
                continue
            if any(part in lname for part in ("language_model.layers", "gemma_expert.model.layers")):
                candidates.append((name.replace(".", "_"), module))
        if not candidates:
            raise RuntimeError("Could not locate pi0.5 transformer layers for activation hooks.")
        return candidates

    def _pool_tensor(self, tensor: Any) -> np.ndarray:
        if not hasattr(tensor, "detach"):
            return np.asarray(tensor, dtype=np.float32).reshape(-1)
        tensor = tensor.detach()
        if tensor.ndim >= 3:
            if self.pooling == "mean":
                tensor = tensor.mean(dim=1)
            elif self.pooling == "last":
                tensor = tensor[:, -1]
            elif self.pooling == "flatten":
                tensor = tensor.reshape(tensor.shape[0], -1)
        if tensor.ndim >= 2:
            tensor = tensor[0]
        return tensor.float().cpu().numpy().reshape(-1).astype(np.float32)


def load_image_np(path: str | Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def _tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    if isinstance(value, dict):
        for key in ("action", "actions", "pred_action"):
            if key in value:
                return _tensor_to_numpy(value[key])
    return np.asarray(value, dtype=np.float32)


def _normalize_action_chunk(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 3:
        value = value[0]
    if value.ndim == 1:
        value = value[None, :]
    if value.ndim != 2:
        raise ValueError(f"Expected action chunk with shape [horizon, action_dim], got {value.shape}.")
    return value


def _hwc_to_chw(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected an HWC image, got shape {image.shape}.")
    return np.ascontiguousarray(np.moveaxis(image, -1, 0))


def _move_tensors_to_device(value: Any, device: Any) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_tensors_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tensors_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors_to_device(item, device) for item in value)
    return value
