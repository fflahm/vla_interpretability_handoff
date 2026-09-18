from __future__ import annotations

import inspect
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import trange

from .envs import LiberoEnv
from .online_rollout import (
    DEFAULT_RENAME_MAP,
    _extract_image,
    _extract_success,
    _instruction_joint_matches,
    _joint_xyz,
    _make_env_pre_post_processors_compat,
    _max_episode_steps,
    _move_tensors_to_device,
    _preprocess_observation_compat,
    _rotate_observation_images,
    _save_episode_video,
    _to_numpy,
    _unwrap_single_env,
)
from .utils import ensure_dir, paligemma_tokenizer_overrides


@dataclass(frozen=True)
class ActivationIntervention:
    """Simple token-region intervention applied to one transformer block output."""

    layer_index: int
    token_start: int | None = None
    token_end: int | None = None
    token_bin_index: int | None = None
    token_bins: int | None = None
    mode: str = "zero"
    scale: float = 0.0


class Pi0LiberoFullTokenRolloutTracer:
    """Runs PI0 in LIBERO and saves full sequence activations per step/layer."""

    def __init__(
        self,
        checkpoint_path: str,
        task: str = "libero_spatial",
        task_id: int = 1,
        device: str = "auto",
        pickup_joint_name: str = "auto",
        place_joint_name: str = "auto",
        instruction: str = "pick up the black bowl from table center and place it on the plate",
        rename_map: dict[str, str] | None = None,
        rotate_images_180: bool = True,
        force_replan_every_step: bool = True,
        replan_interval: int = 1,
        activation_dtype: str = "float16",
        intervention: ActivationIntervention | None = None,
    ) -> None:
        try:
            import torch
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.envs.configs import LiberoEnv as LeRobotLiberoConfig
            from lerobot.envs.factory import make_env
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.pi0 import PI0Policy
        except Exception as exc:
            raise RuntimeError(
                "Online PI0 LIBERO tracing requires LeRobot with PI and LIBERO extras. "
                'Install `lerobot[pi,libero]` on Linux and set `MUJOCO_GL=egl`.'
            ) from exc
        try:
            from lerobot.envs import preprocess_observation
        except ImportError:
            try:
                from lerobot.envs.utils import preprocess_observation
            except ImportError:
                preprocess_observation = lambda observation: _preprocess_observation_compat(observation, torch)
        try:
            from lerobot.envs import make_env_pre_post_processors
        except ImportError:
            try:
                from lerobot.envs.factory import make_env_pre_post_processors
            except ImportError:
                make_env_pre_post_processors = _make_env_pre_post_processors_compat

        if activation_dtype not in ("float16", "float32"):
            raise ValueError(f"`activation_dtype` must be `float16` or `float32`, got `{activation_dtype}`.")
        if replan_interval <= 0:
            raise ValueError(f"`replan_interval` must be positive, got {replan_interval}.")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.device = torch.device(device)
        self.task = task
        self.task_id = task_id
        self.pickup_joint_name = pickup_joint_name
        self.place_joint_name = place_joint_name
        self.fallback_instruction = instruction
        self.rotate_images_180 = rotate_images_180
        self.force_replan_every_step = force_replan_every_step
        self.replan_interval = 1 if force_replan_every_step else replan_interval
        self.activation_dtype = activation_dtype
        self.intervention = intervention
        self.preprocess_observation = preprocess_observation

        cfg_kwargs = {"task": task, "task_ids": [task_id], "max_parallel_tasks": 1}
        signature = inspect.signature(LeRobotLiberoConfig)
        self.env_cfg = LeRobotLiberoConfig(**{key: value for key, value in cfg_kwargs.items() if key in signature.parameters})
        self.env = _unwrap_single_env(make_env(self.env_cfg, n_envs=1))

        config = PreTrainedConfig.from_pretrained(checkpoint_path)
        config.compile_model = False
        self.policy = PI0Policy.from_pretrained(checkpoint_path, config=config).to(self.device).eval()
        rename_map = dict(rename_map or DEFAULT_RENAME_MAP)
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            checkpoint_path,
            preprocessor_overrides={
                "device_processor": {"device": str(self.device)},
                "rename_observations_processor": {"rename_map": rename_map},
                **paligemma_tokenizer_overrides(),
            },
        )
        env_processor_kwargs = {"env_cfg": self.env_cfg, "policy_cfg": self.policy.config}
        processor_signature = inspect.signature(make_env_pre_post_processors)
        filtered_kwargs = {key: value for key, value in env_processor_kwargs.items() if key in processor_signature.parameters}
        self.env_preprocess, self.env_postprocess = make_env_pre_post_processors(**filtered_kwargs)
        self.hidden_modules = _pi0_hidden_modules(self.policy)
        self.token_layout: dict[str, Any] | None = None

    def collect(
        self,
        output_dir: Path,
        num_episodes: int,
        start_seed: int = 0,
        max_steps: int | None = None,
        save_video: bool = True,
        save_wrist_video: bool = False,
        video_fps: int = 10,
        video_format: str = "mp4",
        video_flip_180: bool = True,
        require_mp4: bool = True,
        save_activations: bool = True,
    ) -> dict[str, Any]:
        if num_episodes <= 0:
            raise ValueError(f"`num_episodes` must be positive, got {num_episodes}.")
        if max_steps is not None and max_steps <= 0:
            raise ValueError(f"`max_steps` must be positive when set, got {max_steps}.")
        if video_fps <= 0:
            raise ValueError(f"`video_fps` must be positive, got {video_fps}.")
        if video_format not in ("gif", "mp4"):
            raise ValueError(f"`video_format` must be `gif` or `mp4`, got `{video_format}`.")
        ensure_dir(output_dir)
        summaries = []
        for episode_index in trange(num_episodes, desc="collect online PI0 full-token rollouts"):
            summaries.append(
                self._collect_episode(
                    output_dir=output_dir,
                    episode_index=episode_index,
                    seed=start_seed + episode_index,
                    max_steps=max_steps,
                    save_video=save_video,
                    save_wrist_video=save_wrist_video,
                    video_fps=video_fps,
                    video_format=video_format,
                    video_flip_180=video_flip_180,
                    require_mp4=require_mp4,
                    save_activations=save_activations,
                )
            )
        summary = {
            "policy": "pi0",
            "task": self.task,
            "task_id": self.task_id,
            "instruction": self._instruction(),
            "rotate_images_180": bool(self.rotate_images_180),
            "force_replan_every_step": bool(self.force_replan_every_step),
            "replan_interval": int(self.replan_interval),
            "activation_dtype": self.activation_dtype,
            "save_activations": bool(save_activations),
            "intervention": _intervention_to_dict(self.intervention),
            "num_layers": len(self.hidden_modules),
            "layer_names": [name for name, _ in self.hidden_modules],
            "token_layout": self.token_layout,
            "num_episodes": len(summaries),
            "num_successes": int(sum(bool(item["success"]) for item in summaries)),
            "success_rate": float(np.mean([bool(item["success"]) for item in summaries])),
            "episodes": summaries,
        }
        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()

    def _collect_episode(
        self,
        output_dir: Path,
        episode_index: int,
        seed: int,
        max_steps: int | None,
        save_video: bool,
        save_wrist_video: bool,
        video_fps: int,
        video_format: str,
        video_flip_180: bool,
        require_mp4: bool,
        save_activations: bool,
    ) -> dict[str, Any]:
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        observation, _ = self.env.reset(seed=[seed])
        episode_dir = ensure_dir(output_dir / f"episode_{episode_index:03d}")
        image_dir = ensure_dir(episode_dir / "images")
        activation_dir = ensure_dir(episode_dir / "activations")
        rows: list[dict[str, Any]] = []
        activation_rows: list[dict[str, Any]] = []
        queued_actions: list[np.ndarray] = []
        last_pred_action_chunk: np.ndarray | None = None
        episode_success = False
        episode_reward = 0.0
        episode_max_steps = max_steps or _max_episode_steps(self.env)

        for step in range(episode_max_steps):
            row = self._state_row(observation, image_dir, episode_index, step)
            policy_observation = _rotate_observation_images(observation) if self.rotate_images_180 else observation
            batch = self.preprocess_observation(policy_observation)
            batch["task"] = [self._instruction()]
            batch = self.env_preprocess(batch)
            batch = self.preprocess(batch)
            batch = _move_tensors_to_device(batch, self.device)
            should_replan = not queued_actions or step % self.replan_interval == 0
            if should_replan and hasattr(self.policy, "reset"):
                self.policy.reset()
            if should_replan:
                with self._capture_hidden_states(self.intervention, capture=save_activations) as hidden:
                    with self.torch.inference_mode():
                        pred_action_chunk = self._predict_action_chunk(batch)
                last_pred_action_chunk = pred_action_chunk
                queued_actions = [action.copy() for action in pred_action_chunk[: self.replan_interval]]
                if not queued_actions:
                    raise RuntimeError("PI0 predicted an empty action chunk.")
                step_activation_rows = (
                    _save_full_token_hidden(
                        hidden=hidden,
                        step_dir=ensure_dir(activation_dir / f"step_{step:04d}"),
                        episode_index=episode_index,
                        step=step,
                        dtype=self.activation_dtype,
                    )
                    if save_activations
                    else []
                )
                activation_rows.extend(step_activation_rows)
                if self.token_layout is None and step_activation_rows:
                    self.token_layout = _infer_pi0_token_layout(
                        batch=batch,
                        policy=self.policy,
                        sequence_length=int(step_activation_rows[0]["shape"][0]),
                    )
                    with open(output_dir / "token_layout.json", "w", encoding="utf-8") as f:
                        json.dump(self.token_layout, f, indent=2)
            else:
                step_activation_rows = []

            pred_action = queued_actions.pop(0)
            action_for_env = self.torch.from_numpy(pred_action[None, :]).to(self.device)
            action_transition = self.env_postprocess({"action": action_for_env})
            action_numpy = _to_numpy(action_transition["action"])
            observation, reward, terminated, truncated, info = self.env.step(action_numpy)
            step_success = _extract_success(info)
            episode_success = episode_success or step_success
            episode_reward += float(np.asarray(reward).reshape(-1)[0])
            row.update(
                {
                    "action": action_numpy.reshape(-1).astype(float).tolist(),
                    "policy_pred_action": pred_action.astype(float).tolist(),
                    "policy_pred_action_chunk": last_pred_action_chunk.astype(float).tolist()
                    if last_pred_action_chunk is not None
                    else None,
                    "is_replan": bool(should_replan),
                    "replan_interval": int(self.replan_interval),
                    "queued_actions_remaining": int(len(queued_actions)),
                    "reward": float(np.asarray(reward).reshape(-1)[0]),
                    "terminated": bool(np.asarray(terminated).reshape(-1)[0]),
                    "truncated": bool(np.asarray(truncated).reshape(-1)[0]),
                    "success": bool(step_success),
                    "activation_step_dir": str(activation_dir / f"step_{step:04d}"),
                    "num_activation_layers": len(step_activation_rows),
                }
            )
            rows.append(row)
            if hasattr(self.torch, "cuda") and self.device.type == "cuda":
                self.torch.cuda.empty_cache()
            if bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0]):
                break

        _write_jsonl(episode_dir / "steps.jsonl", rows)
        _write_jsonl(episode_dir / "activation_index.jsonl", activation_rows)
        video_path = (
            _save_episode_video(
                rows=rows,
                episode_dir=episode_dir,
                image_key="image_path",
                stem="episode_agentview_flipped" if video_flip_180 else "episode_agentview",
                fps=video_fps,
                video_format=video_format,
                flip_180=video_flip_180,
                require_mp4=require_mp4,
            )
            if save_video
            else None
        )
        wrist_video_path = (
            _save_episode_video(
                rows=rows,
                episode_dir=episode_dir,
                image_key="wrist_image_path",
                stem="episode_wrist_flipped" if video_flip_180 else "episode_wrist",
                fps=video_fps,
                video_format=video_format,
                flip_180=video_flip_180,
                require_mp4=require_mp4,
            )
            if save_wrist_video
            else None
        )
        return {
            "episode_index": episode_index,
            "seed": seed,
            "num_steps": len(rows),
            "num_activation_files": len(activation_rows),
            "sum_reward": episode_reward,
            "success": episode_success,
            "steps_jsonl": str(episode_dir / "steps.jsonl"),
            "activation_index_jsonl": str(episode_dir / "activation_index.jsonl"),
            "activation_dir": str(activation_dir),
            "video_path": str(video_path) if video_path is not None else None,
            "wrist_video_path": str(wrist_video_path) if wrist_video_path is not None else None,
        }

    def _predict_action_chunk(self, batch: dict[str, Any]) -> np.ndarray:
        if hasattr(self.policy, "predict_action_chunk"):
            pred_action_chunk = self.postprocess(self.policy.predict_action_chunk(batch))
            return _normalize_action_chunk(_to_numpy(pred_action_chunk)).astype(np.float32)
        action = self.postprocess(self.policy.select_action(batch))
        return _to_numpy(action).reshape(1, -1).astype(np.float32)

    def _state_row(self, observation: dict[str, Any], image_dir: Path, episode_index: int, step: int) -> dict[str, Any]:
        image = _extract_image(observation, ("pixels.image", "observation.images.image", "image"))
        wrist_image = _extract_image(
            observation,
            ("pixels.image2", "observation.images.image2", "observation.images.wrist_image", "wrist_image"),
        )
        image_path = image_dir / f"step_{step:04d}.png"
        wrist_path = image_dir / f"step_{step:04d}_wrist.png"
        Image.fromarray(image).save(image_path)
        Image.fromarray(wrist_image).save(wrist_path)
        sim = LiberoEnv._find_attr(self.env, "sim")
        if sim is None:
            raise RuntimeError("Could not locate MuJoCo `sim` while tracing online LIBERO rollout.")
        self._maybe_resolve_target_joints(sim, self._instruction())
        gripper_position = LiberoEnv._extract_gripper_position(observation)
        observation_state = LiberoEnv._extract_observation_state(observation)
        pickup_position = _joint_xyz(sim, self.pickup_joint_name)
        place_position = _joint_xyz(sim, self.place_joint_name)
        return {
            "episode_index": episode_index,
            "step": step,
            "instruction": self._instruction(),
            "image_path": str(image_path),
            "wrist_image_path": str(wrist_path),
            "observation_state": observation_state.astype(float).tolist(),
            "gripper_position": gripper_position.astype(float).tolist(),
            "pickup_target_joint_name": self.pickup_joint_name,
            "pickup_target_position": pickup_position.astype(float).tolist(),
            "pickup_target_offset": (pickup_position - gripper_position).astype(float).tolist(),
            "place_target_joint_name": self.place_joint_name,
            "place_target_position": place_position.astype(float).tolist(),
            "place_target_offset": (place_position - gripper_position).astype(float).tolist(),
        }

    def _maybe_resolve_target_joints(self, sim: Any, instruction: str) -> None:
        if self.pickup_joint_name != "auto" and self.place_joint_name != "auto":
            return
        matches = _instruction_joint_matches(sim, instruction)
        if not matches:
            available = LiberoEnv._mujoco_names(sim)["joints"]
            raise RuntimeError(
                "Could not auto-infer pickup/place target joints from the instruction. "
                "Pass `--pickup-joint-name` and `--place-joint-name` explicitly. "
                f"Instruction: `{instruction}`. Available joints: {available}"
            )
        if self.pickup_joint_name == "auto":
            self.pickup_joint_name = matches[0][1]
            print(f"Auto-inferred pickup joint: {self.pickup_joint_name}", flush=True)
        if self.place_joint_name == "auto":
            self.place_joint_name = matches[-1][1]
            print(f"Auto-inferred place joint: {self.place_joint_name}", flush=True)

    def _instruction(self) -> str:
        for attr in ("task_description", "task"):
            try:
                value = self.env.call(attr)
            except (AttributeError, NotImplementedError):
                continue
            if value:
                return str(value[0])
        return self.fallback_instruction

    @contextmanager
    def _capture_hidden_states(self, intervention: ActivationIntervention | None = None, capture: bool = True) -> Any:
        hidden: dict[str, Any] = {}
        handles = []
        if not capture and intervention is None:
            yield hidden
            return

        def save_hook(name: str, layer_index: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                tensor = output[0] if isinstance(output, tuple) else output
                if intervention is not None and layer_index == intervention.layer_index:
                    tensor = _apply_activation_intervention(tensor, intervention)
                    output = (tensor, *output[1:]) if isinstance(output, tuple) else tensor
                if capture:
                    hidden[name] = tensor.detach() if hasattr(tensor, "detach") else tensor
                return output

            return hook

        for layer_index, (name, module) in enumerate(self.hidden_modules):
            if not capture and (intervention is None or layer_index != intervention.layer_index):
                continue
            handles.append(module.register_forward_hook(save_hook(name, layer_index)))
        try:
            yield hidden
        finally:
            for handle in handles:
                handle.remove()


def _apply_activation_intervention(tensor: Any, intervention: ActivationIntervention) -> Any:
    if not hasattr(tensor, "clone"):
        return tensor
    if intervention.mode not in ("zero", "scale"):
        raise ValueError(f"Unsupported intervention mode `{intervention.mode}`. Use `zero` or `scale`.")
    if tensor.ndim < 2:
        return tensor
    modified = tensor.clone()
    seq_dim = 1 if modified.ndim >= 3 else 0
    seq_len = int(modified.shape[seq_dim])
    if intervention.token_bin_index is not None:
        if intervention.token_bins is None or intervention.token_bins <= 0:
            raise ValueError("`token_bins` must be positive when `token_bin_index` is set.")
        edges = np.linspace(0, seq_len, int(intervention.token_bins) + 1, dtype=int)
        bin_index = max(0, min(int(intervention.token_bin_index), int(intervention.token_bins) - 1))
        start = int(edges[bin_index])
        end = int(edges[bin_index + 1])
    else:
        if intervention.token_start is None or intervention.token_end is None:
            raise ValueError("Set either `token_bin_index`/`token_bins` or `token_start`/`token_end`.")
        start = int(intervention.token_start)
        end = int(intervention.token_end)
    start = max(0, min(start, seq_len))
    end = max(start, min(end, seq_len))
    if end <= start:
        return modified
    index = [slice(None)] * modified.ndim
    index[seq_dim] = slice(start, end)
    if intervention.mode == "zero":
        modified[tuple(index)] = 0
    else:
        modified[tuple(index)] = modified[tuple(index)] * float(intervention.scale)
    return modified


def _intervention_to_dict(intervention: ActivationIntervention | None) -> dict[str, Any] | None:
    if intervention is None:
        return None
    return {
        "layer_index": int(intervention.layer_index),
        "token_start": None if intervention.token_start is None else int(intervention.token_start),
        "token_end": None if intervention.token_end is None else int(intervention.token_end),
        "token_bin_index": None if intervention.token_bin_index is None else int(intervention.token_bin_index),
        "token_bins": None if intervention.token_bins is None else int(intervention.token_bins),
        "mode": intervention.mode,
        "scale": float(intervention.scale),
    }


def _pi0_hidden_modules(policy: Any) -> list[tuple[str, Any]]:
    core = getattr(policy, "model", policy)
    expert = getattr(core, "paligemma_with_expert", None)
    candidates: list[tuple[str, Any]] = []
    if expert is not None:
        paligemma = getattr(getattr(getattr(expert, "paligemma", None), "model", None), "language_model", None)
        paligemma_layers = getattr(paligemma, "layers", None)
        if paligemma_layers is not None:
            candidates.extend((f"paligemma_layer_{index:02d}", layer) for index, layer in enumerate(paligemma_layers))
        action_expert = getattr(getattr(getattr(expert, "gemma_expert", None), "model", None), "layers", None)
        if action_expert is not None:
            candidates.extend((f"expert_layer_{index:02d}", layer) for index, layer in enumerate(action_expert))
    if candidates:
        return candidates
    for name, module in policy.named_modules():
        lname = name.lower()
        if ".layers." not in lname:
            continue
        if any(part in lname for part in ("language_model.layers", "gemma_expert.model.layers", "transformer.layers")):
            candidates.append((name.replace(".", "_"), module))
    if not candidates:
        raise RuntimeError("Could not locate PI0 transformer layers for full-token rollout tracing.")
    return candidates


def _save_full_token_hidden(
    hidden: dict[str, Any],
    step_dir: Path,
    episode_index: int,
    step: int,
    dtype: str,
) -> list[dict[str, Any]]:
    rows = []
    np_dtype = np.float16 if dtype == "float16" else np.float32
    for layer_index, (layer_name, tensor) in enumerate(hidden.items()):
        array = _hidden_to_numpy(tensor, np_dtype)
        path = step_dir / f"{layer_index:03d}_{layer_name}.npy"
        np.save(path, array)
        rows.append(
            {
                "episode_index": episode_index,
                "step": step,
                "layer_index": layer_index,
                "layer_name": layer_name,
                "path": str(path),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        )
    return rows


def _hidden_to_numpy(tensor: Any, dtype: Any) -> np.ndarray:
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
        if tensor.ndim >= 3:
            tensor = tensor[0]
        return tensor.float().cpu().numpy().astype(dtype, copy=False)
    array = np.asarray(tensor)
    if array.ndim >= 3:
        array = array[0]
    return array.astype(dtype, copy=False)


def _normalize_action_chunk(value: np.ndarray) -> np.ndarray:
    chunk = np.asarray(value, dtype=np.float32)
    if chunk.ndim == 3 and chunk.shape[0] == 1:
        chunk = chunk[0]
    if chunk.ndim == 1:
        chunk = chunk[None, :]
    if chunk.ndim != 2:
        raise ValueError(f"Expected a PI0 action chunk shaped [horizon, action_dim], got {chunk.shape}.")
    return chunk


def _infer_pi0_token_layout(batch: dict[str, Any], policy: Any, sequence_length: int) -> dict[str, Any]:
    layout: dict[str, Any] = {
        "sequence_length": int(sequence_length),
        "status": "unresolved",
        "source": "batch_introspection",
        "regions": [],
        "batch_summary": _summarize_nested(batch),
        "config_summary": _summarize_object(getattr(policy, "config", None)),
    }
    token_types = _find_sequence_tensor(batch, sequence_length, key_patterns=("token_type", "token_type_ids"))
    if token_types is not None:
        values = np.asarray(token_types).reshape(-1).astype(int)
        regions = _regions_from_values(values, prefix="token_type")
        layout.update({"status": "exact_token_type_ids", "source": "token_type_ids", "regions": regions})
        return layout

    mask_regions: list[dict[str, Any]] = []
    for key, value in _iter_leaf_tensors(batch):
        key_l = key.lower()
        if "mask" not in key_l:
            continue
        array = _to_cpu_numpy(value)
        if array is None:
            continue
        array = np.asarray(array)
        if array.ndim >= 2 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 1 or array.shape[0] != sequence_length:
            continue
        if array.dtype != np.bool_:
            unique = np.unique(array)
            if not np.all(np.isin(unique, [0, 1])):
                continue
            array = array.astype(bool)
        for start, end in _true_spans(array):
            mask_regions.append({"name": key.replace(".", "_"), "start": int(start), "end": int(end)})
    if mask_regions:
        layout.update({"status": "mask_spans", "source": "sequence_length_boolean_masks", "regions": mask_regions})
        return layout
    pi0_regions = _infer_pi0_multimodal_regions_from_batch(batch, getattr(policy, "config", None), sequence_length)
    if pi0_regions:
        layout.update({"status": "verified_pi0_embed_prefix_suffix_order", "source": "pi0_modeling_pi0_embed_prefix_suffix", "regions": pi0_regions})
        return layout
    return layout


def _find_sequence_tensor(batch: Any, sequence_length: int, key_patterns: tuple[str, ...]) -> np.ndarray | None:
    for key, value in _iter_leaf_tensors(batch):
        key_l = key.lower()
        if not any(pattern in key_l for pattern in key_patterns):
            continue
        array = _to_cpu_numpy(value)
        if array is None:
            continue
        array = np.asarray(array)
        if array.ndim >= 2 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 1 and array.shape[0] == sequence_length:
            return array
    return None


def _iter_leaf_tensors(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        rows: list[tuple[str, Any]] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_iter_leaf_tensors(child, child_prefix))
        return rows
    return [(prefix, value)]


def _to_cpu_numpy(value: Any) -> np.ndarray | None:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return None


def _regions_from_values(values: np.ndarray, prefix: str) -> list[dict[str, Any]]:
    regions = []
    if values.size == 0:
        return regions
    start = 0
    current = int(values[0])
    for index in range(1, len(values) + 1):
        next_value = int(values[index]) if index < len(values) else None
        if next_value != current:
            regions.append({"name": f"{prefix}_{current}", "start": int(start), "end": int(index), "value": current})
            start = index
            current = -1 if next_value is None else next_value
    return regions


def _true_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    spans = []
    start = None
    for index, value in enumerate(mask.tolist() + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            spans.append((start, index))
            start = None
    return spans


def _infer_pi0_multimodal_regions_from_batch(batch: dict[str, Any], config: Any, sequence_length: int) -> list[dict[str, Any]]:
    language = batch.get("observation.language.tokens")
    if language is None or not hasattr(language, "shape"):
        return []
    language_len = int(language.shape[-1])
    configured_image_keys = [str(key) for key in getattr(config, "image_features", []) or []]
    image_keys = [key for key in configured_image_keys if key in batch and hasattr(batch[key], "shape")]
    if not image_keys:
        image_keys = [key for key in batch if str(key).startswith("observation.images.") and hasattr(batch[key], "shape")]
    if not image_keys:
        return []
    empty_cameras = int(getattr(config, "empty_cameras", 0) or 0)
    num_image_blocks = len(image_keys) + empty_cameras
    remaining = int(sequence_length) - language_len
    if remaining <= 0 or num_image_blocks <= 0 or remaining % num_image_blocks != 0:
        return []
    image_tokens = remaining // num_image_blocks
    regions = []
    cursor = 0
    for key in image_keys:
        name = str(key).replace("observation.images.", "")
        regions.append({"name": name, "start": int(cursor), "end": int(cursor + image_tokens), "stream": "prefix"})
        cursor += image_tokens
    for index in range(empty_cameras):
        regions.append({"name": f"empty_camera_{index}", "start": int(cursor), "end": int(cursor + image_tokens), "stream": "prefix"})
        cursor += image_tokens
    regions.append({"name": "language", "start": int(cursor), "end": int(cursor + language_len), "stream": "prefix"})
    cursor += language_len
    chunk_size = int(getattr(config, "chunk_size", 0) or getattr(config, "n_action_steps", 0) or 0)
    if chunk_size > 0:
        regions.extend(
            [
                {"name": "state", "start": 0, "end": 1, "stream": "expert", "sequence_length": chunk_size + 1},
                {"name": "action_chunk", "start": 1, "end": chunk_size + 1, "stream": "expert", "sequence_length": chunk_size + 1},
            ]
        )
    return regions if cursor == int(sequence_length) else []


def _summarize_nested(value: Any, depth: int = 0) -> Any:
    if depth > 4:
        return "..."
    if isinstance(value, dict):
        return {str(key): _summarize_nested(child, depth + 1) for key, child in value.items()}
    if hasattr(value, "shape"):
        dtype = getattr(value, "dtype", None)
        device = getattr(value, "device", None)
        return {"type": type(value).__name__, "shape": list(value.shape), "dtype": str(dtype), "device": str(device)}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "len": len(value), "items": [_summarize_nested(item, depth + 1) for item in list(value)[:4]]}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"type": type(value).__name__}


def _summarize_object(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    summary = {}
    for key in dir(value):
        if key.startswith("_"):
            continue
        try:
            item = getattr(value, key)
        except Exception:
            continue
        if isinstance(item, (str, int, float, bool, type(None))):
            summary[key] = item
        elif isinstance(item, (list, tuple)) and len(item) <= 16 and all(isinstance(x, (str, int, float, bool, type(None))) for x in item):
            summary[key] = list(item)
    return summary


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
