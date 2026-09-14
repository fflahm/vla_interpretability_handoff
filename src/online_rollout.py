from __future__ import annotations

import inspect
import json
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import trange

from .activations import stack_layer_activations
from .envs import LiberoEnv, _flatten_keys, _get_nested, _instruction_object_match, _normalize_libero_object_name
from .utils import ensure_dir


DEFAULT_RENAME_MAP = {"observation.images.image2": "observation.images.wrist_image"}
POOLING_MODES = ("mean", "last", "flatten")


@dataclass(frozen=True)
class Pi05ActivationIntervention:
    """Intervene on one PI0.5 tower's local layer and equal-width token bin."""

    tower: str
    layer_index: int
    token_bin_index: int
    token_bins: int = 96
    mode: str = "zero"
    scale: float = 0.0


class Pi05LiberoRolloutTracer:
    """Runs one online LIBERO task with PI0.5 and records stepwise state plus replan activations."""

    def __init__(
        self,
        checkpoint_path: str,
        task: str = "libero_object",
        task_id: int = 0,
        device: str = "auto",
        pickup_joint_name: str = "auto",
        place_joint_name: str = "auto",
        instruction: str = "pick up the alphabet soup and place it in the basket",
        rename_map: dict[str, str] | None = None,
        pooling: str = "mean",
        rotate_images_180: bool = True,
        intervention: Pi05ActivationIntervention | None = None,
    ) -> None:
        try:
            import torch
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.envs.configs import LiberoEnv as LeRobotLiberoConfig
            from lerobot.envs.factory import make_env
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.pi05 import PI05Policy
        except Exception as exc:
            raise RuntimeError(
                "Online PI0.5 LIBERO tracing requires LeRobot with PI and LIBERO extras. "
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
            except ImportError as exc:
                make_env_pre_post_processors = _make_env_pre_post_processors_compat

        if pooling not in POOLING_MODES:
            raise ValueError(f"`pooling` must be one of {POOLING_MODES}, got `{pooling}`.")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.device = torch.device(device)
        self.task = task
        self.task_id = task_id
        self.pooling = pooling
        self.pickup_joint_name = pickup_joint_name
        self.place_joint_name = place_joint_name
        self.fallback_instruction = instruction
        self.preprocess_observation = preprocess_observation
        self.rotate_images_180 = rotate_images_180
        self.intervention = intervention

        cfg_kwargs = {"task": task, "task_ids": [task_id], "max_parallel_tasks": 1}
        signature = inspect.signature(LeRobotLiberoConfig)
        self.env_cfg = LeRobotLiberoConfig(**{key: value for key, value in cfg_kwargs.items() if key in signature.parameters})
        self.env = _unwrap_single_env(make_env(self.env_cfg, n_envs=1))

        config = PreTrainedConfig.from_pretrained(checkpoint_path)
        config.compile_model = False
        # The PI0.5 checkpoint ties the language embedding; strict loading can
        # report that shared tensor as missing and leave a randomly initialized
        # policy in some LeRobot versions. Match the validated offline wrapper.
        self.policy = PI05Policy.from_pretrained(
            checkpoint_path, config=config, strict=False
        ).to(self.device).eval()
        rename_map = dict(rename_map or DEFAULT_RENAME_MAP)
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            checkpoint_path,
            preprocessor_overrides={
                "device_processor": {"device": str(self.device)},
                "rename_observations_processor": {"rename_map": rename_map},
            },
        )
        env_processor_kwargs = {"env_cfg": self.env_cfg, "policy_cfg": self.policy.config}
        processor_signature = inspect.signature(make_env_pre_post_processors)
        filtered_kwargs = {key: value for key, value in env_processor_kwargs.items() if key in processor_signature.parameters}
        self.env_preprocess, self.env_postprocess = make_env_pre_post_processors(**filtered_kwargs)
        self.hidden_modules = _hidden_modules(self.policy)

    def collect(
        self,
        output_dir: Path,
        num_episodes: int,
        start_seed: int = 0,
        max_steps: int | None = None,
        save_video: bool = True,
        save_wrist_video: bool = False,
        video_fps: int = 10,
        video_format: str = "gif",
        video_flip_180: bool = True,
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
        for episode_index in trange(num_episodes, desc="collect online PI0.5 rollouts"):
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
                )
            )
        summary = {
            "task": self.task,
            "task_id": self.task_id,
            "instruction": self._instruction(),
            "rotate_images_180": bool(self.rotate_images_180),
            "num_episodes": len(summaries),
            "num_successes": int(sum(bool(item["success"]) for item in summaries)),
            "success_rate": float(np.mean([bool(item["success"]) for item in summaries])),
            "intervention": _pi05_intervention_to_dict(self.intervention),
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
    ) -> dict[str, Any]:
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        observation, _ = self.env.reset(seed=[seed])
        episode_dir = ensure_dir(output_dir / f"episode_{episode_index:03d}")
        image_dir = ensure_dir(episode_dir / "images")
        rows = []
        hidden_by_replan = []
        replan_steps = []
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
            with self._capture_hidden_states() as hidden:
                with self.torch.inference_mode():
                    action = self.policy.select_action(batch)
                    action = self.postprocess(action)
            action_transition = self.env_postprocess({"action": action})
            action_numpy = _to_numpy(action_transition["action"])
            observation, reward, terminated, truncated, info = self.env.step(action_numpy)
            step_success = _extract_success(info)
            episode_success = episode_success or step_success
            episode_reward += float(np.asarray(reward).reshape(-1)[0])
            pooled_hidden = {name: _pool_tensor(tensor, self.pooling) for name, tensor in hidden.items()}
            if pooled_hidden:
                hidden_by_replan.append(pooled_hidden)
                replan_steps.append(step)
            row.update(
                {
                    "action": action_numpy.reshape(-1).astype(float).tolist(),
                    "reward": float(np.asarray(reward).reshape(-1)[0]),
                    "terminated": bool(np.asarray(terminated).reshape(-1)[0]),
                    "truncated": bool(np.asarray(truncated).reshape(-1)[0]),
                    "success": bool(step_success),
                    "is_replan": bool(pooled_hidden),
                }
            )
            rows.append(row)
            if bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0]):
                break

        with open(episode_dir / "steps.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        video_path = (
            _save_episode_video(
                rows=rows,
                episode_dir=episode_dir,
                image_key="image_path",
                stem="episode_agentview_flipped" if video_flip_180 else "episode_agentview",
                fps=video_fps,
                video_format=video_format,
                flip_180=video_flip_180,
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
            )
            if save_wrist_video
            else None
        )
        if hidden_by_replan:
            x_layers, layer_names = stack_layer_activations(hidden_by_replan)
            np.savez_compressed(
                episode_dir / "replan_activations.npz",
                X_layers=x_layers,
                layer_names=np.asarray(layer_names),
                step_indices=np.asarray(replan_steps, dtype=np.int32),
            )
        return {
            "episode_index": episode_index,
            "seed": seed,
            "num_steps": len(rows),
            "num_replans": len(replan_steps),
            "sum_reward": episode_reward,
            "success": episode_success,
            "steps_jsonl": str(episode_dir / "steps.jsonl"),
            "activations_npz": str(episode_dir / "replan_activations.npz") if hidden_by_replan else None,
            "video_path": str(video_path) if video_path is not None else None,
            "wrist_video_path": str(wrist_video_path) if wrist_video_path is not None else None,
        }

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
    def _capture_hidden_states(self) -> Any:
        hidden: dict[str, Any] = {}
        handles = []

        def save_hook(name: str):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                tensor = output[0] if isinstance(output, tuple) else output
                modified = _apply_pi05_activation_intervention(tensor, name, self.intervention)
                hidden[name] = modified.detach() if hasattr(modified, "detach") else modified
                if modified is tensor:
                    return None
                if isinstance(output, tuple):
                    return (modified, *output[1:])
                return modified

            return hook

        for name, module in self.hidden_modules:
            handles.append(module.register_forward_hook(save_hook(name)))
        try:
            yield hidden
        finally:
            for handle in handles:
                handle.remove()


def _apply_pi05_activation_intervention(
    tensor: Any,
    layer_name: str,
    intervention: Pi05ActivationIntervention | None,
) -> Any:
    if intervention is None or not hasattr(tensor, "clone"):
        return tensor
    expected_name = f"{intervention.tower}_layer_{int(intervention.layer_index):02d}"
    if layer_name != expected_name:
        return tensor
    if intervention.tower not in ("paligemma", "expert"):
        raise ValueError(f"Unsupported PI0.5 tower `{intervention.tower}`.")
    if intervention.mode not in ("zero", "scale"):
        raise ValueError(f"Unsupported intervention mode `{intervention.mode}`.")
    if intervention.token_bins <= 0:
        raise ValueError("`token_bins` must be positive.")
    if tensor.ndim < 2:
        return tensor
    modified = tensor.clone()
    seq_dim = 1 if modified.ndim >= 3 else 0
    seq_len = int(modified.shape[seq_dim])
    effective_bins = min(int(intervention.token_bins), seq_len)
    if not 0 <= int(intervention.token_bin_index) < effective_bins:
        raise ValueError(
            f"Bin {intervention.token_bin_index} is unavailable for {layer_name}: "
            f"sequence length {seq_len}, effective bins {effective_bins}."
        )
    edges = np.linspace(0, seq_len, effective_bins + 1, dtype=int)
    start = int(edges[int(intervention.token_bin_index)])
    end = int(edges[int(intervention.token_bin_index) + 1])
    index = [slice(None)] * modified.ndim
    index[seq_dim] = slice(start, end)
    if intervention.mode == "zero":
        modified[tuple(index)] = 0
    else:
        modified[tuple(index)] *= float(intervention.scale)
    return modified


def _pi05_intervention_to_dict(intervention: Pi05ActivationIntervention | None) -> dict[str, Any] | None:
    if intervention is None:
        return None
    return {
        "tower": intervention.tower,
        "layer_index": int(intervention.layer_index),
        "token_bin_index": int(intervention.token_bin_index),
        "token_bins": int(intervention.token_bins),
        "mode": intervention.mode,
        "scale": float(intervention.scale),
    }


def _unwrap_single_env(env: Any) -> Any:
    while isinstance(env, dict):
        if not env:
            raise RuntimeError("LeRobot make_env returned an empty environment dictionary.")
        env = next(iter(env.values()))
    return env


def _hidden_modules(policy: Any) -> list[tuple[str, Any]]:
    core = getattr(policy, "model", policy)
    expert = getattr(core, "paligemma_with_expert", None)
    candidates = []
    if expert is not None:
        paligemma = getattr(getattr(getattr(expert, "paligemma", None), "model", None), "language_model", None)
        paligemma_layers = getattr(paligemma, "layers", None)
        if paligemma_layers is not None:
            candidates.extend((f"paligemma_layer_{index:02d}", layer) for index, layer in enumerate(paligemma_layers))
        action_expert = getattr(getattr(getattr(expert, "gemma_expert", None), "model", None), "layers", None)
        if action_expert is not None:
            candidates.extend((f"expert_layer_{index:02d}", layer) for index, layer in enumerate(action_expert))
    if not candidates:
        raise RuntimeError("Could not locate PI0.5 transformer layers for online rollout tracing.")
    return candidates


def _extract_image(observation: dict[str, Any], keys: tuple[str, ...]) -> np.ndarray:
    for key in keys:
        value = _get_nested(observation, key)
        if value is None:
            continue
        image = np.asarray(value)
        if image.ndim == 4:
            image = image[0]
        if image.ndim == 3 and image.shape[0] in (1, 3, 4):
            image = np.moveaxis(image, 0, -1)
        return image.astype(np.uint8)
    raise RuntimeError(f"Could not find image among {keys}. Available observation keys: {_flatten_keys(observation)}")


def _rotate_observation_images(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _rotate_observation_images(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rotate_observation_images(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_rotate_observation_images(item) for item in value)
    if isinstance(value, np.ndarray) and _looks_like_image_array(value):
        return np.flip(value, axis=_image_spatial_axes(value)).copy()
    return value


def _looks_like_image_array(value: np.ndarray) -> bool:
    if value.ndim == 3:
        return value.shape[-1] in (1, 3, 4) or value.shape[0] in (1, 3, 4)
    if value.ndim == 4:
        return value.shape[-1] in (1, 3, 4) or value.shape[1] in (1, 3, 4)
    return False


def _image_spatial_axes(value: np.ndarray) -> tuple[int, int]:
    if value.ndim == 3 and value.shape[-1] in (1, 3, 4):
        return (0, 1)
    if value.ndim == 3 and value.shape[0] in (1, 3, 4):
        return (1, 2)
    if value.ndim == 4 and value.shape[-1] in (1, 3, 4):
        return (1, 2)
    if value.ndim == 4 and value.shape[1] in (1, 3, 4):
        return (2, 3)
    raise ValueError(f"Cannot infer image spatial axes for shape {value.shape}.")


def _save_episode_video(
    rows: list[dict[str, Any]],
    episode_dir: Path,
    image_key: str,
    stem: str,
    fps: int,
    video_format: str,
    flip_180: bool,
    require_mp4: bool = False,
) -> Path | None:
    frames = []
    for row in rows:
        image_path = Path(row[image_key])
        if not image_path.exists():
            continue
        frame = np.asarray(Image.open(image_path).convert("RGB"))
        if flip_180:
            frame = np.flip(frame, axis=(0, 1)).copy()
        frames.append(frame)
    if not frames:
        return None
    if video_format == "mp4":
        mp4_path = episode_dir / f"{stem}.mp4"
        tmp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_path = Path(tmp_file.name)
        tmp_file.close()
        try:
            import imageio.v3 as iio

            iio.imwrite(tmp_path, np.stack(frames, axis=0), fps=fps)
            shutil.copyfile(tmp_path, mp4_path)
            return mp4_path
        except Exception as exc:
            if require_mp4:
                raise RuntimeError(
                    "Could not write MP4 video. Install `imageio imageio-ffmpeg` in the rollout environment, "
                    "or rerun with `--no-require-mp4` / `video_format: gif`."
                ) from exc
            print(f"Could not write MP4 video with imageio; falling back to GIF. Error: {exc}", flush=True)
        finally:
            tmp_path.unlink(missing_ok=True)
    gif_path = episode_dir / f"{stem}.gif"
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=max(1, int(round(1000 / fps))),
        loop=0,
    )
    return gif_path


def _joint_xyz(sim: Any, joint_name: str) -> np.ndarray:
    try:
        return np.asarray(sim.data.get_joint_qpos(joint_name), dtype=np.float32)[:3]
    except Exception as exc:
        available = LiberoEnv._mujoco_names(sim)["joints"]
        raise RuntimeError(
            f"Could not read target joint `{joint_name}` from MuJoCo simulator. Available joints: {available}"
        ) from exc


def _instruction_joint_matches(sim: Any, instruction: str) -> list[tuple[int, str]]:
    normalized_instruction = instruction.lower().replace("_", " ")
    matches: list[tuple[int, str]] = []
    for joint_name in LiberoEnv._mujoco_names(sim)["joints"]:
        object_name = _normalize_libero_object_name(joint_name)
        match = _instruction_object_match(normalized_instruction, object_name)
        if match is not None:
            matches.append((match, joint_name))
    return sorted(matches)


def _extract_success(info: dict[str, Any]) -> bool:
    if "final_info" in info:
        final_info = info["final_info"]
        if isinstance(final_info, (list, tuple, np.ndarray)) and len(final_info):
            final_info = final_info[0]
        if isinstance(final_info, dict) and "is_success" in final_info:
            return bool(np.asarray(final_info["is_success"]).reshape(-1)[0])
    if "is_success" in info:
        return bool(np.asarray(info["is_success"]).reshape(-1)[0])
    return False


def _max_episode_steps(env: Any) -> int:
    try:
        value = env.call("_max_episode_steps")
        return int(np.asarray(value).reshape(-1)[0])
    except (AttributeError, NotImplementedError):
        value = LiberoEnv._find_attr(env, "_max_episode_steps")
        if value is None:
            raise RuntimeError("Could not infer the LIBERO episode step limit. Pass `--max-steps` explicitly.")
        return int(value)


def _pool_tensor(tensor: Any, pooling: str) -> np.ndarray:
    if not hasattr(tensor, "detach"):
        return np.asarray(tensor, dtype=np.float32).reshape(-1)
    tensor = tensor.detach()
    if tensor.ndim >= 3:
        if pooling == "mean":
            tensor = tensor.mean(dim=1)
        elif pooling == "last":
            tensor = tensor[:, -1]
        elif pooling == "flatten":
            tensor = tensor.reshape(tensor.shape[0], -1)
    if tensor.ndim >= 2:
        tensor = tensor[0]
    return tensor.float().cpu().numpy().reshape(-1).astype(np.float32)


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


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _preprocess_observation_compat(observation: dict[str, Any], torch: Any) -> dict[str, Any]:
    """Backport LeRobot's observation conversion for releases without envs.utils."""
    converted: dict[str, Any] = {}
    pixels = observation.get("pixels")
    if pixels is not None:
        images = pixels if isinstance(pixels, dict) else {"image": pixels}
        for name, value in images.items():
            image = torch.from_numpy(np.asarray(value))
            if image.ndim == 3:
                image = image.unsqueeze(0)
            image = image.permute(0, 3, 1, 2).contiguous().float() / 255
            converted[f"observation.images.{name}"] = image
    if "robot_state" in observation:
        converted["observation.robot_state"] = _nested_numpy_to_tensor(observation["robot_state"], torch)
    if "agent_pos" in observation:
        state = torch.from_numpy(np.asarray(observation["agent_pos"])).float()
        converted["observation.state"] = state.unsqueeze(0) if state.ndim == 1 else state
    return converted


def _nested_numpy_to_tensor(value: Any, torch: Any) -> Any:
    if isinstance(value, dict):
        return {key: _nested_numpy_to_tensor(item, torch) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    return value


def _make_env_pre_post_processors_compat(env_cfg: Any, policy_cfg: Any) -> tuple[Any, Any]:
    del policy_cfg
    if hasattr(env_cfg, "get_env_processors"):
        return env_cfg.get_env_processors()
    raise RuntimeError(
        "This LeRobot build does not expose LIBERO environment processors. Update LeRobot or share the output of "
        "`python -c \"from lerobot.envs.configs import LiberoEnv; print(dir(LiberoEnv))\"`."
    )
