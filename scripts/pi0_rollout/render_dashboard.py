#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", required=True)
    parser.add_argument("--rollout-dir", default=None, help="Optional local rollout dir used to resolve image paths.")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["pickup_offset", "place_offset", "action", "policy_pred_action", "progress"],
    )
    parser.add_argument("--token-bins", type=int, default=96)
    parser.add_argument(
        "--token-map",
        default="auto",
        help=(
            "Token-region labels for the activation heatmap. Use `auto`, `none`, or comma-separated "
            "NAME:START:END entries in original token indices, e.g. text:0:48,image1:48:304."
        ),
    )
    parser.add_argument("--activation-vlim", type=float, default=None, help="Fixed absolute z-score color limit.")
    parser.add_argument("--probe-vmax", type=float, default=None, help="Fixed probe-error colorbar maximum.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--output", default=None)
    parser.add_argument("--format", choices=("mp4", "gif"), default="mp4")
    parser.add_argument("--tmp-dir", default=None, help="Local temp dir for MP4 encoding before copying to the final path.")
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    output = Path(args.output) if args.output else analysis_dir / f"pi0_dynamic_episode_{args.episode_index:03d}_dashboard.{args.format}"
    render_episode_dashboard_video(
        analysis_dir=analysis_dir,
        rollout_dir=Path(args.rollout_dir) if args.rollout_dir else None,
        episode_index=int(args.episode_index),
        targets=tuple(args.targets),
        fps=int(args.fps),
        output_path=output,
        video_format=args.format,
        tmp_dir=Path(args.tmp_dir) if args.tmp_dir else None,
        token_bins=int(args.token_bins),
        token_map_spec=str(args.token_map),
        activation_vlim=args.activation_vlim,
        probe_vmax=args.probe_vmax,
    )
    print(f"Saved dashboard video to {output}")


def render_episode_dashboard_video(
    analysis_dir: Path,
    rollout_dir: Path | None,
    episode_index: int,
    targets: tuple[str, ...],
    fps: int,
    output_path: Path,
    video_format: str,
    tmp_dir: Path | None,
    token_bins: int,
    token_map_spec: str,
    activation_vlim: float | None,
    probe_vmax: float | None,
) -> None:
    samples = pd.read_csv(analysis_dir / "pi0_dynamic_samples.csv")
    summary = pd.read_csv(analysis_dir / "pi0_dynamic_probe_summary.csv")
    manifest = pd.read_csv(analysis_dir / "pi0_dynamic_activation_manifest.csv")
    episode = samples[samples["episode_index"].astype(int) == episode_index].copy()
    if episode.empty:
        raise ValueError(f"Episode {episode_index} not found in {analysis_dir / 'pi0_dynamic_samples.csv'}.")
    episode = episode.sort_values("step")
    episode_manifest = manifest[manifest["episode_index"].astype(int) == episode_index].copy()
    if episode_manifest.empty:
        raise ValueError(f"Episode {episode_index} has no activation manifest rows.")
    probe_tables = load_probe_prediction_tables(analysis_dir, targets, episode_index)
    best_layers = best_layers_by_target(summary, targets)
    activation_cache: dict[int, tuple[np.ndarray, int]] = {}
    probe_cache: dict[int, tuple[np.ndarray, list[str], int]] = {}
    token_count = infer_num_tokens(episode_manifest)
    token_layout = load_token_layout(analysis_dir, rollout_dir)
    token_regions = parse_token_regions(token_map_spec, token_count, token_bins, token_layout)
    fixed_activation_vlim = (
        float(activation_vlim)
        if activation_vlim is not None
        else estimate_activation_vlim(episode_manifest, rollout_dir, token_bins, activation_cache)
    )
    fixed_probe_vmax = float(probe_vmax) if probe_vmax is not None else estimate_probe_vmax(probe_tables, probe_cache)

    frames = []
    max_step = int(episode["step"].max())
    for row in episode.to_dict("records"):
        step = int(row["step"])
        image = load_step_image(row, rollout_dir)
        activation_heat, activation_step = activation_heatmap_for_step(
            episode_manifest=episode_manifest,
            rollout_dir=rollout_dir,
            step=step,
            token_bins=token_bins,
            cache=activation_cache,
        )
        probe_error_heat, probe_targets, probe_step = probe_error_heatmap_for_step(
            probe_tables=probe_tables,
            step=step,
            cache=probe_cache,
        )
        frame = draw_dashboard_frame(
            image=image,
            step=step,
            activation_step=activation_step,
            probe_step=probe_step,
            max_step=max_step,
            episode_index=episode_index,
            success=bool(episode["success"].max()),
            is_replan=bool(row.get("is_replan", False)),
            activation_heat=activation_heat,
            probe_error_heat=probe_error_heat,
            probe_targets=probe_targets,
            best_layers=best_layers,
            token_regions=token_regions,
            activation_vlim=fixed_activation_vlim,
            probe_vmax=fixed_probe_vmax,
        )
        frames.append(frame)
    write_video(frames, output_path, fps=fps, video_format=video_format, tmp_dir=tmp_dir)


def load_probe_prediction_tables(analysis_dir: Path, targets: tuple[str, ...], episode_index: int) -> dict[str, pd.DataFrame]:
    tables = {}
    for target in targets:
        path = analysis_dir / f"pi0_dynamic_probe_{target}_sample_predictions.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame = frame[frame["episode_index"].astype(int) == episode_index].copy()
        if frame.empty:
            continue
        frame["error_norm"] = [
            float(
                np.linalg.norm(
                    parse_prediction_value(row["prediction"])
                    - parse_prediction_value(row["ground_truth"])
                )
            )
            for _, row in frame.iterrows()
        ]
        tables[target] = frame.sort_values(["step", "layer"])
    return tables


def parse_prediction_value(value: object) -> np.ndarray:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return np.asarray([float(value)], dtype=np.float32)
    parsed = ast.literal_eval(str(value))
    if isinstance(parsed, (int, float)):
        return np.asarray([float(parsed)], dtype=np.float32)
    return np.asarray(parsed, dtype=np.float32).reshape(-1)


def best_layers_by_target(summary: pd.DataFrame, targets: tuple[str, ...]) -> dict[str, int]:
    best = {}
    for target in targets:
        match = summary[summary["target"].astype(str) == target]
        if not match.empty:
            best[target] = int(match.iloc[0]["best_layer"])
    return best


def estimate_activation_vlim(
    episode_manifest: pd.DataFrame,
    rollout_dir: Path | None,
    token_bins: int,
    cache: dict[int, tuple[np.ndarray, int]],
) -> float:
    values = []
    for step in sorted(int(x) for x in episode_manifest["step"].drop_duplicates().tolist()):
        heat, _ = activation_heatmap_for_step(episode_manifest, rollout_dir, step, token_bins, cache)
        values.append(np.abs(heat).reshape(-1))
    if not values:
        return 1.0
    finite = np.concatenate(values)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    return max(1e-6, float(np.nanpercentile(finite, 98)))


def estimate_probe_vmax(
    probe_tables: dict[str, pd.DataFrame],
    cache: dict[int, tuple[np.ndarray, list[str], int]],
) -> float:
    all_steps = sorted({int(x) for frame in probe_tables.values() for x in frame["step"].drop_duplicates().tolist()})
    values = []
    for step in all_steps:
        heat, _, _ = probe_error_heatmap_for_step(probe_tables, step, cache)
        finite = heat[np.isfinite(heat)]
        if finite.size:
            values.append(finite)
    if not values:
        return 1.0
    finite = np.concatenate(values)
    return max(1e-6, float(np.nanpercentile(finite, 95)))


def infer_num_tokens(episode_manifest: pd.DataFrame) -> int:
    if episode_manifest.empty or "shape" not in episode_manifest.columns:
        return 0
    shape = ast.literal_eval(str(episode_manifest.iloc[0]["shape"]))
    return int(shape[0]) if shape else 0


def load_token_layout(analysis_dir: Path, rollout_dir: Path | None) -> dict[str, Any] | None:
    candidates = [analysis_dir / "token_layout.json"]
    if rollout_dir is not None:
        candidates.append(rollout_dir / "token_layout.json")
    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def parse_token_regions(
    token_map_spec: str,
    num_tokens: int,
    token_bins: int,
    token_layout: dict[str, Any] | None,
) -> list[tuple[str, float, float, str]]:
    if token_map_spec.lower() in ("", "none"):
        return []
    if token_map_spec.lower() == "auto":
        layout_regions = token_layout.get("regions", []) if token_layout else []
        if layout_regions:
            raw_regions = [
                (
                    str(region.get("name", f"region_{index}")),
                    int(region["start"]),
                    int(region["end"]),
                    str(region.get("stream", "prefix")),
                    int(region.get("sequence_length", num_tokens)),
                )
                for index, region in enumerate(layout_regions)
                if "start" in region and "end" in region
            ]
        elif token_layout:
            raw_regions = infer_pi0_regions_from_token_layout_summary(token_layout)
        elif num_tokens == 816:
            # Conservative fallback from observed PI0 batch shapes only. This gives lengths, but the
            # true order should be verified from PI0 model masks/source when token_layout.json has regions.
            raw_regions = [
                ("visual 1", 0, 256, "prefix", 816),
                ("visual 2", 256, 512, "prefix", 816),
                ("visual 3", 512, 768, "prefix", 816),
                ("language", 768, 816, "prefix", 816),
            ]
        elif num_tokens > 0:
            raw_regions = [("all tokens", 0, num_tokens, "prefix", num_tokens)]
        else:
            raw_regions = []
    else:
        raw_regions = []
        for item in token_map_spec.split(","):
            parts = item.strip().split(":")
            if len(parts) != 3:
                raise ValueError(f"Bad --token-map entry `{item}`. Expected NAME:START:END.")
            raw_regions.append((parts[0], int(parts[1]), int(parts[2]), "prefix", num_tokens))
    if num_tokens <= 0:
        return []
    regions = []
    for label, start, end, stream, region_sequence_length in raw_regions:
        denominator = max(1, int(region_sequence_length))
        start_bin = max(0.0, min(float(token_bins), float(start) / float(denominator) * float(token_bins)))
        end_bin = max(0.0, min(float(token_bins), float(end) / float(denominator) * float(token_bins)))
        if end_bin > start_bin:
            regions.append((label, start_bin, end_bin, stream))
    return regions


def infer_pi0_regions_from_token_layout_summary(token_layout: dict[str, Any]) -> list[tuple[str, int, int, str, int]]:
    sequence_length = int(token_layout.get("sequence_length", 0) or 0)
    batch_summary = token_layout.get("batch_summary", {})
    config_summary = token_layout.get("config_summary", {})
    language_shape = batch_summary.get("observation.language.tokens", {}).get("shape")
    if not sequence_length or not language_shape:
        return []
    language_len = int(language_shape[-1])
    image_keys = [
        key
        for key, value in batch_summary.items()
        if str(key).startswith("observation.images.") and isinstance(value, dict) and "shape" in value
    ]
    empty_cameras = int(config_summary.get("empty_cameras", 0) or 0)
    num_image_blocks = len(image_keys) + empty_cameras
    remaining = sequence_length - language_len
    if remaining <= 0 or num_image_blocks <= 0 or remaining % num_image_blocks != 0:
        return []
    image_tokens = remaining // num_image_blocks
    regions: list[tuple[str, int, int, str, int]] = []
    cursor = 0
    for key in image_keys:
        label = str(key).replace("observation.images.", "")
        regions.append((label, cursor, cursor + image_tokens, "prefix", sequence_length))
        cursor += image_tokens
    for index in range(empty_cameras):
        regions.append((f"empty_camera_{index}", cursor, cursor + image_tokens, "prefix", sequence_length))
        cursor += image_tokens
    regions.append(("language", cursor, cursor + language_len, "prefix", sequence_length))
    cursor += language_len
    chunk_size = int(config_summary.get("chunk_size", 0) or config_summary.get("n_action_steps", 0) or 0)
    if chunk_size > 0:
        expert_len = chunk_size + 1
        regions.extend(
            [
                ("state", 0, 1, "expert", expert_len),
                ("action_chunk", 1, expert_len, "expert", expert_len),
            ]
        )
    return regions if cursor == sequence_length else []


def activation_heatmap_for_step(
    episode_manifest: pd.DataFrame,
    rollout_dir: Path | None,
    step: int,
    token_bins: int,
    cache: dict[int, tuple[np.ndarray, int]],
) -> tuple[np.ndarray, int]:
    available_steps = sorted(int(x) for x in episode_manifest["step"].drop_duplicates().tolist())
    if not available_steps:
        return np.zeros((1, token_bins), dtype=np.float32), step
    chosen_step = max([x for x in available_steps if x <= step], default=available_steps[0])
    if chosen_step in cache:
        return cache[chosen_step]
    frame = episode_manifest[episode_manifest["step"].astype(int) == chosen_step].sort_values("layer_index")
    rows = []
    for row in frame.to_dict("records"):
        path = resolve_activation_path(row, rollout_dir)
        if not path.exists():
            rows.append(np.zeros(token_bins, dtype=np.float32))
            continue
        array = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
        token_norm = np.linalg.norm(array, axis=-1)
        rows.append(bin_vector(token_norm, token_bins))
    heat = np.stack(rows, axis=0).astype(np.float32)
    heat = (heat - heat.mean(axis=1, keepdims=True)) / (heat.std(axis=1, keepdims=True) + 1e-8)
    heat = np.nan_to_num(heat, nan=0.0, posinf=0.0, neginf=0.0)
    cache[chosen_step] = (heat, chosen_step)
    return heat, chosen_step


def probe_error_heatmap_for_step(
    probe_tables: dict[str, pd.DataFrame],
    step: int,
    cache: dict[int, tuple[np.ndarray, list[str], int]],
) -> tuple[np.ndarray, list[str], int]:
    all_steps = sorted({int(x) for frame in probe_tables.values() for x in frame["step"].drop_duplicates().tolist()})
    if not all_steps:
        return np.zeros((1, 1), dtype=np.float32), [], step
    chosen_step = max([x for x in all_steps if x <= step], default=all_steps[0])
    if chosen_step in cache:
        return cache[chosen_step]
    targets = list(probe_tables)
    max_layer = max(int(frame["layer"].max()) for frame in probe_tables.values())
    heat = np.full((len(targets), max_layer + 1), np.nan, dtype=np.float32)
    for target_i, target in enumerate(targets):
        current = probe_tables[target][probe_tables[target]["step"].astype(int) == chosen_step]
        for row in current.to_dict("records"):
            heat[target_i, int(row["layer"])] = float(row["error_norm"])
    cache[chosen_step] = (heat, targets, chosen_step)
    return heat, targets, chosen_step


def resolve_activation_path(row: dict[str, object], rollout_dir: Path | None) -> Path:
    path = Path(str(row["path"]))
    if path.exists():
        return path
    if rollout_dir is None:
        return path
    episode_index = int(row["episode_index"])
    step = int(row["step"])
    return rollout_dir / f"episode_{episode_index:03d}" / "activations" / f"step_{step:04d}" / path.name


def bin_vector(values: np.ndarray, bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) == bins:
        return values
    edges = np.linspace(0, len(values), bins + 1, dtype=int)
    output = np.zeros(bins, dtype=np.float32)
    for index in range(bins):
        start, end = int(edges[index]), int(edges[index + 1])
        if end <= start:
            output[index] = values[min(start, len(values) - 1)]
        else:
            output[index] = float(values[start:end].mean())
    return output


def load_step_image(row: dict[str, object], rollout_dir: Path | None) -> np.ndarray:
    candidates = [Path(str(row.get("image_path", "")))]
    if rollout_dir is not None:
        episode_index = int(row["episode_index"])
        step = int(row["step"])
        candidates.append(rollout_dir / f"episode_{episode_index:03d}" / "images" / f"step_{step:04d}.png")
    for path in candidates:
        if path.exists():
            return np.asarray(Image.open(path).convert("RGB"))
    return np.full((224, 224, 3), 235, dtype=np.uint8)


def draw_dashboard_frame(
    image: np.ndarray,
    step: int,
    activation_step: int,
    probe_step: int,
    max_step: int,
    episode_index: int,
    success: bool,
    is_replan: bool,
    activation_heat: np.ndarray,
    probe_error_heat: np.ndarray,
    probe_targets: list[str],
    best_layers: dict[str, int],
    token_regions: list[tuple[str, float, float, str]],
    activation_vlim: float,
    probe_vmax: float,
) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13.5, 7.2), dpi=120)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.65], height_ratios=[1.15, 0.85])
    ax_img = fig.add_subplot(grid[:, 0])
    ax_act = fig.add_subplot(grid[0, 1])
    ax_probe = fig.add_subplot(grid[1, 1])

    ax_img.imshow(np.flip(image, axis=(0, 1)))
    ax_img.axis("off")
    status = "success" if success else "failure"
    replan = "replan" if is_replan else "queued action"
    ax_img.set_title(f"Episode {episode_index:03d} | step {step}/{max_step} | {status} | {replan}", fontsize=11)

    act_limit = max(1e-6, float(activation_vlim))
    act = ax_act.imshow(activation_heat, aspect="auto", origin="lower", cmap="coolwarm", vmin=-act_limit, vmax=act_limit)
    ax_act.set_title(f"Activation at replan step {activation_step}: layer x token-bin norm z-score")
    ax_act.set_xlabel("Token bins")
    ax_act.set_ylabel("Layer")
    if activation_heat.shape[0] > 18:
        ax_act.axhline(17.5, color="black", linewidth=1.0, alpha=0.65)
        ax_act.text(
            0.002,
            18.0 / max(1, activation_heat.shape[0]),
            "expert stream",
            transform=ax_act.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color="black",
        )
    for label, start_bin, end_bin, stream in token_regions:
        line_style = "-" if stream == "prefix" else "--"
        alpha = 0.45 if stream == "prefix" else 0.35
        ax_act.axvline(start_bin - 0.5, color="black", linewidth=0.7, alpha=alpha, linestyle=line_style)
        ax_act.axvline(end_bin - 0.5, color="black", linewidth=0.7, alpha=alpha, linestyle=line_style)
        center = (start_bin + end_bin) / 2.0 - 0.5
        y_pos = 1.02 if stream == "prefix" else -0.16
        va = "bottom" if stream == "prefix" else "top"
        ax_act.text(
            center,
            y_pos,
            label,
            transform=ax_act.get_xaxis_transform(),
            ha="center",
            va=va,
            fontsize=8,
            rotation=0,
        )
    fig.colorbar(act, ax=ax_act, fraction=0.025, pad=0.02, label="z")

    probe_img = ax_probe.imshow(probe_error_heat, aspect="auto", cmap="magma_r", vmin=0.0, vmax=max(probe_vmax, 1e-6))
    for target_i, target in enumerate(probe_targets):
        if target in best_layers:
            ax_probe.scatter(best_layers[target], target_i, marker="*", s=90, c="cyan", edgecolors="black", linewidths=0.5)
    ax_probe.set_yticks(np.arange(len(probe_targets)))
    ax_probe.set_yticklabels([target.replace("_", " ") for target in probe_targets], fontsize=8)
    ax_probe.set_title(f"Probe error at replan step {probe_step}: target x layer (lower is better)")
    ax_probe.set_xlabel("Layer")
    ax_probe.set_ylabel("Probe target")
    fig.colorbar(probe_img, ax=ax_probe, fraction=0.025, pad=0.02, label="error")

    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def write_video(frames: list[np.ndarray], output_path: Path, fps: int, video_format: str, tmp_dir: Path | None = None) -> None:
    ensure_dir(output_path.parent)
    if not frames:
        raise ValueError("No frames to write.")
    if video_format == "mp4":
        try:
            import imageio.v2 as imageio

            with tempfile.TemporaryDirectory(dir=str(tmp_dir) if tmp_dir else None) as temp_dir:
                temp_path = Path(temp_dir) / output_path.name
                with imageio.get_writer(
                    temp_path,
                    fps=fps,
                    codec="libx264",
                    macro_block_size=1,
                    ffmpeg_log_level="error",
                    output_params=["-pix_fmt", "yuv420p"],
                ) as writer:
                    for frame in frames:
                        writer.append_data(np.asarray(frame, dtype=np.uint8))
                shutil.copyfile(temp_path, output_path)
            return
        except Exception as exc:
            raise RuntimeError(
                "Could not write MP4. Install `imageio imageio-ffmpeg`, use a local `--tmp-dir`, "
                "or use `--format gif`."
            ) from exc
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=max(1, int(round(1000 / fps))),
        loop=0,
    )


if __name__ == "__main__":
    main()
