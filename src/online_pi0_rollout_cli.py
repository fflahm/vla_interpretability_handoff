from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .online_pi0_rollout import Pi0LiberoFullTokenRolloutTracer
from .online_rollout import DEFAULT_RENAME_MAP
from .utils import load_config, resolve_path, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--pi0-path", default=None, help="Local LeRobot PI0 checkpoint directory or Hugging Face repo id.")
    parser.add_argument("--task", default=None, help="LIBERO suite, such as libero_spatial.")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--start-seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--pickup-joint-name", default=None)
    parser.add_argument("--place-joint-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--rotate-images-180", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--force-replan-every-step", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--replan-interval", type=int, default=None)
    parser.add_argument("--activation-dtype", choices=("float16", "float32"), default=None)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-wrist-video", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--video-fps", type=int, default=None)
    parser.add_argument("--video-format", choices=("gif", "mp4"), default=None)
    parser.add_argument("--video-flip-180", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--require-mp4", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-activations", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--rename-map",
        default=None,
        help='JSON rename map for policy inputs, e.g. \'{"observation.images.image2":"observation.images.wrist_image"}\'.',
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(resolve_path(args.config, root))
    rollout_cfg = cfg["online_pi0"]
    seed = int(args.start_seed if args.start_seed is not None else rollout_cfg.get("start_seed", cfg["seed"]))
    set_seed(seed)
    checkpoint_path = args.pi0_path or os.environ.get("VLA_PI0_PATH") or rollout_cfg.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("Provide `--pi0-path`, set VLA_PI0_PATH, or configure online_pi0.checkpoint_path.")
    rename_map = json.loads(args.rename_map) if args.rename_map else dict(rollout_cfg.get("rename_map", DEFAULT_RENAME_MAP))
    tracer = Pi0LiberoFullTokenRolloutTracer(
        checkpoint_path=str(checkpoint_path),
        task=str(args.task or rollout_cfg.get("task", "libero_spatial")),
        task_id=int(args.task_id if args.task_id is not None else rollout_cfg.get("task_id", 1)),
        device=str(args.device or cfg["model"].get("device", "auto")),
        pickup_joint_name=str(args.pickup_joint_name or rollout_cfg.get("pickup_joint_name", "auto")),
        place_joint_name=str(args.place_joint_name or rollout_cfg.get("place_joint_name", "auto")),
        instruction=str(args.instruction or rollout_cfg.get("instruction", "")),
        rename_map=rename_map,
        rotate_images_180=bool(
            args.rotate_images_180
            if args.rotate_images_180 is not None
            else rollout_cfg.get("rotate_images_180", True)
        ),
        force_replan_every_step=bool(
            args.force_replan_every_step
            if args.force_replan_every_step is not None
            else rollout_cfg.get("force_replan_every_step", True)
        ),
        replan_interval=int(args.replan_interval if args.replan_interval is not None else rollout_cfg.get("replan_interval", 1)),
        activation_dtype=str(args.activation_dtype or rollout_cfg.get("activation_dtype", "float16")),
    )
    try:
        summary = tracer.collect(
            output_dir=resolve_path(args.output_dir or rollout_cfg["output_dir"], root),
            num_episodes=int(args.num_episodes if args.num_episodes is not None else rollout_cfg.get("num_episodes", 2)),
            start_seed=seed,
            max_steps=args.max_steps if args.max_steps is not None else rollout_cfg.get("max_steps"),
            save_video=bool(
                args.save_video if args.save_video is not None else rollout_cfg.get("save_video", True)
            ),
            save_wrist_video=bool(
                args.save_wrist_video
                if args.save_wrist_video is not None
                else rollout_cfg.get("save_wrist_video", False)
            ),
            video_fps=int(args.video_fps if args.video_fps is not None else rollout_cfg.get("video_fps", 10)),
            video_format=str(args.video_format or rollout_cfg.get("video_format", "mp4")),
            video_flip_180=bool(
                args.video_flip_180
                if args.video_flip_180 is not None
                else rollout_cfg.get("video_flip_180", True)
            ),
            require_mp4=bool(
                args.require_mp4 if args.require_mp4 is not None else rollout_cfg.get("require_mp4", True)
            ),
            save_activations=bool(
                args.save_activations
                if args.save_activations is not None
                else rollout_cfg.get("save_activations", True)
            ),
        )
    finally:
        tracer.close()
    print(json.dumps(summary, indent=2))
