#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data import save_states
from src.envs import LiberoDatasetEnv, LiberoEnv, SyntheticToyEnv
from src.utils import load_config, resolve_path, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--env", default=None, choices=["synthetic", "libero", "libero_dataset"])
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--dataset-hdf5", default=None, help="LIBERO demonstration HDF5 for `--env libero_dataset`.")
    parser.add_argument("--dataset-dir", default=None, help="Directory of LIBERO task HDF5 files for multi-task sampling.")
    parser.add_argument("--target-joint-name", default=None, help="Optional target free-joint override for LIBERO.")
    parser.add_argument("--target-role", choices=["pickup", "place"], default=None, help="Select the source or destination object.")
    parser.add_argument(
        "--min-frame-fraction",
        type=float,
        default=None,
        help="For `libero_dataset`, sample after this trajectory fraction, e.g. 0.5 for place states.",
    )
    parser.add_argument(
        "--max-frame-fraction",
        type=float,
        default=None,
        help="For `libero_dataset`, sample only the first fraction of each trajectory, e.g. 0.5 for approach states.",
    )
    parser.add_argument(
        "--early-frame-indices",
        nargs="+",
        type=int,
        default=None,
        help="For `libero_dataset`, sample only exact early frame indices, e.g. 0 1 2 3.",
    )
    args = parser.parse_args()

    cfg = load_config(resolve_path(args.config, ROOT))
    set_seed(int(cfg["seed"]))
    env_name = args.env or cfg["env"]["name"]
    num_samples = args.num_samples or int(cfg["data"]["num_samples"])

    state_path = resolve_path(cfg["data"]["state_path"], ROOT)
    image_dir = resolve_path(cfg["data"]["image_dir"], ROOT)

    if env_name == "synthetic":
        env = SyntheticToyEnv(
            image_size=int(cfg["data"]["image_size"]),
            instruction=str(cfg["data"]["instruction"]),
            min_pixel_distance=int(cfg["env"]["min_pixel_distance"]),
            background_randomization=bool(cfg["env"]["background_randomization"]),
            seed=int(cfg["seed"]),
        )
    elif env_name == "libero":
        env = LiberoEnv(
            image_size=int(cfg["data"]["image_size"]),
            instruction=str(cfg["data"]["instruction"]),
            task=str(cfg["env"].get("task", "libero_object")),
            task_id=int(cfg["env"].get("task_id", 0)),
            target_body_name=cfg["env"].get("target_body_name"),
            target_joint_name=args.target_joint_name or cfg["env"].get("target_joint_name"),
            target_site_name=cfg["env"].get("target_site_name"),
            camera_key=str(cfg["env"].get("camera_key", "observation.images.image")),
            wrist_camera_key=str(cfg["env"].get("wrist_camera_key", "pixels.image2")),
            seed=int(cfg["seed"]),
        )
    else:
        dataset_hdf5 = args.dataset_hdf5 or cfg["env"].get("dataset_hdf5")
        dataset_dir = args.dataset_dir or cfg["env"].get("dataset_dir")
        if not dataset_hdf5 and not dataset_dir:
            raise ValueError(
                "Provide `--dataset-hdf5`, `--dataset-dir`, or configure the corresponding `env` value "
                "for `--env libero_dataset`."
            )
        env = LiberoDatasetEnv(
            hdf5_path=Path(dataset_hdf5).expanduser() if dataset_hdf5 else None,
            dataset_dir=Path(dataset_dir).expanduser() if dataset_dir else None,
            image_size=int(cfg["data"]["image_size"]),
            target_joint_name=args.target_joint_name or cfg["env"].get("target_joint_name"),
            target_role=args.target_role or str(cfg["env"].get("target_role", "pickup")),
            min_frame_fraction=(
                args.min_frame_fraction if args.min_frame_fraction is not None else float(cfg["env"].get("min_frame_fraction", 0.0))
            ),
            max_frame_fraction=(
                args.max_frame_fraction if args.max_frame_fraction is not None else float(cfg["env"].get("max_frame_fraction", 1.0))
            ),
            early_frame_indices=tuple(args.early_frame_indices) if args.early_frame_indices is not None else None,
            seed=int(cfg["seed"]),
        )

    samples = env.collect(num_samples, image_dir)
    save_states(state_path, samples)
    print(f"Saved {len(samples)} states to {state_path}")


if __name__ == "__main__":
    main()
