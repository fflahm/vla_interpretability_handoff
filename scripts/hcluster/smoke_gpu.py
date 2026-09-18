#!/usr/bin/env python
"""Cheap GPU + checkpoint-config smoke for rjob workers."""
from __future__ import annotations

import os

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv  # noqa: F401
from lerobot.policies.pi0 import PI0Policy  # noqa: F401
from lerobot.policies.pi05 import PI05Policy  # noqa: F401


def main() -> None:
    print("torch", torch.__version__, "cuda_built", torch.version.cuda)
    print("cuda_available", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available in this worker")
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
    x = torch.zeros(8, 8, device="cuda")
    print("matmul_ok", float((x @ x).sum().item()))
    for name, key in (("pi05", "PI05_PATH"), ("pi0", "PI0_PATH")):
        path = os.environ[key]
        cfg = PreTrainedConfig.from_pretrained(path)
        print(name, type(cfg).__name__, path)
    print("gpu_smoke_ok")


if __name__ == "__main__":
    main()
