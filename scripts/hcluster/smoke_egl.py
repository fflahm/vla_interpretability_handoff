#!/usr/bin/env python
"""Headless EGL probe for rjob GPU workers."""
from __future__ import annotations

import os

print("MUJOCO_GL", os.environ.get("MUJOCO_GL"))
print("PYOPENGL_PLATFORM", os.environ.get("PYOPENGL_PLATFORM"))
print("NVIDIA_DRIVER_CAPABILITIES", os.environ.get("NVIDIA_DRIVER_CAPABILITIES"))

from OpenGL import EGL

print("EGL module", EGL)
query = getattr(EGL, "eglQueryString", None)
if query is None:
    raise SystemExit("EGL.eglQueryString is missing; libEGL did not load")
print("eglQueryString", query)

import mujoco

print("mujoco", mujoco.__version__, "GLContext", getattr(mujoco, "GLContext", None))
print("egl_probe_ok")
