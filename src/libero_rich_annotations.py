"""Auditable, sampling-oriented annotations for LIBERO demonstration HDF5 files.

The extractor deliberately separates recorded facts, XML/BDDL metadata, derived
quantities, and simulator-only facts through per-field provenance and validity.
It never interprets geometric proximity as MuJoCo contact.
"""
from __future__ import annotations

import json
import os
import random
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
from PIL import Image

ANNOTATION_VERSION = "libero-rich-v1.1"
DEFAULT_FUTURE_HORIZONS = (1, 4, 8, 16)
DEFAULT_ACTION_CHUNK = 16
LOCAL_LIBERO_ROOT = Path(
    os.environ.get("LIBERO_ROOT", "/data/tos/guoshengyu/vla/libero/LIBERO")
)


def native(value: Any) -> Any:
    """Recursively convert NumPy/HDF5 values to JSON-native values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    return value


def field(value: Any, valid: bool, provenance: str, **extra: Any) -> dict[str, Any]:
    return {"value": native(value), "valid": bool(valid), "provenance": provenance, **native(extra)}


@dataclass(frozen=True)
class DemoRef:
    suite: str
    task: str
    hdf5_path: Path
    demo_key: str
    length: int

    @property
    def episode_id(self) -> str:
        return f"{self.suite}/{self.task}/{self.demo_key}"


def discover_hdf5(input_path: Path, suite_filters: Sequence[str] = ()) -> list[tuple[str, Path]]:
    """Discover task HDF5s below either one suite directory or a suite root."""
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"LIBERO input does not exist: {input_path}")
    files = sorted(input_path.rglob("*_demo.hdf5")) if input_path.is_dir() else [input_path]
    filters = {x.lower() for x in suite_filters}
    found: list[tuple[str, Path]] = []
    for path in files:
        suite = path.parent.name
        if filters and suite.lower() not in filters:
            continue
        found.append((suite, path))
    if not found:
        suffix = f" matching suites {sorted(filters)}" if filters else ""
        raise FileNotFoundError(f"No *_demo.hdf5 files found under {input_path}{suffix}")
    return found


def inventory_demos(files: Sequence[tuple[str, Path]]) -> list[DemoRef]:
    refs: list[DemoRef] = []
    for suite, path in files:
        with h5py.File(path, "r") as handle:
            if "data" not in handle:
                raise ValueError(f"Missing /data group in {path}")
            for key in sorted(handle["data"], key=_demo_sort_key):
                group = handle["data"][key]
                length = int(group["actions"].shape[0])
                refs.append(DemoRef(suite, path.name.removesuffix("_demo.hdf5"), path, key, length))
    return refs


def _demo_sort_key(key: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", key)
    return (int(match.group(1)) if match else 10**9, key)


def balanced_demo_selection(demos: Sequence[DemoRef], limit: int, seed: int) -> list[DemoRef]:
    """Seeded round-robin over tasks after independently shuffling each task."""
    if limit < 1:
        raise ValueError("--num-demos must be positive")
    groups: dict[tuple[str, str], list[DemoRef]] = defaultdict(list)
    for demo in demos:
        groups[(demo.suite, demo.task)].append(demo)
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(groups[key])
    chosen: list[DemoRef] = []
    while len(chosen) < min(limit, len(demos)):
        advanced = False
        for key in keys:
            if groups[key] and len(chosen) < limit:
                chosen.append(groups[key].pop())
                advanced = True
        if not advanced:
            break
    return chosen


def _top_change_indices(values: np.ndarray, count: int, threshold: float = 0.0) -> list[int]:
    if len(values) < 2:
        return []
    scores = np.linalg.norm(np.diff(values, axis=0), axis=-1) if values.ndim > 1 else np.abs(np.diff(values))
    indices = np.argsort(scores)[::-1]
    return [int(i + 1) for i in indices[:count] if float(scores[i]) > threshold]


def select_frame_indices(
    actions: np.ndarray,
    ee_pos: np.ndarray | None,
    limit: int,
) -> tuple[list[int], dict[int, list[str]]]:
    """Combine temporal coverage with action/gripper/trajectory event candidates."""
    length = len(actions)
    if limit < 1 or length < 1:
        return [], {}
    limit = min(limit, length)
    reasons: dict[int, list[str]] = defaultdict(list)
    quantiles = np.linspace(0, length - 1, limit)
    for idx in np.rint(quantiles).astype(int):
        reasons[int(idx)].append("time_quantile")
    event_budget = max(2, limit // 3)
    event_lists: list[list[int]] = []
    if actions.shape[1] >= 7:
        event_lists.append(_top_change_indices(actions[:, 6], event_budget, 0.2))
        for idx in event_lists[-1]:
            reasons[idx].append("gripper_command_change")
    event_lists.append(_top_change_indices(actions[:, :6], event_budget))
    for idx in event_lists[-1]:
        reasons[idx].append("large_action_change")
    if ee_pos is not None and len(ee_pos) == length:
        velocity = np.diff(ee_pos, axis=0, prepend=ee_pos[:1])
        event_lists.append(_top_change_indices(velocity, event_budget))
        for idx in event_lists[-1]:
            reasons[idx].append("trajectory_keypoint")
    # Reserve the strongest candidate from each available event detector.
    selected_set: set[int] = set()
    for event_indices in event_lists:
        if event_indices and len(selected_set) < limit:
            selected_set.add(event_indices[0])
    # Rank remaining events first, retain broad quantile coverage, enforce a hard cap.
    candidates = list(reasons)
    candidates.sort(key=lambda i: (-len(reasons[i]), min(abs(i - q) for q in quantiles), i))
    for candidate in candidates:
        if len(selected_set) >= limit:
            break
        selected_set.add(candidate)
    selected = sorted(selected_set)
    return selected, {i: reasons[i] for i in selected}


def _parse_sexpr(text: str) -> list[Any]:
    tokens = re.findall(r"\(|\)|[^\s()]+", re.sub(r";[^\n]*", "", text))
    stack: list[list[Any]] = [[]]
    for token in tokens:
        if token == "(":
            child: list[Any] = []
            stack[-1].append(child)
            stack.append(child)
        elif token == ")":
            if len(stack) == 1:
                raise ValueError("Unbalanced ')' in BDDL")
            stack.pop()
        else:
            stack[-1].append(token)
    if len(stack) != 1:
        raise ValueError("Unbalanced '(' in BDDL")
    return stack[0]


def _find_section(tree: Any, name: str) -> list[Any] | None:
    if isinstance(tree, list):
        if tree and str(tree[0]).lower() == name.lower():
            return tree
        for item in tree:
            found = _find_section(item, name)
            if found is not None:
                return found
    return None


def parse_bddl(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"valid": False, "path": str(path) if path else None, "warnings": ["BDDL file not found"]}
    text = path.read_text(encoding="utf-8")
    tree = _parse_sexpr(text)
    objects: list[dict[str, str]] = []
    fixtures: list[dict[str, str]] = []
    for section_name, target in ((":objects", objects), (":fixtures", fixtures)):
        section = _find_section(tree, section_name) or []
        pending: list[str] = []
        i = 1
        while i < len(section):
            token = section[i]
            if token == "-" and i + 1 < len(section):
                category = str(section[i + 1])
                target.extend({"instance": name, "category": category} for name in pending)
                pending = []
                i += 2
            else:
                if isinstance(token, str):
                    pending.append(token)
                i += 1
    language = _find_section(tree, ":language")
    interest = _find_section(tree, ":obj_of_interest") or []
    goal = _find_section(tree, ":goal")
    init = _find_section(tree, ":init")
    return {
        "valid": True,
        "path": str(path),
        "language": " ".join(map(str, language[1:])) if language else None,
        "objects": objects,
        "fixtures": fixtures,
        "objects_of_interest": [str(x) for x in interest[1:]],
        "init_ast": native(init[1:] if init else []),
        "goal_ast": native(goal[1:] if goal else []),
        "goal_predicates": _flatten_predicates(goal[1:] if goal else []),
        "provenance": "bddl",
    }


def _flatten_predicates(node: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if isinstance(node, list):
        if node and isinstance(node[0], str) and node[0].lower() not in {"and", "or", "not"}:
            result.append({"name": node[0], "arguments": native(node[1:])})
        else:
            for item in node[1:] if node and isinstance(node[0], str) else node:
                result.extend(_flatten_predicates(item))
    return result


def parse_model_xml(xml_text: str) -> dict[str, Any]:
    """Extract entities, free-joint initial poses, assets, materials and cameras."""
    root = ET.fromstring(xml_text)
    materials = {
        el.get("name"): {"rgba": _floats(el.get("rgba")), "texture": el.get("texture")}
        for el in root.findall(".//asset/material") if el.get("name")
    }
    entities: list[dict[str, Any]] = []
    for body in root.findall(".//body"):
        name = body.get("name")
        if not name:
            continue
        joints = body.findall("./joint")
        free = body.find("./freejoint")
        if free is None:
            free = next((j for j in joints if j.get("type", "hinge") == "free"), None)
        geoms = body.findall(".//geom")
        entities.append({
            "instance": name,
            "entity_type": "body",
            "movable": free is not None,
            "free_joint": free.get("name") if free is not None else None,
            "initial_pose": {"position": _floats(body.get("pos")), "quaternion_wxyz": _floats(body.get("quat"))},
            "appearance": [{
                "geom": g.get("name"), "type": g.get("type"), "size": _floats(g.get("size")),
                "rgba": _floats(g.get("rgba")), "material": g.get("material"),
                "mesh": g.get("mesh"),
            } for g in geoms],
        })
    cameras = [{
        "name": c.get("name"), "position": _floats(c.get("pos")),
        "quaternion_wxyz": _floats(c.get("quat")), "fovy": _number(c.get("fovy")),
        "mode": c.get("mode"),
    } for c in root.findall(".//camera")]
    meshes = [{
        "name": m.get("name"), "file": m.get("file"), "scale": _floats(m.get("scale"))
    } for m in root.findall(".//asset/mesh")]
    return {"entities": entities, "materials": materials, "meshes": meshes, "cameras": cameras}


def _floats(text: str | None) -> list[float] | None:
    return [float(x) for x in text.split()] if text else None


def _number(text: str | None) -> float | None:
    return float(text) if text is not None else None


_LIBERO_PATH_ATTR = re.compile(r'(?:file|meshdir|texturedir)="[^"]+"')
_LIBERO_OBJECT_ALIAS = re.compile(r'(?<!new_)salad_dressing')


def repair_libero_object_names(xml_text: str) -> tuple[str, list[dict[str, str]]]:
    """Align hdf5 MuJoCo identifiers with objects created from current BDDL.

    LIVING_ROOM_SCENE4 demos name geoms/meshes ``salad_dressing_*``; the BDDL
    env instantiates ``NewSaladDressing`` and looks up ``new_salad_dressing_*``.
    Asset ``file=`` / ``meshdir=`` / ``texturedir=`` paths are left unchanged.
    """
    replacements: list[dict[str, str]] = []
    saved_paths: list[str] = []

    def stash(match: re.Match[str]) -> str:
        saved_paths.append(match.group(0))
        return f"__LIBERO_PATH_{len(saved_paths) - 1}__"

    protected = _LIBERO_PATH_ATTR.sub(stash, xml_text)
    rewritten = _LIBERO_OBJECT_ALIAS.sub("new_salad_dressing", protected)
    if rewritten != protected:
        replacements.append({"old": "salad_dressing", "new": "new_salad_dressing"})
    for index, original in enumerate(saved_paths):
        rewritten = rewritten.replace(f"__LIBERO_PATH_{index}__", original, 1)
    return rewritten, replacements


def repair_asset_paths(
    xml_text: str, assets_root: Path, robosuite_assets_root: Path | None = None,
) -> tuple[str, list[dict[str, str]]]:
    """Rewrite stale LIBERO and robosuite asset paths from dataset machines."""
    replacements: list[dict[str, str]] = []
    pattern = re.compile(r'(?P<prefix>(?:file|meshdir|texturedir)=")(?P<path>/[^"]+)(?P<suffix>")')
    def replace(match: re.Match[str]) -> str:
        old = match.group("path")
        marker = next((m for m in ("/chiliocosm/assets/", "/libero/assets/") if m in old), None)
        target_root = assets_root
        if "/robosuite/models/assets/" in old and robosuite_assets_root is not None:
            marker = "/robosuite/models/assets/"
            target_root = robosuite_assets_root
        if marker is None:
            return match.group(0)
        relative = old.split(marker, 1)[1]
        new = str(target_root / relative)
        replacements.append({"old": old, "new": new})
        return f'{match.group("prefix")}{new}{match.group("suffix")}'
    xml_text = pattern.sub(replace, xml_text)
    xml_text, name_replacements = repair_libero_object_names(xml_text)
    replacements.extend(name_replacements)
    return xml_text, replacements


def robosuite_assets_root() -> Path | None:
    try:
        import robosuite
    except ImportError:
        return None
    return Path(robosuite.__file__).resolve().parent / "models" / "assets"


def free_joint_qpos_map(xml_text: str) -> dict[str, tuple[int, int]]:
    """Compile XML and return free-joint slices in flattened ``MjSimState``.

    robosuite's flattened state stores simulation time before qpos, hence +1
    relative to ``model.jnt_qposadr``.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_string(xml_text)
    result: dict[str, tuple[int, int]] = {}
    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        start = 1 + int(model.jnt_qposadr[joint_id])
        if name:
            result[name] = (start, start + 7)
    return result


def extract_free_joint_poses(
    flattened_state: np.ndarray, xml_text: str,
    address_map: Mapping[str, tuple[int, int]] | None = None,
) -> dict[str, dict[str, Any]]:
    poses: dict[str, dict[str, Any]] = {}
    for name, (start, end) in (address_map or free_joint_qpos_map(xml_text)).items():
        if end > len(flattened_state):
            raise ValueError(f"State length {len(flattened_state)} is too short for free joint {name} qpos[{start}:{end}]")
        qpos = np.asarray(flattened_state[start:end], dtype=float)
        poses[name] = {
            "position": native(qpos[:3]), "quaternion_wxyz": native(qpos[3:7]),
            "flattened_state_address": [start, end],
            "qpos_address": [start - 1, end - 1], "valid": True,
            "provenance": "sim_state_qpos+compiled_mujoco_address",
        }
    return poses


def action_labels(action: np.ndarray) -> dict[str, Any]:
    action = np.asarray(action, dtype=float)
    if action.shape != (7,):
        raise ValueError(f"Expected LIBERO 7D action, got shape {action.shape}")
    tn, rn = float(np.linalg.norm(action[:3])), float(np.linalg.norm(action[3:6]))
    labels: list[str] = []
    if tn > 0.02:
        labels.append("translate")
    if rn > 0.02:
        labels.append("rotate")
    # Verified against recorded LIBERO gripper qpos: -1=open and +1=close.
    labels.append("open_gripper" if action[6] < -0.2 else "close_gripper" if action[6] > 0.2 else "hold_gripper")
    if tn <= 0.02 and rn <= 0.02:
        labels.append("stationary")
    return {
        "raw_7d": native(action), "translation_norm": tn, "rotation_norm": rn,
        "gripper_command": float(action[6]), "gripper_sign_convention": "-1=open,+1=close",
        "primitive_labels": labels,
    }


def fixed_chunk(array: np.ndarray, start: int, horizon: int) -> tuple[list[Any], list[bool]]:
    width = array.shape[1:] if array.ndim > 1 else ()
    out = np.zeros((horizon,) + width, dtype=array.dtype)
    available = max(0, min(horizon, len(array) - start))
    if available:
        out[:available] = array[start:start + available]
    return native(out), [i < available for i in range(horizon)]


def future_labels(
    actions: np.ndarray, ee_pos: np.ndarray, phases: Sequence[int], events: Mapping[str, int],
    frame_index: int, horizons: Sequence[int] = DEFAULT_FUTURE_HORIZONS,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for horizon in horizons:
        target = frame_index + horizon
        valid = target < len(actions)
        result[str(horizon)] = {
            "valid": valid,
            "action": native(actions[target]) if valid else None,
            "eef_delta": native(ee_pos[target] - ee_pos[frame_index]) if valid else None,
            "phase": int(phases[target]) if valid else None,
            "censored": not valid,
        }
    next_events = {name: idx - frame_index for name, idx in events.items() if idx >= frame_index}
    return {"horizons": result, "time_to_event_steps": next_events, "event_censored": not bool(next_events)}


def infer_phase_sequence(actions: np.ndarray, ee_pos: np.ndarray, gripper_qpos: np.ndarray) -> tuple[list[int], dict[str, int]]:
    """Offline-smoothed spatial pick/place phase estimate with event anchors.

    It is intentionally marked heuristic by callers. Phases are monotone 0..10;
    event anchors derive from gripper command/state and motion, not t/T alone.
    """
    length = len(actions)
    if length == 0:
        return [], {}
    command = actions[:, 6]
    close_candidates = np.flatnonzero(command > 0.2)
    open_candidates = np.flatnonzero(command < -0.2)
    close = int(close_candidates[0]) if len(close_candidates) else max(1, length // 3)
    open_after = open_candidates[open_candidates > close + 2]
    release = int(open_after[0]) if len(open_after) else max(close + 1, 4 * length // 5)
    release = min(release, length - 1)
    speed = np.linalg.norm(np.diff(ee_pos, axis=0, prepend=ee_pos[:1]), axis=1)
    moving = np.flatnonzero(speed > max(0.002, float(np.median(speed) * 1.5)))
    motion_start = int(moving[0]) if len(moving) else 0
    settle_candidates = moving[moving > release]
    settle = min(length - 1, int(settle_candidates[-1] + 1)) if len(settle_candidates) else length - 1
    anchors = {
        "motion_start": motion_start, "grasp_command": close,
        "release_command": release, "settled": settle,
    }
    points = [0, motion_start, max(motion_start, close - 2), close, min(close + 2, length - 1),
              (close + release) // 2, max(close, release - 2), release,
              min(release + 2, length - 1), settle, length - 1]
    points = np.maximum.accumulate(points)
    phases = np.interp(np.arange(length), points, np.arange(11))
    phases = np.maximum.accumulate(np.rint(phases).astype(int)).clip(0, 10)
    return phases.tolist(), anchors


PHASE_DETAILS = (
    "initial", "approach_source", "pregrasp", "grasp", "lift", "transport",
    "approach_goal", "release", "retreat", "settle", "complete",
)


def _resolve_bddl(h5: h5py.File, task: str, suite: str, libero_root: Path) -> Path | None:
    raw = native(h5["data"].attrs.get("bddl_file_name", ""))
    candidates = [Path(raw)] if raw else []
    candidates.extend([
        libero_root / "libero" / "libero" / "bddl_files" / suite / f"{task}.bddl",
        libero_root / "libero" / "libero" / "bddl_files" / Path(str(raw)).name,
    ])
    return next((p for p in candidates if p.exists()), None)


class SimulatorReplay:
    """Replay a recorded trajectory in its exact per-demo MuJoCo model."""

    def __init__(self, bddl_path: Path, repaired_xml: str) -> None:
        from libero.libero.envs.env_wrapper import ControlEnv

        self.wrapper = ControlEnv(
            bddl_file_name=str(bddl_path), use_camera_obs=False,
            has_renderer=False, has_offscreen_renderer=False, hard_reset=False,
        )
        self.wrapper.reset_from_xml_string(repaired_xml)
        self.wrapper.sim.reset()
        self.env = self.wrapper.env
        self.sim = self.wrapper.sim
        self.goal_states = list(self.env.parsed_problem.get("goal_state", []))
        self.entity_body_ids = {str(k): int(v) for k, v in self.env.obj_body_id.items()}
        self.body_to_entity = self._body_to_entity_map()

    def close(self) -> None:
        self.wrapper.close()

    def _body_to_entity_map(self) -> dict[int, str]:
        result: dict[int, str] = {}
        entity_ids = {body_id: name for name, body_id in self.entity_body_ids.items()}
        for body_id in range(self.sim.model.nbody):
            cursor = body_id
            while cursor > 0:
                if cursor in entity_ids:
                    result[body_id] = entity_ids[cursor]
                    break
                cursor = int(self.sim.model.body_parentid[cursor])
            name = (self.sim.model.body_id2name(body_id) or "").lower()
            if body_id not in result and any(x in name for x in ("gripper", "finger", "hand")):
                result[body_id] = "robot_gripper"
        return result

    def replay(
        self, states: np.ndarray, actions: np.ndarray, selected: Sequence[int],
        ee_pos: np.ndarray,
    ) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
        selected_set = set(selected)
        details: dict[int, dict[str, Any]] = {}
        source = self.goal_states[0][1] if self.goal_states and len(self.goal_states[0]) >= 2 else None
        source_z: list[float] = []
        source_contact: list[bool] = []
        held: list[bool] = []
        successes: list[bool] = []
        for idx, state in enumerate(states):
            self.sim.set_state_from_flattened(np.asarray(state, dtype=float))
            self.sim.forward()
            predicates = [bool(self.env._eval_predicate(goal)) for goal in self.goal_states]
            success = bool(all(predicates)) if predicates else bool(self.wrapper.check_success())
            contacts = self._contacts()
            touched = sorted({
                side for contact in contacts for side in contact["entities"]
                if side not in {"robot_gripper", "unmapped"} and "robot_gripper" in contact["entities"]
            })
            contact_source = bool(source and source in touched)
            held_now = contact_source and bool(actions[idx, 6] > 0.2)
            source_pose = self._body_pose(source)
            source_z.append(float(source_pose["position"][2]) if source_pose else float("nan"))
            source_contact.append(contact_source)
            held.append(held_now)
            successes.append(success)
            if idx in selected_set:
                body_poses = {
                    name: pose for name in self.entity_body_ids
                    if (pose := self._body_pose(name)) is not None
                }
                details[idx] = {
                    "body_poses": body_poses, "contacts": contacts,
                    "contact_graph": _contact_graph(contacts),
                    "gripper_contact_entities": touched,
                    "held_object": source if held_now else None,
                    "held_valid": True,
                    "held_provenance": "heuristic_true_contact+recorded_close_command",
                    "predicate_status": [
                        {"predicate": native(goal), "satisfied": value}
                        for goal, value in zip(self.goal_states, predicates)
                    ],
                    "satisfied_fraction": sum(predicates) / len(predicates) if predicates else None,
                    "all_success": success,
                    "eef_object": _eef_offsets(body_poses, ee_pos[idx]),
                    "goal_relevant_pairs": _goal_pairs(self.goal_states, body_poses),
                }
        phases, anchors, evidence = infer_sim_phase_sequence(
            actions, np.asarray(source_z), source_contact, held, successes,
        )
        return details, {"phases": phases, "anchors": anchors, "phase_evidence": evidence}

    def _body_pose(self, name: str | None) -> dict[str, Any] | None:
        if name is None or name not in self.entity_body_ids:
            return None
        body_id = self.entity_body_ids[name]
        return {
            "position": native(self.sim.data.body_xpos[body_id]),
            "quaternion_wxyz": native(self.sim.data.body_xquat[body_id]),
            "body_id": body_id, "valid": True, "provenance": "sim_replay_body_pose",
        }

    def _contacts(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for index in range(int(self.sim.data.ncon)):
            contact = self.sim.data.contact[index]
            geom_ids = (int(contact.geom1), int(contact.geom2))
            body_ids = tuple(int(self.sim.model.geom_bodyid[g]) for g in geom_ids)
            entities = [self.body_to_entity.get(b, "unmapped") for b in body_ids]
            # Robot self-contact, floor contact, and contacts internal to one
            # entity are not useful annotation edges. Keeping only mapped,
            # inter-entity contacts makes the JSONL auditable without copying
            # dozens of low-level solver contacts into every frame.
            if "unmapped" in entities or entities[0] == entities[1]:
                continue
            result.append({
                "geom_ids": list(geom_ids),
                "geom_names": [self.sim.model.geom_id2name(g) or f"geom_{g}" for g in geom_ids],
                "body_ids": list(body_ids),
                "body_names": [self.sim.model.body_id2name(b) or f"body_{b}" for b in body_ids],
                "entities": entities,
                "distance": float(contact.dist), "provenance": "mujoco_contact",
            })
        return result


def infer_sim_phase_sequence(
    actions: np.ndarray, source_z: np.ndarray, source_contact: Sequence[bool],
    held: Sequence[bool], successes: Sequence[bool],
) -> tuple[list[int], dict[str, int], dict[str, Any]]:
    """Infer phases from replay contact, held, lift, release and success events."""
    length = len(actions)
    finite_z = source_z[np.isfinite(source_z)]
    initial_z = float(np.median(finite_z[: min(5, len(finite_z))])) if len(finite_z) else float("nan")
    grasp_ix = np.flatnonzero(np.asarray(source_contact) & (actions[:, 6] > 0.2))
    held_ix = np.flatnonzero(held)
    lift_ix = np.flatnonzero(source_z > initial_z + 0.03) if np.isfinite(initial_z) else np.array([], dtype=int)
    grasp = int(grasp_ix[0]) if len(grasp_ix) else None
    held_at = int(held_ix[0]) if len(held_ix) else grasp
    lift_after = lift_ix[lift_ix >= (held_at or 0)]
    lift = int(lift_after[0]) if len(lift_after) else held_at
    open_ix = np.flatnonzero(actions[:, 6] < -0.2)
    open_after = open_ix[open_ix > (grasp if grasp is not None else length // 2)]
    release = int(open_after[0]) if len(open_after) else None
    success_ix = np.flatnonzero(successes)
    success = int(success_ix[0]) if len(success_ix) else None
    anchors = {
        name: value for name, value in {
            "grasp_contact_closed": grasp, "held": held_at, "lift": lift,
            "release_open_command": release, "success": success,
        }.items() if value is not None
    }
    if grasp is None or release is None:
        phases, fallback = infer_phase_sequence(actions, np.zeros((length, 3)), np.zeros((length, 2)))
        return phases, {**fallback, **anchors}, {
            "source": "sim_replay_partial+action_fallback", "evidence_count": len(anchors),
            "missing": [x for x in ("grasp_contact_closed", "release_open_command") if x not in anchors],
        }
    points = [
        0, max(0, grasp - 3), max(0, grasp - 1), grasp,
        lift if lift is not None else min(grasp + 2, length - 1),
        (grasp + release) // 2, max(grasp, release - 2), release,
        min(release + 2, length - 1), success if success is not None else length - 1, length - 1,
    ]
    points = np.maximum.accumulate(np.clip(points, 0, length - 1))
    phases = np.maximum.accumulate(np.rint(np.interp(np.arange(length), points, np.arange(11))).astype(int))
    if success is not None:
        phases[success:] = 10
    return phases.clip(0, 10).tolist(), anchors, {
        "source": "sim_replay_events", "evidence_count": len(anchors), "initial_source_z": initial_z,
    }


def _contact_graph(contacts: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    pairs = {tuple(sorted(map(str, contact["entities"]))) for contact in contacts}
    return [{"entity_a": a, "entity_b": b, "provenance": "mujoco_contact"} for a, b in sorted(pairs)]


def _eef_offsets(body_poses: Mapping[str, Mapping[str, Any]], eef: np.ndarray) -> list[dict[str, Any]]:
    result = []
    for name, pose in body_poses.items():
        delta = np.asarray(pose["position"], dtype=float) - eef
        result.append({
            "object": name, "offset": native(delta), "distance": float(np.linalg.norm(delta)),
            "valid": True, "provenance": "sim_replay_body_pose+hdf5_eef",
        })
    return result


def _goal_pairs(
    goals: Sequence[Sequence[str]], body_poses: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for goal in goals:
        if len(goal) != 3 or goal[1] not in body_poses or goal[2] not in body_poses:
            continue
        delta = np.asarray(body_poses[goal[2]]["position"]) - np.asarray(body_poses[goal[1]]["position"])
        result.append({
            "predicate": goal[0], "entity_a": goal[1], "entity_b": goal[2],
            "a_to_b_offset": native(delta), "distance": float(np.linalg.norm(delta)),
            "valid": True, "provenance": "sim_replay_body_pose",
        })
    return result


class RichAnnotationExtractor:
    def __init__(
        self, output_dir: Path, frames_per_demo: int = 16, simulator: str = "auto",
        libero_root: Path = LOCAL_LIBERO_ROOT, action_chunk: int = DEFAULT_ACTION_CHUNK,
        future_horizons: Sequence[int] = DEFAULT_FUTURE_HORIZONS,
    ) -> None:
        if simulator not in {"auto", "required", "off"}:
            raise ValueError("simulator must be auto, required, or off")
        self.output_dir = output_dir.resolve()
        self.frames_per_demo = frames_per_demo
        self.simulator = simulator
        self.libero_root = libero_root.resolve()
        self.action_chunk = action_chunk
        self.future_horizons = tuple(future_horizons)
        self.warnings: list[str] = []
        self.sim_failures: list[dict[str, str]] = []

    def run(self, selected: Sequence[DemoRef], seed: int, inventory_count: int) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_root = self.output_dir / "images"
        image_root.mkdir(exist_ok=True)
        episodes_path = self.output_dir / "episodes.jsonl"
        frames_path = self.output_dir / "frames.jsonl"
        manifest: list[dict[str, Any]] = []
        frame_count = 0
        with episodes_path.open("w", encoding="utf-8") as episodes_file, frames_path.open("w", encoding="utf-8") as frames_file:
            for ref in selected:
                episode, rows, selection = self._extract_demo(ref, image_root)
                episodes_file.write(json.dumps(native(episode), ensure_ascii=False) + "\n")
                for row in rows:
                    frames_file.write(json.dumps(native(row), ensure_ascii=False) + "\n")
                frame_count += len(rows)
                manifest.append(selection)
        manifest_doc = {
            "annotation_version": ANNOTATION_VERSION, "seed": seed,
            "strategy": "seeded balanced task round-robin; frame quantiles plus event candidates",
            "inventory_demo_count": inventory_count, "selected_demo_count": len(selected),
            "episodes": manifest,
        }
        _write_json(self.output_dir / "selection_manifest.json", manifest_doc)
        summary = {
            "annotation_version": ANNOTATION_VERSION,
            "selected_demos": len(selected), "exported_frames": frame_count,
            "suites": sorted({x.suite for x in selected}),
            "tasks": sorted({x.task for x in selected}),
            "simulator": {
                "requested": self.simulator,
                "status": "off" if self.simulator == "off" else ("degraded" if self.sim_failures else "available"),
                "failures": self.sim_failures,
                "note": "No proximity value is reported as contact.",
            },
            "warnings": sorted(set(self.warnings)),
            "outputs": ["episodes.jsonl", "frames.jsonl", "selection_manifest.json", "images/"],
        }
        _write_json(self.output_dir / "summary.json", summary)
        return summary

    def _extract_demo(self, ref: DemoRef, image_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        with h5py.File(ref.hdf5_path, "r") as handle:
            data = handle["data"]
            demo = data[ref.demo_key]
            obs = demo["obs"]
            actions = np.asarray(demo["actions"], dtype=float)
            ee_pos = np.asarray(obs["ee_pos"], dtype=float)
            ee_ori = np.asarray(obs["ee_ori"], dtype=float)
            joint = np.asarray(obs["joint_states"], dtype=float)
            gripper = np.asarray(obs["gripper_states"], dtype=float)
            phases, anchors = infer_phase_sequence(actions, ee_pos, gripper)
            phase_evidence: dict[str, Any] = {
                "source": "recorded_action+eef_fallback", "evidence_count": len(anchors),
            }
            indices, reasons = select_frame_indices(actions, ee_pos, self.frames_per_demo)
            bddl_path = _resolve_bddl(handle, ref.task, ref.suite, self.libero_root)
            bddl = parse_bddl(bddl_path)
            xml_text = native(demo.attrs.get("model_file", ""))
            assets_root = self.libero_root / "libero" / "libero" / "assets"
            repaired_xml, replacements = repair_asset_paths(xml_text, assets_root, robosuite_assets_root())
            xml_info = parse_model_xml(repaired_xml)
            states = np.asarray(demo["states"], dtype=float)
            qpos_map = free_joint_qpos_map(repaired_xml)
            direct_free_poses = [
                extract_free_joint_poses(states[idx], repaired_xml, qpos_map) for idx in indices
            ]
            sim_details: dict[int, dict[str, Any]] = {}
            sim_valid = False
            sim_reason = "simulator_off"
            replay: SimulatorReplay | None = None
            if self.simulator != "off":
                try:
                    if bddl_path is None:
                        raise FileNotFoundError(f"Cannot replay without BDDL for {ref.episode_id}")
                    replay = SimulatorReplay(bddl_path, repaired_xml)
                    sim_details, timeline = replay.replay(states, actions, indices, ee_pos)
                    phases = timeline["phases"]
                    anchors = timeline["anchors"]
                    phase_evidence = timeline["phase_evidence"]
                    sim_valid = True
                    sim_reason = "sim_replay"
                except Exception as exc:
                    sim_reason = f"simulator_replay_failed: {type(exc).__name__}: {exc}"
                    self.sim_failures.append({"episode_id": ref.episode_id, "error": sim_reason})
                    if self.simulator == "required":
                        raise RuntimeError(f"{sim_reason} ({ref.episode_id})") from exc
                    self.warnings.append("Simulator replay failed for at least one demo; fields were degraded")
                finally:
                    if replay is not None:
                        replay.close()
            entities = self._merge_entities(xml_info, bddl)
            velocities = _finite_difference(ee_pos)
            joint_vel = _finite_difference(joint)
            gripper_width = np.sum(np.abs(gripper), axis=1)
            rows: list[dict[str, Any]] = []
            episode_dir = image_root / _safe(ref.suite) / _safe(ref.task) / _safe(ref.demo_key)
            episode_dir.mkdir(parents=True, exist_ok=True)
            for selected_offset, idx in enumerate(indices):
                image_paths = {}
                for source, public_name in (("agentview_rgb", "agentview"), ("eye_in_hand_rgb", "wrist")):
                    target = episode_dir / f"{idx:05d}_{public_name}.png"
                    Image.fromarray(np.asarray(obs[source][idx])).save(target)
                    image_paths[public_name] = {
                        "path": str(target.relative_to(self.output_dir)),
                        "raw_dataset": f"obs/{source}", "raw_frame_index": idx,
                        "transform": {"operation": "none", "shape_hwc": list(obs[source].shape[1:]),
                                      "color_space": "RGB", "dtype": "uint8"},
                    }
                phase = phases[idx]
                sim_frame = sim_details.get(idx)
                frame_entities = []
                for entity in entities:
                    item = dict(entity)
                    joint_pose = (
                        direct_free_poses[selected_offset].get(entity["free_joint"])
                        if entity.get("free_joint") else None
                    )
                    item["free_joint_qpos_pose"] = joint_pose or field(None, False, "not_a_free_joint_body")
                    pose = None
                    if sim_frame:
                        matched = next((
                            name for name in sim_frame["body_poses"]
                            if entity["instance"] == name or entity["instance"].startswith(name + "_")
                        ), None)
                        pose = sim_frame["body_poses"].get(matched) if matched else None
                    item["current_pose"] = pose or joint_pose or field(None, False, sim_reason)
                    frame_entities.append(item)
                chunk, chunk_mask = fixed_chunk(actions, idx, self.action_chunk)
                rows.append({
                    "annotation_version": ANNOTATION_VERSION,
                    "sample": {
                        "episode_id": ref.episode_id, "suite": ref.suite, "task": ref.task,
                        "hdf5": str(ref.hdf5_path), "demo": ref.demo_key, "frame": idx,
                        "T": ref.length, "time_step": idx, "progress": idx / max(1, ref.length - 1),
                        "selection_reasons": reasons[idx],
                    },
                    "images": image_paths,
                    "objects": frame_entities,
                    "spatial": ({
                        "eef_object": sim_frame["eef_object"],
                        "goal_relevant_pairs": field(sim_frame["goal_relevant_pairs"], True, "sim_replay_body_pose"),
                        "body_poses": field(sim_frame["body_poses"], True, "sim_replay_body_pose"),
                        "contacts": field(sim_frame["contacts"], True, "mujoco_contact"),
                        "contact_graph": field(sim_frame["contact_graph"], True, "mujoco_contact"),
                        "predicates": field(sim_frame["predicate_status"], True, "libero_eval_predicate"),
                    } if sim_frame else self._spatial(frame_entities, ee_pos[idx], sim_reason)),
                    "goal": {
                        "ast": bddl.get("goal_ast", []), "predicates": bddl.get("goal_predicates", []),
                        "predicate_status": field(
                            sim_frame["predicate_status"] if sim_frame else None,
                            sim_frame is not None, "libero_eval_predicate" if sim_frame else sim_reason,
                        ),
                        "satisfied_fraction": field(
                            sim_frame["satisfied_fraction"] if sim_frame else None,
                            sim_frame is not None, "libero_eval_predicate" if sim_frame else sim_reason,
                        ),
                        "all_success": field(
                            sim_frame["all_success"] if sim_frame else None,
                            sim_frame is not None, "libero_eval_predicate" if sim_frame else sim_reason,
                        ),
                    },
                    "action": {
                        **action_labels(actions[idx]),
                        "chunk": chunk, "chunk_mask": chunk_mask, "chunk_horizon": self.action_chunk,
                        "provenance": "hdf5_recorded",
                    },
                    "phase": {
                        "index": phase, "family": "spatial_pick_place", "detail": PHASE_DETAILS[phase],
                        "provenance": phase_evidence["source"],
                        "confidence": None,
                        "evidence_count": phase_evidence["evidence_count"],
                        "evidence_quality": phase_evidence,
                        "anchor": [name for name, value in anchors.items() if value == idx],
                    },
                    "body": {
                        "eef_position": field(ee_pos[idx], True, "hdf5_recorded"),
                        "eef_orientation_axis_angle": field(ee_ori[idx], True, "hdf5_recorded"),
                        "joint_position": field(joint[idx], True, "hdf5_recorded"),
                        "gripper_qpos": field(gripper[idx], True, "hdf5_recorded"),
                        "eef_velocity_per_step": field(velocities[idx], True, "finite_difference"),
                        "joint_velocity_per_step": field(joint_vel[idx], True, "finite_difference"),
                        "gripper_width_proxy": field(gripper_width[idx], True, "sum_abs_recorded_qpos",
                                                       warning="not metric finger separation"),
                        "gripper_state": field(_gripper_state(gripper_width, idx), True, "offline_heuristic"),
                        "motion_state": field(_motion_state(velocities[idx]), True, "finite_difference_heuristic"),
                        "contact": field(
                            sim_frame["contacts"] if sim_frame else None,
                            sim_frame is not None, "mujoco_contact" if sim_frame else sim_reason,
                        ),
                        "held_object": field(
                            sim_frame["held_object"] if sim_frame else None,
                            sim_frame is not None,
                            sim_frame["held_provenance"] if sim_frame else sim_reason,
                        ),
                    },
                    "future": future_labels(actions, ee_pos, phases, anchors, idx, self.future_horizons),
                    "quality": {
                        "direct_observation_valid": True, "simulator_valid": sim_frame is not None,
                        "provenance": ["hdf5_recorded", "bddl", "model_xml", "derived", sim_reason],
                        "warnings": (
                            ["held remains a contact+command heuristic; phase is event-derived"]
                            if sim_frame else ["phase/gripper state are heuristics", "contact/predicates unavailable"]
                        ),
                    },
                })
            episode = {
                "annotation_version": ANNOTATION_VERSION, "episode_id": ref.episode_id,
                "suite": ref.suite, "task": ref.task, "hdf5": str(ref.hdf5_path),
                "demo": ref.demo_key, "T": ref.length, "instruction": bddl.get("language"),
                "bddl": bddl, "entities": entities, "camera_metadata": xml_info["cameras"],
                "asset_metadata": {"meshes": xml_info["meshes"], "materials": xml_info["materials"],
                                   "path_replacements": replacements},
                "phase_anchors": anchors, "phase_evidence": phase_evidence,
                "simulator": {"valid": sim_valid, "provenance": sim_reason},
            }
            selection = {
                "episode_id": ref.episode_id, "T": ref.length, "frames": indices,
                "frame_reasons": {str(i): reasons[i] for i in indices},
            }
            return episode, rows, selection

    @staticmethod
    def _merge_entities(xml_info: Mapping[str, Any], bddl: Mapping[str, Any]) -> list[dict[str, Any]]:
        declarations = {x["instance"]: (x["category"], "movable") for x in bddl.get("objects", [])}
        declarations.update({x["instance"]: (x["category"], "fixture") for x in bddl.get("fixtures", [])})
        interests = set(bddl.get("objects_of_interest", []))
        goal_args = {arg for p in bddl.get("goal_predicates", []) for arg in p["arguments"]}
        xml_entities = {entity["instance"]: entity for entity in xml_info["entities"]}
        result = []
        for instance, (category, declared_type) in declarations.items():
            candidates = [
                name for name in xml_entities
                if name == instance or name.startswith(instance + "_")
            ]
            if not candidates:
                continue
            # LIBERO object root bodies conventionally use `<instance>_main`.
            # Select one canonical body per BDDL entity instead of copying every
            # nested mesh/part body into every frame.
            name = min(
                candidates,
                key=lambda candidate: (
                    candidate != instance,
                    candidate != f"{instance}_main",
                    len(candidate),
                    candidate,
                ),
            )
            entity = dict(xml_entities[name])
            appearance = entity.pop("appearance")
            result.append({
                **entity, "bddl_instance": instance, "category": category,
                "declared_entity_type": declared_type,
                "goal_roles": {
                    "object_of_interest": instance in interests, "goal_argument": instance in goal_args,
                },
                "affordance_candidates": _affordances(category, entity["movable"]),
                "appearance": {
                    "geoms": appearance, "category": category,
                    "color_semantics": field(None, False, "not_in_source"),
                    "shape_semantics": field(None, False, "not_in_source"),
                },
                "provenance": "model_xml+bddl",
            })
        return result

    @staticmethod
    def _spatial(entities: Sequence[Mapping[str, Any]], eef: np.ndarray, sim_reason: str) -> dict[str, Any]:
        offsets = []
        for entity in entities:
            pose = entity.get("initial_pose", {})
            pos = pose.get("position") if pose else None
            if entity.get("movable") and pos and len(pos) == 3:
                delta = np.asarray(pos) - eef
                offsets.append({
                    "object": entity["instance"], "offset": native(delta),
                    "distance": float(np.linalg.norm(delta)),
                    "valid": False, "provenance": "xml_initial_pose_only",
                    "warning": "not a per-frame body pose",
                })
        return {
            "eef_object": offsets,
            "goal_relevant_pairs": field(None, False, sim_reason),
            "body_poses": field(None, False, sim_reason),
            "contacts": field(None, False, sim_reason),
            "predicates": field(None, False, sim_reason),
        }


def _finite_difference(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values)
    return np.gradient(values, axis=0)


def _gripper_state(widths: np.ndarray, idx: int) -> str:
    lo, hi = np.quantile(widths, [0.2, 0.8])
    return "closed" if widths[idx] <= lo else "open" if widths[idx] >= hi else "partial"


def _motion_state(velocity: np.ndarray) -> str:
    speed = float(np.linalg.norm(velocity))
    return "stationary" if speed < 0.002 else "moving_slow" if speed < 0.02 else "moving"


def _affordances(category: str | None, movable: bool) -> list[str]:
    result = ["grasp", "move"] if movable else ["support"]
    text = (category or "").lower()
    if any(x in text for x in ("bowl", "plate", "cup", "container")):
        result.append("contain_or_support_candidate")
    if "drawer" in text or "cabinet" in text:
        result.append("open_close_candidate")
    return result


def _safe(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(native(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def extract_dataset(
    input_path: Path, output_dir: Path, num_demos: int = 100, frames_per_demo: int = 16,
    seed: int = 42, suite_filters: Sequence[str] = (), simulator: str = "auto",
    libero_root: Path = LOCAL_LIBERO_ROOT,
) -> dict[str, Any]:
    files = discover_hdf5(input_path, suite_filters)
    inventory = inventory_demos(files)
    selected = balanced_demo_selection(inventory, num_demos, seed)
    extractor = RichAnnotationExtractor(output_dir, frames_per_demo, simulator, libero_root)
    return extractor.run(selected, seed, len(inventory))
