"""Per-frame scalar statistics for PI0.5 ablation frames (no self-occlusion)."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .libero_rich_annotations import (
    LOCAL_LIBERO_ROOT,
    SimulatorReplay,
    _finite_difference,
    _resolve_bddl,
    infer_phase_sequence,
    native,
    parse_bddl,
    repair_asset_paths,
    robosuite_assets_root,
)
from .utils import ensure_dir, log, write_json


DEFAULT_DELTA_WINDOW = 8
DISTMAX_M = 2.0

# Franka Panda arm limits used when the compiled model is unavailable.
PANDA_JOINT_LIMITS = np.array(
    [
        [-2.8973, 2.8973],
        [-1.7628, 1.7628],
        [-2.8973, 2.8973],
        [-3.0718, -0.0698],
        [-2.8973, 2.8973],
        [-0.0175, 3.7525],
        [-2.8973, 2.8973],
    ],
    dtype=np.float64,
)

TABLE_TOKENS = ("table", "floor", "ground", "mount0_")
ROBOT_PREFIXES = ("robot0_", "gripper0_")

GRIPPER_CLOSED, GRIPPER_PARTIAL, GRIPPER_OPEN = 0, 1, 2

STAT_COLUMNS = [
    "row_index",
    "sample_id",
    "suite",
    "task",
    "demo_key",
    "libero_frame_index",
    "episode_id",
    "hdf5_path",
    "episode_T",
    "progress",
    "phase_index",
    "phase_source",
    "motion_joint_l2",
    "motion_ee_pos_l2",
    "motion_ee_ori_rad",
    "motion_gripper_l2",
    "motion_valid",
    "vel_joint_l2",
    "vel_ee_pos_l2",
    "vel_ee_ori_rad",
    "vel_gripper_l2",
    "k_limit",
    "k_change",
    "k_change_valid",
    "gripper_width",
    "gripper_state_code",
    "gripper_command",
    "d_ee_target",
    "d_ee_nearest_obj",
    "d_target_nearest_link",
    "d_body_env",
    "contact_nontable",
    "target_held",
    "target_name",
    "sim_valid",
    "sim_error",
]


def load_sampled_frame_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {"row_index", "demo_key", "libero_frame_index", "hdf5_path"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"sampled_frames.csv missing columns: {sorted(missing)}")
    return table.sort_values("row_index").reset_index(drop=True)


def group_frames_by_demo(sampled: pd.DataFrame) -> dict[tuple[str, str], list[tuple[int, int]]]:
    grouped: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for item in sampled.itertuples(index=False):
        grouped[(str(item.hdf5_path), str(item.demo_key))].append(
            (int(item.row_index), int(item.libero_frame_index))
        )
    return grouped


def consecutive_l2(values: np.ndarray, index: int) -> tuple[float, bool]:
    """||x[t+1]-x[t]||_2. Invalid on the last frame."""
    values = np.asarray(values, dtype=np.float64)
    if index < 0 or index >= len(values) - 1:
        return float("nan"), False
    delta = values[index + 1] - values[index]
    return float(np.linalg.norm(delta.reshape(-1))), True


def axis_angle_geodesic(a: np.ndarray, b: np.ndarray) -> float:
    """Rotation-vector geodesic distance in radians."""
    from scipy.spatial.transform import Rotation

    ra = Rotation.from_rotvec(np.asarray(a, dtype=np.float64).reshape(3))
    rb = Rotation.from_rotvec(np.asarray(b, dtype=np.float64).reshape(3))
    return float((ra.inv() * rb).magnitude())


def consecutive_ori(values: np.ndarray, index: int) -> tuple[float, bool]:
    values = np.asarray(values, dtype=np.float64)
    if index < 0 or index >= len(values) - 1:
        return float("nan"), False
    return axis_angle_geodesic(values[index], values[index + 1]), True


def joint_limit_proximity(q: np.ndarray, limits: np.ndarray = PANDA_JOINT_LIMITS) -> float:
    """K_limit in [0, 1]: 0 at mid-range, 1 at a joint limit."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    limits = np.asarray(limits, dtype=np.float64)
    n = min(len(q), len(limits))
    q = q[:n]
    lo = limits[:n, 0]
    hi = limits[:n, 1]
    half = np.maximum((hi - lo) / 2.0, 1e-8)
    dist = np.minimum(q - lo, hi - q)
    dist = np.maximum(dist, 0.0)
    return float(np.max(1.0 - dist / half))


def configuration_change(q: np.ndarray, index: int, delta: int) -> tuple[float, bool]:
    q = np.asarray(q, dtype=np.float64)
    target = index + int(delta)
    if index < 0 or target >= len(q) or delta <= 0:
        return float("nan"), False
    return float(np.linalg.norm((q[target] - q[index]).reshape(-1))), True


def gripper_width(gripper_qpos: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(np.asarray(gripper_qpos, dtype=np.float64)), axis=-1)


def gripper_state_code(widths: np.ndarray, index: int) -> int:
    lo, hi = np.quantile(widths, [0.2, 0.8])
    width = float(widths[index])
    if width <= lo:
        return GRIPPER_CLOSED
    if width >= hi:
        return GRIPPER_OPEN
    return GRIPPER_PARTIAL


def is_table_name(name: str) -> bool:
    lowered = (name or "").lower()
    return any(token in lowered for token in TABLE_TOKENS)


def is_robot_name(name: str) -> bool:
    text = name or ""
    return text.startswith(ROBOT_PREFIXES) and "mount" not in text.lower()


def contact_is_nontable_robot_object(
    body_names: Sequence[str],
    entities: Sequence[str] | None = None,
) -> bool:
    names = [str(x) for x in body_names]
    ents = [str(x) for x in (entities if entities is not None else names)]
    robot = [is_robot_name(n) or e == "robot_gripper" for n, e in zip(names, ents)]
    table = [is_table_name(n) or is_table_name(e) for n, e in zip(names, ents)]
    return any(robot) and any((not r) and (not t) for r, t in zip(robot, table))


def panda_limits_from_sim(sim: Any) -> np.ndarray:
    limits = []
    for joint_id in range(int(sim.model.njnt)):
        name = _model_id2name(sim.model, "joint", joint_id)
        if not str(name).startswith("robot0_joint"):
            continue
        lo, hi = np.asarray(sim.model.jnt_range[joint_id], dtype=np.float64)
        if hi - lo < 1e-6:
            continue
        limits.append([float(lo), float(hi)])
    if len(limits) >= 7:
        return np.asarray(limits[:7], dtype=np.float64)
    return PANDA_JOINT_LIMITS.copy()


def hdf5_scalars_for_index(
    *,
    index: int,
    joint: np.ndarray,
    ee_pos: np.ndarray,
    ee_ori: np.ndarray,
    gripper: np.ndarray,
    actions: np.ndarray,
    phases: Sequence[int],
    delta_window: int,
    limits: np.ndarray,
    phase_source: str = "hdf5",
) -> dict[str, Any]:
    length = len(joint)
    widths = gripper_width(gripper)
    motion_q, motion_ok = consecutive_l2(joint, index)
    motion_ee, _ = consecutive_l2(ee_pos, index)
    motion_ori, _ = consecutive_ori(ee_ori, index)
    motion_grip, _ = consecutive_l2(gripper, index)
    k_change, k_ok = configuration_change(joint, index, delta_window)
    vel_q = _row_l2(_finite_difference(joint), index)
    vel_ee = _row_l2(_finite_difference(ee_pos), index)
    vel_ori = _ori_speed(ee_ori, index)
    vel_grip = _row_l2(_finite_difference(gripper), index)
    progress = float(index / max(length - 1, 1))
    return {
        "episode_T": int(length),
        "progress": progress,
        "phase_index": int(phases[index]) if index < len(phases) else -1,
        "phase_source": phase_source,
        "motion_joint_l2": motion_q,
        "motion_ee_pos_l2": motion_ee,
        "motion_ee_ori_rad": motion_ori,
        "motion_gripper_l2": motion_grip,
        "motion_valid": int(motion_ok),
        "vel_joint_l2": vel_q,
        "vel_ee_pos_l2": vel_ee,
        "vel_ee_ori_rad": vel_ori,
        "vel_gripper_l2": vel_grip,
        "k_limit": joint_limit_proximity(joint[index], limits),
        "k_change": k_change,
        "k_change_valid": int(k_ok),
        "gripper_width": float(widths[index]),
        "gripper_state_code": gripper_state_code(widths, index),
        "gripper_command": float(actions[index, 6]) if actions.shape[-1] > 6 else float("nan"),
    }


def resolve_target_name(
    bddl: dict[str, Any],
    entity_names: Iterable[str],
    goal_states: Sequence[Any] | None = None,
) -> str | None:
    names = [str(x) for x in entity_names]
    candidates: list[str] = []
    if goal_states:
        first = goal_states[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            candidates.append(str(first[1]))
    candidates.extend(str(x) for x in (bddl.get("objects_of_interest") or []) if x)
    for pred in bddl.get("goal_predicates") or []:
        candidates.extend(str(a) for a in (pred.get("arguments") or []))
    candidates.extend(str(x.get("instance")) for x in bddl.get("objects") or [] if x.get("instance"))
    for candidate in candidates:
        if candidate in names:
            return candidate
        matched = next((n for n in names if n == candidate or n.startswith(f"{candidate}_")), None)
        if matched:
            return matched
    movable = [n for n in names if not is_table_name(n)]
    return movable[0] if movable else (names[0] if names else None)


def sim_proximity_and_contact(
    replay: SimulatorReplay,
    *,
    ee_pos: np.ndarray,
    gripper_command: float,
    target_name: str | None,
    distmax: float = DISTMAX_M,
) -> dict[str, Any]:
    sim = replay.sim
    entity_names = list(replay.entity_body_ids)
    object_names = [name for name in entity_names if not is_table_name(name)]
    target = target_name if target_name in replay.entity_body_ids else resolve_target_name(
        {},
        entity_names,
        getattr(replay, "goal_states", None),
    )
    ee = np.asarray(ee_pos, dtype=np.float64).reshape(3)
    d_ee_target = float("nan")
    d_ee_nearest = float("nan")
    d_target_link = float("nan")
    if object_names:
        distances = []
        for name in object_names:
            pos = np.asarray(sim.data.body_xpos[replay.entity_body_ids[name]], dtype=np.float64)
            distances.append(float(np.linalg.norm(pos - ee)))
        d_ee_nearest = min(distances)
    if target is not None and target in replay.entity_body_ids:
        tpos = np.asarray(sim.data.body_xpos[replay.entity_body_ids[target]], dtype=np.float64)
        d_ee_target = float(np.linalg.norm(tpos - ee))
        link_dists = []
        for body_id in range(int(sim.model.nbody)):
            name = _model_id2name(sim.model, "body", body_id)
            if not is_robot_name(name):
                continue
            link_dists.append(float(np.linalg.norm(np.asarray(sim.data.body_xpos[body_id]) - tpos)))
        if link_dists:
            d_target_link = min(link_dists)
    d_body_env = _min_robot_object_geom_distance(sim, replay, distmax)
    contact_nontable, gripper_hits = _contact_flags(sim, replay)
    target_held = int(bool(target) and target in gripper_hits and gripper_command > 0.2)
    return {
        "d_ee_target": d_ee_target,
        "d_ee_nearest_obj": d_ee_nearest,
        "d_target_nearest_link": d_target_link,
        "d_body_env": d_body_env,
        "contact_nontable": float(contact_nontable),
        "target_held": float(target_held),
        "target_name": target or "",
        "sim_valid": 1,
    }


def extract_demo_stats(
    hdf5_path: Path,
    demo_key: str,
    frame_indices: Sequence[int],
    *,
    suite: str = "libero_spatial",
    task: str = "",
    delta_window: int = DEFAULT_DELTA_WINDOW,
    simulator: str = "auto",
    replay_mode: str = "selected",
    libero_root: Path = LOCAL_LIBERO_ROOT,
) -> list[dict[str, Any]]:
    import h5py

    if simulator not in {"auto", "required", "off"}:
        raise ValueError("simulator must be auto, required, or off")
    if replay_mode not in {"selected", "full"}:
        raise ValueError("replay_mode must be selected or full")

    task = task or hdf5_path.name.removesuffix("_demo.hdf5")
    with h5py.File(hdf5_path, "r") as handle:
        demo = handle["data"][demo_key]
        obs = demo["obs"]
        actions = np.asarray(demo["actions"], dtype=np.float64)
        ee_pos = np.asarray(obs["ee_pos"], dtype=np.float64)
        ee_ori = np.asarray(obs["ee_ori"], dtype=np.float64)
        joint = _arm_joints(obs["joint_states"])
        gripper = np.asarray(obs["gripper_states"], dtype=np.float64)
        xml_text = native(demo.attrs.get("model_file", ""))
        bddl_path = _resolve_bddl(handle, task, suite, Path(libero_root))
        if bddl_path is not None and not bddl_path.is_file():
            bddl_path = None
        states = None
        if simulator != "off":
            states = np.asarray(demo["states"], dtype=np.float64)
    bddl = parse_bddl(bddl_path)
    phases, _ = infer_phase_sequence(actions, ee_pos, gripper)
    phase_source = "hdf5"
    limits = PANDA_JOINT_LIMITS
    replay: SimulatorReplay | None = None
    sim_error = ""
    if simulator != "off":
        try:
            if bddl_path is None:
                raise FileNotFoundError(f"No BDDL for {hdf5_path}")
            assets_root = Path(libero_root) / "libero" / "libero" / "assets"
            repaired_xml, _ = repair_asset_paths(xml_text, assets_root, robosuite_assets_root())
            replay = SimulatorReplay(bddl_path, repaired_xml)
            limits = panda_limits_from_sim(replay.sim)
            if replay_mode == "full" and states is not None:
                _, timeline = replay.replay(states, actions, list(frame_indices), ee_pos)
                phases = timeline["phases"]
                phase_source = str(timeline.get("phase_evidence", {}).get("source", "sim_events"))
        except Exception as exc:
            sim_error = f"{type(exc).__name__}: {exc}"
            if simulator == "required":
                raise
            replay = None
    target_name = None
    if replay is not None:
        target_name = resolve_target_name(bddl, replay.entity_body_ids, replay.goal_states)
    elif bddl.get("valid"):
        target_name = resolve_target_name(bddl, [str(x.get("instance")) for x in bddl.get("objects") or []])

    unique_indices = list(dict.fromkeys(int(i) for i in frame_indices))
    by_index: dict[int, dict[str, Any]] = {}
    try:
        for index in unique_indices:
            row = hdf5_scalars_for_index(
                index=index,
                joint=joint,
                ee_pos=ee_pos,
                ee_ori=ee_ori,
                gripper=gripper,
                actions=actions,
                phases=phases,
                delta_window=delta_window,
                limits=limits,
                phase_source=phase_source,
            )
            row.update(_empty_sim_fields(target_name or "", sim_error))
            if replay is not None and states is not None and 0 <= index < len(states):
                replay.sim.set_state_from_flattened(np.asarray(states[index], dtype=float))
                replay.sim.forward()
                row.update(
                    sim_proximity_and_contact(
                        replay,
                        ee_pos=ee_pos[index],
                        gripper_command=float(actions[index, 6]),
                        target_name=target_name,
                    )
                )
                qvel = _arm_qvel_l2(replay.sim)
                if np.isfinite(qvel):
                    row["vel_joint_l2"] = qvel
                row["sim_error"] = ""
                row["phase_source"] = phase_source
            by_index[index] = row
    finally:
        if replay is not None:
            replay.close()
    return [by_index[int(index)] for index in frame_indices]


def extract_sampled_frame_stats(
    sampled: pd.DataFrame,
    *,
    delta_window: int = DEFAULT_DELTA_WINDOW,
    simulator: str = "auto",
    replay_mode: str = "selected",
    libero_root: Path = LOCAL_LIBERO_ROOT,
    max_frames: int | None = None,
) -> pd.DataFrame:
    sampled = sampled.sort_values("row_index").reset_index(drop=True)
    if max_frames is not None:
        sampled = sampled.head(int(max_frames))
    grouped = group_frames_by_demo(sampled)
    meta: dict[tuple[str, str], dict[str, Any]] = {}
    for item in sampled.itertuples(index=False):
        key = (str(item.hdf5_path), str(item.demo_key))
        sample_ids = meta.setdefault(
            key,
            {
                "suite": getattr(item, "suite", "libero_spatial"),
                "task": getattr(item, "task", ""),
                "episode_id": getattr(item, "episode_id", ""),
                "sample_ids": {},
            },
        )["sample_ids"]
        sample_ids[int(item.libero_frame_index)] = int(getattr(item, "sample_id", -1))

    from tqdm import tqdm

    out_rows: list[dict[str, Any]] = []
    for (hdf5_path, demo_key), pairs in tqdm(grouped.items(), desc="frame stats by demo"):
        indices = [frame for _, frame in pairs]
        info = meta[(hdf5_path, demo_key)]
        try:
            stats = extract_demo_stats(
                Path(hdf5_path),
                demo_key,
                indices,
                suite=str(info["suite"] or "libero_spatial"),
                task=str(info["task"] or ""),
                delta_window=delta_window,
                simulator=simulator,
                replay_mode=replay_mode,
                libero_root=Path(libero_root),
            )
        except Exception as exc:
            log(f"Failed {hdf5_path} {demo_key}: {type(exc).__name__}: {exc}")
            stats = [
                {
                    **_failed_hdf5_fields(),
                    **_empty_sim_fields("", f"{type(exc).__name__}: {exc}"),
                }
                for _ in indices
            ]
        for (row_index, frame_index), payload in zip(pairs, stats):
            row = dict(payload)
            row.update(
                {
                    "row_index": int(row_index),
                    "sample_id": int(info["sample_ids"].get(frame_index, -1)),
                    "suite": info["suite"],
                    "task": info["task"],
                    "demo_key": demo_key,
                    "libero_frame_index": int(frame_index),
                    "episode_id": info["episode_id"],
                    "hdf5_path": hdf5_path,
                }
            )
            out_rows.append(row)
    result = pd.DataFrame(out_rows).sort_values("row_index").reset_index(drop=True)
    return result.reindex(columns=STAT_COLUMNS)


def write_frame_stats(
    table: pd.DataFrame,
    output_dir: Path,
    *,
    extra_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    csv_path = output_dir / "frame_stats.csv"
    table.to_csv(csv_path, index=False)
    n = len(table)
    summary = {
        "n_frames": n,
        "n_demos": int(table[["hdf5_path", "demo_key"]].drop_duplicates().shape[0]) if n else 0,
        "sim_valid_frames": int(pd.to_numeric(table.get("sim_valid"), errors="coerce").fillna(0).sum()) if n else 0,
        "motion_valid_frames": int(pd.to_numeric(table.get("motion_valid"), errors="coerce").fillna(0).sum()) if n else 0,
        "columns": list(table.columns),
        "csv": str(csv_path),
    }
    if extra_summary:
        summary.update(extra_summary)
    write_json(output_dir / "summary.json", summary)
    return summary


def _arm_joints(joint: np.ndarray) -> np.ndarray:
    joint = np.asarray(joint, dtype=np.float64)
    if joint.ndim == 1:
        return joint[:7]
    return joint[:, :7] if joint.shape[-1] > 7 else joint


def _row_l2(values: np.ndarray, index: int) -> float:
    return float(np.linalg.norm(np.asarray(values[index], dtype=np.float64).reshape(-1)))


def _ori_speed(ee_ori: np.ndarray, index: int) -> float:
    ee_ori = np.asarray(ee_ori, dtype=np.float64)
    if len(ee_ori) == 1:
        return 0.0
    if index <= 0:
        return axis_angle_geodesic(ee_ori[0], ee_ori[1])
    if index >= len(ee_ori) - 1:
        return axis_angle_geodesic(ee_ori[-2], ee_ori[-1])
    return 0.5 * (
        axis_angle_geodesic(ee_ori[index - 1], ee_ori[index])
        + axis_angle_geodesic(ee_ori[index], ee_ori[index + 1])
    )


def _contact_flags(sim: Any, replay: SimulatorReplay) -> tuple[bool, set[str]]:
    gripper_hits: set[str] = set()
    nontable = False
    for index in range(int(sim.data.ncon)):
        contact = sim.data.contact[index]
        geom_ids = (int(contact.geom1), int(contact.geom2))
        body_ids = [int(sim.model.geom_bodyid[g]) for g in geom_ids]
        names = [_model_id2name(sim.model, "body", b) for b in body_ids]
        entities = [replay.body_to_entity.get(b, names[i]) for i, b in enumerate(body_ids)]
        if contact_is_nontable_robot_object(names, entities):
            nontable = True
        if "robot_gripper" in entities:
            for ent, is_robot in zip(entities, [is_robot_name(n) or e == "robot_gripper" for n, e in zip(names, entities)]):
                if not is_robot and not is_table_name(ent):
                    gripper_hits.add(ent)
    return nontable, gripper_hits


def _min_robot_object_geom_distance(sim: Any, replay: SimulatorReplay, distmax: float) -> float:
    robot_geoms, object_geoms = _collision_geom_sets(sim, replay)
    if not robot_geoms or not object_geoms:
        return _min_robot_object_body_distance(sim, replay)
    try:
        import mujoco

        if not hasattr(mujoco, "mj_geomDistance"):
            return _min_robot_object_body_distance(sim, replay)
        best = float(distmax)
        fromto = np.empty(6, dtype=np.float64)
        for g1 in robot_geoms:
            for g2 in object_geoms:
                dist = float(mujoco.mj_geomDistance(sim.model, sim.data, g1, g2, float(distmax), fromto))
                if dist < best:
                    best = dist
        return best
    except Exception:
        return _min_robot_object_body_distance(sim, replay)


def _collision_geom_sets(sim: Any, replay: SimulatorReplay) -> tuple[list[int], list[int]]:
    robot_geoms: list[int] = []
    object_geoms: list[int] = []
    for geom_id in range(int(sim.model.ngeom)):
        if int(sim.model.geom_group[geom_id]) != 0:
            continue
        body_id = int(sim.model.geom_bodyid[geom_id])
        body_name = _model_id2name(sim.model, "body", body_id)
        entity = replay.body_to_entity.get(body_id, body_name)
        if is_table_name(body_name) or is_table_name(entity):
            continue
        if is_robot_name(body_name):
            robot_geoms.append(geom_id)
        elif body_id in replay.body_to_entity and replay.body_to_entity[body_id] != "robot_gripper":
            object_geoms.append(geom_id)
    return robot_geoms, object_geoms


def _min_robot_object_body_distance(sim: Any, replay: SimulatorReplay) -> float:
    object_ids = [
        body_id
        for name, body_id in replay.entity_body_ids.items()
        if not is_table_name(name)
    ]
    if not object_ids:
        return float("nan")
    best = float("inf")
    for body_id in range(int(sim.model.nbody)):
        name = _model_id2name(sim.model, "body", body_id)
        if not is_robot_name(name):
            continue
        rpos = np.asarray(sim.data.body_xpos[body_id], dtype=np.float64)
        for oid in object_ids:
            dist = float(np.linalg.norm(rpos - np.asarray(sim.data.body_xpos[oid], dtype=np.float64)))
            if dist < best:
                best = dist
    return best if np.isfinite(best) else float("nan")


def _arm_qvel_l2(sim: Any) -> float:
    speeds = []
    for joint_id in range(int(sim.model.njnt)):
        name = _model_id2name(sim.model, "joint", joint_id)
        if not str(name).startswith("robot0_joint"):
            continue
        adr = int(sim.model.jnt_dofadr[joint_id])
        speeds.append(float(sim.data.qvel[adr]))
    if not speeds:
        return float("nan")
    return float(np.linalg.norm(speeds))


def _model_id2name(model: Any, kind: str, index: int) -> str:
    method = getattr(model, f"{kind}_id2name", None)
    if callable(method):
        return method(index) or ""
    try:
        import mujoco

        obj = {
            "body": mujoco.mjtObj.mjOBJ_BODY,
            "joint": mujoco.mjtObj.mjOBJ_JOINT,
            "geom": mujoco.mjtObj.mjOBJ_GEOM,
        }[kind]
        return mujoco.mj_id2name(model, obj, index) or ""
    except Exception:
        return ""


def _empty_sim_fields(target_name: str = "", sim_error: str = "") -> dict[str, Any]:
    return {
        "d_ee_target": float("nan"),
        "d_ee_nearest_obj": float("nan"),
        "d_target_nearest_link": float("nan"),
        "d_body_env": float("nan"),
        "contact_nontable": float("nan"),
        "target_held": float("nan"),
        "target_name": target_name,
        "sim_valid": 0,
        "sim_error": sim_error,
    }


def _failed_hdf5_fields() -> dict[str, Any]:
    return {
        "episode_T": -1,
        "progress": float("nan"),
        "phase_index": -1,
        "phase_source": "failed",
        "motion_joint_l2": float("nan"),
        "motion_ee_pos_l2": float("nan"),
        "motion_ee_ori_rad": float("nan"),
        "motion_gripper_l2": float("nan"),
        "motion_valid": 0,
        "vel_joint_l2": float("nan"),
        "vel_ee_pos_l2": float("nan"),
        "vel_ee_ori_rad": float("nan"),
        "vel_gripper_l2": float("nan"),
        "k_limit": float("nan"),
        "k_change": float("nan"),
        "k_change_valid": 0,
        "gripper_width": float("nan"),
        "gripper_state_code": -1,
        "gripper_command": float("nan"),
    }
