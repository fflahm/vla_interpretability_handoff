"""Human-readable reports for LIBERO rich-annotation JSONL records."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield non-empty JSONL records with useful parse errors."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc


def select_frame(
    frames_path: Path,
    *,
    sample_index: int | None = None,
    episode_id: str | None = None,
    frame_index: int | None = None,
) -> dict[str, Any]:
    """Select one frame by JSONL row or by episode/frame identity."""
    if sample_index is not None and sample_index < 0:
        raise ValueError("sample_index must be non-negative")
    if sample_index is None and (episode_id is None or frame_index is None):
        raise ValueError("Specify sample_index, or both episode_id and frame_index")

    for row_index, record in enumerate(read_jsonl(frames_path)):
        sample = record.get("sample", {})
        if sample_index is not None and row_index == sample_index:
            return record
        if (
            sample_index is None
            and sample.get("episode_id") == episode_id
            and int(sample.get("frame", -1)) == frame_index
        ):
            return record
    selector = (
        f"sample_index={sample_index}"
        if sample_index is not None
        else f"episode_id={episode_id!r}, frame={frame_index}"
    )
    raise LookupError(f"No frame matched {selector} in {frames_path}")


def select_episode(episodes_path: Path, episode_id: str) -> dict[str, Any]:
    """Load the episode record associated with a selected frame."""
    for record in read_jsonl(episodes_path):
        if record.get("episode_id") == episode_id:
            return record
    raise LookupError(f"Episode {episode_id!r} not found in {episodes_path}")


def combined_record(
    annotation_dir: Path,
    *,
    sample_index: int | None = None,
    episode_id: str | None = None,
    frame_index: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the matching episode and frame records."""
    annotation_dir = annotation_dir.resolve()
    frame = select_frame(
        annotation_dir / "frames.jsonl",
        sample_index=sample_index,
        episode_id=episode_id,
        frame_index=frame_index,
    )
    selected_episode_id = str(frame["sample"]["episode_id"])
    episode = select_episode(annotation_dir / "episodes.jsonl", selected_episode_id)
    return episode, frame


def render_markdown(
    episode: Mapping[str, Any],
    frame: Mapping[str, Any],
    annotation_dir: Path,
    *,
    include_raw: bool = False,
) -> str:
    """Render one joined episode/frame record as a compact Markdown report."""
    sample = frame["sample"]
    body = frame.get("body", {})
    phase = frame.get("phase", {})
    goal = frame.get("goal", {})
    action = frame.get("action", {})
    spatial = frame.get("spatial", {})
    quality = frame.get("quality", {})
    lines: list[str] = [
        f"# LIBERO frame report: `{sample['episode_id']} @ {sample['frame']}`",
        "",
        "## Overview",
        "",
        _table(
            ["Field", "Value"],
            [
                ["Suite", sample.get("suite")],
                ["Task", sample.get("task")],
                ["Demo", sample.get("demo")],
                ["Frame", f"{sample.get('frame')} / {int(sample.get('T', 1)) - 1}"],
                ["Progress", _number(sample.get("progress"), 4)],
                ["Instruction", episode.get("instruction")],
                ["Selection reasons", ", ".join(sample.get("selection_reasons", []))],
                ["Annotation version", frame.get("annotation_version")],
                ["Simulator", episode.get("simulator", {}).get("provenance")],
            ],
        ),
        "",
        "## Images",
        "",
    ]
    for view in ("agentview", "wrist"):
        image = frame.get("images", {}).get(view)
        if not image:
            continue
        image_path = (annotation_dir / image["path"]).resolve()
        lines.extend(
            [
                f"### {view}",
                "",
                f"![{view}]({image_path})",
                "",
                f"- File: `{image_path}`",
                f"- Source: `{image.get('raw_dataset')}[{image.get('raw_frame_index')}]`",
                f"- Export transform: `{json.dumps(image.get('transform', {}), ensure_ascii=False)}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Goal and phase",
            "",
            _table(
                ["Field", "Value", "Validity / provenance"],
                [
                    ["Goal predicates", _goal_predicates(goal), _provenance(goal.get("predicate_status"))],
                    ["Satisfied fraction", _value(goal.get("satisfied_fraction")), _provenance(goal.get("satisfied_fraction"))],
                    ["All success", _value(goal.get("all_success")), _provenance(goal.get("all_success"))],
                    ["Phase", f"{phase.get('index')} · {phase.get('detail')}", phase.get("provenance")],
                    ["Phase family", phase.get("family"), f"evidence_count={phase.get('evidence_count')}"],
                    ["Event anchor", ", ".join(phase.get("anchor", [])) or "—", "exact selected-frame anchor"],
                ],
            ),
            "",
            "### Episode event anchors",
            "",
            "```json",
            json.dumps(episode.get("phase_anchors", {}), indent=2, ensure_ascii=False),
            "```",
            "",
            "## Robot body",
            "",
            _table(
                ["Label", "Value", "Provenance"],
                [
                    ["EEF position", _vector(_value(body.get("eef_position"))), _provenance(body.get("eef_position"))],
                    ["EEF orientation", _vector(_value(body.get("eef_orientation_axis_angle"))), _provenance(body.get("eef_orientation_axis_angle"))],
                    ["EEF velocity / step", _vector(_value(body.get("eef_velocity_per_step"))), _provenance(body.get("eef_velocity_per_step"))],
                    ["Joint position", _vector(_value(body.get("joint_position"))), _provenance(body.get("joint_position"))],
                    ["Joint velocity / step", _vector(_value(body.get("joint_velocity_per_step"))), _provenance(body.get("joint_velocity_per_step"))],
                    ["Gripper qpos", _vector(_value(body.get("gripper_qpos"))), _provenance(body.get("gripper_qpos"))],
                    ["Gripper width proxy", _number(_value(body.get("gripper_width_proxy")), 5), _provenance(body.get("gripper_width_proxy"))],
                    ["Gripper state", _value(body.get("gripper_state")), _provenance(body.get("gripper_state"))],
                    ["Motion state", _value(body.get("motion_state")), _provenance(body.get("motion_state"))],
                    ["Held object", _value(body.get("held_object")), _provenance(body.get("held_object"))],
                ],
            ),
            "",
            "## Action",
            "",
            _table(
                ["Label", "Value"],
                [
                    ["Raw 7D", _vector(action.get("raw_7d"))],
                    ["Translation norm", _number(action.get("translation_norm"), 5)],
                    ["Rotation norm", _number(action.get("rotation_norm"), 5)],
                    ["Gripper command", f"{action.get('gripper_command')} ({action.get('gripper_sign_convention')})"],
                    ["Primitives", ", ".join(action.get("primitive_labels", []))],
                    ["Action chunk", f"horizon={action.get('chunk_horizon')}, valid={sum(action.get('chunk_mask', []))}"],
                    ["Provenance", action.get("provenance")],
                ],
            ),
            "",
            "## Objects",
            "",
            _object_table(frame.get("objects", []), spatial),
            "",
            "## Spatial relations and contacts",
            "",
            "### Goal-relevant pairs",
            "",
            _goal_pair_table(_value(spatial.get("goal_relevant_pairs")) or []),
            "",
            "### Entity contact graph",
            "",
            _contact_table(_value(spatial.get("contact_graph")) or []),
            "",
            f"Raw mapped MuJoCo contacts in this frame: **{len(_value(spatial.get('contacts')) or [])}**",
            "",
            "## Future labels",
            "",
            _future_table(frame.get("future", {})),
            "",
            "## Quality and provenance",
            "",
            _table(
                ["Field", "Value"],
                [
                    ["Direct observation valid", quality.get("direct_observation_valid")],
                    ["Simulator valid", quality.get("simulator_valid")],
                    ["Sources", ", ".join(map(str, quality.get("provenance", [])))],
                    ["Warnings", "; ".join(quality.get("warnings", [])) or "None"],
                ],
            ),
            "",
        ]
    )
    if include_raw:
        lines.extend(
            [
                "## Raw joined record",
                "",
                "<details>",
                "<summary>Expand episode and frame JSON</summary>",
                "",
                "```json",
                json.dumps({"episode": episode, "frame": frame}, indent=2, ensure_ascii=False),
                "```",
                "",
                "</details>",
                "",
            ]
        )
    return "\n".join(lines)


def _value(item: Any) -> Any:
    return item.get("value") if isinstance(item, Mapping) and "value" in item else item


def _provenance(item: Any) -> str:
    if not isinstance(item, Mapping):
        return "—"
    validity = "valid" if item.get("valid") else "invalid"
    return f"{validity} · {item.get('provenance', 'unknown')}"


def _number(value: Any, precision: int = 3) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def _vector(value: Any, precision: int = 4) -> str:
    if value is None:
        return "—"
    if not isinstance(value, (list, tuple)):
        return str(value)
    return "[" + ", ".join(_number(x, precision) for x in value) + "]"


def _goal_predicates(goal: Mapping[str, Any]) -> str:
    statuses = _value(goal.get("predicate_status")) or []
    if statuses:
        return "; ".join(
            f"{item.get('predicate')} = {item.get('satisfied')}" for item in statuses
        )
    predicates = goal.get("predicates", [])
    return "; ".join(f"{p.get('name')}({', '.join(p.get('arguments', []))})" for p in predicates) or "—"


def _object_table(objects: list[Mapping[str, Any]], spatial: Mapping[str, Any]) -> str:
    distances = {
        item["object"]: item.get("distance")
        for item in spatial.get("eef_object", [])
        if "object" in item
    }
    rows = []
    for item in objects:
        pose = item.get("current_pose", {})
        roles = [name for name, enabled in item.get("goal_roles", {}).items() if enabled]
        rows.append(
            [
                item.get("bddl_instance") or item.get("instance"),
                item.get("category"),
                item.get("declared_entity_type"),
                ", ".join(roles) or "—",
                _vector(pose.get("position") if isinstance(pose, Mapping) else None),
                _number(distances.get(item.get("bddl_instance") or item.get("instance")), 4),
                pose.get("provenance") if isinstance(pose, Mapping) else "—",
            ]
        )
    return _table(
        ["Instance", "Category", "Type", "Goal roles", "Position xyz", "EEF distance", "Pose source"],
        rows,
    )


def _goal_pair_table(pairs: list[Mapping[str, Any]]) -> str:
    rows = [
        [
            item.get("predicate"),
            item.get("entity_a"),
            item.get("entity_b"),
            _vector(item.get("a_to_b_offset")),
            _number(item.get("distance"), 4),
        ]
        for item in pairs
    ]
    return _table(["Predicate", "Entity A", "Entity B", "A→B offset", "Distance"], rows)


def _contact_table(contacts: list[Mapping[str, Any]]) -> str:
    rows = [
        [item.get("entity_a"), item.get("entity_b"), item.get("provenance")]
        for item in contacts
    ]
    return _table(["Entity A", "Entity B", "Source"], rows)


def _future_table(future: Mapping[str, Any]) -> str:
    rows = []
    for horizon, item in sorted(
        future.get("horizons", {}).items(), key=lambda pair: int(pair[0])
    ):
        rows.append(
            [
                horizon,
                item.get("valid"),
                item.get("phase"),
                _vector(item.get("eef_delta")),
                _vector(item.get("action")),
                item.get("censored"),
            ]
        )
    return _table(["Horizon", "Valid", "Phase", "EEF delta", "Action", "Censored"], rows)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    def clean(value: Any) -> str:
        text = "—" if value is None or value == "" else str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    rendered = [
        "| " + " | ".join(clean(x) for x in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    rendered.extend("| " + " | ".join(clean(x) for x in row) + " |" for row in rows)
    if not rows:
        rendered.append("| " + " | ".join(["None"] + ["—"] * (len(headers) - 1)) + " |")
    return "\n".join(rendered)
