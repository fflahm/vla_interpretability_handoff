from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.libero_rich_annotations import (
    DemoRef,
    action_labels,
    balanced_demo_selection,
    fixed_chunk,
    extract_free_joint_poses,
    future_labels,
    infer_phase_sequence,
    parse_model_xml,
    select_frame_indices,
    extract_dataset,
)


class RichAnnotationsTest(unittest.TestCase):
    def test_xml_free_joint_and_appearance(self) -> None:
        xml = """<mujoco><asset><material name="red" rgba="1 0 0 1"/></asset>
        <worldbody><body name="bowl" pos="1 2 3"><freejoint name="bowl_joint"/>
        <geom name="g" type="sphere" size=".1" material="red"/></body>
        <body name="table"><geom type="box" size="1 1 .1"/></body>
        <camera name="agentview" fovy="45"/></worldbody></mujoco>"""
        parsed = parse_model_xml(xml)
        bowl = next(x for x in parsed["entities"] if x["instance"] == "bowl")
        self.assertTrue(bowl["movable"])
        self.assertEqual(bowl["free_joint"], "bowl_joint")
        self.assertEqual(bowl["initial_pose"]["position"], [1.0, 2.0, 3.0])

    def test_balanced_selection_is_reproducible(self) -> None:
        demos = [
            DemoRef("suite", task, Path(f"{task}.hdf5"), f"demo_{i}", 20)
            for task in ("a", "b", "c") for i in range(5)
        ]
        first = balanced_demo_selection(demos, 8, 7)
        second = balanced_demo_selection(demos, 8, 7)
        self.assertEqual(first, second)
        counts = {task: sum(x.task == task for x in first) for task in ("a", "b", "c")}
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertNotEqual([x.demo_key for x in first], [f"demo_{i}" for i in range(5)] + ["demo_0"] * 3)

    def test_frame_selection_bounded_sorted_and_event_aware(self) -> None:
        actions = np.zeros((40, 7))
        actions[17:, 6] = -1
        actions[29:, :3] = 0.5
        ee = np.zeros((40, 3))
        ee[25:, 0] = 1
        indices, reasons = select_frame_indices(actions, ee, 8)
        self.assertEqual(indices, sorted(set(indices)))
        self.assertEqual(len(indices), 8)
        all_reasons = {r for values in reasons.values() for r in values}
        self.assertIn("gripper_command_change", all_reasons)
        self.assertIn("trajectory_keypoint", all_reasons)

    def test_action_future_masks_and_phase(self) -> None:
        actions = np.zeros((12, 7))
        actions[3:9, 6] = 1
        actions[9:, 6] = -1
        ee = np.stack([np.arange(12), np.zeros(12), np.zeros(12)], axis=1) * 0.01
        gripper = np.ones((12, 2))
        phases, events = infer_phase_sequence(actions, ee, gripper)
        self.assertEqual(len(phases), 12)
        self.assertTrue(all(a <= b for a, b in zip(phases, phases[1:])))
        self.assertTrue(all(0 <= x <= 10 for x in phases))
        labels = action_labels(actions[3])
        self.assertIn("close_gripper", labels["primitive_labels"])
        self.assertEqual(labels["gripper_sign_convention"], "-1=open,+1=close")
        self.assertIn("open_gripper", action_labels(actions[9])["primitive_labels"])
        chunk, mask = fixed_chunk(actions, 10, 4)
        self.assertEqual(mask, [True, True, False, False])
        future = future_labels(actions, ee, phases, events, 10, (1, 4))
        self.assertTrue(future["horizons"]["1"]["valid"])
        self.assertFalse(future["horizons"]["4"]["valid"])
        self.assertTrue(future["horizons"]["4"]["censored"])

    def test_free_joint_pose_from_flattened_state(self) -> None:
        xml = """<mujoco><worldbody><body name="robot"><joint name="hinge"/>
        <geom type="sphere" size=".1"/></body><body name="object"><freejoint name="object_joint"/>
        <geom type="sphere" size=".1"/></body></worldbody></mujoco>"""
        state = np.array([0.0, 0.25, 1.0, 2.0, 3.0, 0.7, 0.1, 0.2, 0.3, 99.0])
        pose = extract_free_joint_poses(state, xml)["object_joint"]
        self.assertEqual(pose["position"], [1.0, 2.0, 3.0])
        self.assertEqual(pose["quaternion_wxyz"], [0.7, 0.1, 0.2, 0.3])
        self.assertEqual(pose["qpos_address"], [1, 8])
        self.assertEqual(pose["flattened_state_address"], [2, 9])
        self.assertTrue(pose["valid"])

    @unittest.skipUnless(os.environ.get("RUN_LIBERO_INTEGRATION") == "1", "set RUN_LIBERO_INTEGRATION=1")
    def test_simulator_output_schema(self) -> None:
        dataset = Path(
            "/data/tos/guoshengyu/vla/libero/libero_spatial/"
            "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5"
        )
        if not dataset.exists():
            self.skipTest("local LIBERO integration dataset unavailable")
        with tempfile.TemporaryDirectory() as directory:
            summary = extract_dataset(dataset, Path(directory), 1, 2, 4, simulator="required")
            row = json.loads((Path(directory) / "frames.jsonl").read_text().splitlines()[0])
            self.assertEqual(summary["simulator"]["status"], "available")
            self.assertTrue(row["quality"]["simulator_valid"])
            self.assertTrue(row["spatial"]["body_poses"]["value"])
            self.assertIsInstance(row["spatial"]["contacts"]["value"], list)
            self.assertTrue(row["goal"]["predicate_status"]["value"])


if __name__ == "__main__":
    unittest.main()
