from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.pi05_frame_stats import (
    PANDA_JOINT_LIMITS,
    STAT_COLUMNS,
    axis_angle_geodesic,
    configuration_change,
    consecutive_l2,
    contact_is_nontable_robot_object,
    extract_demo_stats,
    extract_sampled_frame_stats,
    group_frames_by_demo,
    hdf5_scalars_for_index,
    joint_limit_proximity,
    load_sampled_frame_table,
    write_frame_stats,
)


def _mid_q() -> np.ndarray:
    return 0.5 * (PANDA_JOINT_LIMITS[:, 0] + PANDA_JOINT_LIMITS[:, 1])


def _write_demo_hdf5(path: Path, *, length: int = 12) -> None:
    joint = np.zeros((length, 7), dtype=np.float64)
    joint[:, 0] = np.linspace(0.0, 0.7, length)
    ee_pos = np.zeros((length, 3), dtype=np.float64)
    ee_pos[:, 0] = np.linspace(0.0, 0.12, length)
    ee_ori = np.zeros((length, 3), dtype=np.float64)
    ee_ori[:, 2] = np.linspace(0.0, 0.5, length)
    gripper = np.stack([np.linspace(0.04, 0.01, length), np.linspace(0.04, 0.01, length)], axis=1)
    actions = np.zeros((length, 7), dtype=np.float64)
    actions[:4, 6] = -1.0
    actions[4:9, 6] = 1.0
    actions[9:, 6] = -1.0
    states = np.zeros((length, 8), dtype=np.float64)
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data/demo_0")
        demo.create_dataset("actions", data=actions)
        demo.create_dataset("states", data=states)
        obs = demo.create_group("obs")
        obs.create_dataset("joint_states", data=joint)
        obs.create_dataset("ee_pos", data=ee_pos)
        obs.create_dataset("ee_ori", data=ee_ori)
        obs.create_dataset("gripper_states", data=gripper)
        demo.attrs["model_file"] = "<mujoco></mujoco>"
        handle["data"].attrs["bddl_file_name"] = ""


class FrameStatFormulasTest(unittest.TestCase):
    def test_k_limit_center_and_bound(self) -> None:
        self.assertAlmostEqual(joint_limit_proximity(_mid_q()), 0.0, places=6)
        at_limit = _mid_q()
        at_limit[0] = PANDA_JOINT_LIMITS[0, 1]
        self.assertAlmostEqual(joint_limit_proximity(at_limit), 1.0, places=6)
        halfway = _mid_q()
        half = (PANDA_JOINT_LIMITS[3, 1] - PANDA_JOINT_LIMITS[3, 0]) / 2.0
        halfway[3] = PANDA_JOINT_LIMITS[3, 0] + 0.5 * half
        self.assertAlmostEqual(joint_limit_proximity(halfway), 0.5, places=6)

    def test_axis_angle_geodesic_quarter_turn(self) -> None:
        dist = axis_angle_geodesic(np.zeros(3), np.array([0.0, 0.0, 0.5 * math.pi]))
        self.assertAlmostEqual(dist, 0.5 * math.pi, places=6)

    def test_consecutive_l2_censors_last_frame(self) -> None:
        values = np.array([[0.0], [3.0], [7.0]])
        dist, valid = consecutive_l2(values, 0)
        self.assertTrue(valid)
        self.assertAlmostEqual(dist, 3.0)
        dist, valid = consecutive_l2(values, 2)
        self.assertFalse(valid)
        self.assertTrue(math.isnan(dist))

    def test_k_change_window(self) -> None:
        q = np.zeros((10, 7))
        q[8] = 2.0
        dist, valid = configuration_change(q, 0, 8)
        self.assertTrue(valid)
        self.assertAlmostEqual(dist, float(np.linalg.norm(q[8])))
        dist, valid = configuration_change(q, 3, 8)
        self.assertFalse(valid)
        self.assertTrue(math.isnan(dist))

    def test_contact_classifier_excludes_table(self) -> None:
        self.assertTrue(
            contact_is_nontable_robot_object(
                ("robot0_link6", "akita_black_bowl_1"),
                ("robot_gripper", "akita_black_bowl_1"),
            )
        )
        self.assertFalse(
            contact_is_nontable_robot_object(
                ("robot0_link0", "wooden_table"),
                ("robot0_link0", "wooden_table"),
            )
        )
        self.assertFalse(
            contact_is_nontable_robot_object(
                ("akita_black_bowl_1", "plate_1"),
                ("akita_black_bowl_1", "plate_1"),
            )
        )


class SampledFrameGroupingTest(unittest.TestCase):
    def test_groups_by_hdf5_and_demo(self) -> None:
        table = pd.DataFrame(
            {
                "row_index": [2, 0, 1],
                "sample_id": [9, 7, 8],
                "demo_key": ["demo_1", "demo_0", "demo_1"],
                "libero_frame_index": [4, 1, 9],
                "hdf5_path": ["/a.hdf5", "/a.hdf5", "/a.hdf5"],
            }
        )
        grouped = group_frames_by_demo(table)
        self.assertEqual(set(grouped), {("/a.hdf5", "demo_0"), ("/a.hdf5", "demo_1")})
        self.assertEqual(grouped[("/a.hdf5", "demo_1")], [(2, 4), (1, 9)])

    def test_load_sampled_frame_table_requires_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sampled.csv"
            pd.DataFrame({"row_index": [0]}).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_sampled_frame_table(path)


class SyntheticHdf5ExtractTest(unittest.TestCase):
    def test_extracts_hdf5_scalars_without_simulator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hdf5_path = Path(tmp) / "pick_up_the_bowl_demo.hdf5"
            _write_demo_hdf5(hdf5_path, length=12)
            rows = extract_demo_stats(
                hdf5_path,
                "demo_0",
                [0, 3, 11],
                simulator="off",
                delta_window=8,
            )
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["motion_valid"], 1)
            self.assertGreater(rows[0]["motion_joint_l2"], 0.0)
            self.assertGreater(rows[0]["motion_ee_ori_rad"], 0.0)
            self.assertEqual(rows[2]["motion_valid"], 0)
            self.assertTrue(math.isnan(rows[2]["motion_joint_l2"]))
            self.assertEqual(rows[0]["k_change_valid"], 1)
            self.assertEqual(rows[2]["k_change_valid"], 0)
            self.assertEqual(rows[0]["sim_valid"], 0)
            self.assertTrue(math.isnan(rows[0]["d_body_env"]))
            self.assertTrue(0 <= rows[1]["phase_index"] <= 10)
            self.assertAlmostEqual(rows[0]["progress"], 0.0)
            self.assertAlmostEqual(rows[2]["progress"], 1.0)

    def test_sampled_table_roundtrip_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hdf5_path = Path(tmp) / "pick_up_the_bowl_demo.hdf5"
            _write_demo_hdf5(hdf5_path, length=12)
            sampled = pd.DataFrame(
                {
                    "row_index": [1, 0],
                    "sample_id": [101, 100],
                    "suite": ["libero_spatial", "libero_spatial"],
                    "task": ["pick_up_the_bowl", "pick_up_the_bowl"],
                    "demo_key": ["demo_0", "demo_0"],
                    "libero_frame_index": [11, 0],
                    "episode_id": ["x/demo_0", "x/demo_0"],
                    "hdf5_path": [str(hdf5_path), str(hdf5_path)],
                }
            )
            table = extract_sampled_frame_stats(sampled, simulator="off", delta_window=8)
            self.assertEqual(list(table.columns), STAT_COLUMNS)
            self.assertEqual(list(table["row_index"]), [0, 1])
            self.assertEqual(list(table["libero_frame_index"]), [0, 11])
            self.assertEqual(int(table.iloc[0]["motion_valid"]), 1)
            self.assertEqual(int(table.iloc[1]["motion_valid"]), 0)
            self.assertGreater(int(table.iloc[0]["episode_T"]), 1)
            self.assertEqual(table.iloc[0]["sim_error"], "")
            summary = write_frame_stats(table, Path(tmp) / "out")
            self.assertEqual(summary["n_frames"], 2)
            self.assertTrue((Path(tmp) / "out" / "frame_stats.csv").exists())

    def test_hdf5_scalars_match_direct_formulas(self) -> None:
        joint = np.zeros((5, 7))
        joint[1] = 0.2
        ee_pos = np.zeros((5, 3))
        ee_ori = np.zeros((5, 3))
        gripper = np.ones((5, 2)) * 0.03
        actions = np.zeros((5, 7))
        row = hdf5_scalars_for_index(
            index=0,
            joint=joint,
            ee_pos=ee_pos,
            ee_ori=ee_ori,
            gripper=gripper,
            actions=actions,
            phases=[0, 1, 2, 3, 4],
            delta_window=1,
            limits=PANDA_JOINT_LIMITS,
        )
        self.assertAlmostEqual(row["motion_joint_l2"], np.linalg.norm(joint[1]))
        self.assertEqual(row["phase_index"], 0)


class SampledCsvSmokeTest(unittest.TestCase):
    def test_two_real_frames_simulator_off(self) -> None:
        csv_path = Path("outputs/ablation/pi05_probe_guided_frames/sampled_frames.csv")
        if not csv_path.exists():
            self.skipTest("sampled_frames.csv is not present")
        sampled = load_sampled_frame_table(csv_path).head(2)
        hdf5_path = Path(sampled.iloc[0]["hdf5_path"])
        if not hdf5_path.exists():
            self.skipTest(f"LIBERO hdf5 missing: {hdf5_path}")
        table = extract_sampled_frame_stats(sampled, simulator="off", delta_window=8)
        self.assertEqual(len(table), 2)
        self.assertEqual(list(table["row_index"]), list(sampled["row_index"]))
        self.assertTrue((table["episode_T"] > 1).all())
        self.assertTrue(table["k_limit"].between(0.0, 1.0).all())
        self.assertEqual(int(table["sim_valid"].sum()), 0)
        self.assertIn("motion_ee_ori_rad", table.columns)
        self.assertIn("d_body_env", table.columns)
        self.assertIn("contact_nontable", table.columns)
        self.assertIn("phase_index", table.columns)


if __name__ == "__main__":
    unittest.main()
