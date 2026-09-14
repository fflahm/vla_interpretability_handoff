from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.pi05_frame_stat_groups import (
    GROUP1_LARGE,
    GROUP2_NEAR_ZERO,
    GROUP3_SMALL,
    PLOT_DIMENSIONS,
    UNASSIGNED,
    assign_impact_groups,
    attach_impact_groups,
    merge_stats_and_impact,
    short_task_label,
    write_group_outputs,
)


class ImpactGroupingTest(unittest.TestCase):
    def test_equal_size_tails_and_near_zero(self) -> None:
        values = np.concatenate(
            [
                np.linspace(-2.0, -1.0, 20),
                np.linspace(-0.8, -0.3, 40),
                np.linspace(-0.1, 0.1, 20),
                np.linspace(1.0, 2.0, 20),
            ]
        )
        labels = assign_impact_groups(values, tail_frac=0.2)
        self.assertEqual(int((labels == GROUP1_LARGE).sum()), 20)
        self.assertEqual(int((labels == GROUP2_NEAR_ZERO).sum()), 20)
        self.assertEqual(int((labels == GROUP3_SMALL).sum()), 20)
        self.assertEqual(int((labels == UNASSIGNED).sum()), 40)
        self.assertGreater(values[labels == GROUP1_LARGE].min(), values[labels == GROUP2_NEAR_ZERO].max())
        self.assertGreater(values[labels == GROUP2_NEAR_ZERO].min(), values[labels == GROUP3_SMALL].max())
        self.assertLess(np.mean(np.abs(values[labels == GROUP2_NEAR_ZERO])), 0.12)

    def test_merge_requires_aligned_row_index(self) -> None:
        stats = pd.DataFrame({"row_index": [0, 1], "task": ["a", "b"], "progress": [0.1, 0.2]})
        impact = pd.DataFrame({"frame_row": [0, 1], "good_minus_bad": [1.0, -1.0]})
        merged = merge_stats_and_impact(stats, impact)
        self.assertEqual(list(merged["good_minus_bad"]), [1.0, -1.0])
        bad = pd.DataFrame({"frame_row": [0, 2], "good_minus_bad": [1.0, -1.0]})
        with self.assertRaises(ValueError):
            merge_stats_and_impact(stats, bad)

    def test_short_task_label(self) -> None:
        name = "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"
        self.assertEqual(short_task_label(name), "on the stove")


class GroupPlotSmokeTest(unittest.TestCase):
    def test_writes_one_png_per_dimension(self) -> None:
        rng = np.random.default_rng(0)
        n = 60
        stats = pd.DataFrame(
            {
                "row_index": np.arange(n),
                "task": rng.choice(["pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate", "other_task"], n),
                "progress": rng.uniform(0, 1, n),
                "phase_index": rng.integers(0, 11, n),
                "motion_joint_l2": rng.random(n),
                "motion_ee_pos_l2": rng.random(n),
                "motion_ee_ori_rad": rng.random(n),
                "motion_gripper_l2": rng.random(n),
                "motion_valid": np.ones(n, dtype=int),
                "vel_joint_l2": rng.random(n),
                "vel_ee_pos_l2": rng.random(n),
                "vel_ee_ori_rad": rng.random(n),
                "vel_gripper_l2": rng.random(n),
                "k_limit": rng.random(n),
                "k_change": rng.random(n),
                "k_change_valid": np.ones(n, dtype=int),
                "gripper_width": rng.random(n),
                "gripper_state_code": rng.integers(0, 3, n),
                "gripper_command": rng.choice([-1.0, 1.0], n),
                "d_ee_target": rng.random(n),
                "d_ee_nearest_obj": rng.random(n),
                "d_target_nearest_link": rng.random(n),
                "d_body_env": rng.random(n),
                "contact_nontable": rng.integers(0, 2, n),
                "target_held": rng.integers(0, 2, n),
            }
        )
        impact = pd.DataFrame(
            {
                "frame_row": np.arange(n),
                "good_minus_bad": np.linspace(-2.0, 2.0, n),
            }
        )
        table = attach_impact_groups(merge_stats_and_impact(stats, impact), tail_frac=0.2)
        with tempfile.TemporaryDirectory() as tmp:
            summary = write_group_outputs(table, Path(tmp), dpi=80, bins=8)
            self.assertEqual(len(summary["figures"]), len(PLOT_DIMENSIONS))
            for path in summary["figures"]:
                self.assertTrue(Path(path).exists())
                self.assertGreater(Path(path).stat().st_size, 100)

    def test_real_csvs_if_present(self) -> None:
        stats_path = Path("outputs/ablation/pi05_probe_guided_frames/frame_stats/frame_stats.csv")
        impact_path = Path("outputs/ablation/pi05_probe_guided_frames/frame_good_minus_bad.csv")
        if not stats_path.exists() or not impact_path.exists():
            self.skipTest("real frame_stats outputs are not present")
        stats = pd.read_csv(stats_path)
        impact = pd.read_csv(impact_path)
        table = attach_impact_groups(merge_stats_and_impact(stats, impact), tail_frac=0.2)
        counts = table["impact_group"].value_counts().to_dict()
        self.assertEqual(counts[GROUP1_LARGE], 200)
        self.assertEqual(counts[GROUP2_NEAR_ZERO], 200)
        self.assertEqual(counts[GROUP3_SMALL], 200)
        self.assertEqual(counts[UNASSIGNED], 400)
        self.assertGreater(
            table.loc[table.impact_group == GROUP1_LARGE, "good_minus_bad"].min(),
            table.loc[table.impact_group == GROUP2_NEAR_ZERO, "good_minus_bad"].max(),
        )


if __name__ == "__main__":
    unittest.main()
