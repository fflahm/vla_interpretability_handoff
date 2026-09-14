from __future__ import annotations

import unittest

import numpy as np

from src.libero_self_occupancy import (
    OccupancyGridSpec,
    evenly_spaced_indices,
    hard_iou,
    parse_index_spec,
    soft_iou,
    stratified_episode_split,
    token_position_bins,
)


class OccupancyGridSpecTest(unittest.TestCase):
    def test_soften_averages_subvoxels(self) -> None:
        spec = OccupancyGridSpec(
            size=2,
            lower=(0.0, 0.0, 0.0),
            upper=(2.0, 2.0, 2.0),
            supersample=2,
        )
        high = np.zeros((4, 4, 4), dtype=bool)
        high[:2, :2, :2] = True
        high[2, 2, 2] = True

        soft = spec.soften(high)

        self.assertEqual(soft.shape, (2, 2, 2))
        self.assertEqual(float(soft[0, 0, 0]), 1.0)
        self.assertEqual(float(soft[1, 1, 1]), 0.125)

    def test_centers_are_inside_fixed_bounds(self) -> None:
        spec = OccupancyGridSpec(size=4, supersample=2)
        centers = spec.high_resolution_centers()
        self.assertEqual(centers.shape, (8, 8, 8, 3))
        self.assertTrue(np.all(centers[..., 0] > spec.lower[0]))
        self.assertTrue(np.all(centers[..., 0] < spec.upper[0]))


class SelectionAndMetricsTest(unittest.TestCase):
    def test_evenly_spaced_indices_exclude_endpoints(self) -> None:
        self.assertEqual(evenly_spaced_indices(10, 4), [2, 4, 5, 7])

    def test_iou_is_one_for_identical_soft_targets(self) -> None:
        target = np.asarray([[[[0.0, 0.25], [0.5, 1.0]]]], dtype=np.float32)
        self.assertAlmostEqual(soft_iou(target, target), 1.0)
        self.assertAlmostEqual(hard_iou(target, target), 1.0)

    def test_soft_iou_penalizes_false_occupancy(self) -> None:
        target = np.asarray([[[[0.0, 1.0]]]], dtype=np.float32)
        prediction = np.asarray([[[[1.0, 1.0]]]], dtype=np.float32)
        self.assertAlmostEqual(soft_iou(target, prediction), 0.5)

    def test_token_bins_never_empty_and_keep_hidden_vector(self) -> None:
        tokens = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
        binned = token_position_bins(tokens, requested_bins=96)
        self.assertEqual(binned.shape, (2, 5, 3))
        np.testing.assert_array_equal(binned, tokens)

    def test_token_bins_are_equal_width_means(self) -> None:
        tokens = np.arange(8, dtype=np.float32).reshape(1, 8, 1)
        np.testing.assert_allclose(
            token_position_bins(tokens, requested_bins=3).reshape(-1),
            [0.5, 3.0, 6.0],
        )

    def test_stratified_split_has_no_episode_leakage(self) -> None:
        rows = []
        for task in ("task_a", "task_b"):
            for demo in range(4):
                for frame in range(2):
                    rows.append({
                        "task": task,
                        "episode_id": f"libero_spatial/{task}/demo_{demo}",
                        "frame_index": frame,
                    })
        train, test = stratified_episode_split(rows, 0.25, 7)
        train_episodes = {rows[i]["episode_id"] for i in train}
        test_episodes = {rows[i]["episode_id"] for i in test}
        self.assertFalse(train_episodes & test_episodes)
        for task in ("task_a", "task_b"):
            self.assertTrue(any(rows[i]["task"] == task for i in train))
            self.assertTrue(any(rows[i]["task"] == task for i in test))

    def test_parse_index_spec_supports_lists_ranges_and_strides(self) -> None:
        self.assertEqual(parse_index_spec("all", 8), list(range(8)))
        self.assertEqual(parse_index_spec("0,8,16", 96), [0, 8, 16])
        self.assertEqual(parse_index_spec("0-3", 96), [0, 1, 2, 3])
        self.assertEqual(parse_index_spec("0-15:8", 96), [0, 8])
        self.assertEqual(parse_index_spec("0,80,95", 50), [0])


if __name__ == "__main__":
    unittest.main()
