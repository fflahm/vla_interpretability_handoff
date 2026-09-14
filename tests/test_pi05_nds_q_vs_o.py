from __future__ import annotations

import unittest

import numpy as np

from src.pi05_nds_q_vs_o import (
    occupancy_ckpt_is_reusable,
    occupancy_frequency_baseline,
    d_o_minus_q,
    make_q_baseline_per_sample,
    nds,
    per_sample_mse,
    shuffle_rows_by_split,
    summarize_nds_rows,
)


class NdsFormulaTest(unittest.TestCase):
    def test_nds_is_one_minus_ratio_and_not_clipped(self) -> None:
        self.assertAlmostEqual(nds(0.2, 0.5), 0.6)
        self.assertLess(nds(0.8, 0.5), 0.0)
        self.assertAlmostEqual(d_o_minus_q(0.1, 0.4), -0.3)

    def test_q_baseline_is_r2_against_train_mean_zero(self) -> None:
        q_z = np.asarray([[1.0, -1.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        test_idx = np.asarray([1, 2])
        base = make_q_baseline_per_sample(q_z, test_idx)
        np.testing.assert_allclose(base, per_sample_mse(np.zeros((2, 2)), q_z[test_idx]))
        pred = np.zeros((2, 2), dtype=np.float32)
        self.assertAlmostEqual(nds(float(per_sample_mse(pred, q_z[test_idx]).mean()), float(base.mean())), 0.0)

    def test_occupancy_baseline_uses_train_split_only(self) -> None:
        occupancy = np.zeros((4, 16, 16, 16), dtype=np.float32)
        occupancy[0, 0, 0, 0] = 1.0
        occupancy[1, 0, 0, 0] = 1.0
        occupancy[2, 0, 0, 0] = 1.0
        occupancy[3, 1, 0, 0] = 1.0
        train_idx = np.asarray([0, 1])
        p = occupancy_frequency_baseline(occupancy, train_idx)
        self.assertAlmostEqual(float(p[0]), 1.0 - 1e-6, places=5)
        self.assertLess(float(p[16 * 16]), 0.01)

    def test_shuffle_breaks_sample_alignment(self) -> None:
        h = np.arange(20, dtype=np.float32).reshape(10, 2)
        train_idx = np.arange(7)
        test_idx = np.arange(7, 10)
        shuffled = shuffle_rows_by_split(h, train_idx, test_idx, seed=0)
        self.assertFalse(np.array_equal(shuffled[train_idx], h[train_idx]))
        self.assertEqual(set(map(tuple, shuffled[train_idx].tolist())), set(map(tuple, h[train_idx].tolist())))

    def test_reuse_requires_matching_bottleneck(self) -> None:
        import torch

        ckpt = {
            "output_dim": 4096,
            "state_dict": {"0.weight": torch.zeros(64, 8)},
        }
        self.assertTrue(occupancy_ckpt_is_reusable(ckpt, 64))
        self.assertFalse(occupancy_ckpt_is_reusable(ckpt, 32))

    def test_summary_keeps_negative_d(self) -> None:
        summary = summarize_nds_rows(
            [
                {"arm": "real", "tower": "expert", "nds_q": 0.5, "nds_o": 0.1, "d_o_minus_q": -0.4},
                {"arm": "real", "tower": "expert", "nds_q": 0.1, "nds_o": 0.2, "d_o_minus_q": 0.1},
            ]
        )
        self.assertAlmostEqual(summary["d_mean"], -0.15)
        self.assertAlmostEqual(summary["frac_q_dominant"], 0.5)
        self.assertAlmostEqual(summary["frac_o_dominant"], 0.5)


class QDecoderTrainTest(unittest.TestCase):
    def test_q_decoder_mse_beats_mean_baseline(self) -> None:
        import torch
        from src.pi05_nds_q_vs_o import train_bottleneck_regressor

        rng = np.random.default_rng(0)
        h = rng.normal(size=(80, 8)).astype(np.float32)
        q = (h[:, :2] * 0.5).astype(np.float32)
        q = np.concatenate([q, rng.normal(size=(80, 6)).astype(np.float32) * 0.01], axis=1)
        train_idx = np.arange(64)
        test_idx = np.arange(64, 80)
        result = train_bottleneck_regressor(
            x=h,
            y=q,
            train_idx=train_idx,
            test_idx=test_idx,
            hidden=16,
            epochs=8,
            batch_size=16,
            learning_rate=2e-3,
            seed=0,
            device=torch.device("cpu"),
            kind="q",
        )
        base = make_q_baseline_per_sample(q, test_idx)
        score = nds(float(result["per_sample"].mean()), float(base.mean()))
        self.assertGreater(score, 0.0)


if __name__ == "__main__":
    unittest.main()
