from __future__ import annotations

import unittest

import numpy as np

from src.pi05_occupancy_cmi import (
    annotate_history,
    assemble_features,
    decoder_csv_metrics,
    decoder_description,
    ic_hat,
    make_decoder,
    normalize_decoder_architecture,
    probe_seed,
    resolve_bin_indices,
    summarize_ic,
    train_decoder,
    zscore_train,
)


class FeatureAssemblyTest(unittest.TestCase):
    def test_q_only_keeps_q_dim(self) -> None:
        q = np.ones((4, 8), dtype=np.float32)
        got = assemble_features(h=None, q=q, arm="q", matched_params=False)
        self.assertEqual(got.shape, (4, 8))

    def test_hq_concat_and_matched_q_zeros(self) -> None:
        h = np.arange(12, dtype=np.float32).reshape(3, 4)
        q = np.ones((3, 2), dtype=np.float32)
        hq = assemble_features(h=h, q=q, arm="hq", matched_params=False)
        self.assertEqual(hq.shape, (3, 6))
        np.testing.assert_array_equal(hq[:, :4], h)
        q_matched = assemble_features(h=h, q=q, arm="q", matched_params=True)
        self.assertEqual(q_matched.shape, hq.shape)
        np.testing.assert_array_equal(q_matched[:, :4], np.zeros((3, 4)))
        np.testing.assert_array_equal(q_matched[:, 4:], q)


class IcEstimatorTest(unittest.TestCase):
    def test_ic_is_bce_q_minus_bce_hq(self) -> None:
        self.assertAlmostEqual(ic_hat(0.40, 0.25), 0.15)
        self.assertLess(ic_hat(0.20, 0.30), 0.0)

    def test_seed_formula_matches_script_23(self) -> None:
        self.assertEqual(probe_seed(42, 3, 8), 42 + 3000 + 8)

    def test_summarize_fraction_positive(self) -> None:
        rows = [
            {"tower": "paligemma", "ic_hat": 0.1, "test_bce_q": 0.4, "test_bce_hq": 0.3},
            {"tower": "paligemma", "ic_hat": -0.05, "test_bce_q": 0.4, "test_bce_hq": 0.45},
            {"tower": "expert", "ic_hat": 0.2, "test_bce_q": 0.4, "test_bce_hq": 0.2},
        ]
        summary = summarize_ic(rows)
        self.assertEqual(summary["n"], 3)
        self.assertAlmostEqual(summary["frac_ic_gt_0"], 2 / 3)


class NormalizationAndBinsTest(unittest.TestCase):
    def test_zscore_uses_train_split_only(self) -> None:
        x = np.asarray([[0.0], [10.0], [20.0]], dtype=np.float32)
        train_idx = np.asarray([0, 1])
        z, mean, std = zscore_train(x, train_idx)
        self.assertAlmostEqual(float(mean[0]), 5.0)
        self.assertAlmostEqual(float(z[0, 0]), (0.0 - 5.0) / float(std[0]))

    def test_stored_and_quarter_bin_specs(self) -> None:
        stored = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88]
        self.assertEqual(resolve_bin_indices("stored", stored, 96), stored)
        quarter = resolve_bin_indices("auto-1/4", stored, 96)
        self.assertEqual(len(quarter), 3)
        self.assertTrue(set(quarter).issubset(stored))


class TrainLoopTest(unittest.TestCase):
    def _toy(self):
        rng = np.random.default_rng(0)
        n = 80
        x = rng.normal(size=(n, 8)).astype(np.float32)
        occupancy = (rng.random((n, 16, 16, 16)) > 0.98).astype(np.float32)
        return x, occupancy, np.arange(64), np.arange(64, 80)

    def test_fixed_epochs_records_every_epoch(self) -> None:
        import torch

        x, occupancy, train_idx, test_idx = self._toy()
        result = train_decoder(
            x=x,
            occupancy=occupancy,
            train_idx=train_idx,
            test_idx=test_idx,
            epochs=3,
            batch_size=16,
            learning_rate=2e-3,
            seed=0,
            device=torch.device("cpu"),
            pos_weight=torch.tensor(1.0),
            patience=0,
        )
        self.assertEqual([row["epoch"] for row in result["history"]], [1, 2, 3])
        self.assertFalse(result["early_stopped"])
        self.assertEqual(result["stopped_epoch"], 3)
        self.assertNotIn("history", decoder_csv_metrics(result))
        rows = annotate_history(result["history"], arm="q", probe_seed=0)
        self.assertEqual(rows[0]["arm"], "q")
        self.assertEqual(len(rows), 3)

    def test_early_stop_uses_train_loss_not_a_val_split(self) -> None:
        import torch

        x, occupancy, train_idx, test_idx = self._toy()
        result = train_decoder(
            x=x,
            occupancy=occupancy,
            train_idx=train_idx,
            test_idx=test_idx,
            epochs=20,
            batch_size=16,
            learning_rate=2e-3,
            seed=0,
            device=torch.device("cpu"),
            pos_weight=torch.tensor(1.0),
            patience=2,
            min_delta=1e9,
        )
        self.assertTrue(result["early_stopped"])
        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(result["stopped_epoch"], 3)
        self.assertEqual(len(result["history"]), 3)


class DecoderArchitectureTest(unittest.TestCase):
    def test_mlp_and_linear_shapes(self) -> None:
        import torch
        from torch import nn

        self.assertEqual(normalize_decoder_architecture("MLP"), "mlp")
        self.assertEqual(normalize_decoder_architecture("lin"), "linear")
        self.assertEqual(decoder_description("mlp", 64), "Linear(in,64)-GELU-Linear(64,4096)")
        self.assertEqual(decoder_description("linear"), "Linear(in,4096)")
        mlp = make_decoder(8, torch.device("cpu"), architecture="mlp", hidden_size=64)
        linear = make_decoder(8, torch.device("cpu"), architecture="linear")
        self.assertIsInstance(mlp, nn.Sequential)
        self.assertIsInstance(linear, nn.Linear)
        x = torch.zeros(2, 8)
        self.assertEqual(tuple(mlp(x).shape), (2, 4096))
        self.assertEqual(tuple(linear(x).shape), (2, 4096))
        self.assertEqual(sum(p.numel() for p in linear.parameters()), 8 * 4096 + 4096)

    def test_linear_train_loop_records_architecture(self) -> None:
        import torch

        rng = np.random.default_rng(0)
        x = rng.normal(size=(80, 8)).astype(np.float32)
        occupancy = (rng.random((80, 16, 16, 16)) > 0.98).astype(np.float32)
        result = train_decoder(
            x=x,
            occupancy=occupancy,
            train_idx=np.arange(64),
            test_idx=np.arange(64, 80),
            epochs=1,
            batch_size=16,
            learning_rate=2e-3,
            seed=0,
            device=torch.device("cpu"),
            pos_weight=torch.tensor(1.0),
            architecture="linear",
        )
        self.assertEqual(result["architecture"], "linear")
        self.assertEqual(result["decoder"], "Linear(in,4096)")
        self.assertEqual(len(result["history"]), 1)
        self.assertEqual(result["input_dim"], 8)


if __name__ == "__main__":
    unittest.main()
