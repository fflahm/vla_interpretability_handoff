from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from src.online_rollout import Pi05ActivationIntervention, _apply_pi05_activation_intervention
from src.pi05_probe_ablation import (
    compare_rollouts,
    select_layer_matched_probe_pairs,
    select_probe_conditions,
)


class ProbeConditionSelectionTest(unittest.TestCase):
    def test_selects_disjoint_extremes_and_averages_expert_flow_times(self) -> None:
        rows = []
        for tower in ("paligemma", "expert"):
            for layer in range(3):
                for bin_index in range(4):
                    base = layer * 4 + bin_index
                    flows = (None,) if tower == "paligemma" else (0.1, 0.5, 1.0)
                    for flow in flows:
                        condition = (
                            f"paligemma/layer_{layer:02d}/static"
                            if flow is None
                            else f"expert/layer_{layer:02d}/t={flow:.1f}"
                        )
                        rows.append({"condition": condition, "bin": bin_index, "soft_iou": float(base)})
        selected = select_probe_conditions(pd.DataFrame(rows), per_tower=2)
        self.assertEqual(len(selected), 8)
        self.assertEqual(set(selected.groupby(["tower", "probe_group"]).size()), {2})
        expert = selected[selected["tower"] == "expert"]
        self.assertTrue((expert["probe_measurements"] == 3).all())
        self.assertFalse(
            selected.duplicated(["tower", "layer_index", "token_bin_index"]).any()
        )


class LayerMatchedSelectionTest(unittest.TestCase):
    def _metrics(self) -> pd.DataFrame:
        rows = []
        for tower in ("paligemma", "expert"):
            for layer in range(10):
                for bin_index in (0, 8, 12, 16, 20, 24, 32, 40, 48, 80, 88):
                    # 0/8/88 stay low so two bad bins still leave a 0.03 IoU gap
                    # after the good-bin repeat cap. Pali bin 80 is globally best
                    # so the cap can trigger; Expert >48 is inflated so the
                    # expert_max_bin filter is observable.
                    if bin_index == 0:
                        score = 0.04 + 0.001 * layer
                    elif bin_index == 8:
                        score = 0.06 + 0.001 * layer
                    elif bin_index == 88:
                        score = 0.02
                    else:
                        score = 0.20 + 0.02 * bin_index / 8.0 + 0.002 * layer
                    if tower == "expert" and bin_index > 48:
                        score = 0.99
                    condition = (
                        f"paligemma/layer_{layer:02d}/static"
                        if tower == "paligemma"
                        else f"expert/layer_{layer:02d}/t=0.5"
                    )
                    rows.append({"condition": condition, "bin": bin_index, "soft_iou": score})
        return pd.DataFrame(rows)

    def test_pairs_share_layer_and_respect_constraints(self) -> None:
        selected = select_layer_matched_probe_pairs(
            self._metrics(),
            pairs_per_tower=5,
            bins_per_group=2,
            min_iou_gap=0.03,
            max_good_bin_repeats=2,
            expert_max_bin=48,
        )
        self.assertEqual(len(selected), 40)
        for tower, subset in selected.groupby("tower"):
            self.assertEqual(subset["layer_index"].nunique(), 5)
            pairs = subset.groupby("pair_id")
            self.assertEqual(len(pairs), 10)
            for _, pair in pairs:
                layers = set(pair["layer_index"])
                self.assertEqual(len(layers), 1)
                self.assertEqual(set(pair["probe_group"]), {"best", "worst"})
                self.assertEqual(len(pair), 2)
                gap = float(pair["iou_gap"].iloc[0])
                self.assertGreaterEqual(gap, 0.03)
            for _, layer_rows in subset.groupby("layer_index"):
                self.assertEqual(int((layer_rows["probe_group"] == "best").sum()), 2)
                self.assertEqual(int((layer_rows["probe_group"] == "worst").sum()), 2)
            good_bins = subset.loc[subset["probe_group"] == "best", "token_bin_index"]
            self.assertLessEqual(int((good_bins == 80).sum()), 2)
        expert_bins = selected.loc[selected["tower"] == "expert", "token_bin_index"]
        self.assertTrue((expert_bins <= 48).all())
        layers = sorted(selected.loc[selected["tower"] == "paligemma", "layer_index"].unique())
        self.assertGreaterEqual(layers[-1] - layers[0], 4)
        self.assertEqual(set(selected["selection_mode"]), {"layer_matched"})
        self.assertFalse(
            selected.duplicated(["tower", "layer_index", "token_bin_index"]).any()
        )


class Pi05InterventionTest(unittest.TestCase):
    def test_zeroes_only_requested_tower_layer_and_effective_bin(self) -> None:
        tensor = torch.ones(1, 50, 3)
        intervention = Pi05ActivationIntervention(
            tower="expert", layer_index=2, token_bin_index=8, token_bins=96
        )
        untouched = _apply_pi05_activation_intervention(tensor, "expert_layer_01", intervention)
        self.assertIs(untouched, tensor)
        modified = _apply_pi05_activation_intervention(tensor, "expert_layer_02", intervention)
        self.assertTrue(torch.equal(modified[:, 8, :], torch.zeros(1, 3)))
        self.assertEqual(int(torch.count_nonzero(modified).item()), 49 * 3)

    def test_rejects_bin_outside_short_expert_sequence(self) -> None:
        intervention = Pi05ActivationIntervention(
            tower="expert", layer_index=0, token_bin_index=50, token_bins=96
        )
        with self.assertRaisesRegex(ValueError, "effective bins 50"):
            _apply_pi05_activation_intervention(
                torch.ones(1, 50, 2), "expert_layer_00", intervention
            )


class RolloutComparisonTest(unittest.TestCase):
    def test_compares_matched_episode_steps(self) -> None:
        baseline = {
            "summary": {"success_rate": 1.0},
            "steps": [
                {
                    "episode_index": 0,
                    "step": 0,
                    "action": [0.0, 0.0],
                    "gripper_position": [0.0, 0.0, 0.0],
                }
            ],
        }
        observed = {
            "summary": {"success_rate": 0.0},
            "steps": [
                {
                    "episode_index": 0,
                    "step": 0,
                    "action": [3.0, 4.0],
                    "gripper_position": [0.0, 0.0, 2.0],
                }
            ],
        }
        result = compare_rollouts(baseline, observed)
        self.assertAlmostEqual(result["mean_policy_action_delta_l2"], 5.0)
        self.assertAlmostEqual(result["mean_env_action_delta_l2"], 5.0)
        self.assertAlmostEqual(result["mean_gripper_position_delta_l2"], 2.0)
        self.assertAlmostEqual(result["causal_impact_score"], 22.0)


if __name__ == "__main__":
    unittest.main()
