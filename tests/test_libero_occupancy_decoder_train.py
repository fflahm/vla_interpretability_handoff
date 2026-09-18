from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.libero_self_occupancy import parse_index_spec
from src.occupancy_decoder import (
    bce_pos_weight,
    iter_split_rows,
    list_extract_conditions,
    load_condition_activations,
    load_occupancy_targets,
    occupancy_bce_dice_loss,
    occupancy_loss,
    save_torch_inplace,
    split_indices,
    sync_tree_inplace,
    train_one_probe,
    load_expert_layer,
    load_paligemma_layer,
)


def _fake_extract(root: Path) -> None:
    occupancy_root = root / "occupancy"
    act_root = root / "activations_root"
    suite = "libero_90"
    task = "KITCHEN_FAKE_stack_bowls"
    split = {
        "seed": 0,
        "suite": suite,
        "frames_per_demo": 4,
        "tasks": [
            {
                "suite": suite,
                "task": task,
                "train": ["demo_0", "demo_1"],
                "test": ["demo_2"],
                "ablation": ["demo_3"],
            }
        ],
    }
    (act_root).mkdir(parents=True)
    (act_root / "split.json").write_text(json.dumps(split), encoding="utf-8")
    (act_root / "capture_config.json").write_text(
        json.dumps({"paligemma_bins": 3, "expert_bins": 2, "flow_times": [1.0, 0.5]}),
        encoding="utf-8",
    )
    rng = np.random.default_rng(0)
    for demo_i, demo in enumerate(["demo_0", "demo_1", "demo_2", "demo_3"]):
        gt_dir = occupancy_root / "gt" / suite / task / demo
        act_dir = act_root / "activations" / suite / task / demo
        gt_dir.mkdir(parents=True)
        act_dir.mkdir(parents=True)
        occ = np.zeros((4, 16, 16, 16), dtype=np.float16)
        occ[:, : 2 + demo_i, :2, :2] = 1
        np.save(gt_dir / "occupancy.npy", occ)
        pali = rng.normal(size=(18, 4, 3, 2048)).astype(np.float16)
        scale = occ.mean(axis=(1, 2, 3)).astype(np.float16)
        pali[:, :, 0, 0] += 3.0 * scale[None, :]
        np.save(act_dir / "paligemma.npy", pali)
        expert = rng.normal(size=(18, 2, 4, 2, 1024)).astype(np.float16)
        np.save(act_dir / "expert.npy", expert)


class ExtractSplitTest(unittest.TestCase):
    def test_train_test_ignore_ablation(self) -> None:
        split = {
            "suite": "libero_90",
            "tasks": [
                {
                    "suite": "libero_90",
                    "task": "task_a",
                    "train": ["demo_0", "demo_1"],
                    "test": ["demo_2"],
                    "ablation": ["demo_3"],
                }
            ],
        }
        rows = iter_split_rows(split, frames=2)
        self.assertEqual(len(rows), 6)
        self.assertEqual({row.split for row in rows}, {"train", "test"})
        self.assertNotIn("demo_3", {row.demo_key for row in rows})
        train_idx, test_idx = split_indices(rows)
        self.assertEqual(len(train_idx), 4)
        self.assertEqual(len(test_idx), 2)
        self.assertTrue(set(train_idx).isdisjoint(test_idx))

    def test_condition_names_match_run_full(self) -> None:
        conditions = list_extract_conditions(
            towers=["paligemma", "expert"],
            layer_indices=[0],
            flow_times=[1.0, 0.8],
        )
        names = [item[0] for item in conditions]
        self.assertEqual(
            names,
            [
                "paligemma/layer_00/static",
                "expert/layer_00/t=1.0",
                "expert/layer_00/t=0.8",
            ],
        )

    def test_bin_spec(self) -> None:
        self.assertEqual(parse_index_spec("0", max_value=97), [0])


class DecoderTrainSmokeTest(unittest.TestCase):
    def test_loss_and_probe_on_fake_extract(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fake_extract(root)
            split = json.loads((root / "activations_root" / "split.json").read_text())
            rows = iter_split_rows(split, frames=4)
            train_idx, test_idx = split_indices(rows)
            occupancy = load_occupancy_targets(root / "occupancy", rows)
            x_all = load_condition_activations(
                root / "activations_root",
                rows,
                tower="paligemma",
                layer=0,
                time_index=None,
                bin_indices=[0],
            )
            self.assertEqual(occupancy.shape, (12, 16, 16, 16))
            self.assertEqual(x_all.shape, (12, 1, 2048))
            self.assertEqual(len(train_idx), 8)
            self.assertEqual(len(test_idx), 4)

            logits = torch.zeros(2, 4096)
            yb = torch.zeros(2, 4096)
            yb[0, :8] = 1
            pos = torch.tensor(2.0)
            loss, value = occupancy_bce_dice_loss(logits, yb, pos)
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(value, 0.0)
            bce_only, bce_value = occupancy_loss(logits, yb, pos, kind="bce")
            self.assertTrue(torch.isfinite(bce_only))
            self.assertLess(bce_value, value)

            payload, history, metrics = train_one_probe(
                x=x_all[:, 0],
                occupancy=occupancy,
                train_idx=train_idx,
                test_idx=test_idx,
                condition="paligemma/layer_00/static",
                bin_index=0,
                column=0,
                epochs=2,
                batch_size=4,
                learning_rate=2e-3,
                bottleneck=64,
                seed=0,
                device="cpu",
                pos_weight_value=bce_pos_weight(occupancy.reshape(len(occupancy), -1)[train_idx]),
            )
            self.assertEqual(len(history), 2)
            self.assertIn("state_dict", payload)
            self.assertTrue(np.isfinite(metrics["soft_iou"]))
            dest = root / "decoder.pt"
            save_torch_inplace(dest, payload)
            self.assertGreater(dest.stat().st_size, 0)

            pali = load_paligemma_layer(
                root / "activations_root",
                rows,
                layer=0,
                bin_indices=[0, 2],
            )
            self.assertEqual(pali.shape, (12, 2, 2048))
            self.assertEqual(pali.dtype, np.float16)
            expert = load_expert_layer(
                root / "activations_root",
                rows,
                layer=0,
                bin_indices=[0],
                n_times=2,
            )
            self.assertEqual(expert.shape, (12, 2, 1, 1024))
            src = root / "sync_src"
            dst = root / "sync_dst"
            src.mkdir()
            (src / "keep.txt").write_text("ok", encoding="utf-8")
            (src / "layer_cache").mkdir()
            (src / "layer_cache" / "big.npy").write_bytes(b"123")
            stats = sync_tree_inplace(src, dst, skip_dir_names=("layer_cache",))
            self.assertEqual(stats["copied"], 1)
            self.assertTrue((dst / "keep.txt").exists())
            self.assertFalse((dst / "layer_cache" / "big.npy").exists())


if __name__ == "__main__":
    unittest.main()
