from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.online_rollout import Pi05ActivationIntervention
from src.pi05_frame_ablation import (
    OccupancyFrame,
    Pi05OfflineChunkPredictor,
    chunk_delta_l2,
    frame_provenance_table,
    load_occupancy_frames,
    load_selected_bins,
    load_task_hdf5_paths,
    per_frame_good_minus_bad,
    plot_chunk_delta_histograms,
    run_offline_frame_ablation,
    sample_frames,
    summarize_bin_deltas,
)


def _write_png(path: Path) -> None:
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(path)


def _write_jsonl(path: Path, n: int) -> None:
    image_dir = path.parent / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    dummy = image_dir / "dummy.png"
    _write_png(dummy)
    rows = []
    for index in range(n):
        rows.append(
            {
                "sample_id": index,
                "demo_key": f"demo_{index // 3}",
                "frame_index": index,
                "image_path": str(dummy),
                "wrist_image_path": str(dummy),
                "instruction": "pick up the bowl",
                "observation_state": [0.1 * index] * 8,
                "task": "pick_up_the_bowl",
                "suite": "libero_spatial",
                "episode_id": f"libero_spatial/pick_up_the_bowl/demo_{index // 3}",
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _selected(n_bins: int = 4) -> pd.DataFrame:
    rows = []
    for index in range(n_bins):
        rows.append(
            {
                "tower": "paligemma" if index < 2 else "expert",
                "layer_index": index,
                "token_bin_index": 8 * index,
                "probe_group": "best" if index % 2 == 0 else "worst",
                "probe_rank": index + 1,
                "probe_soft_iou": 0.4 - 0.05 * index,
                "pair_id": index // 2 + 1,
                "global_layer_index": index,
            }
        )
    return pd.DataFrame(rows)


class OccupancyFrameSamplingTest(unittest.TestCase):
    def test_loads_jsonl_without_occupancy_and_samples_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _write_jsonl(run_dir / "samples.jsonl", 12)
            frames = load_occupancy_frames(run_dir)
            self.assertEqual(len(frames), 12)
            first = sample_frames(frames, 5, np.random.default_rng(0))
            second = sample_frames(frames, 5, np.random.default_rng(0))
            self.assertEqual([frame.sample_id for frame in first], [frame.sample_id for frame in second])
            self.assertEqual(len({frame.sample_id for frame in first}), 5)
            table = frame_provenance_table(
                first,
                {"pick_up_the_bowl": "/data/pick_up_the_bowl_demo.hdf5"},
            )
            self.assertEqual(list(table["row_index"]), list(range(5)))
            self.assertTrue(
                {"sample_id", "suite", "task", "demo_key", "libero_frame_index", "hdf5_path", "episode_id"}.issubset(
                    table.columns
                )
            )
            row = table.iloc[0]
            self.assertEqual(row["suite"], "libero_spatial")
            self.assertEqual(row["task"], "pick_up_the_bowl")
            self.assertEqual(row["hdf5_path"], "/data/pick_up_the_bowl_demo.hdf5")
            self.assertEqual(int(row["libero_frame_index"]), int(first[0].frame_index))
            self.assertEqual(row["demo_key"], first[0].demo_key)
            self.assertEqual(int(row["sample_id"]), int(first[0].sample_id))


class ChunkDeltaPipelineTest(unittest.TestCase):
    def test_per_bin_chunk_l2_and_compact_outputs(self) -> None:
        selected = _selected(4)
        calls = {"forwards": 0, "select_action": 0}

        def predict(frame, intervention, images):
            calls["forwards"] += 1
            self.assertIsNotNone(images)
            chunk = np.zeros((5, 7), dtype=np.float32)
            chunk[:, 0] = float(frame.sample_id)
            if intervention is None:
                return chunk
            chunk = chunk + 1.0 + 0.25 * intervention.token_bin_index
            if intervention.tower == "expert":
                chunk = chunk + 0.5
            return chunk

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            png = out / "dummy.png"
            _write_png(png)
            frames = [
                OccupancyFrame(
                    sample_id=i,
                    demo_key="d",
                    frame_index=i,
                    image_path=str(png),
                    wrist_image_path=str(png),
                    instruction="do it",
                    observation_state=[0.0] * 8,
                )
                for i in range(6)
            ]
            sample_ids, chunk_l2 = run_offline_frame_ablation(
                frames,
                selected,
                predict,
                checkpoint_path=out / "frame_deltas.npz",
                save_every=2,
            )
            self.assertEqual(calls["select_action"], 0)
            self.assertEqual(calls["forwards"], 6 * (1 + 4))
            self.assertEqual(sample_ids.tolist(), list(range(6)))
            self.assertEqual(chunk_l2.shape, (6, 4))
            expected = chunk_delta_l2(np.zeros((5, 7)) + 1.0, np.zeros((5, 7)))
            self.assertAlmostEqual(float(chunk_l2[0, 0]), expected, places=5)
            packed = np.load(out / "frame_deltas.npz")
            self.assertEqual(
                set(packed.files),
                {"sample_ids", "chunk_l2", "completed_frames", "libero_frame_index"},
            )
            self.assertEqual(packed["libero_frame_index"].tolist(), list(range(6)))
            metrics = summarize_bin_deltas(selected, chunk_l2)
            self.assertEqual(len(metrics), 4)
            plot_chunk_delta_histograms(selected, chunk_l2, out / "figures")
            self.assertTrue((out / "figures" / "01_chunk_delta_hist_good_vs_bad.png").exists())
            self.assertTrue((out / "figures" / "02_chunk_delta_hist_by_tower.png").exists())
            self.assertTrue((out / "figures" / "03_frame_good_minus_bad_hist.png").exists())
            diffs = per_frame_good_minus_bad(selected, chunk_l2)
            self.assertEqual(len(diffs), 6)
            # best bins are even columns with smaller token_bin_index than their worst pair.
            self.assertTrue(np.isfinite(diffs["good_minus_bad"]).all())

    def test_selected_csv_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selected.csv"
            _selected(3).to_csv(path, index=False)
            loaded = load_selected_bins(path)
            self.assertEqual(list(loaded["token_bin_index"]), [0, 8, 16])


class FreshChunkPredictorTest(unittest.TestCase):
    def test_resets_and_never_calls_select_action(self) -> None:
        layer = torch.nn.Identity()
        counts = {"reset": 0, "chunk": 0, "select": 0}

        class Policy(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                paligemma = torch.nn.Module()
                paligemma.language_model = torch.nn.Module()
                paligemma.language_model.layers = torch.nn.ModuleList([layer])
                expert = torch.nn.Module()
                expert.model = torch.nn.Module()
                expert.model.layers = torch.nn.ModuleList([torch.nn.Identity()])
                bundle = torch.nn.Module()
                bundle.paligemma = paligemma
                bundle.gemma_expert = expert
                self.model = torch.nn.Module()
                self.model.paligemma_with_expert = bundle

            def reset(self) -> None:
                counts["reset"] += 1

            def predict_action_chunk(self, batch):
                counts["chunk"] += 1
                _ = layer(torch.zeros(1, 8, 3))
                return torch.zeros(1, 4, 7)

            def select_action(self, batch):
                counts["select"] += 1
                raise AssertionError("select_action would reuse the previous chunk")

        wrapper = SimpleNamespace(
            torch=torch,
            device_obj=torch.device("cpu"),
            policy=Policy(),
            preprocess=lambda frame: frame,
            postprocess=lambda value: value,
            _make_frame=lambda image, instruction, metadata: {"ok": True},
        )
        predictor = Pi05OfflineChunkPredictor("unused", wrapper=wrapper)
        frame = OccupancyFrame(0, "d", 0, "a.png", "w.png", "task", [0.0] * 8)
        images = {"image": np.zeros((4, 4, 3), dtype=np.uint8), "wrist_image": np.zeros((4, 4, 3), dtype=np.uint8)}
        chunk = predictor.predict(frame, None, images)
        self.assertEqual(chunk.shape, (4, 7))
        intervened = predictor.predict(
            frame,
            Pi05ActivationIntervention("paligemma", 0, 0, token_bins=8, mode="zero"),
            images,
        )
        self.assertEqual(intervened.shape, (4, 7))
        self.assertEqual(counts["reset"], 2)
        self.assertEqual(counts["chunk"], 2)
        self.assertEqual(counts["select"], 0)


class TaskHdf5LookupTest(unittest.TestCase):
    def test_reads_gt_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "gt_metadata.json").write_text(
                json.dumps(
                    {
                        "tasks": [
                            {
                                "hdf5_path": "/libero/libero_spatial/pick_up_the_bowl_demo.hdf5",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            mapping = load_task_hdf5_paths(run_dir)
            self.assertEqual(
                mapping["pick_up_the_bowl"],
                "/libero/libero_spatial/pick_up_the_bowl_demo.hdf5",
            )


class ChunkDeltaMathTest(unittest.TestCase):
    def test_flattens_full_chunk(self) -> None:
        baseline = np.zeros((2, 3), dtype=np.float32)
        ablated = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=np.float32)
        self.assertAlmostEqual(chunk_delta_l2(ablated, baseline), 5.0)
        with self.assertRaises(ValueError):
            chunk_delta_l2(np.zeros((2, 3)), np.zeros((3, 2)))


if __name__ == "__main__":
    unittest.main()
