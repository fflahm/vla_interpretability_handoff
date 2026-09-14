from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.libero_annotation_report import combined_record, render_markdown, select_frame


class LiberoAnnotationReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        episode = {
            "episode_id": "libero_spatial/task/demo_1",
            "instruction": "put the bowl on the plate",
            "phase_anchors": {"grasp": 5},
            "simulator": {"provenance": "sim_replay"},
        }
        frame = {
            "annotation_version": "test",
            "sample": {
                "episode_id": episode["episode_id"],
                "suite": "libero_spatial",
                "task": "task",
                "demo": "demo_1",
                "frame": 7,
                "T": 20,
                "progress": 7 / 19,
                "selection_reasons": ["time_quantile"],
            },
            "images": {},
            "objects": [],
            "spatial": {
                "eef_object": [],
                "goal_relevant_pairs": {"value": [], "valid": True, "provenance": "sim"},
                "contact_graph": {"value": [], "valid": True, "provenance": "sim"},
                "contacts": {"value": [], "valid": True, "provenance": "sim"},
            },
            "goal": {
                "predicates": [{"name": "On", "arguments": ["bowl", "plate"]}],
                "predicate_status": {
                    "value": [{"predicate": ["On", "bowl", "plate"], "satisfied": False}],
                    "valid": True,
                    "provenance": "sim",
                },
                "satisfied_fraction": {"value": 0.0, "valid": True, "provenance": "sim"},
                "all_success": {"value": False, "valid": True, "provenance": "sim"},
            },
            "phase": {
                "index": 3,
                "detail": "grasp",
                "family": "spatial_pick_place",
                "provenance": "sim_replay_events",
                "evidence_count": 3,
                "anchor": [],
            },
            "body": {},
            "action": {
                "raw_7d": [0, 0, 0, 0, 0, 0, 1],
                "primitive_labels": ["close_gripper"],
                "chunk_mask": [True],
                "chunk_horizon": 1,
            },
            "future": {"horizons": {}},
            "quality": {
                "direct_observation_valid": True,
                "simulator_valid": True,
                "provenance": ["hdf5", "sim"],
                "warnings": [],
            },
        }
        (self.root / "episodes.jsonl").write_text(json.dumps(episode) + "\n", encoding="utf-8")
        (self.root / "frames.jsonl").write_text(json.dumps(frame) + "\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_join_and_markdown(self) -> None:
        episode, frame = combined_record(self.root, sample_index=0)
        report = render_markdown(episode, frame, self.root)
        self.assertIn("LIBERO frame report", report)
        self.assertIn("put the bowl on the plate", report)
        self.assertIn("On", report)
        self.assertIn("close_gripper", report)
        self.assertNotIn("Raw joined record", report)
        raw_report = render_markdown(episode, frame, self.root, include_raw=True)
        self.assertIn("Raw joined record", raw_report)

    def test_select_by_episode_and_frame(self) -> None:
        frame = select_frame(
            self.root / "frames.jsonl",
            episode_id="libero_spatial/task/demo_1",
            frame_index=7,
        )
        self.assertEqual(frame["phase"]["index"], 3)


if __name__ == "__main__":
    unittest.main()
