import json
import unittest
from unittest import mock

import numpy as np

from custom_nodes.Yeban_MOSS_BBox import moss_bbox_v3


class MossBBoxV3ShotCacheTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((48, 32, 3), dtype=np.uint8)
        self.cache = {}
        self.backend_calls = []
        self.facts = {
            "P1": "A seated person in a blue dress, torso leaning against a counter.",
            "P1_source_crop": "The full figure is visible with a little floor beneath the feet.",
            "O1": "A dark counter beside the person.",
            "background_caption": "A warm interior with a counter and ceiling lamp.",
            "smart_reverse_prompt": "A faithful source-image prompt describing the seated pose and blue dress.",
        }
        self.patches = [
            mock.patch.object(moss_bbox_v3.nodes, "_normalise_rgb_batch", return_value=[self.image]),
            mock.patch.object(moss_bbox_v3, "resolve_backend", return_value=("model-x", "url", "key", "profile-x")),
            mock.patch.object(moss_bbox_v3, "detect_boxes", return_value=([{"x": 5, "y": 4, "width": 16, "height": 40}], [{"x": 2, "y": 20, "width": 28, "height": 8}])),
            mock.patch.object(moss_bbox_v3.nodes, "_read_cache", side_effect=lambda key, schema_version: self.cache.get(key)),
            mock.patch.object(moss_bbox_v3.nodes, "_write_cache", side_effect=lambda key, data, schema_version: self.cache.__setitem__(key, data)),
            mock.patch.object(moss_bbox_v3.nodes, "_response_text_for_repair", side_effect=lambda raw: raw),
            mock.patch.object(moss_bbox_v3.nodes, "_moss_layout_draft_to_prompt", side_effect=self.compile_layout),
            mock.patch.object(moss_bbox_v3.nodes, "_call_backend", side_effect=self.call_backend),
        ]
        for patcher in self.patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def call_backend(self, instruction, image, **kwargs):
        self.backend_calls.append(instruction)
        return json.dumps(self.facts)

    @staticmethod
    def compile_layout(draft_text, width, height, **kwargs):
        draft = json.loads(draft_text)
        return json.dumps({
            "width": width, "height": height,
            "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx",
            "summary": {"caption": draft["summary"]["caption"], "canvas": "single"},
            "items": [
                {"label": "object", "caption": draft["background_caption"],
                 "bbox": [-height // 2, -width // 2, height // 2, width // 2]},
                *[
                    {"label": item["label"], "caption": item["caption"],
                     "bbox": [-height * 0.4, -width * 0.2, height * 0.4, width * 0.2]}
                    for item in draft["items"]
                ],
            ],
        })

    def run_node(self, shot_size, use_cache=True, additional_requirements=""):
        return moss_bbox_v3.YebanMossBBoxStandaloneV3().write_layout(
            image=self.image, layout_aspect="reference_aspect", layout_long_edge=64,
            shot_size=shot_size, backend="local_grok",
            backend_model="model-x", custom_model="", reasoning="low",
            subject_setting="", photography_style="", additional_requirements=additional_requirements,
            timeout=30, use_cache=use_cache, cache_seed=0, target_word_count=500,
        )

    def test_local_llama_backend_and_model_inputs_are_exposed(self):
        schema = moss_bbox_v3.YebanMossBBoxStandaloneV3.INPUT_TYPES()
        self.assertIn("local_llama", schema["required"]["backend"][0])
        self.assertIn("llama_model", schema["optional"])
        self.assertIn("llama_parameters", schema["optional"])

    def test_changing_shot_size_reuses_source_facts_and_smart_prompt(self):
        reference = self.run_node("reference")
        wide = self.run_node("wide")

        self.assertEqual(len(self.backend_calls), 1)
        reference_layout = json.loads(reference[0])
        wide_layout = json.loads(wide[0])
        self.assertNotEqual(reference_layout, wide_layout)
        reference_person = next(item for item in reference_layout["items"] if item["label"] == "character")
        wide_person = next(item for item in wide_layout["items"] if item["label"] == "character")
        self.assertIn("Original source framing: The full figure is visible", reference_person["caption"])
        self.assertNotIn("Original source framing:", wide_person["caption"])
        self.assertEqual(reference[3], wide[3])
        self.assertIn("person item captions remain crop-neutral", self.backend_calls[0])
        self.assertNotIn("Selected shot size", self.backend_calls[0])

    def test_shot_sizes_apply_the_approved_frame_height_targets(self):
        expected = {"close": 0.90, "environmental": 0.58, "wide": 0.34}
        for shot_size, target in expected.items():
            with self.subTest(shot_size=shot_size):
                layout = {
                    "summary": {"caption": "A room.", "canvas": "single"},
                    "items": [
                        {"label": "object", "caption": "Room", "bbox": [-50, -50, 50, 50]},
                        {"label": "character", "caption": "Person", "bbox": [-40, -20, 40, 20]},
                        {"label": "object", "caption": "Seat", "bbox": [-10, 25, 20, 45]},
                    ],
                }
                background_box = layout["items"][0]["bbox"][:]
                moss_bbox_v3._apply_shot_size(layout, shot_size, "center", 100, 100)
                person = next(item["bbox"] for item in layout["items"] if item["label"] == "character")
                self.assertEqual(person[2] - person[0], round(100 * target))
                self.assertEqual(layout["items"][0]["bbox"], background_box)
                self.assertIn(f"about {target:.0%}", layout["summary"]["caption"])

    def test_shot_layout_keeps_character_last_and_locks_pose_before_scene_details(self):
        for shot_size in ("close", "environmental", "wide"):
            with self.subTest(shot_size=shot_size):
                layout = {
                    "summary": {"caption": "A room.", "canvas": "single"},
                    "items": [
                        {"label": "object", "caption": "Room", "bbox": [-50, -50, 50, 50]},
                        {"label": "character", "caption": "Person", "bbox": [-40, -20, 40, 20]},
                        {"label": "object", "caption": "Seat", "bbox": [-10, 25, 20, 45]},
                    ],
                }
                moss_bbox_v3._apply_shot_size(layout, shot_size, "center", 100, 100)
                self.assertEqual(layout["items"][0]["caption"], "Room")
                self.assertEqual(layout["items"][-1]["label"], "character")
                self.assertIn("Change only the person's scale", layout["summary"]["caption"])
                self.assertIn("Preserve the source pose", layout["summary"]["caption"])

    def test_every_shot_outputs_the_bbox_layout_instead_of_discarding_coordinates(self):
        for shot_size in ("reference", "close", "environmental", "wide"):
            with self.subTest(shot_size=shot_size):
                prompt, _, _, _ = self.run_node(shot_size)
                layout = json.loads(prompt)
                self.assertEqual(layout["bbox_coord_space"], "abs_centralized")
                self.assertEqual(layout["bbox_format"], "yxyx")
                self.assertTrue(any(item["label"] == "character" for item in layout["items"]))

    def test_reference_keeps_original_person_box_scale(self):
        layout = {
            "summary": {"caption": "A room.", "canvas": "single"},
            "items": [{"label": "character", "caption": "Person", "bbox": [-30, -20, 50, 20]}],
        }
        original_box = layout["items"][0]["bbox"][:]
        moss_bbox_v3._apply_shot_size(layout, "reference", "center", 100, 100)
        self.assertEqual(layout["items"][0]["bbox"], original_box)

    def test_enlarged_character_boxes_are_shifted_inside_canvas_without_changing_target_height(self):
        layout = {
            "summary": {"caption": "A room.", "canvas": "single"},
            "items": [
                {"label": "object", "caption": "Room", "bbox": [-50, -50, 50, 50]},
                {"label": "character", "caption": "Person", "bbox": [10, -10, 30, 10]},
                {"label": "object", "caption": "Lamp", "bbox": [-45, 35, -30, 49]},
            ],
        }
        moss_bbox_v3._apply_shot_size(layout, "close", "center", 100, 100)
        person = next(item["bbox"] for item in layout["items"] if item["label"] == "character")
        self.assertEqual(person[2] - person[0], 90)
        self.assertGreaterEqual(person[0], -50)
        self.assertLessEqual(person[2], 50)
        self.assertGreaterEqual(person[1], -50)
        self.assertLessEqual(person[3], 50)
        for item in layout["items"]:
            top, left, bottom, right = item["bbox"]
            self.assertGreaterEqual(top, -50)
            self.assertGreaterEqual(left, -50)
            self.assertLessEqual(bottom, 50)
            self.assertLessEqual(right, 50)

    def test_disabling_cache_still_repeats_backend_call(self):
        self.run_node("reference", use_cache=False)
        self.run_node("wide", use_cache=False)
        self.assertEqual(len(self.backend_calls), 2)

    def test_changed_fact_instruction_invalidates_cached_analysis(self):
        self.run_node("reference")
        self.run_node("wide", additional_requirements="Keep the background brighter.")
        self.assertEqual(len(self.backend_calls), 2)


if __name__ == "__main__":
    unittest.main()
