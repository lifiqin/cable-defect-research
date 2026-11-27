import json
import tempfile
import unittest
from pathlib import Path

from cable_research.prepare_external import materialize
from cable_research.split import create_split


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        (self.data / "images/train").mkdir(parents=True)
        (self.data / "labels/train").mkdir(parents=True)
        self.folds = [["a0", "a1"], ["b0", "b1"], ["c0", "c1"]]
        stems = [s for fold in self.folds for s in fold]
        (self.data / "train_ready.txt").write_text("\n".join(f"images/train/{s}.jpg" for s in stems), encoding="utf-8")
        for stem in stems:
            (self.data / "images/train" / f"{stem}.jpg").write_bytes(b"test")
            (self.data / "labels/train" / f"{stem}.txt").write_text(f"{stem[-1]} 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        self.plan = {"folds": self.folds, "groups": [{"fold": i, "stems": fold[:]} for i, fold in enumerate(self.folds, 1)]}
        self.plan_file = self.root / "plan.json"
        self.save_plan()

    def save_plan(self):
        self.plan_file.write_text(json.dumps(self.plan), encoding="utf-8")

    def test_no_groups_cross_train_val(self):
        yaml_file, counts = create_split(self.data, self.plan_file, self.root / "out", 2)
        self.assertTrue(yaml_file.is_file())
        self.assertEqual(counts["train"]["images"], 4)
        self.assertEqual(counts["val"]["images"], 2)
        train = (yaml_file.parent / "train.txt").read_text(encoding="utf-8")
        val = (yaml_file.parent / "val.txt").read_text(encoding="utf-8")
        self.assertNotIn("b0.jpg", train)
        self.assertIn("b0.jpg", val)
        self.assertNotIn("/images/test/", train + val)

    def test_overlapping_folds_fail_before_creating_output(self):
        self.plan["folds"][0].append("b0")
        self.save_plan()
        out = self.root / "out"
        with self.assertRaisesRegex(ValueError, "more than one fold"):
            create_split(self.data, self.plan_file, out, 2)
        self.assertFalse(out.exists())

    def test_group_spanning_folds_is_rejected(self):
        self.plan["groups"][0]["stems"].append("b0")
        self.save_plan()
        with self.assertRaisesRegex(ValueError, "group straddles"):
            create_split(self.data, self.plan_file, self.root / "out", 2)

    def test_out_of_bounds_box_fails_before_writing_manifests(self):
        (self.data / "labels/train/a0.txt").write_text("0 0.01 0.5 0.9 0.1\n", encoding="utf-8")
        out = self.root / "out"
        with self.assertRaisesRegex(ValueError, "extends beyond"):
            create_split(self.data, self.plan_file, out, 2)
        self.assertFalse(out.exists())

    def test_external_layout_keeps_fold_separation_and_bytes(self):
        out = self.root / "external"
        counts = materialize(self.data, self.plan_file, out, 2)
        self.assertEqual((counts["train"]["images"], counts["val"]["images"]), (4, 2))
        self.assertTrue((out / "data.yaml").is_file())
        self.assertEqual((out / "valid/images/b0.jpg").read_bytes(), b"test")
        self.assertFalse((out / "train/images/b0.jpg").exists())
        self.assertFalse((out / "test").exists())


if __name__ == "__main__":
    unittest.main()
