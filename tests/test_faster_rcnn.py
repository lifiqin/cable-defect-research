import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from cable_research.faster_rcnn import ScratchDataset, UltralyticsAP, collate, decode_yolo


class FasterRCNNTests(unittest.TestCase):
    def test_ultralytics_ap_matches_by_class_and_iou(self):
        metric = UltralyticsAP()
        metric.update({"boxes": torch.tensor([[0., 0., 10., 10.]]), "labels": torch.tensor([1]),
                       "scores": torch.tensor([0.9])},
                      {"boxes": torch.tensor([[0., 0., 10., 10.], [20., 20., 30., 30.]]),
                       "labels": torch.tensor([1, 2])})
        result = metric.compute()
        self.assertGreater(result["class_map50_95"]["1"], 0.99)
        self.assertEqual(result["class_map50_95"]["2"], 0.0)
        self.assertGreater(result["map50"], 0.49)
        self.assertLess(result["map50"], 0.51)

    def test_ultralytics_ap_handles_no_predictions(self):
        metric = UltralyticsAP()
        metric.update({"boxes": torch.empty((0, 4)), "labels": torch.empty(0, dtype=torch.int64),
                       "scores": torch.empty(0)},
                      {"boxes": torch.tensor([[0., 0., 10., 10.]]), "labels": torch.tensor([1])})
        self.assertEqual(metric.compute()["map50_95"], 0.0)

    def test_decode_maps_background_and_clamps_rounding(self):
        boxes, labels = decode_yolo(["0 0.5 0.5 1.0000000 0.5", "1 0.5 0.5 0.2 0.4"], 100, 50)
        self.assertTrue(torch.equal(labels, torch.tensor([1, 2])))
        self.assertTrue(torch.allclose(boxes[0], torch.tensor([0.0, 12.5, 100.0, 37.5])))
        self.assertEqual(decode_yolo([], 100, 50)[0].shape, (0, 4))
        with self.assertRaises(ValueError):
            decode_yolo(["0 0.0 0.5 0.5 0.5"], 100, 50)

    def test_flip_and_collation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "labels/train").mkdir(parents=True)
            image = root / "a.png"
            Image.new("RGB", (100, 50)).save(image)
            (root / "labels/train/a.txt").write_text("0 0.2 0.5 0.2 0.4\n", encoding="utf-8")
            image_list = root / "train.txt"
            image_list.write_text(str(image) + "\n", encoding="utf-8")
            dataset = ScratchDataset(image_list, root, 100, augment=True)
            with patch("cable_research.faster_rcnn.random.random", return_value=0.0):
                pixels, target = dataset[0]
            self.assertEqual(pixels.shape, (3, 50, 100))
            self.assertTrue(torch.allclose(target["boxes"][0], torch.tensor([70.0, 15.0, 90.0, 35.0])))
            images, targets = collate([(pixels, target)])
            self.assertEqual(len(images), len(targets))


if __name__ == "__main__":
    unittest.main()
