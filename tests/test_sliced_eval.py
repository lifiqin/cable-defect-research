import unittest

import torch

from cable_research.sliced_eval import combine, merge_predictions, starts, windows


class SlicedEvaluationTests(unittest.TestCase):
    def test_windows_cover_all_pixels_and_handle_small_images(self):
        self.assertEqual(starts(100, 200, 0.25), [0])
        self.assertEqual(starts(4096, 2048, 0.25), [0, 1536, 2048])
        boxes = windows(3072, 4096, 2048, 0.25)
        self.assertEqual(len(boxes), 6)
        self.assertEqual(boxes[-1], (1024, 2048, 3072, 4096))
        for y in range(0, 4096, 101):
            for x in range(0, 3072, 101):
                self.assertTrue(any(l <= x < r and t <= y < b for l, t, r, b in boxes))
        with self.assertRaises(ValueError):
            starts(100, 40, 0.9)

    def test_class_aware_nms_keeps_distinct_classes(self):
        predictions = [{"boxes": torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.]]),
                        "scores": torch.tensor([0.9, 0.8]), "labels": torch.tensor([1, 1])},
                       {"boxes": torch.tensor([[0., 0., 10., 10.]]), "scores": torch.tensor([0.7]),
                        "labels": torch.tensor([2])}]
        result = combine(predictions, nms_iou=0.7, max_dets=300)
        self.assertEqual(result["labels"].tolist(), [1, 2])
        self.assertEqual(combine([], 0.7, 300)["boxes"].shape, (0, 4))

    def test_rtdetr_full_mode_does_not_apply_extra_nms(self):
        prediction = {"boxes": torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.]]),
                      "scores": torch.tensor([0.9, 0.8]), "labels": torch.tensor([1, 1])}
        self.assertEqual(len(merge_predictions([prediction], "rtdetr", "full", 0.7)["boxes"]), 2)
        self.assertEqual(len(merge_predictions([prediction], "yolo", "full", 0.7)["boxes"]), 1)


if __name__ == "__main__":
    unittest.main()
