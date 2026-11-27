import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from cable_research.p2 import yolo11_p2_config


class P2Tests(unittest.TestCase):
    def test_p2_has_four_detection_strides_without_mutating_base(self):
        from ultralytics import YOLO

        torch.set_num_threads(2)
        base = YOLO("yolo11n.yaml").model.yaml
        output = yolo11_p2_config(base)
        self.assertEqual(base["head"][-1][0], [16, 19, 22])
        self.assertEqual(output["head"][-1][0], [19, 22, 25, 28])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "yolo11n-p2.yaml"
            path.write_text(yaml.safe_dump(output, sort_keys=False), encoding="utf-8")
            model = YOLO(path).model
            self.assertEqual(model.stride.tolist(), [4.0, 8.0, 16.0, 32.0])

    def test_unexpected_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "YOLO11"):
            yolo11_p2_config({"backbone": [], "head": []})


if __name__ == "__main__":
    unittest.main()
