"""Build a YOLO11 P2--P5 comparison topology from the installed base config.

The P2 idea is established prior work; this is an experiment, not an original
architecture claim. Pretrained transfer is partial because the head changes.
"""

import copy


def yolo11_p2_config(base):
    if (len(base.get("backbone", [])) != 11 or len(base.get("head", [])) != 13 or
            base["backbone"][2][2] != "C3k2" or base["head"][-1][2] != "Detect" or
            base.get("scale") not in {"n", "s", "m", "l", "x"}):
        raise ValueError("expected an unmodified scaled YOLO11 architecture")
    cfg = copy.deepcopy(base)
    cfg.pop("yaml_file", None)
    cfg["head"] = [
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 6], 1, "Concat", [1]],
        [-1, 2, "C3k2", [512, False]],
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 4], 1, "Concat", [1]],
        [-1, 2, "C3k2", [256, False]],
        [-1, 1, "nn.Upsample", [None, 2, "nearest"]],
        [[-1, 2], 1, "Concat", [1]],
        [-1, 2, "C3k2", [128, False]],
        [-1, 1, "Conv", [128, 3, 2]],
        [[-1, 16], 1, "Concat", [1]],
        [-1, 2, "C3k2", [256, False]],
        [-1, 1, "Conv", [256, 3, 2]],
        [[-1, 13], 1, "Concat", [1]],
        [-1, 2, "C3k2", [512, False]],
        [-1, 1, "Conv", [512, 3, 2]],
        [[-1, 10], 1, "Concat", [1]],
        [-1, 2, "C3k2", [1024, True]],
        [[19, 22, 25, 28], 1, "Detect", ["nc"]],
    ]
    return cfg
