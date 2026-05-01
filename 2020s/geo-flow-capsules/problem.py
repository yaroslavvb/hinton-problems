"""
Geo / Geo+ moving 2D shapes for flow capsules
(Sabour, Tagliasacchi, Yazdani, Hinton & Fleet 2021).
"""

import numpy as np


def generate_geo_pair(h: int = 64, w: int = 64, n_shapes: int = 3):
    """Render two consecutive frames with known affine flow + ground-truth flow map."""
    raise NotImplementedError


def build_flow_capsule_net():
    raise NotImplementedError


def train_unsupervised(model, data, n_steps: int, lr: float):
    raise NotImplementedError


def part_segmentation_iou(model, data) -> float:
    raise NotImplementedError


if __name__ == "__main__":
    pass
