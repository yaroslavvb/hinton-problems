"""
Geo / Geo+ moving 2D shapes for flow capsules
(Sabour, Tagliasacchi, Yazdani, Hinton & Fleet 2021).

Spec-compatible re-export shim. The implementation lives in
`geo_flow_capsules.py`; we expose the four required entry points here.
"""

from geo_flow_capsules import (
    build_flow_capsule_net,
    fit_flow_capsules,
    generate_geo_pair,
    part_segmentation_iou,
    train_unsupervised,
)


__all__ = [
    "build_flow_capsule_net",
    "fit_flow_capsules",
    "generate_geo_pair",
    "part_segmentation_iou",
    "train_unsupervised",
]


if __name__ == "__main__":
    from geo_flow_capsules import main as _main
    _main()
