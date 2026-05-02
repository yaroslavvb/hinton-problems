"""
Stub-spec interface for the transforming-pairs problem.

Implementation lives in `transforming_pairs.py`. This shim keeps the
spec-mandated function names available for tools that import them.
"""

from transforming_pairs import (
    generate_transformed_pairs,
    build_gated_rbm,
    train,
    visualize_transformation_filters,
)

__all__ = [
    "generate_transformed_pairs",
    "build_gated_rbm",
    "train",
    "visualize_transformation_filters",
]


if __name__ == "__main__":
    from transforming_pairs import main
    main()
