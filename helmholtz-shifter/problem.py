"""
Stub-spec interface for the Helmholtz-shifter problem.

Implementation lives in `helmholtz_shifter.py`. This shim keeps the
spec-mandated function names available for tools that import them.
"""

from helmholtz_shifter import (
    generate_shifter as generate_dataset,
    build_helmholtz_machine,
    wake_sleep,
    inspect_layer3_units,
)

__all__ = [
    "generate_dataset",
    "build_helmholtz_machine",
    "wake_sleep",
    "inspect_layer3_units",
]


if __name__ == "__main__":
    from helmholtz_shifter import main
    main()
