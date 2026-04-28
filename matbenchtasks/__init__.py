"""TRIADS runners for remaining Matbench tasks."""

import warnings


warnings.filterwarnings(
    "ignore",
    message=r".*PymatgenData\(impute_nan=False\).*",
    category=UserWarning,
    module=r"matminer\.utils\.data",
)

__version__ = "0.1.0"
