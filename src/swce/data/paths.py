"""Location of the local copies of the benchmark files."""

import os
from pathlib import Path

DATA_ENV = "SWCE_DATA"
"""Environment variable naming the data root."""


def default_data_root() -> Path:
    """Directory that holds the local copies of the benchmark files and images.

    Returns:
        The path named by the `SWCE_DATA` environment variable, or the relative path `data`,
        resolved against the working directory, when the variable is unset.
    """
    return Path(os.environ.get(DATA_ENV, "data"))


def artifacts_root() -> Path:
    """Directory that holds what the recipes of the methods produce.

    Fitted directions, subspaces and calibrations are written here, one file per method and
    model. The directory sits under the data root, so that one folder shared between machines
    carries both the benchmark files and the outcome of every recipe.

    Returns:
        The folder `artifacts` under the data root.
    """
    return default_data_root() / "artifacts"
