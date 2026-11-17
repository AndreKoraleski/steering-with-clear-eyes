"""Loading of the registry committed with the package."""

from functools import cache
from pathlib import Path

from swce.registry.schema import Registry

REGISTRY_PATH = Path(__file__).parent / "data" / "registry.json"
"""File holding the registry."""


@cache
def load_registry(path: Path = REGISTRY_PATH) -> Registry:
    """Registry read from its file and validated.

    Args:
        path: File holding the registry.

    Returns:
        The validated registry, cached per path.

    Raises:
        pydantic.ValidationError: If the file does not follow the schema.
    """
    return Registry.model_validate_json(path.read_text(encoding="utf-8"))
