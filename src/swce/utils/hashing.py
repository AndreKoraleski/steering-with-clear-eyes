"""Content hashing used to pin benchmark files and to version protocols."""

import hashlib
import json
from typing import Any


def canonical_sha256(value: Any) -> str:
    """SHA-256 of the canonical JSON encoding of a value.

    The encoding sorts mapping keys and omits whitespace, so that equal values hash equally
    whatever the order in which their keys were inserted.

    Args:
        value: Value made of JSON serializable types.

    Returns:
        The hexadecimal digest of the UTF-8 encoded JSON text.

    Raises:
        TypeError: If `value` holds an object that JSON cannot encode.
    """
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()
