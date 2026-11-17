"""Check that the hosts still serve the pinned files.

Small files are downloaded whole and hashed. Large ones are only asked for their size, which is
enough to notice a replaced archive without downloading it every week.
"""

import hashlib
import urllib.request
from enum import StrEnum

from swce.data.download import TIMEOUT
from swce.registry.schema import PinnedFile

FULL_CHECK_BYTES = 64 << 20
"""Largest file that is downloaded whole to be verified."""


class RemoteState(StrEnum):
    """Outcome of comparing a host with a pin.

    Attributes:
        OK: The host serves the pinned content, or the pinned size for a large file.
        CHANGED: The host serves something else.
        UNREACHABLE: The host did not answer.
    """

    OK = "ok"
    CHANGED = "changed"
    UNREACHABLE = "unreachable"


def remote_state(file: PinnedFile) -> RemoteState:
    """Compare what a host serves with a pin.

    Args:
        file: Pinned file to check.

    Returns:
        The state of the file on its host.
    """
    try:
        if file.size <= FULL_CHECK_BYTES:
            with urllib.request.urlopen(file.url, timeout=TIMEOUT) as response:
                content = response.read()
            same = (
                len(content) == file.size
                and hashlib.new(file.algorithm, content).hexdigest() == file.digest
            )
        else:
            request = urllib.request.Request(file.url, method="HEAD")
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                same = int(response.headers.get("Content-Length", -1)) == file.size
    except (OSError, ValueError):
        return RemoteState.UNREACHABLE
    return RemoteState.OK if same else RemoteState.CHANGED
