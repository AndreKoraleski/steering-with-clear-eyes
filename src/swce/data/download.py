"""Download of pinned benchmark files and of single images.

A pinned file is accepted only when its size and digest equal those of the registry. A download
is written next to its target and renamed once verified, so an interrupted download never leaves
a file that looks complete. Every process writes its own partial file, so jobs sharing a data root
can download the same file at once.
"""

import hashlib
import os
import shutil
import time
import urllib.request
from collections.abc import Iterable
from pathlib import Path

from swce.data.paths import default_data_root
from swce.logging import get_logger
from swce.registry.schema import BenchmarkCard, PinnedFile

CHUNK = 1 << 20
"""Number of bytes read at once when hashing and downloading."""
TIMEOUT = 60.0
"""Seconds a host may take to answer."""
ATTEMPTS = 5
"""Number of times a download is tried before its failure is raised."""
WAIT = 2.0
"""Seconds waited after the first failed attempt, doubled after every further one."""

log = get_logger(__name__)


class PinError(RuntimeError):
    """Raised when a file does not match its pin."""


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    """Digest of a file, read in chunks.

    Args:
        path: File to hash.
        algorithm: Name of a hash function of `hashlib`.

    Returns:
        The hexadecimal digest of the content.
    """
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def matches(path: Path, file: PinnedFile) -> bool:
    """Whether a local file equals a pinned one.

    Args:
        path: Local file, which may be missing.
        file: Pin to compare with.

    Returns:
        True when the file exists with the pinned size and digest.
    """
    return (
        path.is_file()
        and path.stat().st_size == file.size
        and file_digest(path, file.algorithm) == file.digest
    )


def _partial(target: Path) -> Path:
    """File a download of this process is written to before it is renamed to its target."""
    return target.with_name(f"{target.name}.{os.getpid()}.part")


def _download(url: str, target: Path) -> None:
    """Write the content of an address to a file, trying again after a failure of the network.

    A long evaluation fetches thousands of images, and one refused connection should not end it.
    The download is tried `ATTEMPTS` times, waiting twice as long after every failure.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(ATTEMPTS):
        try:
            with (
                urllib.request.urlopen(url, timeout=TIMEOUT) as response,
                target.open("wb") as stream,
            ):
                shutil.copyfileobj(response, stream, CHUNK)
            return
        except OSError as error:
            if attempt == ATTEMPTS - 1:
                raise
            log.warning("download failed, trying again", url=url, error=str(error))
            time.sleep(WAIT * 2**attempt)


def fetch(file: PinnedFile, root: Path | None = None) -> Path:
    """Local copy of a pinned file, downloaded when missing.

    Args:
        file: File to obtain.
        root: Data root, or None for the default one.

    Returns:
        The path of the verified local copy.

    Raises:
        PinError: If the local copy or the downloaded content differs from the pin.
        OSError: If the host cannot be reached.
    """
    target = (root or default_data_root()) / file.path
    if target.exists():
        if not matches(target, file):
            raise PinError(f"{target} differs from its pin, delete it to download it again")
        return target
    partial = _partial(target)
    log.info("downloading", path=file.path, bytes=file.size)
    _download(file.url, partial)
    if not matches(partial, file):
        partial.unlink()
        raise PinError(f"{file.url} no longer serves the pinned content of {file.path}")
    partial.replace(target)
    return target


def fetch_image(card: BenchmarkCard, name: str, root: Path | None = None) -> Path:
    """Local copy of one image of a benchmark that serves its images one by one.

    Such images carry no checksum on their host, so an existing local copy is trusted.

    Args:
        card: Benchmark whose `image_url` and `image_dir` locate the image.
        name: File name of the image.
        root: Data root, or None for the default one.

    Returns:
        The path of the local copy.

    Raises:
        ValueError: If the benchmark serves no single images.
        OSError: If the host cannot be reached.
    """
    if card.image_url is None:
        raise ValueError(f"benchmark {card.key} serves no single images")
    target = (root or default_data_root()) / card.image_dir / name
    if not target.exists():
        partial = _partial(target)
        _download(card.image_url.format(name=name), partial)
        partial.replace(target)
    return target


def fetch_images(card: BenchmarkCard, names: Iterable[str], root: Path | None = None) -> Path:
    """Local folder of a benchmark holding the named images, downloaded when missing.

    Args:
        card: Benchmark serving its images one by one.
        names: File names of the images.
        root: Data root, or None for the default one.

    Returns:
        The folder holding the images.

    Raises:
        ValueError: If the benchmark serves no single images.
        OSError: If the host cannot be reached.
    """
    for name in names:
        fetch_image(card, name, root)
    return (root or default_data_root()) / card.image_dir
