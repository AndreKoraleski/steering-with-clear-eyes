"""Hooks that modify the residual stream of a subject while it runs."""

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from torch import Tensor, nn

Edit = Callable[[Tensor], Tensor]
"""Function of the residual stream after a block, of shape `(batch, positions, width)`."""


def _edited(output: Any, edit: Edit) -> Any:
    """Output of a block with its residual stream replaced by the edited one."""
    if isinstance(output, Tensor):
        return edit(output)
    if isinstance(output, tuple):
        return (edit(output[0]), *output[1:])
    raise TypeError(f"cannot steer a block returning {type(output).__name__}")


@contextmanager
def edit_layers(layers: Sequence[nn.Module], edits: Mapping[int, Edit]) -> Iterator[None]:
    """Apply edits to the output of chosen blocks for the duration of a context.

    The hooks are removed when the context ends, even after an error, so the subject is left
    unmodified.

    Args:
        layers: Blocks of the subject, from the first to the last.
        edits: Edit of the residual stream after a block, by the index of the block.

    Yields:
        Nothing. The subject is steered inside the context.

    Raises:
        IndexError: If an index names no block.
    """
    handles = []
    try:
        for index, edit in edits.items():
            if not 0 <= index < len(layers):
                raise IndexError(f"layer {index} out of range for {len(layers)} layers")

            def hook(_module: nn.Module, _inputs: Any, output: Any, edit: Edit = edit) -> Any:
                return _edited(output, edit)

            handles.append(layers[index].register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def add_vector(vector: Tensor, scale: float = 1.0) -> Edit:
    """Edit adding a fixed vector at every position.

    Args:
        vector: Direction of shape `(width,)`.
        scale: Factor applied to the vector.

    Returns:
        An edit that adds the scaled vector, moved to the device and type of the residual stream.
    """

    def edit(hidden: Tensor) -> Tensor:
        return hidden + scale * vector.to(device=hidden.device, dtype=hidden.dtype)

    return edit
