"""Reference conditions every steering method is compared with."""

from contextlib import AbstractContextManager, nullcontext
from typing import ClassVar

import torch

from swce.models.subject import Subject
from swce.steering.hooks import add_vector, edit_layers
from swce.steering.steerer import Steerer


class Unmodified(Steerer):
    """The subject as released, which every gain and loss is measured against."""

    name: ClassVar[str] = "unmodified"
    description: ClassVar[str] = "The model without any intervention."

    fitted_on: tuple[str, ...] | None = ()

    def attach(self, subject: Subject) -> AbstractContextManager[None]:
        """Leave the subject as it is.

        Args:
            subject: Model to leave unmodified.

        Returns:
            A context that changes nothing.
        """
        return nullcontext()


class RandomDirection(Steerer):
    """A direction drawn at random, added with the norm of the method it controls for.

    A gain this control also obtains comes from perturbing the hidden states and not from the
    information the direction of a method carries.

    Attributes:
        norms: Norm of the added vector by decoder layer index, taken from the method under
            control.
        width: Width of the residual stream of the subject.
        seed: Seed of the draw. Each layer receives its own direction.
        controls: Name of the method whose norms are matched.
    """

    name: ClassVar[str] = "random-direction"
    description: ClassVar[str] = (
        "A direction drawn uniformly on the sphere and added at the norm of a method."
    )

    norms: dict[int, float]
    width: int
    seed: int = 0
    controls: str = ""
    fitted_on: tuple[str, ...] | None = ()

    def vectors(self) -> dict[int, torch.Tensor]:
        """Directions added at every layer.

        Returns:
            For every layer, a vector drawn uniformly on the unit sphere and scaled to the norm
            of that layer. The draw depends on the seed and on the layer only.
        """
        vectors = {}
        for layer, norm in self.norms.items():
            generator = torch.Generator().manual_seed(self.seed * 1_000_003 + layer)
            direction = torch.randn(self.width, generator=generator, dtype=torch.float64)
            vectors[layer] = (norm * direction / direction.norm()).to(torch.float32)
        return vectors

    def attach(self, subject: Subject) -> AbstractContextManager[None]:
        """Add the random directions to the decoder of the subject.

        Args:
            subject: Model to steer.

        Returns:
            A context inside which the directions are added at every position of their layers.
        """
        edits = {layer: add_vector(vector) for layer, vector in self.vectors().items()}
        return edit_layers(subject.decoder_layers(), edits)
