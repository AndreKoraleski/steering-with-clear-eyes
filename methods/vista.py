"""VISTA, visual information steering with a vector computed for every image.

Source: Li et al., The Hidden Life of Tokens: Reducing Hallucination of Large Vision-Language
    Models via Visual Information Steering, ICML 2025, arXiv 2502.03628. Official code at
    https://github.com/LzVv123456/VISTA, commit efcf499919e066755e7c33778fbfd864c204329c.
Licence: MIT.
Settings: `run_chair.sh` and `run_pope.sh` of the official code for LLaVA-1.5. CHAIR uses a
    strength of 0.17 and POPE of 0.01, both with a logit weight of 0.3 over the layers 25 to 30,
    the defaults of `chair_eval.py`. The paper reports MME without a script, so MME is run at
    both settings.

The method needs no data. Before every generation it runs the prompt with and without the image
and takes the difference of the residual stream at the last position after every decoder block as
the visual steering vector of that block. The vector is added to the output of the feed forward
module of the block on the unit sphere, so the norm of that output is kept. The logits are then
mixed with those read from intermediate layers.

The official code passes the single difference through a principal component analysis of one
sample. The centred sample is zero, so the analysis adds to the difference whatever unit vector
the linear algebra library returns for a zero matrix, over all layers at once. The same
operations are applied here, so the vectors equal the official ones on the same machine.
"""

from collections.abc import Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from typing import ClassVar

import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from torch import Tensor

from swce.models.subject import Subject, residual_width
from swce.steering.hooks import Edit, edit_layers
from swce.steering.steerer import InapplicableError, Steerer


def steering_vectors(with_image: Tensor, without_image: Tensor) -> Tensor:
    """Visual steering vectors as the official `obtain_vsv` computes them for one image.

    Args:
        with_image: Residual stream at the last position after every decoder block for the
            prompt with its image, of shape `(layers, width)`.
        without_image: The same for the prompt without its image.

    Returns:
        The vectors of every block, of shape `(layers, width)`.
    """
    sample = (with_image - without_image).reshape(1, -1).float()
    mean = sample.mean(dim=0, keepdim=True)
    left, _, right = torch.linalg.svd(sample - mean, full_matrices=False)
    largest = torch.argmax(torch.abs(left), dim=0)
    signs = torch.sign(left[largest, torch.arange(left.shape[1])])
    components = (right * signs.view(-1, 1))[:1]
    direction: Tensor = (components.sum(dim=0, keepdim=True) + mean).mean(dim=0)
    return direction.view(with_image.shape)


def steer(vector: Tensor, strength: float) -> Edit:
    """Edit of the official `VSVLayer` for one visual steering vector.

    Args:
        vector: Visual steering vector of the block, of shape `(width,)`.
        strength: Coefficient of the vector.

    Returns:
        An edit that moves the output along the vector on the unit sphere, more strongly where
        the output points away from it, and restores the norm of the output.
    """

    def edit(hidden: Tensor) -> Tensor:
        state = hidden.float()
        direction = vector.to(state.device)
        norm = torch.norm(state, p=2, dim=-1, keepdim=True)
        opposition = F.cosine_similarity(state, -direction[None, None, :], dim=-1).clamp(min=0)
        shift = strength * (1.0 + opposition.unsqueeze(-1)) * F.normalize(direction, dim=-1)
        moved: Tensor = F.normalize(F.normalize(state, p=2, dim=-1) + shift, p=2, dim=-1) * norm
        return moved.to(hidden.dtype)

    return edit


class Vista(Steerer):
    """VISTA at one published setting.

    Attributes:
        strength: Coefficient of the visual steering vector.
        logits_alpha: Weight of the logits read from intermediate layers.
        logits_layers: First and last decoder block, both included, whose residual streams are
            read through the unembedding and averaged.
    """

    name: ClassVar[str] = "vista"
    description: ClassVar[str] = "VISTA (Li et al., ICML 2025), per image steering vector."

    strength: float
    logits_alpha: float = 0.3
    logits_layers: tuple[int, int] = (25, 30)
    fitted_on: tuple[str, ...] | None = ()

    def attach(self, subject: Subject) -> AbstractContextManager[None]:
        """Steer every generation of the subject with the vector of its own image.

        Args:
            subject: Model to steer, which must expose its residual stream, its feed forward
                modules and its unembedding.

        Returns:
            A context inside which every generation is steered.
        """
        return self._attached(subject)

    def vectors_for(self, subject: Subject, image: Image.Image, prompt: str) -> Tensor:
        """Steering vectors of one input.

        Args:
            subject: Model about to answer, with no vector set.
            image: Image of the input.
            prompt: Prompt of the input.

        Returns:
            The vector of every decoder block, of shape `(layers, width)`.
        """
        return steering_vectors(subject.residuals(image, prompt), subject.residuals(None, prompt))

    @contextmanager
    def _attached(self, subject: Subject) -> Iterator[None]:
        vectors: dict[int, Tensor] = {}
        read: dict[int, Tensor] = {}
        first, last = self.logits_layers
        head = subject.unembedding()

        def listen(image: Image.Image, prompt: str) -> None:
            # Both passes run unsteered, since the edits do nothing while no vector is set.
            vectors.clear()
            vectors.update(dict(enumerate(self.vectors_for(subject, image, prompt))))

        def steered(index: int) -> Edit:
            def edit(hidden: Tensor) -> Tensor:
                found = vectors.get(index)
                return hidden if found is None else steer(found, self.strength)(hidden)

            return edit

        def remember(index: int) -> Edit:
            def edit(hidden: Tensor) -> Tensor:
                read[index] = hidden
                return hidden

            return edit

        def mix(logits: Tensor) -> Tensor:
            if not vectors or len(read) <= last - first:
                return logits
            weight: Tensor = head.weight  # type: ignore[assignment]
            states = torch.stack([read[index] for index in range(first, last + 1)])
            bias = getattr(head, "bias", None)
            augmented = F.linear(states.to(weight.dtype), weight, bias).mean(dim=0)
            return self.logits_alpha * augmented + (1 - self.logits_alpha) * logits

        mlps = subject.decoder_mlps()
        if last >= len(mlps):
            raise InapplicableError(
                f"logits are read from blocks {first} to {last} of {len(mlps)} blocks"
            )
        with ExitStack() as stack:
            stack.enter_context(
                edit_layers(mlps, {index: steered(index) for index in range(len(mlps))})
            )
            stack.enter_context(
                edit_layers(
                    subject.decoder_layers(),
                    {index: remember(index) for index in range(first, last + 1)},
                )
            )
            stack.enter_context(edit_layers([head], {0: mix}))
            subject.listeners.append(listen)
            try:
                yield
            finally:
                subject.listeners.remove(listen)


class VistaRandom(Vista):
    """VISTA with its visual steering vectors replaced by random directions.

    The edit normalizes its vector, so only the direction matters. The strength and the logit
    mixing are those of the method. A gain this control also obtains does not come from the
    difference the image makes to the residual stream.

    Attributes:
        seed: Seed of the draw of the directions, one per decoder block.
    """

    name: ClassVar[str] = "vista-random"
    description: ClassVar[str] = "VISTA with random vectors, as a control."

    seed: int = 0

    def vectors_for(self, subject: Subject, image: Image.Image, prompt: str) -> Tensor:
        """Random directions, the same for every input.

        Args:
            subject: Model about to answer.
            image: Image of the input, unused.
            prompt: Prompt of the input, unused.

        Returns:
            One direction per decoder block, of shape `(layers, width)`.
        """
        generator = torch.Generator().manual_seed(self.seed)
        shape = (len(subject.decoder_mlps()), residual_width(subject))
        return torch.randn(shape, generator=generator)


def control() -> list[Steerer]:
    """Random vector controls of the settings of `build`.

    Returns:
        One control per setting, with the same strength, tasks and models.
    """
    return [
        VistaRandom(**setting.model_dump(exclude={"fitted_on"}))
        for setting in build()
        if isinstance(setting, Vista)
    ]


def build() -> list[Steerer]:
    """Settings the official code publishes for LLaVA-1.5, and its defaults for other models.

    Returns:
        The CHAIR setting, the POPE setting, and the defaults of `chair_eval.py` with its steering
        and its logit mixing switched on, used on models without a published setting.
    """
    model = ("llava-1.5-7b",)
    return [
        Vista(setting="chair", strength=0.17, published_for=model, tasks=("chair", "mme")),
        Vista(
            setting="pope", strength=0.01, published_for=model, tasks=("pope-*", "repope-*", "mme")
        ),
        Vista(setting="code-default", strength=0.1, fallback=True),
    ]
