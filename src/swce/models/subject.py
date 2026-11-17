"""Interface of the vision language models that are steered and evaluated.

A subject is frozen. It answers an image and a prompt with its own default generation settings,
repeatably, and exposes the blocks of its language decoder and of its vision encoder, so that
steering methods can attach hooks to them without knowing the architecture.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence

from PIL import Image
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn
from torch.nn import functional

Listener = Callable[[Image.Image, str], None]
"""Function called with the image and the prompt before every generation."""


class Generation(BaseModel):
    """Answer of a subject to one image and prompt.

    Attributes:
        text: Generated text, without the prompt.
        tokens: Number of generated tokens.
        yes: Probability of the token yes at the first generated position, or None when the
            subject does not expose it.
        no: Probability of the token no at the first generated position, or None.
        lens_yes: Probability of yes read from the residual stream after every decoder block at
            the position that predicts the answer, or None when the lens was not recorded.
        lens_no: The same for no.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    tokens: int
    yes: float | None = None
    no: float | None = None
    lens_yes: tuple[float, ...] | None = None
    lens_no: tuple[float, ...] | None = None


class Subject(ABC):
    """Frozen vision language model under evaluation.

    Attributes:
        name: Unique name of the model, which keys the leaderboard.
        description: One sentence naming the checkpoint, its revision and its precision.
        lens: Whether every generation also records the logit lens of its prompt, read from
            the pass over the prompt that the generation runs anyway.
    """

    name: str
    description: str
    lens: bool = False

    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int,
        settings: Mapping[str, object] | None = None,
    ) -> Generation:
        """Answer one image and prompt with the default generation settings of the model.

        Implementations call `notify` with the image and the prompt before generating.

        Args:
            image: Image in RGB.
            prompt: Text of the user turn, placed in the default template of the model.
            max_new_tokens: Largest number of tokens to generate.
            settings: Generation settings replacing the defaults, for the steps of a method
                whose official code fixes them. An evaluation never passes any.

        Returns:
            The generated answer.
        """

    @abstractmethod
    def decoder_layers(self) -> Sequence[nn.Module]:
        """Blocks of the language decoder, from the first to the last.

        Returns:
            Modules whose output, or the first element of whose output, is the residual stream
            after the block, of shape `(batch, positions, width)`.
        """

    def decoder_mlps(self) -> Sequence[nn.Module]:
        """Feed forward module of every decoder block.

        Returns:
            Modules whose output is what the block adds to the residual stream after attention.

        Raises:
            NotImplementedError: If the subject does not expose them.
        """
        raise NotImplementedError(f"{self.name} exposes no decoder feed forward modules")

    def encoder_mlps(self) -> Sequence[nn.Module]:
        """Feed forward module of every vision encoder block.

        Returns:
            Modules with the same convention as `decoder_mlps`.

        Raises:
            NotImplementedError: If the subject does not expose them.
        """
        raise NotImplementedError(f"{self.name} exposes no encoder feed forward modules")

    def unembedding(self) -> nn.Module:
        """Module mapping the final hidden state to the logits of the vocabulary.

        Returns:
            A linear module with a `weight` of shape `(vocabulary, width)`.

        Raises:
            NotImplementedError: If the subject does not expose it.
        """
        raise NotImplementedError(f"{self.name} exposes no unembedding")

    def settings(self) -> dict[str, object]:
        """Settings of the subject that decide its answers, recorded with every run.

        Returns:
            Values that can be written as JSON, such as its precision, its seed and its
            generation settings. Empty when the subject has none to report.
        """
        return {}

    def final_norm(self) -> nn.Module:
        """Normalization the decoder applies before its unembedding.

        Returns:
            The module mapping the residual stream after the last block to the state the
            unembedding reads.

        Raises:
            NotImplementedError: If the subject does not expose it.
        """
        raise NotImplementedError(f"{self.name} exposes no final normalization")

    def first_tokens(self, words: Sequence[str]) -> list[int]:
        """Identifiers of the tokens an answer starting with one of the words can start with.

        Args:
            words: Words, such as the spellings of yes.

        Returns:
            The identifiers, without repetition.

        Raises:
            NotImplementedError: If the subject does not expose its vocabulary.
        """
        raise NotImplementedError(f"{self.name} exposes no vocabulary")

    def token_text(self, identifiers: Sequence[int]) -> list[str]:
        """Text of tokens of the vocabulary.

        Args:
            identifiers: Identifiers of the tokens.

        Returns:
            The text of each token.

        Raises:
            NotImplementedError: If the subject does not expose its vocabulary.
        """
        raise NotImplementedError(f"{self.name} exposes no vocabulary")

    def residuals(self, image: Image.Image | None, prompt: str, answer: str = "") -> Tensor:
        """Residual stream at the last position after every decoder block.

        Args:
            image: Image in RGB, or None to run the same prompt with the image tokens removed.
            prompt: Text of the user turn.
            answer: Text appended after the template as the beginning of the answer.

        Returns:
            A tensor of shape `(layers, width)` in single precision on the processor.

        Raises:
            NotImplementedError: If the subject does not expose it.
        """
        raise NotImplementedError(f"{self.name} exposes no residual stream")

    @property
    def listeners(self) -> list[Listener]:
        """Functions called before every generation, in order.

        A steering method whose intervention depends on the input registers one while attached.
        """
        found: list[Listener] = self.__dict__.setdefault("_listeners", [])
        return found

    def notify(self, image: Image.Image, prompt: str) -> None:
        """Call every listener with the input of a generation.

        Args:
            image: Image about to be answered.
            prompt: Prompt about to be answered.
        """
        for listener in list(self.listeners):
            listener(image, prompt)


def residual_width(subject: Subject) -> int:
    """Width of the residual stream of the decoder of a subject.

    Args:
        subject: Model exposing its unembedding.

    Returns:
        The number of features the unembedding reads.
    """
    weight: Tensor = subject.unembedding().weight  # type: ignore[assignment]
    return int(weight.shape[1])


def unembed(subject: Subject, states: Tensor) -> Tensor:
    """Logits the unembedding of a subject gives to residual states.

    The states go through the final normalization and then through the weights of the
    unembedding, and not through its module. A steering method may have attached a hook to that
    module, as VISTA does to mix logits, and a reading of the stream must not trigger it.

    Args:
        subject: Model whose final normalization and unembedding read the states.
        states: Residual states of shape `(..., width)`, on any device and of any type.

    Returns:
        Logits of shape `(..., vocabulary)`, on the device and of the type of the unembedding.
    """
    head = subject.unembedding()
    weight: Tensor = head.weight  # type: ignore[assignment]
    bias = getattr(head, "bias", None)
    normed = subject.final_norm()(states.to(device=weight.device, dtype=weight.dtype))
    logits: Tensor = functional.linear(normed, weight, bias)
    return logits
