"""Nullu, projection of the feed forward output onto the null space of a hallucination subspace.

Source: Yang et al., Nullu: Mitigating Object Hallucinations in Large Vision-Language Models via
    HalluSpace Projection, CVPR 2025, arXiv 2412.13817. Official code at
    https://github.com/Ziwei-Zheng/Nullu, commit dcb1ef4ea44e659a769d72caa499ddde23eab877.
Licence: MIT.
Settings: Section 9 of the paper and the command of the official README for LLaVA-1.5, with the
    four leading singular vectors and the decoder blocks 16 to 31, for every benchmark. The paper
    gives a rank for MiniGPT-4 and mPLUG-Owl2 as well and for no other model. It chooses the rank
    and the blocks by a sweep whose grid and criterion it does not state, so on any other model
    the defaults of the official `model_edit.py` are used, a rank of eight over the same blocks.

The official code edits the weights. It multiplies the output projection of the feed forward
module of every chosen block by the projector onto the null space of the subspace, which is the
same as projecting the output of the module. The projection is applied here as a hook, so the
weights stay untouched and the subject is unmodified once the context ends.

`fit` builds the subspaces of any model as the official `model_run.py` and `model_edit.py` do.
Every pair of the LURE file distributed with the official code gives a truthful and a
hallucinated description of one image. The residual stream after every block is averaged over all
positions of the conversation holding the description, and the subspace of a block is spanned by
the leading right singular vectors of half the differences between the two averages. The official
command uses all 5000 pairs, which the paper does not state.
"""

import json
from collections.abc import Iterable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, ClassVar

import torch
from PIL import Image
from torch import Tensor
from tqdm import tqdm

from swce.data.download import fetch, fetch_images
from swce.data.paths import artifacts_root
from swce.models.subject import Subject, residual_width
from swce.registry import load_registry
from swce.steering.hooks import Edit, edit_layers
from swce.steering.steerer import InapplicableError, Steerer

PAIRS = load_registry().file("methods/nullu/hallucination5k_train.jsonl")
"""Truthful and hallucinated descriptions of 5000 COCO train2014 images, from LURE."""
QUESTION = "Please describe this image in detail."
"""Question the official code asks with every description."""
KEPT = 64
"""Number of leading singular vectors written for every block, more than any published rank."""


def training_images(names: Iterable[str]) -> Path:
    """Folder of the COCO train2014 images a step reads, fetched from the official host.

    Args:
        names: File names of the images.

    Returns:
        The folder holding the images under the data root.
    """
    return fetch_images(load_registry().benchmark("coco-train2014"), names)


def subspaces_path(subject: str) -> Path:
    """File of the subspaces built here for a subject.

    Args:
        subject: Name of the subject.

    Returns:
        The path under the artifacts folder of the data root.
    """
    return artifacts_root() / f"nullu_{subject}.pt"


def project_out(basis: Tensor) -> Edit:
    """Edit removing the component of the output inside a subspace.

    Args:
        basis: Orthonormal vectors spanning the subspace, of shape `(width, rank)`.

    Returns:
        An edit that subtracts from the output its projection onto the subspace.
    """

    def edit(hidden: Tensor) -> Tensor:
        vectors = basis.to(device=hidden.device, dtype=hidden.dtype)
        return hidden - (hidden @ vectors) @ vectors.T

    return edit


class Nullu(Steerer):
    """Nullu at one setting.

    Attributes:
        rank: Number of leading singular vectors spanning the subspace of every block.
        first_layer: First decoder block that is edited.
        last_layer: Last decoder block that is edited, included.
        subspaces: File of singular vectors by block index, or None for the one built for the
            subject.
    """

    name: ClassVar[str] = "nullu"
    description: ClassVar[str] = "Nullu (Yang et al., CVPR 2025), null space projection."

    rank: int = 4
    first_layer: int = 16
    last_layer: int = 31
    subspaces: Path | None = None
    fitted_on: tuple[str, ...] | None = ("lure-pairs", "coco-train2014")

    def prepare(self, subject: Subject) -> None:
        """Build the subspaces of the subject when they are missing.

        Args:
            subject: Model about to be steered.
        """
        if self.subspaces is None and not subspaces_path(subject.name).exists():
            fit(subject)

    def attach(self, subject: Subject) -> AbstractContextManager[None]:
        """Project the output of the feed forward module of every chosen block.

        Args:
            subject: Model to steer.

        Returns:
            A context inside which the outputs are projected.

        Raises:
            FileNotFoundError: If the subspaces were not built, so that the unmodified model is
                never evaluated under the name of the method.
            InapplicableError: If the subject has fewer blocks than the setting edits.
        """
        path = self.subspaces or subspaces_path(subject.name)
        if not path.exists():
            raise FileNotFoundError(f"build the subspaces of {self.name} first, at {path}")
        mlps = subject.decoder_mlps()
        if self.last_layer >= len(mlps):
            raise InapplicableError(
                f"blocks {self.first_layer} to {self.last_layer} of {len(mlps)} blocks"
            )
        loaded: dict[int, Tensor] = torch.load(path, map_location="cpu", weights_only=True)
        edits = {
            layer: project_out(loaded[layer][:, : self.rank])
            for layer in range(self.first_layer, self.last_layer + 1)
        }
        return edit_layers(mlps, edits)


class NulluRandom(Nullu):
    """Nullu with a random subspace of the same rank projected out of the same blocks.

    The control needs no fitting. A gain it also obtains does not come from what the fitted
    subspaces carry.

    Attributes:
        seed: Seed of the draw of the subspaces.
    """

    name: ClassVar[str] = "nullu-random"
    description: ClassVar[str] = "Nullu with random subspaces, as a control."

    seed: int = 0
    fitted_on: tuple[str, ...] | None = ()

    def prepare(self, subject: Subject) -> None:
        """Do nothing, since a random subspace is drawn without any data.

        Args:
            subject: Model about to be steered.
        """

    def attach(self, subject: Subject) -> AbstractContextManager[None]:
        """Project a random subspace out of the feed forward output of every chosen block.

        Args:
            subject: Model to steer.

        Returns:
            A context inside which the outputs are projected.

        Raises:
            InapplicableError: If the subject has fewer blocks than the setting edits.
        """
        mlps = subject.decoder_mlps()
        if self.last_layer >= len(mlps):
            raise InapplicableError(
                f"blocks {self.first_layer} to {self.last_layer} of {len(mlps)} blocks"
            )
        width = residual_width(subject)
        edits = {}
        for layer in range(self.first_layer, self.last_layer + 1):
            generator = torch.Generator().manual_seed(self.seed * 1_000_003 + layer)
            drawn = torch.randn((width, self.rank), generator=generator, dtype=torch.float64)
            edits[layer] = project_out(torch.linalg.qr(drawn).Q.float())
        return edit_layers(mlps, edits)


def subspaces(truthful: Tensor, hallucinated: Tensor, kept: int = KEPT) -> dict[int, Tensor]:
    """Singular vectors of every block from the averaged states of the pairs.

    Args:
        truthful: Residual stream after every block, averaged over the positions of the
            conversation with the truthful description, of shape `(pairs, layers, width)`.
        hallucinated: The same with the hallucinated description.
        kept: Number of leading singular vectors kept for every block.

    Returns:
        For every block index, the leading right singular vectors of half the differences, of
        shape `(width, kept)`.
    """
    differences = ((truthful - hallucinated) / 2).float()
    found = {}
    for layer in range(differences.shape[1]):
        _, _, right = torch.linalg.svd(differences[:, layer], full_matrices=False)
        found[layer] = right.T[:, :kept].contiguous()
    return found


def _conversation(subject: Any, answer: str) -> str:
    """Text of the question and of a description given as the complete answer."""
    if subject.name.startswith("llava-1.5"):
        # The official conversation template closes the answer with the end of sequence token.
        closing: str = subject.processor.tokenizer.eos_token
        return f"{subject.template(QUESTION)} {answer}{closing}"
    content = [{"type": "image"}, {"type": "text", "text": QUESTION}]
    messages = [{"role": "user", "content": content}, {"role": "assistant", "content": answer}]
    text: str = subject.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return text


def fit(
    subject: Any,
    images: Path | None = None,
    output: Path | None = None,
    samples: int | None = None,
) -> Path:
    """Build the subspaces of a subject from the pairs the authors distribute.

    This follows the official scripts and has not been run on real weights yet.

    Args:
        subject: A `HuggingFaceSubject`.
        images: Folder holding the COCO train2014 images the pairs name, or None to fetch
            them from the official host.
        output: File to write the subspaces to, or None for the artifacts folder.
        samples: Number of leading pairs used, or None for all of them, as the official command.

    Returns:
        The path of the written file.
    """
    lines = fetch(PAIRS).read_text(encoding="utf-8").splitlines()
    pairs = [json.loads(line) for line in lines if line.strip()][:samples]
    images = images or training_images(pair["image"] for pair in pairs)
    truthful: list[Tensor] = []
    hallucinated: list[Tensor] = []
    for pair in tqdm(pairs, desc="Nullu subspaces"):
        with Image.open(images / pair["image"]) as opened:
            image = opened.convert("RGB")
        for description, states in ((pair["value"], truthful), (pair["h_value"], hallucinated)):
            inputs = subject.inputs(image, _conversation(subject, description))
            states.append(subject.all_states(inputs)[1:].mean(dim=1))
    output = output or subspaces_path(subject.name)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(subspaces(torch.stack(truthful), torch.stack(hallucinated)), output)
    return output


def control() -> list[Steerer]:
    """Random subspace controls of the settings of `build`.

    Returns:
        One control per setting, with the same rank, blocks and models.
    """
    return [
        NulluRandom(**setting.model_dump(exclude={"fitted_on"}))
        for setting in build()
        if isinstance(setting, Nullu)
    ]


def build() -> list[Steerer]:
    """Setting the paper publishes for LLaVA-1.5, and the defaults of the code for other models.

    Returns:
        The published setting, used for every benchmark, and the defaults of `model_edit.py`, a
        rank of eight over the same blocks, used on models without a published setting.
    """
    return [
        Nullu(published_for=("llava-1.5-7b",)),
        Nullu(setting="code-default", rank=8, fallback=True),
    ]
