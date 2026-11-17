"""VTI, visual and textual intervention with fixed directions in every layer.

Source: Liu et al., Reducing Hallucinations in Large Vision-Language Models via Latent Space
    Steering, ICLR 2025, arXiv 2410.15778. Official code at https://github.com/shengliu66/VTI,
    commit bb5dfee33a9e555f22af9faa7b3495c0ac3df69e.
Licence: none declared. The code is reimplemented here from its description and behaviour, and
    the demonstration file is downloaded from the official repository and never redistributed.
Settings: Appendix A of the paper. CHAIR uses a visual strength of 0.2 and a textual strength of
    0.4, and every other experiment 0.9 for both. Directions are fitted on 50 demonstrations with
    50 random masks of 99% of the patches, as the paper and the defaults of the official script
    state. MME is not reported in the paper, so it is run at both settings.

The official code wraps the feed forward module of every block of the vision encoder and of the
language decoder. It adds the normalized direction to the normalized output and restores the norm
of the output. The wrapper multiplies the shift by an extra factor of 0.1 in both branches, which
the paper does not mention and which is kept here.

Three details of the official fitting are kept because they change the directions.

- The directions are the mean difference plus the first principal component of the differences.
- Every demonstration is given the question of the last sampled demonstration.
- The differences of the vision encoder are reshaped from layers by tokens to tokens by layers
  without a permutation, so the principal component is computed over chunks that mix both.

The official script draws its demonstrations with the global random state after loading the
model, so the exact sample of the authors cannot be recovered. The sample here is drawn with the
same seed of 42 from a generator of its own.
"""

import json
import random
from collections.abc import Iterable, Iterator, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager
from pathlib import Path
from typing import ClassVar

import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from torch import Tensor

from swce.data.download import fetch, fetch_images
from swce.data.paths import artifacts_root
from swce.models.subject import Subject
from swce.registry import load_registry
from swce.steering.hooks import Edit, edit_layers
from swce.steering.steerer import InapplicableError, Steerer

DEMONSTRATIONS = load_registry().file("methods/vti/hallucination_vti_demos.jsonl")
"""Images of COCO train2014 with a faithful and a hallucinated description of each."""
PATCH = 14
"""Side in pixels of a patch of the vision encoder of LLaVA-1.5."""
SHIFT_FACTOR = 0.1
"""Factor the official code applies to the shift of every wrapped module."""


def training_images(names: Iterable[str]) -> Path:
    """Folder of the COCO train2014 images a step reads, fetched from the official host.

    Args:
        names: File names of the images.

    Returns:
        The folder holding the images under the data root.
    """
    return fetch_images(load_registry().benchmark("coco-train2014"), names)


def directions_path(subject: str) -> Path:
    """File of the directions fitted here for a subject, under the keys `visual` and `textual`.

    Args:
        subject: Name of the subject.

    Returns:
        The path under the artifacts folder of the data root.
    """
    return artifacts_root() / f"vti_{subject}.pt"


def shift(direction: Tensor, strength: float) -> Edit:
    """Edit of the official `VTILayer` for one direction.

    Args:
        direction: Direction of the block, of shape `(width,)`, or `(positions, width)` for the
            vision encoder, where every token has its own.
        strength: Coefficient of the direction.

    Returns:
        An edit that moves the output along the direction on the unit sphere and restores the
        norm of the output.
    """

    def edit(hidden: Tensor) -> Tensor:
        state = hidden.float()
        norm = torch.norm(state, dim=-1).unsqueeze(-1)
        move = strength * F.normalize(direction.to(state.device), dim=-1)
        moved: Tensor = F.normalize(F.normalize(state, dim=-1) + SHIFT_FACTOR * move, dim=-1) * norm
        return moved.to(hidden.dtype)

    return edit


def principal_direction(differences: Tensor) -> Tensor:
    """Direction the official code derives from the differences of the demonstrations.

    Args:
        differences: Differences of shape `(demonstrations, features)`, or
            `(batch, demonstrations, features)` to treat every batch entry on its own.

    Returns:
        The mean difference plus the first principal component, of shape `(features,)` or
        `(batch, features)`. The sign of the component makes its largest loading in absolute
        value positive in the left singular vector, as in the official code.
    """
    batched = differences if differences.ndim == 3 else differences.unsqueeze(0)
    batched = batched.float()
    mean = batched.mean(dim=1, keepdim=True)
    left, _, right = torch.linalg.svd(batched - mean, full_matrices=False)
    largest = torch.argmax(torch.abs(left), dim=1, keepdim=True)
    signs = torch.sign(torch.gather(left, 1, largest))
    component = (right * signs.view(right.shape[0], -1, 1))[:, :1]
    direction: Tensor = (component + mean).squeeze(1)
    return direction if differences.ndim == 3 else direction[0]


def mask_patches(pixels: Tensor, indices: Tensor, patch: int = 14) -> Tensor:
    """Image with the chosen patches replaced by the mean of every channel.

    Args:
        pixels: Preprocessed image of shape `(channels, height, width)`.
        indices: Indices of the patches to replace, counted row by row.
        patch: Side of a patch in pixels.

    Returns:
        A copy of the image with the patches replaced.
    """
    masked = pixels.clone()
    mean = pixels.mean(dim=(1, 2), keepdim=True)
    per_row = pixels.shape[2] // patch
    for index in indices.tolist():
        top, left = (index // per_row) * patch, (index % per_row) * patch
        masked[:, top : top + patch, left : left + patch] = mean.expand(-1, patch, patch)
    return masked


class Vti(Steerer):
    """VTI at one published setting.

    Attributes:
        visual: Strength of the visual directions, in the vision encoder. Zero disables them.
        textual: Strength of the textual directions, in the language decoder. Zero disables them.
        directions: File holding the fitted directions, or None for the one fitted for the
            subject.
    """

    name: ClassVar[str] = "vti"
    description: ClassVar[str] = "VTI (Liu et al., ICLR 2025), fixed directions in every layer."

    visual: float
    textual: float
    directions: Path | None = None
    fitted_on: tuple[str, ...] | None = ("coco-train2014",)

    def prepare(self, subject: Subject) -> None:
        """Fit the directions of the subject when they are missing.

        Args:
            subject: Model about to be steered.

        Raises:
            InapplicableError: If the visual branch is asked for on a subject that gives every
                image its own number of tokens.
        """
        if self.directions is None and not directions_path(subject.name).exists():
            fit(subject, visual=bool(self.visual))

    def attach(self, subject: Subject) -> AbstractContextManager[None]:
        """Shift the output of the feed forward module of every block.

        Args:
            subject: Model to steer.

        Returns:
            A context inside which the encoder and the decoder are steered.

        Raises:
            FileNotFoundError: If the directions were not fitted, so that the unmodified model is
                never evaluated under the name of the method.
        """
        path = self.directions or directions_path(subject.name)
        if not path.exists():
            raise FileNotFoundError(f"fit the directions of {self.name} first, at {path}")
        return self._attached(subject, path)

    @contextmanager
    def _attached(self, subject: Subject, path: Path) -> Iterator[None]:
        loaded: dict[str, Tensor] = torch.load(path, map_location="cpu", weights_only=True)
        with ExitStack() as stack:
            for strength, key, mlps in (
                (self.visual, "visual", subject.encoder_mlps),
                (self.textual, "textual", subject.decoder_mlps),
            ):
                if not strength:
                    continue
                if key not in loaded:
                    raise InapplicableError(f"the {key} directions of VTI were not fitted")
                modules = mlps()
                if len(modules) != len(loaded[key]):
                    raise InapplicableError(
                        f"{len(loaded[key])} {key} directions for {len(modules)} blocks"
                    )
                edits = {index: shift(d, strength) for index, d in enumerate(loaded[key])}
                stack.enter_context(edit_layers(modules, edits))
            yield


class VtiRandom(Vti):
    """VTI with its fitted directions replaced by random ones of the same shape.

    The edit normalizes its directions, so only their orientation matters. The strengths are
    those of the method. A gain this control also obtains does not come from what the fitted
    directions carry.

    Attributes:
        seed: Seed of the draw of the directions.
    """

    name: ClassVar[str] = "vti-random"
    description: ClassVar[str] = "VTI with random directions, as a control."

    seed: int = 0
    fitted_on: tuple[str, ...] | None = ()

    def _random_path(self, subject: str) -> Path:
        """File of the random directions of a subject for the seed of the control."""
        return artifacts_root() / f"vti_{subject}_random{self.seed}.pt"

    def prepare(self, subject: Subject) -> None:
        """Fit the directions of the method, then draw random ones of the same shape.

        Args:
            subject: Model about to be steered.
        """
        super().prepare(subject)
        path = self._random_path(subject.name)
        if not path.exists():
            fitted = self.directions or directions_path(subject.name)
            loaded: dict[str, Tensor] = torch.load(fitted, map_location="cpu", weights_only=True)
            generator = torch.Generator().manual_seed(self.seed)
            drawn = {
                key: torch.randn(found.shape, generator=generator) for key, found in loaded.items()
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(drawn, path)

    def attach(self, subject: Subject) -> AbstractContextManager[None]:
        """Shift the output of the feed forward modules along the random directions.

        Args:
            subject: Model to steer.

        Returns:
            A context inside which the encoder and the decoder are steered.

        Raises:
            FileNotFoundError: If the control was not prepared for the subject.
        """
        path = self._random_path(subject.name)
        if not path.exists():
            raise FileNotFoundError(f"prepare {self.name} for {subject.name} first")
        return self._attached(subject, path)


def visual_directions(clean: Tensor, masked: Tensor) -> Tensor:
    """Visual directions from the hidden states of the vision encoder.

    Args:
        clean: Hidden states of the clean images, of shape `(demonstrations, layers + 1, tokens,
            width)`, the first layer being the embeddings.
        masked: Hidden states averaged over the masked copies of every image, of the same shape.

    Returns:
        Directions of shape `(layers, tokens, width)`, without the embeddings.
    """
    demonstrations, layers, tokens, width = clean.shape
    # The official code reshapes without permuting, which is kept as it is.
    differences = (masked - clean).reshape(demonstrations, tokens, layers * width).transpose(0, 1)
    return principal_direction(differences).view(layers, tokens, width)[1:]


def textual_directions(faithful: Tensor, hallucinated: Tensor) -> Tensor:
    """Textual directions from the hidden states of the language decoder.

    Args:
        faithful: Hidden states at the last position for the faithful descriptions, of shape
            `(demonstrations, layers + 1, width)`, the first layer being the embeddings.
        hallucinated: The same for the hallucinated descriptions.

    Returns:
        Directions of shape `(layers, width)`, without the embeddings.
    """
    demonstrations, layers, width = faithful.shape
    differences = (faithful - hallucinated).reshape(demonstrations, layers * width)
    return principal_direction(differences).view(layers, width)[1:]


def fit(
    subject: Subject,
    images: Path | None = None,
    output: Path | None = None,
    *,
    demonstrations: int = 50,
    trials: int = 50,
    mask_ratio: float = 0.99,
    seed: int = 42,
    visual: bool = True,
) -> Path:
    """Fit the directions of a subject and write them.

    This follows the official script and has not been run on the real weights yet.

    Args:
        subject: A `HuggingFaceSubject` whose images all yield the same number of tokens.
        images: Folder holding the COCO train2014 images the demonstrations name, or None to
            fetch them from the official host.
        output: File to write the directions to, or None for the artifacts folder.
        demonstrations: Number of demonstrations drawn from the official file.
        trials: Number of random masks averaged for every image.
        mask_ratio: Fraction of the patches every mask replaces.
        seed: Seed of the sample of demonstrations and of the masks.
        visual: Whether to fit the visual directions. Without them only the textual branch can
            be used, which is the default of the official script.

    Returns:
        The path of the written file.

    Raises:
        InapplicableError: If the subject gives every image its own number of tokens, since the
            visual directions are fitted per image token.
    """
    from swce.models.hf import HuggingFaceSubject

    if not isinstance(subject, HuggingFaceSubject):
        raise TypeError("the directions of VTI are fitted on a HuggingFaceSubject")
    if visual and not subject.fixed_grid:
        raise InapplicableError(
            f"VTI fits one visual direction per image token, and {subject.name} gives every "
            "image its own number of tokens"
        )
    lines = fetch(DEMONSTRATIONS).read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    sample: Sequence[dict[str, str]] = random.Random(seed).sample(rows, demonstrations)
    images = images or training_images(row["image"] for row in sample)
    generator = torch.Generator().manual_seed(seed)
    # The official code gives every demonstration the question of the last one.
    text = subject.template(sample[-1]["question"])
    clean, masked, faithful, hallucinated = [], [], [], []

    def states(batch: Tensor) -> Tensor:
        vision = subject.model.get_submodule("model.vision_tower")
        moved = batch.unsqueeze(0).to(subject.device, subject.dtype)
        out = vision(moved, output_hidden_states=True)
        return torch.cat([state.float().cpu() for state in out.hidden_states], dim=0)

    with torch.inference_mode():
        for row in sample:
            with Image.open(images / row["image"]) as opened:
                image = opened.convert("RGB")
            if visual:
                processed = subject.processor.image_processor(image, return_tensors="pt")
                pixels: Tensor = processed["pixel_values"][0]
                patches = (pixels.shape[-1] * pixels.shape[-2]) // PATCH**2
                clean.append(states(pixels))
                copies = []
                for _ in range(trials):
                    kept = int(mask_ratio * patches)
                    chosen = torch.randperm(patches, generator=generator)[:kept]
                    copies.append(states(mask_patches(pixels, chosen, PATCH)))
                masked.append(torch.stack(copies).mean(dim=0))
            # The official prompt appends the description right after the template.
            faithful.append(subject.hidden_states(image, text + row["value"]))
            hallucinated.append(subject.hidden_states(image, text + row["h_value"]))
    directions = {"textual": textual_directions(torch.stack(faithful), torch.stack(hallucinated))}
    if visual:
        directions["visual"] = visual_directions(torch.stack(clean), torch.stack(masked))
    output = output or directions_path(subject.name)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(directions, output)
    return output


def control() -> list[Steerer]:
    """Random direction controls of the settings of `build`.

    Returns:
        One control per setting, with the same strengths, tasks and models.
    """
    return [
        VtiRandom(**setting.model_dump(exclude={"fitted_on"}))
        for setting in build()
        if isinstance(setting, Vti)
    ]


def build() -> list[Steerer]:
    """Settings the paper publishes for LLaVA-1.5, and the defaults of the code for other models.

    Returns:
        The CHAIR setting, the setting of every other experiment, named after POPE, and the
        defaults of the official script, a textual strength of 0.8 without the visual branch,
        used on models without a published setting.
    """
    return [
        Vti(
            setting="chair",
            visual=0.2,
            textual=0.4,
            published_for=("llava-1.5-7b",),
            tasks=("chair", "mme"),
        ),
        Vti(
            setting="pope",
            visual=0.9,
            textual=0.9,
            published_for=("llava-1.5-7b",),
            tasks=("pope-*", "repope-*", "mme"),
        ),
        Vti(setting="code-default", visual=0.0, textual=0.8, fallback=True),
    ]
