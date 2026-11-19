"""States of the residual stream at every position of a corpus, for training autoencoders.

The corpus holds COCO train2014 images only, which no evaluation task reads.

- The detailed descriptions of the Nullu pairs, 5000 images, read after the prompt of CHAIR.
- The yes or no questions REVIS calibrates on, read up to the position of the answer.
- Descriptions the model writes itself of some of the images of the Nullu pairs.

A tenth of the images, chosen by a hash of their name, is held out from training, so that the
checks read sequences the autoencoders have not seen.

Every position is tagged as an image token, a token of the prompt or a token of the response.
The first position, which holds the start of the sequence, is dropped, and so is any position
whose state is more than ten times as long as the median of its sequence, since a few positions
of these models carry states thousands of times longer than the others and would dominate the
loss. Image tokens outnumber the others, so only a sample of them is kept from every sequence.
"""

import hashlib
import json
import random
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn

from swce.data.download import fetch, fetch_image
from swce.data.paths import artifacts_root
from swce.models.hf import HuggingFaceSubject
from swce.registry import load_registry

DESCRIBE = "Please describe this image in detail."
"""Prompt of CHAIR, under which the descriptions are read."""
IMAGE, PROMPT, RESPONSE = 0, 1, 2
"""Tags of the positions."""
TYPES = ("image", "prompt", "response")
"""Name of every tag."""
IMAGE_TOKENS = 128
"""Image tokens kept from every sequence."""
OUTLIER = 10.0
"""Ratio to the median norm of its sequence above which a position is dropped."""


@dataclass(frozen=True)
class Example:
    """One sequence of the corpus.

    Attributes:
        image: File name of a COCO train2014 image.
        prompt: Text of the user turn.
        response: Text read after the template, empty for a question.
        kind: `description`, `own` or `question`.
    """

    image: str
    prompt: str
    response: str
    kind: str


def held_out(image: str) -> bool:
    """Whether an image belongs to the tenth held out from training."""
    return hashlib.sha256(image.encode()).digest()[0] % 10 == 0


def sae_root(subject: str) -> Path:
    """Folder of the autoencoders and the corpus of a subject, under the artifacts."""
    return artifacts_root() / "sae" / subject


def _benchmark_file(key: str, name: str) -> Path:
    card = load_registry().benchmark(key)
    (found,) = [file for file in card.files if file.path.endswith(name)]
    return fetch(found)


def descriptions() -> list[Example]:
    """The detailed descriptions of the Nullu pairs."""
    path = _benchmark_file("nullu-data", "hallucination5k_train.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [Example(row["image"], DESCRIBE, row["value"], "description") for row in rows]


def questions() -> list[Example]:
    """The yes or no questions REVIS calibrates on."""
    path = _benchmark_file("revis-data", "coco_pope_calibration_merged.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [Example(row["image"], row["text"], "", "question") for row in rows]


def own_descriptions(
    subject: HuggingFaceSubject, images: Sequence[str], path: Path, max_new_tokens: int = 512
) -> list[Example]:
    """Descriptions the subject writes of images, kept in a file and resumed from it.

    Args:
        subject: Model that describes the images, greedily as in CHAIR.
        images: File names of the images.
        path: File of the descriptions, one JSON object per line.
        max_new_tokens: Largest length of a description, that of CHAIR.

    Returns:
        One example per image.
    """
    done: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["image"]] = row["response"]
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = [image for image in images if image not in done]
    fetch_all(missing)
    with path.open("a", encoding="utf-8") as out:
        for image in missing:
            answer = subject.generate(open_image(image), DESCRIBE, max_new_tokens)
            done[image] = answer.text
            out.write(json.dumps({"image": image, "response": answer.text}) + "\n")
            out.flush()
    return [Example(image, DESCRIBE, done[image], "own") for image in images]


def fetch_all(images: Sequence[str], workers: int = 16) -> None:
    """Download the COCO train2014 images that are missing, several at a time."""
    card = load_registry().benchmark("coco-train2014")
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda name: fetch_image(card, name), sorted(set(images))))


def open_image(name: str) -> Image.Image:
    """A COCO train2014 image in RGB, downloaded when missing."""
    card = load_registry().benchmark("coco-train2014")
    with Image.open(fetch_image(card, name)) as image:
        return image.convert("RGB")


def image_token(subject: HuggingFaceSubject) -> int:
    """Identifier of the token that stands for one image token in the input of the model."""
    config = subject.model.config
    for name in ("image_token_index", "image_token_id"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise ValueError(f"{subject.name} names no image token in its configuration")


def encode(
    subject: HuggingFaceSubject, image: Image.Image, prompt: str, response: str
) -> tuple[dict[str, Tensor], Tensor]:
    """Inputs of one sequence and the tag of each of its positions.

    Args:
        subject: Model that reads the sequence.
        image: Image in RGB.
        prompt: Text of the user turn.
        response: Text read after the template.

    Returns:
        The keyword arguments of the model and the tags, of shape `(positions,)`.
    """
    template = subject.template(prompt)
    inputs = subject.inputs(image, template + response)
    identifiers = inputs["input_ids"][0]
    tags = torch.full_like(identifiers, PROMPT)
    if response:
        start = subject.inputs(image, template)["input_ids"].shape[1]
        tags[start:] = RESPONSE
    tags[identifiers == image_token(subject)] = IMAGE
    return inputs, tags.cpu()


@contextmanager
def capture(layers: Sequence[nn.Module], blocks: Sequence[int]) -> Iterator[dict[int, Tensor]]:
    """Output of some blocks at every position of the next pass, while the context is open.

    Args:
        layers: Every block of the decoder.
        blocks: Indices of the blocks to read.

    Yields:
        A mapping filled with the output of every block, `(positions, width)`, on its device.
    """
    found: dict[int, Tensor] = {}

    def hook(index: int) -> Any:
        def read(_module: nn.Module, _inputs: Any, output: Any) -> None:
            state = output[0] if isinstance(output, tuple) else output
            found[index] = state[0].detach()

        return read

    handles = [layers[index].register_forward_hook(hook(index)) for index in blocks]
    try:
        yield found
    finally:
        for handle in handles:
            handle.remove()


def keep(states: Tensor, tags: Tensor, image_tokens: int, seed: int) -> Tensor:
    """Positions of one sequence that are kept for training.

    Args:
        states: States of shape `(positions, blocks, width)`.
        tags: Tag of every position.
        image_tokens: Image tokens to keep, chosen at random.
        seed: Seed of that choice.

    Returns:
        The indices of the kept positions, in order.
    """
    norms = states.float().norm(dim=-1)
    regular = (norms <= OUTLIER * norms.median(dim=0).values).all(dim=1).cpu()
    regular[0] = False
    images = [int(index) for index in torch.nonzero(regular & (tags == IMAGE)).flatten()]
    chosen = random.Random(seed).sample(images, min(image_tokens, len(images)))
    others = torch.nonzero(regular & (tags != IMAGE)).flatten().tolist()
    return torch.tensor(sorted(chosen + others), dtype=torch.long)


@torch.inference_mode()
def read(
    subject: HuggingFaceSubject,
    example: Example,
    blocks: Sequence[int],
    *,
    image_tokens: int = IMAGE_TOKENS,
    seed: int = 0,
) -> "Reading":
    """States of the kept positions of one example at some blocks.

    Args:
        subject: Model that reads the example.
        example: Sequence to read.
        blocks: Indices of the blocks to read.
        image_tokens: Image tokens to keep.
        seed: Seed of the choice of image tokens, mixed with the image.

    Returns:
        The reading of the example.
    """
    inputs, tags = encode(subject, open_image(example.image), example.prompt, example.response)
    with capture(subject.decoder_layers(), blocks) as found:
        output = subject.model(**inputs)
    states = torch.stack([found[block] for block in blocks], dim=1)
    mixed = int.from_bytes(hashlib.sha256(f"{seed}:{example.image}".encode()).digest()[:4])
    kept = keep(states, tags, image_tokens, mixed)
    return Reading(
        states=states[kept.to(states.device)],
        tags=tags[kept],
        positions=kept,
        identifiers=inputs["input_ids"][0].cpu(),
        inputs=inputs,
        logits=output.logits[0],
    )


@dataclass
class Reading:
    """What one pass over an example gives.

    Attributes:
        states: States of the kept positions, `(kept, blocks, width)`, on the model's device.
        tags: Tag of every kept position.
        positions: Index of every kept position in the sequence.
        identifiers: Tokens of the whole sequence.
        inputs: Keyword arguments of the model for the sequence.
        logits: Logits of the model at every position of the sequence.
    """

    states: Tensor
    tags: Tensor
    positions: Tensor
    identifiers: Tensor
    inputs: dict[str, Tensor]
    logits: Tensor
