"""REVIS, a gated visual direction added at one decoder layer.

Source: Wu et al., REVIS: Sparse Latent Steering to Mitigate Object Hallucination in Large
    Vision-Language Models, ICML 2026, arXiv 2602.11824. Official code at
    https://github.com/antgroup/Revis, commit 75f4d29d402d4c1c4dbb2c69e88e3aa50cd0c9cc.
Licence: Apache 2.0.
Settings: Appendix C.3 of the paper sets the strength to 1.1 for LLaVA-1.5, and Section 5.1 sets
    it to 1.6 for Qwen2.5-VL-7B. No strength is published for any other model, on which the
    default of 2.5 of the official script is used. The risk factor of 1.0 is the default of the
    official code.

The method has three steps, each available here for any model.

1. `compute_vectors` builds the visual vector of every layer from the 100 pairs of faithful and
   hallucinated descriptions the authors distribute. For LLaVA-1.5 the vectors the authors
   release are downloaded instead, pinned by their digest.
2. `calibrate` selects the layer and the gate threshold on the calibration questions the authors
   distribute, as `layer_select.py` does. The paper names layer 31 in Appendix C.3 and reports its
   main numbers at layer 30 in Table 11 for LLaVA-1.5 and gives no threshold for it, so the
   calibration is the only source of both.
3. The hook adds the vector wherever the risk, the negated cosine similarity between the residual
   stream and the vector, exceeds the threshold. The vector keeps its norm.

The calibration defaults are those of `layer_select.sh`, the 85th percentile over 300 questions,
searching down to layer 10. The paper states a value of 0.8 for the percentile and 100 images.

The official script answers the calibration questions once per layer, although the answers do not
depend on the layer under observation. They are generated once here and every layer is observed
in the same pass, which gives the same risks.

The official code holds two prompt constructions for Qwen models, and its entry point routes
every Qwen checkpoint to the one written for Qwen2.5. The one written for Qwen3 is used here for
Qwen3-VL. The description standing for an answer without knowledge is drawn at random without a
seed in the official code, and with a seed here.
"""

import json
import random
from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from sklearn.decomposition import PCA
from torch import Tensor
from tqdm import tqdm

from swce.data.download import fetch, fetch_images
from swce.data.paths import artifacts_root
from swce.models.subject import Subject
from swce.registry import load_registry
from swce.steering.hooks import Edit, edit_layers
from swce.steering.steerer import InapplicableError, Steerer

RELEASED = {"llava-1.5-7b": load_registry().file("methods/revis/llava15_none_image.pt")}
"""Vectors the authors release, by subject name."""
PAIRS = load_registry().file("methods/revis/OH_sampled_data_100.json")
"""Pairs of faithful and hallucinated descriptions of 100 COCO train2014 images."""
QUESTIONS = load_registry().file("methods/revis/coco_pope_calibration_merged.jsonl")
"""Object presence questions on COCO train2014 images used by the calibration."""
QUESTION = "Describe the image in detail."
"""Question the official code asks with every description."""
UNKNOWN = (
    "I cannot answer this question because there is no image provided.",
    "Without an image, I cannot describe the scene.",
    "Please provide an image so I can describe it.",
    "I don't see any image here.",
    "The visual information is missing.",
)
"""Descriptions standing for an answer without knowledge, one drawn for every pair."""


def gate(vector: Tensor, strength: float, tau: float, gamma: float) -> Edit:
    """Edit of the official `MADRSteeringHook_visual_only`.

    Args:
        vector: Visual vector of the layer, of shape `(width,)`, with its own norm.
        strength: Coefficient of the vector where the gate opens.
        tau: Risk above which the gate opens.
        gamma: Factor of the risk.

    Returns:
        An edit that adds the scaled vector at the positions whose risk exceeds the threshold.
    """

    def edit(hidden: Tensor) -> Tensor:
        direction = vector.to(device=hidden.device, dtype=hidden.dtype)
        confidence = torch.matmul(F.normalize(hidden, dim=-1), F.normalize(direction, dim=-1))
        risk = -(gamma * confidence).unsqueeze(-1)
        # The official hook scales the vector in single precision and adds the result to a
        # delta of the type of the stream, which rounds differently from scaling in half
        # precision.
        delta = torch.zeros_like(hidden)
        delta += ((risk > tau).float() * strength) * direction
        return hidden + delta

    return edit


def training_images(names: Iterable[str]) -> Path:
    """Folder of the COCO train2014 images a step reads, fetched from the official host.

    Args:
        names: File names of the images.

    Returns:
        The folder holding the images under the data root.
    """
    return fetch_images(load_registry().benchmark("coco-train2014"), names)


def vectors_path(subject: str) -> Path:
    """File of the vectors computed here for a subject.

    Args:
        subject: Name of the subject.

    Returns:
        The path under the artifacts folder of the data root.
    """
    return artifacts_root() / f"revis_{subject}.pt"


def calibration_path(subject: str) -> Path:
    """File of the calibration outcome of a subject, with the keys `layer` and `tau`.

    Args:
        subject: Name of the subject.

    Returns:
        The path under the artifacts folder of the data root.
    """
    return artifacts_root() / f"revis_{subject}.json"


def load_vectors(subject: str, path: Path | None = None) -> dict[int, dict[str, Tensor]]:
    """Vectors of a subject by hidden state index, one above the block index.

    Args:
        subject: Name of the subject.
        path: File of vectors, or None for the computed ones, else the released ones.

    Returns:
        For every hidden state index, the vectors named `truth`, `language` and `visual`.

    Raises:
        FileNotFoundError: If no vector is released for the subject and none was computed.
    """
    if path is None:
        path = vectors_path(subject)
        if not path.exists():
            if subject not in RELEASED:
                raise FileNotFoundError(f"compute the vectors of REVIS for {subject} first")
            path = fetch(RELEASED[subject])
    loaded: dict[int, dict[str, Tensor]] = torch.load(path, map_location="cpu", weights_only=True)
    return loaded


class Revis(Steerer):
    """REVIS at one setting.

    Attributes:
        strength: Coefficient of the vector.
        layer: Index of the decoder block whose output is steered, or None for the one the
            calibration of the subject selects.
        tau: Risk above which the gate opens, or None for the calibrated one.
        gamma: Factor of the risk.
        vectors: File of vectors by hidden state index, or None for the vectors computed for the
            subject, else the released ones.
    """

    name: ClassVar[str] = "revis"
    description: ClassVar[str] = "REVIS (Wu et al., ICML 2026), gated visual vector at one layer."

    strength: float
    layer: int | None = None
    tau: float | None = None
    gamma: float = 1.0
    vectors: Path | None = None
    fitted_on: tuple[str, ...] | None = ("lure-pairs", "coco-train2014")

    def prepare(self, subject: Subject) -> None:
        """Compute the vectors and run the calibration of the subject when they are missing.

        Args:
            subject: Model about to be steered.

        Raises:
            InapplicableError: If no layer of the subject passes the calibration.
        """
        released = self.vectors is not None or subject.name in RELEASED
        if not released and not vectors_path(subject.name).exists():
            compute_vectors(subject)
        calibrated = self.layer is not None and self.tau is not None
        if not calibrated and not calibration_path(subject.name).exists():
            calibrate(subject, vectors=self.vectors)

    def attach(self, subject: Subject) -> AbstractContextManager[None]:
        """Add the gated vector to the output of one decoder block.

        Args:
            subject: Model to steer.

        Returns:
            A context inside which the block is steered.

        Raises:
            FileNotFoundError: If the subject has neither released nor computed vectors, or
                was not calibrated while the setting leaves the layer or the threshold open.
        """
        return self._attach(subject)

    def _attach(self, subject: Subject) -> AbstractContextManager[None]:
        """Add the gated vector of `vectors` at the calibrated layer and threshold."""
        layer, tau = self.layer, self.tau
        if layer is None or tau is None:
            path = calibration_path(subject.name)
            if not path.exists():
                raise FileNotFoundError(f"calibrate {self.name} for {subject.name} first")
            calibrated = json.loads(path.read_text(encoding="utf-8"))
            layer = int(calibrated["layer"]) if layer is None else layer
            tau = float(calibrated["tau"]) if tau is None else tau
        vector = load_vectors(subject.name, self.vectors)[layer + 1]["visual"]
        edit = gate(vector, self.strength, tau, self.gamma)
        return edit_layers(subject.decoder_layers(), {layer: edit})


class RevisRandom(Revis):
    """REVIS with its visual vector replaced by a random direction of the same norm.

    The layer, the threshold, the strength and the gate are those of the method. A gain this
    control also obtains does not come from what the visual vector carries.

    Attributes:
        seed: Seed of the draw of the directions.
    """

    name: ClassVar[str] = "revis-random"
    description: ClassVar[str] = "REVIS with a random vector of the same norm, as a control."

    seed: int = 0
    fitted_on: tuple[str, ...] | None = ()

    def prepare(self, subject: Subject) -> None:
        """Run the recipe of the method, then draw the random vectors of the subject.

        Args:
            subject: Model about to be steered.
        """
        super().prepare(subject)
        path = self._random_path(subject.name)
        if not path.exists():
            loaded = load_vectors(subject.name, self.vectors)
            random_vectors = {}
            for index, named in loaded.items():
                vector = named["visual"]
                generator = torch.Generator().manual_seed(self.seed * 1_000_003 + index)
                drawn = torch.randn(vector.shape, generator=generator, dtype=torch.float64)
                scaled = (drawn / drawn.norm() * vector.double().norm()).to(vector.dtype)
                random_vectors[index] = {"visual": scaled}
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(random_vectors, path)

    def _random_path(self, subject: str) -> Path:
        """File of the random vectors of a subject for the seed of the control."""
        return artifacts_root() / f"revis_{subject}_random{self.seed}.pt"

    def attach(self, subject: Subject) -> AbstractContextManager[None]:
        """Add the gated random vector to the output of the block of the method.

        Args:
            subject: Model to steer.

        Returns:
            A context inside which the block is steered.

        Raises:
            FileNotFoundError: If the control was not prepared for the subject.
        """
        path = self._random_path(subject.name)
        if not path.exists():
            raise FileNotFoundError(f"prepare {self.name} for {subject.name} first")
        return self.model_copy(update={"vectors": path})._attach(subject)


def principal(samples: Tensor) -> Tensor:
    """First principal component of the samples, scaled to their mean norm.

    This is the official `get_pure_direction_with_pca`, including its two special cases.

    Args:
        samples: Vectors of shape `(samples, width)`.

    Returns:
        The mean of fewer than two samples, zero for samples without variance, and otherwise
        the component, of shape `(width,)`.
    """
    if samples.shape[0] < 2:
        return samples.mean(dim=0)
    if torch.allclose(samples.std(dim=0), torch.zeros_like(samples[0]), atol=1e-6):
        return torch.zeros_like(samples[0])
    component = PCA(n_components=1).fit(samples.cpu().float().numpy()).components_[0]
    norm = torch.linalg.norm(samples, dim=1).mean().item()
    direction: Tensor = (torch.from_numpy(component) * norm).float()
    return direction


def combine(
    faithful: Tensor, hallucinated: Tensor, plain: Tensor, prior: Tensor, unknown: Tensor
) -> dict[str, Tensor]:
    """Vectors of one layer from the hidden states of the pairs.

    Args:
        faithful: States of the faithful descriptions with their image, `(pairs, width)`.
        hallucinated: States of the hallucinated descriptions with their image.
        plain: States of the faithful descriptions without the image.
        prior: States of the hallucinated descriptions without the image.
        unknown: States of the descriptions without knowledge, without the image.

    Returns:
        The vectors `truth`, `language` and `visual`. The visual vector is the principal
        component of the difference an image makes to a faithful description, once its
        projection on the direction of the language prior is removed from every sample.
    """
    direction = F.normalize(principal(prior - unknown), dim=-1)
    raw = faithful - plain
    purified = raw - (raw @ direction.unsqueeze(-1)) * direction.unsqueeze(0)
    return {
        "truth": principal(faithful - hallucinated),
        "language": principal(unknown - prior),
        "visual": principal(purified),
    }


def _states(subject: Any, question: str, answer: str, image: Image.Image | None) -> Tensor:
    """Hidden states at the last position, as the official code builds its prompts."""
    processor = subject.processor
    if subject.name.startswith("qwen3"):
        # The official branch for Qwen3 keeps an empty image entry and its own question when
        # there is no image.
        text = question if image is not None else "Describe this image."
        user = [{"type": "image", "image": image}, {"type": "text", "text": text}]
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        encoded = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
    else:
        if subject.name.startswith("llava-1.5"):
            # The official prompt of LLaVA-1.5 keeps the image placeholder without an image.
            prompt = f"USER: <image>\n{question} ASSISTANT: {answer}"
        else:
            content = [{"type": "text", "text": question}]
            if image is not None:
                content.insert(0, {"type": "image"})
            messages = [
                {"role": "user", "content": content},
                {"role": "assistant", "content": answer},
            ]
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        if image is None:
            # The official code calls the processor without an image. Recent versions of the
            # library refuse a text that still holds an image placeholder, where earlier ones
            # tokenized it, so the tokenizer is called directly.
            encoded = processor.tokenizer([prompt], return_tensors="pt", padding=True)
        else:
            encoded = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
    found: Tensor = subject.all_states(subject.moved(encoded))[:, -1]
    return found


def compute_vectors(
    subject: Any, images: Path | None = None, output: Path | None = None, seed: int = 0
) -> Path:
    """Compute the vectors of a subject from the pairs the authors distribute.

    This follows `vector_calculator.py` and has not been run on real weights yet.

    Args:
        subject: A `HuggingFaceSubject`.
        images: Folder holding the COCO train2014 images the pairs name, or None to fetch
            them from the official host.
        output: File to write the vectors to, or None for the artifacts folder.
        seed: Seed of the draw of the descriptions without knowledge.

    Returns:
        The path of the written file, keyed by hidden state index.
    """
    pairs = json.loads(fetch(PAIRS).read_text(encoding="utf-8"))["annotations"]
    images = images or training_images(f"{pair['image_id']}.jpg" for pair in pairs)
    generator = random.Random(seed)
    names = ("faithful", "hallucinated", "plain", "prior", "unknown")
    states: dict[str, list[Tensor]] = {name: [] for name in names}
    for pair in tqdm(pairs, desc="REVIS vectors"):
        path = images / f"{pair['image_id']}.jpg"
        if not path.exists():
            continue
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        states["faithful"].append(_states(subject, QUESTION, pair["caption"], image))
        states["hallucinated"].append(_states(subject, QUESTION, pair["h_caption"], image))
        states["plain"].append(_states(subject, QUESTION, pair["caption"], None))
        states["prior"].append(_states(subject, QUESTION, pair["h_caption"], None))
        states["unknown"].append(_states(subject, QUESTION, generator.choice(UNKNOWN), None))
    stacked = {name: torch.stack(found) for name, found in states.items()}
    layers = stacked["faithful"].shape[1]
    vectors = {
        layer: combine(*(stacked[name][:, layer] for name in names)) for layer in range(layers)
    }
    output = output or vectors_path(subject.name)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vectors, output)
    return output


def select(
    risks: Mapping[int, Sequence[float]],
    categories: Sequence[str],
    *,
    percentile: float,
    min_layer: int,
) -> tuple[int, float] | None:
    """Layer and threshold the official backward search selects.

    Args:
        risks: Risk of every answered question at the last prompt position, by block index.
        categories: Outcome of every question, `TP`, `FP`, `TN` or `Other`.
        percentile: Percentile of the risks of correct answers taken as the threshold, from 0
            to 100.
        min_layer: Lowest block index the search reaches.

    Returns:
        The deepest block whose mean risk is higher on false positives than on correct answers,
        with its threshold, or None when no block qualifies.
    """
    for layer in sorted(risks, reverse=True):
        if layer < min_layer:
            break
        factual = [r for r, c in zip(risks[layer], categories, strict=True) if c in ("TP", "TN")]
        hallucinated = [r for r, c in zip(risks[layer], categories, strict=True) if c == "FP"]
        if not factual or not hallucinated:
            continue
        if np.mean(hallucinated) - np.mean(factual) > 0:
            return layer, float(np.percentile(factual, percentile))
    return None


def calibrate(
    subject: Subject,
    images: Path | None = None,
    *,
    vectors: Path | None = None,
    questions: Path | None = None,
    output: Path | None = None,
    percentile: float = 85.0,
    max_samples: int = 300,
    min_layer: int = 10,
) -> tuple[int, float]:
    """Select the layer and the threshold of a subject and write them.

    Args:
        subject: Model to calibrate.
        images: Folder holding the COCO train2014 images the questions name, or None to
            fetch them from the official host.
        vectors: File of vectors, or None for the computed ones, else the released ones.
        questions: File of calibration questions, or None for the one the authors distribute.
        output: File to write the outcome to, or None for the artifacts folder.
        percentile: Percentile of the risks of correct answers taken as the threshold.
        max_samples: Number of leading questions used.
        min_layer: Lowest block index the search reaches.

    Returns:
        The selected block index and threshold.

    Raises:
        InapplicableError: If no block separates false positives from correct answers.
    """
    loaded = load_vectors(subject.name, vectors)
    lines = (questions or fetch(QUESTIONS)).read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()][:max_samples]
    images = images or training_images(dict.fromkeys(row["image"] for row in rows))
    blocks = subject.decoder_layers()
    observed = [layer for layer in range(min_layer, len(blocks)) if layer + 1 in loaded]
    captured: dict[int, float] = {}

    def observe(layer: int) -> Edit:
        vector = loaded[layer + 1]["visual"]

        def edit(hidden: Tensor) -> Tensor:
            # Only the first pass of a generation, over the whole prompt, is observed.
            if layer not in captured:
                last = hidden[:, -1, :]
                unit = F.normalize(vector.to(device=last.device, dtype=last.dtype), dim=-1)
                captured[layer] = -torch.matmul(F.normalize(last, dim=-1), unit).item()
            return hidden

        return edit

    risks: dict[int, list[float]] = {layer: [] for layer in observed}
    categories: list[str] = []
    outcomes = {("yes", "yes"): "TP", ("no", "yes"): "FP", ("no", "no"): "TN"}
    with edit_layers(blocks, {layer: observe(layer) for layer in observed}):
        for row in tqdm(rows, desc="REVIS calibration"):
            captured.clear()
            with Image.open(images / row["image"]) as opened:
                image = opened.convert("RGB")
            prompt = row["text"] + " Please answer yes or no."
            text = subject.generate(image, prompt, 5, settings={"do_sample": False}).text.lower()
            answer = "yes" if "yes" in text else ("no" if "no" in text else "unknown")
            categories.append(outcomes.get((row["label"].lower(), answer), "Other"))
            for layer in observed:
                risks[layer].append(captured[layer])
    selected = select(risks, categories, percentile=percentile, min_layer=min_layer)
    if selected is None:
        raise InapplicableError(
            f"no layer of {subject.name} separates false positives from correct answers"
        )
    output = output or calibration_path(subject.name)
    output.parent.mkdir(parents=True, exist_ok=True)
    outcome = {"layer": selected[0], "tau": selected[1], "percentile": percentile}
    output.write_text(json.dumps(outcome, indent=2) + "\n", encoding="utf-8", newline="\n")
    return selected


def build() -> list[Steerer]:
    """Setting the paper publishes for LLaVA-1.5, and the defaults of the code for other models.

    Both leave the layer and the threshold to the calibration, as the recipe does.

    Returns:
        The published strength of 1.1, and the default strength of 2.5 of the official script,
        used on models without a published setting.
    """
    return [
        Revis(strength=1.1, published_for=("llava-1.5-7b",)),
        Revis(setting="code-default", strength=2.5, fallback=True),
    ]


def control() -> list[Steerer]:
    """Random vector controls of the settings of `build`.

    Returns:
        One control per setting, with the same strength and models.
    """
    return [
        RevisRandom(**setting.model_dump(exclude={"fitted_on"}))
        for setting in build()
        if isinstance(setting, Revis)
    ]
