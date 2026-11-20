"""Where the answer looks: norm-weighted attention from the answer to the image on POPE.

The weight of attention says how much a head mixes a position in, not how much that position
changes the output. A few image tokens of LLaVA receive large weights whatever the question,
and add little. Weighting each weight by the norm of what the head writes from that position,
the transformed value of Kobayashi et al. (Attention is Not Only a Weight, 2020), measures what
actually reaches the answer. The shares are summed over the heads of every layer.

The image tokens are mapped back to the image through the geometry of the processor, so the
attention can be split between the region of the object asked about, when it is present, and
the regions of the objects of the image. The processor of LLaVA-1.5 on the Hugging Face hub
resizes the shortest side of an image to 336 pixels and crops the centre, where the original
code pads the image to a square, so a part of a wide or tall image is never seen. The share of
the boxes of every object inside the crop is reported as well.
"""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from tqdm import tqdm

from swce.diagnostics.coco import object_of
from swce.models.hf import HuggingFaceSubject
from swce.sae.harvest import image_token
from swce.tasks.task import Task


class Grid:
    """Where a box of the original image lands on the grid of image tokens.

    Attributes:
        shortest: Length the shortest side is resized to.
        crop: Side of the square centre crop.
        patch: Side of a patch.
        pad: Whether the image is padded to a square instead of cropped.
    """

    def __init__(self, shortest: int, crop: int, patch: int, *, pad: bool = False) -> None:
        """Describe the geometry of a processor."""
        self.shortest, self.crop, self.patch, self.pad = shortest, crop, patch, pad

    @classmethod
    def of(cls, subject: HuggingFaceSubject) -> "Grid":
        """Geometry of the processor of a subject."""
        processor = subject.processor
        images = processor.image_processor
        size = images.size
        shortest = getattr(size, "shortest_edge", None) or size["shortest_edge"]
        crop = getattr(images.crop_size, "height", None) or images.crop_size["height"]
        pad = bool(getattr(images, "do_pad", False)) and not getattr(images, "do_center_crop", True)
        return cls(int(shortest), int(crop), int(processor.patch_size), pad=pad)

    @property
    def side(self) -> int:
        """Patches along each side of the grid."""
        return self.crop // self.patch

    def place(self, width: int, height: int, box: Sequence[float]) -> tuple[float, ...]:
        """A box of the original image in the pixels of the crop, unclipped."""
        x, y, w, h = box
        if self.pad:
            side = max(width, height)
            scale = self.crop / side
            shift_x, shift_y = (side - width) / 2 * scale, (side - height) / 2 * scale
        else:
            scale = self.shortest / min(width, height)
            shift_x = -(width * scale - self.crop) / 2
            shift_y = -(height * scale - self.crop) / 2
        return (x * scale + shift_x, y * scale + shift_y, w * scale, h * scale)

    def visible(self, width: int, height: int, box: Sequence[float]) -> float:
        """Share of the area of a box inside the crop."""
        x, y, w, h = self.place(width, height, box)
        if w <= 0 or h <= 0:
            return 0.0
        inside_w = max(0.0, min(x + w, self.crop) - max(x, 0.0))
        inside_h = max(0.0, min(y + h, self.crop) - max(y, 0.0))
        # Clipped, since rounding can put a box that fits whole a hair above one.
        return min(1.0, inside_w * inside_h / (w * h))

    def cells(self, width: int, height: int, box: Sequence[float]) -> list[int]:
        """Image tokens, in row-major order, whose patch overlaps a box."""
        x, y, w, h = self.place(width, height, box)
        left, top = max(0.0, x), max(0.0, y)
        right, bottom = min(float(self.crop), x + w), min(float(self.crop), y + h)
        if right <= left or bottom <= top:
            return []
        first_col, last_col = int(left // self.patch), int((right - 1e-9) // self.patch)
        first_row, last_row = int(top // self.patch), int((bottom - 1e-9) // self.patch)
        return [
            row * self.side + col
            for row in range(first_row, last_row + 1)
            for col in range(first_col, last_col + 1)
        ]


class ResizedGrid:
    """Where a box lands on the tokens of a processor that resizes without cropping.

    Qwen3-VL and LLaVA-OneVision-1.5 resize every image to a whole number of patches, keeping its
    aspect, and merge each square of `merge` by `merge` patches into one token of the language
    model, in row-major order. The processor returns the size of the resized image in patches as
    `image_grid_thw`, which fixes where every token lies.

    Attributes:
        rows: Patches along the height of the resized image.
        columns: Patches along its width.
        patch: Side of a patch in pixels.
        merge: Patches merged along each side into one token.
    """

    def __init__(self, rows: int, columns: int, patch: int, merge: int) -> None:
        """Describe the grid of one image."""
        self.rows, self.columns, self.patch, self.merge = rows, columns, patch, merge

    @classmethod
    def of(cls, subject: HuggingFaceSubject, inputs: Mapping[str, Tensor]) -> "ResizedGrid":
        """Grid of the image of one input, from the grid its processor returned."""
        _, rows, columns = (int(value) for value in inputs["image_grid_thw"][0].tolist())
        model: Any = subject.model
        vision = model.config.vision_config
        return cls(rows, columns, int(vision.patch_size), int(vision.spatial_merge_size))

    def _place(self, width: int, height: int, box: Sequence[float]) -> tuple[float, ...]:
        x, y, w, h = box
        scale_x = self.columns * self.patch / width
        scale_y = self.rows * self.patch / height
        return (x * scale_x, y * scale_y, w * scale_x, h * scale_y)

    def visible(self, width: int, height: int, box: Sequence[float]) -> float:
        """Share of a box the resized image keeps, which is all of it."""
        return 1.0

    def cells(self, width: int, height: int, box: Sequence[float]) -> list[int]:
        """Tokens of the language model, in row-major order, whose square overlaps a box."""
        x, y, w, h = self._place(width, height, box)
        cell = self.patch * self.merge
        across, down = self.columns // self.merge, self.rows // self.merge
        left, top = max(0.0, x), max(0.0, y)
        right, bottom = min(float(across * cell), x + w), min(float(down * cell), y + h)
        if right <= left or bottom <= top:
            return []
        first_col, last_col = int(left // cell), int((right - 1e-9) // cell)
        first_row, last_row = int(top // cell), int((bottom - 1e-9) // cell)
        return [
            row * across + col
            for row in range(first_row, last_row + 1)
            for col in range(first_col, last_col + 1)
        ]


def head_value_norms(values: Tensor, output: Tensor, heads: int) -> Tensor:
    """Norm of what every head writes from every position, `(heads, positions)`.

    Args:
        values: Output of the value projection, `(positions, key value heads * dimension)`.
        output: Weight of the output projection, `(width, heads * dimension)`.
        heads: Number of attention heads. Key value heads are repeated when there are fewer.
    """
    dimension = output.shape[1] // heads
    groups = values.shape[1] // dimension
    norms = []
    for head in range(heads):
        value = values[
            :, (head * groups // heads) * dimension : (head * groups // heads + 1) * dimension
        ]
        block = output[:, head * dimension : (head + 1) * dimension]
        norms.append((value.float() @ block.float().T).norm(dim=-1))
    return torch.stack(norms)


@contextmanager
def eager(subject: HuggingFaceSubject) -> Iterator[None]:
    """Compute attention eagerly while open, so that its weights are returned.

    Models written as remote code may ignore the switch, so the forward pass must still ask for
    the weights with `output_attentions`.
    """
    model: Any = subject.model
    before = model.config._attn_implementation
    model.set_attn_implementation("eager")
    try:
        yield
    finally:
        model.set_attn_implementation(before)


@torch.inference_mode()
def answer_attention(subject: HuggingFaceSubject, inputs: Mapping[str, Tensor]) -> Tensor:
    """Norm-weighted attention of the answer position to every position, by layer.

    Returns:
        A tensor of shape `(layers, positions)` whose rows sum to one, in single precision on the
        processor.
    """
    layers = subject.decoder_layers()
    weights: dict[int, Tensor] = {}
    values: dict[int, Tensor] = {}

    def attention_hook(index: int) -> Any:
        def read(_module: nn.Module, _inputs: Any, output: Any) -> None:
            weights[index] = output[1][0, :, -1].float()

        return read

    def value_hook(index: int) -> Any:
        def read(_module: nn.Module, _inputs: Any, output: Any) -> None:
            values[index] = output[0]

        return read

    handles = []
    for index, layer in enumerate(layers):
        attention = layer.get_submodule("self_attn")
        handles.append(attention.register_forward_hook(attention_hook(index)))
        handles.append(attention.get_submodule("v_proj").register_forward_hook(value_hook(index)))
    try:
        with eager(subject):
            subject.model(**inputs, output_attentions=True)
    finally:
        for handle in handles:
            handle.remove()
    rows = []
    for index, layer in enumerate(layers):
        attention = layer.get_submodule("self_attn")
        heads = weights[index].shape[0]
        projection: Tensor = attention.get_submodule("o_proj").weight  # type: ignore[assignment]
        norms = head_value_norms(values[index], projection, heads)
        weighted = (weights[index] * norms.to(weights[index].device)).sum(dim=0)
        rows.append(weighted / weighted.sum())
    return torch.stack(rows).cpu()


def attention_task(
    subject: HuggingFaceSubject,
    task: str,
    answers: pd.DataFrame,
    boxes: Mapping[str, tuple[int, int, list[tuple[str, list[float]]]]],
    *,
    data: Any = None,
) -> pd.DataFrame:
    """Shares of the norm-weighted attention of the answer on the regions of every image.

    Args:
        subject: Model to read.
        task: Key of a POPE task.
        answers: Rows of `swce.diagnostics.prior.margins`, whose margin with the image tells the
            answer of the model.
        boxes: Size and boxes of the images of the task, by file name.
        data: Data root, or None for the default one.

    Returns:
        One row per question and layer, with the `kind` of the answer, and the shares of the
        attention on the first position, on the image, on the object asked about and on the other
        objects of the image, with the share of the object inside the crop.
    """
    bound = Task(task, data)
    fixed = None if not subject.fixed_grid else Grid.of(subject)
    margins = dict(zip(answers["id"].astype(str), answers["margin"], strict=True))
    image = image_token(subject)
    rows = []
    for item in tqdm(bound.items(), desc=f"attention {task}", mininterval=30):
        if item.id not in margins:
            continue
        inputs = subject.inputs(bound.image(item), subject.template(item.prompt))
        identifiers = inputs["input_ids"][0].cpu()
        positions = torch.nonzero(identifiers == image).flatten()
        shares = answer_attention(subject, inputs)
        grid: Grid | ResizedGrid = fixed if fixed is not None else ResizedGrid.of(subject, inputs)
        width, height, objects = boxes.get(item.group, (1, 1, []))
        target = object_of(item.prompt)
        asked = [
            cell
            for name, box in objects
            if name == target
            for cell in grid.cells(width, height, box)
        ]
        others = [
            cell
            for name, box in objects
            if name != target
            for cell in grid.cells(width, height, box)
        ]
        visible = [grid.visible(width, height, box) for name, box in objects if name == target]
        said_yes = margins[item.id] > 0
        expected = item.label == "yes"
        kind = {
            (True, True): "correct_yes",
            (True, False): "wrong_yes",
            (False, False): "correct_no",
            (False, True): "missed",
        }[(said_yes, expected)]
        asked_positions = positions[sorted(set(asked))] if asked else positions[:0]
        other_positions = positions[sorted(set(others) - set(asked))] if others else positions[:0]
        for layer in range(shares.shape[0]):
            row = shares[layer]
            rows.append(
                {
                    "id": item.id,
                    "kind": kind,
                    "layer": layer,
                    "first": float(row[0]),
                    "image": float(row[positions].sum()),
                    "object": float(row[asked_positions].sum()),
                    "others": float(row[other_positions].sum()),
                    "object_cells": len(set(asked)),
                    "object_visible": float(np.mean(visible)) if visible else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def summarize_attention(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean shares by kind and layer, the object share over questions whose object is visible."""
    summary = frame.groupby(["kind", "layer"], as_index=False).agg(
        first=("first", "mean"),
        image=("image", "mean"),
        others=("others", "mean"),
        questions=("id", "nunique"),
    )
    seen = frame[frame["object_cells"] > 0]
    objects = seen.groupby(["kind", "layer"], as_index=False).agg(object=("object", "mean"))
    return summary.merge(objects, on=["kind", "layer"], how="left")


def visibility(frame: pd.DataFrame) -> pd.DataFrame:
    """Share of the present objects of the task by how much of their boxes the crop keeps."""
    present = frame[(frame["layer"] == 0) & frame["object_visible"].notna()]
    bins = [-0.01, 0.0, 0.5, 0.99, 1.0]
    labels = ["outside", "under half", "partly", "whole"]
    counts = pd.cut(present["object_visible"], bins=bins, labels=labels).value_counts(sort=False)
    return pd.DataFrame({"visible": counts.index.astype(str), "questions": counts.to_numpy()})
