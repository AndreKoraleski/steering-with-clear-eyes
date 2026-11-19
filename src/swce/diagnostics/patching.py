"""Activation patching: where a wrong yes on POPE is decided.

Every wrong yes, a question whose object is absent and that the model answers yes, is paired with
a correct no to the same question on another image. LLaVA-1.5 gives every image the same number
of tokens and the text is the same, so the two sequences line up position by position. The
states of the correct run are copied into the hallucinating run, at one decoder block and one
group of positions at a time, and the margin of yes over no at the first answer token is read
again. The share of the gap between the two runs that the copy closes says how much those states
decide the answer.

Three groups of positions are patched: the image tokens, the tokens of the object named in the
question, and the position of the answer. Patching across images swaps the whole image, not only
the object, which the readings must keep in mind.

Qwen3-VL adds features of its vision encoder to the image tokens again in its first decoder
blocks, which it calls DeepStack. A patch of the image tokens at one of those blocks is partly
overwritten by the next injection from the hallucinating image, so those rows are marked
`reinjected`. The other groups of positions are never injected into.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from tqdm import tqdm

from swce.diagnostics.coco import object_of
from swce.diagnostics.prior import margin
from swce.models.hf import HuggingFaceSubject
from swce.sae.harvest import capture, image_token
from swce.tasks.task import Task

GROUPS = ("image", "object", "answer")
"""Groups of positions patched in turn."""


def find_span(identifiers: Sequence[int], piece: Sequence[int]) -> list[int]:
    """Positions of the last occurrence of a run of tokens in a sequence, or none."""
    width = len(piece)
    for start in range(len(identifiers) - width, -1, -1):
        if list(identifiers[start : start + width]) == list(piece):
            return list(range(start, start + width))
    return []


def object_positions(
    subject: HuggingFaceSubject, identifiers: Sequence[int], name: str
) -> list[int]:
    """Positions of the tokens of an object named in a sequence, tried with and without a space."""
    tokenizer = subject.processor.tokenizer
    for spelling in (f" {name}", name):
        found = find_span(identifiers, tokenizer.encode(spelling, add_special_tokens=False))
        if found:
            return found
    return []


@contextmanager
def patched(layer: nn.Module, positions: Tensor, states: Tensor) -> Iterator[None]:
    """Replace the output of a block at some positions with given states, while open.

    Args:
        layer: Decoder block.
        positions: Positions to replace.
        states: States of the other run at that block, `(positions of the run, width)`.
    """

    def hook(_module: nn.Module, _inputs: Any, output: Any) -> Any:
        state = output[0] if isinstance(output, tuple) else output
        state = state.clone()
        index = positions.to(state.device)
        state[0, index] = states.to(state.device, state.dtype)[index]
        return (state, *output[1:]) if isinstance(output, tuple) else state

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def pairs_of(answers: pd.DataFrame, count: int, seed: int = 0) -> list[tuple[str, list[str]]]:
    """Wrong yeses, each with the correct noes to the same question on other images.

    Args:
        answers: Rows of `swce.diagnostics.prior.margins`, whose margin with the image tells the
            answer of the model.
        count: Wrong yeses to draw, at most.
        seed: Seed of the draw and of the order of the partners.

    Returns:
        The identifier of every wrong yes drawn, with those of its partners in a random order.
        A model whose images give varying numbers of tokens can only be patched with a partner
        of the same length, so the partners are tried in turn.
    """
    absent = answers[answers["label"] == "no"]
    wrong = absent[absent["margin"] > 0]
    right = absent[absent["margin"] <= 0]
    generator = np.random.default_rng(seed)
    found = []
    for index in generator.permutation(len(wrong)):
        row = wrong.iloc[index]
        partners = right[(right["object"] == row["object"]) & (right["group"] != row["group"])]
        if len(partners):
            order = generator.permutation(len(partners))
            found.append((str(row["id"]), [str(partners["id"].iloc[i]) for i in order]))
        if len(found) == count:
            break
    return found


def _vision_config(subject: HuggingFaceSubject) -> Any:
    """Configuration of the vision encoder of a subject, or None."""
    model: Any = subject.model
    return getattr(model.config, "vision_config", None)


def patch_pair(
    subject: HuggingFaceSubject, task: Task, wrong: Any, right: Any
) -> list[dict[str, Any]]:
    """Patch the correct run of a pair into its hallucinating run at every block and group."""
    bad = subject.inputs(task.image(wrong), subject.template(wrong.prompt))
    good = subject.inputs(task.image(right), subject.template(right.prompt))
    if bad["input_ids"].shape != good["input_ids"].shape:
        return []
    layers = subject.decoder_layers()
    blocks = list(range(len(layers)))
    with capture(layers, blocks) as found:
        good_margin = margin(subject, good)
    states = {block: found[block].clone() for block in blocks}
    bad_margin = margin(subject, bad)
    identifiers = bad["input_ids"][0].tolist()
    image = image_token(subject)
    groups = {
        "image": [i for i, token in enumerate(identifiers) if token == image],
        "object": object_positions(subject, identifiers, object_of(wrong.prompt)),
        "answer": [len(identifiers) - 1],
    }
    gap = bad_margin - good_margin
    injected = len(getattr(_vision_config(subject), "deepstack_visual_indexes", None) or [])
    rows = []
    for group in GROUPS:
        positions = torch.tensor(groups[group], dtype=torch.long)
        if not len(positions):
            continue
        for block in blocks:
            with patched(layers[block], positions, states[block]):
                changed = margin(subject, bad)
            rows.append(
                {
                    "wrong": wrong.id,
                    "right": right.id,
                    "group": group,
                    "block": block,
                    "positions": len(positions),
                    "margin": bad_margin,
                    "target": good_margin,
                    "patched": changed,
                    "restored": (bad_margin - changed) / gap if gap else float("nan"),
                    "reinjected": group == "image" and block < injected,
                }
            )
    return rows


def patching_task(
    subject: HuggingFaceSubject,
    task: str,
    answers: pd.DataFrame,
    *,
    pairs: int = 200,
    tries: int = 20,
    seed: int = 0,
    data: Any = None,
) -> pd.DataFrame:
    """Patching on pairs of wrong yeses and correct noes of a POPE task.

    Every wrong yes is patched with the first of its partners whose sequence has the same length,
    trying at most `tries` of them. On LLaVA-1.5, whose images all give 576 tokens, the first
    partner always fits. Draws continue until `pairs` wrong yeses are patched or none is left.

    Returns:
        One row per pair, group and block, with the share of the gap of the margin the patch
        restores, one when the patched run answers as the correct one does, and the number of
        partners tried before one fit.
    """
    bound = Task(task, data)
    items = {item.id: item for item in bound.items()}
    rows: list[dict[str, Any]] = []
    done = 0
    progress = tqdm(total=pairs, desc=f"patching {task}", mininterval=30)
    for wrong, partners in pairs_of(answers, len(answers), seed):
        for tried, right in enumerate(partners[:tries], start=1):
            found = patch_pair(subject, bound, items[wrong], items[right])
            if found:
                rows += [dict(row, tried=tried) for row in found]
                done += 1
                progress.update()
                break
        if done == pairs:
            break
    progress.close()
    return pd.DataFrame(rows)


def summarize_patching(frame: pd.DataFrame) -> pd.DataFrame:
    """Median and mean restored share by group and block, and the share of answers flipped."""
    flipped = np.sign(frame["patched"]) != np.sign(frame["margin"])
    return (
        frame.assign(flipped=flipped)
        .groupby(["group", "block"], as_index=False)
        .agg(
            restored=("restored", "median"),
            restored_mean=("restored", "mean"),
            flipped=("flipped", "mean"),
            pairs=("wrong", "nunique"),
        )
    )
