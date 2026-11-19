"""Attention knockout: which layers carry the image to the answer of a POPE question.

In a window of consecutive decoder layers, the attention from one set of positions to another is
blocked, and the margin of yes over no at the first answer token is read again. A route whose
blocking moves the margin carries what the answer reads. Three routes are blocked in turn.

- `answer-image`: the position of the answer reading the image tokens directly.
- `answer-text`: the position of the answer reading the text after the image, the question and
  the template.
- `text-image`: that text reading the image tokens, the indirect route by which the image can
  reach the answer through the question.

The window slides over every layer. Wrong yeses are set against correct answers, to see whether a
wrong yes reads the image less, or later. Unlike the weights of attention, a knockout changes what
the model computes, so the route it finds is one the answer depends on.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from tqdm import tqdm

from swce.diagnostics.prior import margin
from swce.models.hf import HuggingFaceSubject
from swce.sae.harvest import image_token
from swce.tasks.schema import Item
from swce.tasks.task import Task

WINDOW = 9
"""Consecutive layers blocked at once."""
ROUTES = ("answer-image", "answer-text", "text-image")
"""Routes blocked in turn."""


def route_positions(identifiers: Tensor, image: int) -> dict[str, tuple[Tensor, Tensor]]:
    """Positions that read and are read for every route, from the tokens of one sequence.

    Args:
        identifiers: Tokens of the sequence, `(positions,)`.
        image: Identifier of the token standing for one image token.

    Returns:
        For every route, the positions whose attention is blocked and the positions they may no
        longer read.
    """
    count = len(identifiers)
    images = torch.nonzero(identifiers == image).flatten()
    last = torch.tensor([count - 1])
    after = torch.arange(int(images.max()) + 1, count - 1) if len(images) else torch.arange(0)
    return {
        "answer-image": (last, images),
        "answer-text": (last, after),
        "text-image": (after, images),
    }


def blocked_mask(
    count: int, queries: Tensor, keys: Tensor, *, boolean: bool, dtype: torch.dtype, device: Any
) -> Tensor:
    """Causal mask of one sequence with the attention of some positions to others removed.

    Args:
        count: Number of positions.
        queries: Positions whose attention is blocked.
        keys: Positions they may no longer read.
        boolean: Whether the attention takes a mask of allowed positions rather than an additive
            one.
        dtype: Type of an additive mask.
        device: Device of the mask.

    Returns:
        A mask of shape `(1, 1, count, count)`.
    """
    allowed = torch.ones(count, count, dtype=torch.bool, device=device).tril()
    if len(queries) and len(keys):
        allowed[queries.to(device)[:, None], keys.to(device)[None, :]] = False
    allowed.fill_diagonal_(True)
    if boolean:
        return allowed[None, None]
    additive = torch.zeros(count, count, dtype=dtype, device=device)
    additive.masked_fill_(~allowed, torch.finfo(dtype).min)
    return additive[None, None]


@contextmanager
def knocked_out(
    layers: Sequence[nn.Module], window: Sequence[int], queries: Tensor, keys: Tensor
) -> Iterator[None]:
    """Block the attention of `queries` to `keys` in the layers of a window, while open.

    The mask is built for the whole sequence of the next pass, so it suits a single pass over a
    prompt, not a generation that adds positions.
    """

    def hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        assert hidden is not None
        count = hidden.shape[1]
        config = getattr(module, "config", None)
        eager = getattr(config, "_attn_implementation", "sdpa") == "eager"
        given = kwargs.get("attention_mask")
        if given is None:
            mask = blocked_mask(
                count, queries, keys, boolean=not eager, dtype=hidden.dtype, device=hidden.device
            )
        elif given.dtype == torch.bool:
            ours = blocked_mask(
                count, queries, keys, boolean=True, dtype=hidden.dtype, device=given.device
            )
            mask = given & ours
        else:
            ours = blocked_mask(
                count, queries, keys, boolean=False, dtype=given.dtype, device=given.device
            )
            mask = torch.minimum(given, ours)
        kwargs["attention_mask"] = mask
        return args, kwargs

    handles = [
        layers[index].get_submodule("self_attn").register_forward_pre_hook(hook, with_kwargs=True)
        for index in window
    ]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def windows(layers: int, size: int = WINDOW) -> list[list[int]]:
    """Every window of `size` consecutive layers, sliding by one."""
    size = min(size, layers)
    return [list(range(start, start + size)) for start in range(layers - size + 1)]


def knockout_item(
    subject: HuggingFaceSubject, task: Task, item: Item, *, size: int = WINDOW
) -> list[dict[str, Any]]:
    """The margin of one question with every route blocked in every window."""
    inputs = subject.inputs(task.image(item), subject.template(item.prompt))
    base = margin(subject, inputs)
    routes = route_positions(inputs["input_ids"][0].cpu(), image_token(subject))
    layers = subject.decoder_layers()
    rows = []
    for route in ROUTES:
        queries, keys = routes[route]
        for window in windows(len(layers), size):
            with knocked_out(layers, window, queries, keys):
                changed = margin(subject, inputs)
            rows.append(
                {
                    "id": item.id,
                    "route": route,
                    "start": window[0],
                    "end": window[-1],
                    "margin": base,
                    "knocked": changed,
                    "change": changed - base,
                }
            )
    return rows


def knockout_task(
    subject: HuggingFaceSubject,
    task: str,
    answers: pd.DataFrame,
    *,
    per_kind: int = 200,
    size: int = WINDOW,
    seed: int = 0,
    data: Any = None,
) -> pd.DataFrame:
    """Knockout on wrong yeses and on as many correct answers of a POPE task.

    Args:
        subject: Model to read.
        task: Key of a POPE task.
        answers: Rows of `swce.diagnostics.prior.margins` for the task, whose margin with the
            image tells the answer of the model.
        per_kind: Wrong yeses and correct answers to read, at most.
        size: Layers of every window.
        seed: Seed of the sample.
        data: Data root, or None for the default one.

    Returns:
        One row per question, route and window, with the `kind` of the question, `wrong_yes` or
        `correct`.
    """
    said_yes = answers["margin"] > 0
    expected = answers["label"] == "yes"
    kinds = {
        "wrong_yes": answers[said_yes & ~expected],
        "correct": answers[said_yes == expected],
    }
    generator = np.random.default_rng(seed)
    chosen = {
        kind: set(frame["id"].iloc[generator.permutation(len(frame))[:per_kind]])
        for kind, frame in kinds.items()
    }
    bound = Task(task, data)
    rows = []
    for item in tqdm(bound.items(), desc=f"knockout {task}", mininterval=30):
        kind = next((kind for kind, ids in chosen.items() if item.id in ids), None)
        if kind is None:
            continue
        rows += [dict(row, kind=kind) for row in knockout_item(subject, bound, item, size=size)]
    return pd.DataFrame(rows)


def summarize_knockout(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean change of the margin by kind, route and window, and the share of answers it flips."""
    flipped = np.sign(frame["knocked"]) != np.sign(frame["margin"])
    return (
        frame.assign(flipped=flipped)
        .groupby(["kind", "route", "start", "end"], as_index=False)
        .agg(change=("change", "mean"), flipped=("flipped", "mean"), items=("id", "nunique"))
    )
