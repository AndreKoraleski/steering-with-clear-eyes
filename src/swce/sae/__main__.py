"""Train sparse autoencoders on the residual stream of a model and read the methods through them.

`swce-sae train --subject MODULE:FACTORY --blocks B ...` trains one TopK autoencoder per block
and seed, online, on the corpus of `swce.sae.harvest`, and stores them under
`artifacts/sae/MODEL` in the data root, with their training curves.

`swce-sae evaluate --subject MODULE:FACTORY --blocks B ... METHOD ...` runs the four checks of
`swce.sae.checks` and the two readings, and writes them under `sae/MODEL` in the results
directory.

- Which latents the textual direction of VTI moves at every block, read from the decoder rows
  and from the change of the latents on the POPE questions, and confirmed against the direction
  itself and the mean change of the states, which use no autoencoder.
- Which latents tell a hallucinated yes from a correct yes on POPE, chosen on half the images,
  read on the other half, and confirmed against the difference of the mean states.
"""

import argparse
import fnmatch
import heapq
import json
import os
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from tqdm import tqdm

from swce.baselines.reference import Unmodified
from swce.data.paths import DATA_ENV, artifacts_root
from swce.diagnostics.lens import NO_WORDS, YES_WORDS, observe
from swce.diagnostics.lens import read as unembedded
from swce.evaluation.__main__ import load_factory, load_steerers
from swce.evaluation.runner import prepare
from swce.logging import configure_logging
from swce.models.hf import HuggingFaceSubject
from swce.models.subject import residual_width
from swce.sae.checks import (
    Variance,
    best_match,
    cosines,
    explained_by,
    replication,
    separating_latents,
    unexplained,
)
from swce.sae.harvest import (
    RESPONSE,
    TYPES,
    Example,
    descriptions,
    fetch_all,
    held_out,
    own_descriptions,
    questions,
    read,
    sae_root,
)
from swce.sae.model import BATCH, LATENTS, Bank, Trainer, load, save
from swce.steering.steerer import Steerer
from swce.tasks.task import Task

BLOCKS = (9, 13, 24, 30)
"""Blocks of LLaVA-1.5-7B chosen from the probes and the methods."""
SEEDS = (0, 1)
"""Seeds of the autoencoders of every block."""


def corpus(subject: HuggingFaceSubject, own: int) -> list[Example]:
    """Every example of the corpus, with descriptions of the subject of the first images."""
    found = descriptions()
    first = list(dict.fromkeys(example.image for example in found))[:own]
    path = sae_root(subject.name) / "descriptions.jsonl"
    return found + questions() + own_descriptions(subject, first, path)


def train(
    subject: HuggingFaceSubject,
    blocks: Sequence[int],
    seeds: Sequence[int],
    *,
    own: int,
    epochs: int,
    capacity: int,
    latents: int,
    batch: int = BATCH,
    limit: int | None = None,
) -> list[Path]:
    """Train one autoencoder per block and seed and store them under the artifacts.

    Args:
        subject: Model whose stream is read.
        blocks: Blocks to read.
        seeds: Seeds of the autoencoders of every block.
        own: Images the subject describes itself.
        epochs: Passes over the corpus, each keeping other image tokens.
        capacity: States the shuffling buffer holds.
        latents: Latents of every autoencoder.
        batch: States in every step.
        limit: Examples of every pass, or None for all of them, to rehearse.

    Returns:
        The files of the autoencoders.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    examples = [example for example in corpus(subject, own) if not held_out(example.image)]
    if limit:
        examples = random.Random(0).sample(examples, min(limit, len(examples)))
    fetch_all([example.image for example in examples])
    width = residual_width(subject)
    pairs = [(block, seed) for block in blocks for seed in seeds]
    bank = Bank([b for b, _ in pairs], [s for _, s in pairs], width, latents=latents)
    bank.to(subject.device)
    trainer = Trainer(
        bank, [list(blocks).index(b) for b, _ in pairs], capacity=capacity, batch=batch
    )
    folder = sae_root(subject.name)
    tag = "-".join(str(block) for block in blocks)
    for epoch in range(epochs):
        order = random.Random(epoch).sample(examples, len(examples))
        for example in tqdm(order, desc=f"epoch {epoch + 1} of {epochs}", mininterval=30):
            trainer.add(read(subject, example, blocks, seed=epoch).states)
        if epoch == epochs - 1:
            trainer.finish()
        written = save(bank, folder)
        history = pd.DataFrame(trainer.history)
        history.to_csv(folder / f"history-{tag}.csv", index=False, lineterminator="\n")
        if len(history):
            print(history.groupby(["block", "seed"]).tail(1).to_string(index=False), flush=True)
    (folder / f"done-{tag}").write_text(f"{trainer.tokens} tokens in {trainer.steps} steps\n")
    return written


def bank_of(subject: str, blocks: Sequence[int], seeds: Sequence[int], device: Any) -> Bank:
    """The stored autoencoders of the blocks and seeds, on a device."""
    folder = sae_root(subject)
    paths = [folder / f"block{block}-seed{seed}.pt" for block in blocks for seed in seeds]
    return load(paths).to(device)


def pope_states(
    subject: HuggingFaceSubject, steerers: Sequence[Steerer], task: str, limit: int | None
) -> tuple[dict[str, Tensor], np.ndarray, np.ndarray, np.ndarray]:
    """States at the position of the answer to every POPE question, under every method.

    Returns:
        The states of every setting by its label, `(items, layers, width)`, the unmodified ones
        under `unmodified`, whether each item expects yes, whether the unmodified model answers
        yes, and the image of each item.
    """
    bound = Task(task)
    chosen = [
        steerer
        for steerer in prepare(subject, steerers)
        if not isinstance(steerer, Unmodified)
        and (not steerer.tasks or any(fnmatch.fnmatch(task, p) for p in steerer.tasks))
    ]
    labels = ["unmodified", *[f"{s.name}/{s.setting}" for s in chosen]]
    found: dict[str, list[Tensor]] = {label: [] for label in labels}
    positive, answered, groups = [], [], []
    yes, no = subject.first_tokens(YES_WORDS), subject.first_tokens(NO_WORDS)
    for item in tqdm(bound.items(limit), desc=f"read {task}", mininterval=30):
        if item.label not in ("yes", "no"):
            continue
        image = bound.image(item)
        for label, steerer in zip(labels, [None, *chosen], strict=True):
            found[label].append(observe(subject, image, item.prompt, steerer))
        last = unembedded(subject, found["unmodified"][-1][-1:])[0]
        answered.append(bool(last[yes].logsumexp(0) > last[no].logsumexp(0)))
        positive.append(item.label == "yes")
        groups.append(item.group)
    states = {label: torch.stack(values) for label, values in found.items()}
    return states, np.array(positive), np.array(answered), np.array(groups)


def steering_check(bank: Bank, states: dict[str, Tensor]) -> pd.DataFrame:
    """Unexplained variance of the states at the answer position under every setting."""
    rows = []
    device = bank.decoder.device
    for index, block in enumerate(bank.blocks):
        for label, found in states.items():
            at = found[:, block].to(device)
            with torch.no_grad():
                rebuilt = bank.rebuild(index, at)
            rows.append(
                {
                    "block": block,
                    "seed": bank.seeds[index],
                    "setting": label,
                    "unexplained": unexplained(at, rebuilt),
                    "items": len(at),
                }
            )
    return pd.DataFrame(rows)


@torch.no_grad()
def vti_reading(
    bank: Bank, states: dict[str, Tensor], directions: Tensor, top: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """What the textual direction of VTI moves, latent by latent, at every block.

    Returns:
        A summary per block, seed and setting of VTI, and the latents named, with their
        cosine with the direction and with the mean change of the states.
    """
    summary, named = [], []
    device = bank.decoder.device
    for label in [label for label in states if label.startswith("vti/")]:
        for index, block in enumerate(bank.blocks):
            rows = bank.decoder[index]
            direction = directions[block].to(device).float()
            before = states["unmodified"][:, block].to(device)
            after = states[label][:, block].to(device)
            change = (after - before).mean(dim=0)
            moved = (bank.latents_of(index, after) - bank.latents_of(index, before)).mean(dim=0)
            by_row = cosines(direction[None], rows)[0]
            by_change = moved.abs().argsort(descending=True)[:top]
            by_cosine = by_row.abs().argsort(descending=True)[:top]
            summary.append(
                {
                    "setting": label,
                    "block": block,
                    "seed": bank.seeds[index],
                    "direction_vs_change": float(cosines(direction[None], change[None])[0, 0]),
                    "top_cosine": float(by_row.abs().max()),
                    "direction_explained_top": explained_by(direction, rows[by_cosine]),
                    "direction_explained_top5x": explained_by(
                        direction, rows[by_row.abs().argsort(descending=True)[: 5 * top]]
                    ),
                    "change_share_top": float(moved.abs()[by_change].sum() / moved.abs().sum()),
                    "change_explained_top": explained_by(change, rows[by_change]),
                }
            )
            for rank, latent in enumerate(by_change.tolist()):
                named.append(
                    {
                        "setting": label,
                        "block": block,
                        "seed": bank.seeds[index],
                        "rank": rank,
                        "latent": latent,
                        "mean_change": float(moved[latent]),
                        "cosine_direction": float(by_row[latent]),
                        "cosine_state_change": float(
                            cosines(change[None], rows[latent][None])[0, 0]
                        ),
                    }
                )
    return pd.DataFrame(summary), pd.DataFrame(named)


@torch.no_grad()
def hallucination_reading(
    bank: Bank,
    states: Tensor,
    positive: np.ndarray,
    answered: np.ndarray,
    groups: np.ndarray,
    top: int = 20,
) -> pd.DataFrame:
    """Latents that tell a hallucinated yes from a correct yes of the unmodified model."""
    said_yes = answered
    hallucinated = ~positive[said_yes]
    if min(int(hallucinated.sum()), int((~hallucinated).sum())) < 2:
        # Too few answers of either kind to tell them apart, as on a rehearsal.
        return pd.DataFrame(columns=["block", "seed", "latent"])
    frames = []
    device = bank.decoder.device
    for index, block in enumerate(bank.blocks):
        at = states[said_yes][:, block].to(device)
        activations = bank.latents_of(index, at).cpu().numpy()
        chosen = separating_latents(activations, hallucinated, groups[said_yes], top=top)
        mask = torch.from_numpy(hallucinated).to(device)
        difference = at[mask].mean(dim=0) - at[~mask].mean(dim=0)
        rows = bank.decoder[index][torch.from_numpy(chosen.latent.to_numpy()).to(device)]
        chosen["cosine_mean_difference"] = cosines(difference[None], rows)[0].cpu().numpy()
        chosen["block"], chosen["seed"] = block, bank.seeds[index]
        chosen["hallucinated"], chosen["correct"] = (
            int(hallucinated.sum()),
            int((~hallucinated).sum()),
        )
        frames.append(chosen)
    return pd.concat(frames, ignore_index=True)


def seed_check(
    bank: Bank, named: dict[tuple[int, int], list[int]]
) -> tuple[pd.DataFrame, dict[tuple[int, int], Tensor]]:
    """Best match of every latent of one seed in the other, by block."""
    rows: list[dict[str, float]] = []
    matches: dict[tuple[int, int], Tensor] = {}
    for first in range(len(bank.blocks)):
        for second in range(len(bank.blocks)):
            if bank.blocks[first] != bank.blocks[second] or first == second:
                continue
            found = best_match(bank.decoder[first], bank.decoder[second])
            matches[(bank.blocks[first], bank.seeds[first])] = found
            key = (bank.blocks[first], bank.seeds[first])
            chosen = named.get(key, [])
            row: dict[str, float] = {
                "block": bank.blocks[first],
                "seed": bank.seeds[first],
                "other": bank.seeds[second],
            }
            row.update(replication(found))
            if chosen:
                row.update({f"named_{k}": v for k, v in replication(found[chosen]).items()})
            rows.append(row)
    return pd.DataFrame(rows), matches


@torch.no_grad()
def held_out_check(
    subject: HuggingFaceSubject,
    bank: Bank,
    blocks: Sequence[int],
    examples: Sequence[Example],
    named: dict[tuple[int, int], list[int]],
    splice: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reconstruction by token type, dead latents, next token loss and contexts of latents.

    Args:
        subject: Model whose stream is read.
        bank: Autoencoders to check.
        blocks: Blocks of the states read, in order.
        examples: Held-out examples.
        named: Latents of every block and seed whose top contexts are kept.
        splice: Examples with a response on which the next token loss is measured.

    Returns:
        Reconstruction per block, seed and type, the next token loss per block and seed, and the
        top contexts of the named latents.
    """
    device = bank.decoder.device
    count = len(bank.blocks)
    variances = [Variance(bank.decoder.shape[-1]) for _ in range(count)]
    fired = torch.zeros(count, bank.latents, dtype=torch.bool, device=device)
    tokens = 0
    losses: list[dict[str, Any]] = []
    tops: dict[tuple[int, int, int], list[tuple[float, int, str]]] = {}
    spliced = 0
    tokenizer = subject.processor.tokenizer
    for number, example in enumerate(tqdm(examples, desc="held out", mininterval=30)):
        # Every image token is kept, so that the loss check replaces the whole stream.
        reading = read(subject, example, blocks, image_tokens=10**6)
        tokens += len(reading.tags)
        for index in range(count):
            at = reading.states[:, list(blocks).index(bank.blocks[index])]
            latents = bank.latents_of(index, at)
            rebuilt = (latents @ bank.decoder[index] + bank.bias[index]) / bank.scale[index]
            fired[index] |= (latents > 0).any(dim=0)
            for tag, name in enumerate(TYPES):
                mask = (reading.tags == tag).to(device)
                if bool(mask.any()):
                    variances[index].add(f"{example.kind}/{name}", at[mask], rebuilt[mask])
            for latent in named.get((bank.blocks[index], bank.seeds[index]), []):
                values = latents[:, latent]
                best = int(values.argmax())
                value = float(values[best])
                if value <= 0:
                    continue
                key = (bank.blocks[index], bank.seeds[index], latent)
                heap = tops.setdefault(key, [])
                if len(heap) < 8 or value > heap[0][0]:
                    position = int(reading.positions[best])
                    window = reading.identifiers[max(0, position - 8) : position + 1].tolist()
                    text = tokenizer.decode(
                        [w for w in window if w >= 0], skip_special_tokens=False
                    )
                    entry = (value, number, f"{TYPES[int(reading.tags[best])]}: {text}")
                    (heapq.heappush if len(heap) < 8 else heapq.heapreplace)(heap, entry)
        if example.response and spliced < splice:
            spliced += 1
            losses.extend(splice_losses(subject, bank, reading))
    rows = []
    for index in range(count):
        for group, value in variances[index].unexplained().items():
            kind, name = group.split("/")
            rows.append(
                {
                    "block": bank.blocks[index],
                    "seed": bank.seeds[index],
                    "kind": kind,
                    "type": name,
                    "unexplained": value,
                    "tokens": variances[index].count[group],
                }
            )
        rows.append(
            {
                "block": bank.blocks[index],
                "seed": bank.seeds[index],
                "kind": "all",
                "type": "dead",
                "unexplained": float((~fired[index]).float().mean()),
                "tokens": tokens,
            }
        )
    contexts = [
        {"block": b, "seed": s, "latent": latent, "activation": value, "context": text}
        for (b, s, latent), heap in tops.items()
        for value, _, text in sorted(heap, reverse=True)
    ]
    loss = pd.DataFrame(losses)
    if len(loss):
        loss = loss.groupby(["block", "seed", "setting"], as_index=False).agg(
            loss=("loss", "mean"), tokens=("tokens", "sum")
        )
    return pd.DataFrame(rows), loss, pd.DataFrame(contexts)


def splice_losses(subject: HuggingFaceSubject, bank: Bank, reading: Any) -> list[dict[str, Any]]:
    """Next token loss on the response with and without the reconstruction in every block."""
    identifiers = reading.identifiers.to(reading.logits.device)
    response = torch.nonzero(reading.tags == RESPONSE).flatten()
    targets = reading.positions[response]
    targets = targets[targets > 0].to(identifiers.device)

    def loss(logits: Tensor) -> float:
        return float(F.cross_entropy(logits[targets - 1].float(), identifiers[targets]))

    found = [
        {
            "block": b,
            "seed": s,
            "setting": "clean",
            "loss": loss(reading.logits),
            "tokens": len(targets),
        }
        for b, s in zip(bank.blocks, bank.seeds, strict=True)
    ]
    layers = subject.decoder_layers()
    positions = reading.positions.to(identifiers.device)
    for index, block in enumerate(bank.blocks):

        def hook(_module: nn.Module, _inputs: Any, output: Any, index: int = index) -> Any:
            state = output[0] if isinstance(output, tuple) else output
            state = state.clone()
            rebuilt = bank.rebuild(index, state[0, positions])
            state[0, positions] = rebuilt.to(state.dtype)
            return (state, *output[1:]) if isinstance(output, tuple) else state

        handle = layers[block].register_forward_hook(hook)
        try:
            logits = subject.model(**reading.inputs).logits[0]
        finally:
            handle.remove()
        found.append(
            {
                "block": block,
                "seed": bank.seeds[index],
                "setting": "spliced",
                "loss": loss(logits),
                "tokens": len(targets),
            }
        )
    return found


def evaluate(
    subject: HuggingFaceSubject,
    steerers: Sequence[Steerer],
    blocks: Sequence[int],
    seeds: Sequence[int],
    *,
    task: str,
    limit: int | None,
    held: int,
    splice: int,
    own: int,
    results: Path,
    directions: Path | None,
) -> list[Path]:
    """Run the checks and the readings and write them under `sae/MODEL` in the results."""
    torch.backends.cuda.matmul.allow_tf32 = True
    bank = bank_of(subject.name, blocks, seeds, subject.device)
    folder = results / "sae" / subject.name
    folder.mkdir(parents=True, exist_ok=True)
    written = []

    def write(frame: pd.DataFrame, name: str) -> None:
        target = folder / f"{name}.csv"
        frame.to_csv(target, index=False, lineterminator="\n", float_format="%.6g")
        written.append(target)
        print(f"\n{name}\n{frame.to_string(index=False, float_format=lambda v: f'{v:.3f}')}")

    states, positive, answered, groups = pope_states(subject, steerers, task, limit)
    write(steering_check(bank, states), "steering")
    named: dict[tuple[int, int], list[int]] = {}
    if directions is not None and directions.exists():
        loaded = torch.load(directions, map_location="cpu", weights_only=True)
        summary, latents = vti_reading(bank, states, loaded["textual"])
        write(summary, "vti")
        write(latents, "vti-latents")
        for block, seed, latent in latents[["block", "seed", "latent"]].to_numpy().tolist():
            named.setdefault((int(block), int(seed)), []).append(int(latent))
    hallucination = hallucination_reading(bank, states["unmodified"], positive, answered, groups)
    write(hallucination, "hallucination")
    for block, seed, latent in hallucination[["block", "seed", "latent"]].to_numpy().tolist():
        named.setdefault((int(block), int(seed)), []).append(int(latent))
    named = {key: sorted(set(value)) for key, value in named.items()}
    seeds_frame, _ = seed_check(bank, named)
    write(seeds_frame, "replication")
    examples = [e for e in corpus(subject, own) if held_out(e.image)]
    examples = random.Random(0).sample(examples, min(held, len(examples)))
    fetch_all([example.image for example in examples])
    reconstruction, loss, contexts = held_out_check(subject, bank, blocks, examples, named, splice)
    write(reconstruction, "reconstruction")
    write(loss, "splice")
    write(contexts, "contexts")
    (folder / "settings.json").write_text(
        json.dumps(
            {
                "blocks": list(blocks),
                "seeds": list(seeds),
                "latents": bank.latents,
                "k": bank.k,
                "task": task,
                "items": len(positive),
                "held_out": len(examples),
                "splice": splice,
            },
            indent=2,
        )
        + "\n"
    )
    return written


def main(argv: list[str] | None = None) -> int:
    """Run the autoencoders from the command line.

    Args:
        argv: Command line arguments without the program name, or None to read `sys.argv`.

    Returns:
        Exit status, zero on success.
    """
    parser = argparse.ArgumentParser(prog="swce-sae", description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument("methods", nargs="*", help="references of method factories to read")
    parser.add_argument("--subject", required=True, help="reference of the subject factory")
    parser.add_argument("--blocks", type=int, nargs="+", default=list(BLOCKS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--latents", type=int, default=LATENTS)
    parser.add_argument("--own", type=int, default=256, help="images the model describes itself")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=BATCH, help="states in every step")
    parser.add_argument("--capacity", type=int, default=2**18, help="states the buffer holds")
    parser.add_argument("--limit", type=int, default=0, help="examples per pass, 0 for all")
    parser.add_argument("--task", default="pope-adversarial")
    parser.add_argument("--items", type=int, default=0, help="images of the task, 0 for all")
    parser.add_argument("--held", type=int, default=1000, help="held-out examples to check")
    parser.add_argument("--splice", type=int, default=200, help="examples of the loss check")
    parser.add_argument("--directions", type=Path, default=None, help="VTI directions file")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--data", type=Path, default=None, help="data root, default SWCE_DATA")
    args = parser.parse_args(argv)
    configure_logging()
    if args.data is not None:
        os.environ[DATA_ENV] = str(args.data)
    subject = load_factory(args.subject)()
    if not isinstance(subject, HuggingFaceSubject):
        raise TypeError(f"{args.subject} built {type(subject).__name__}, not a HuggingFaceSubject")
    if args.command == "train":
        paths = train(
            subject,
            args.blocks,
            args.seeds,
            own=args.own,
            epochs=args.epochs,
            capacity=args.capacity,
            latents=args.latents,
            batch=args.batch,
            limit=args.limit or None,
        )
    else:
        directions = args.directions or artifacts_root() / f"vti_{subject.name}.pt"
        paths = evaluate(
            subject,
            load_steerers(args.methods) if args.methods else [],
            args.blocks,
            args.seeds,
            task=args.task,
            limit=args.items or None,
            held=args.held,
            splice=args.splice,
            own=args.own,
            results=args.results,
            directions=directions,
        )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
