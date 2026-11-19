"""Checks of trained autoencoders and the readings of latents they support.

Four checks come before any reading of a latent, and were fixed before training.

- Reconstruction by token type on held-out images: the fraction of the variance of the states
  the reconstruction leaves unexplained, with the latents that never fire.
- Reconstruction under steering: the same fraction on the states under a method, which says
  whether the autoencoder can describe the states a method produces at all.
- Replication across seeds: for every latent of one seed, the largest cosine of its decoder row
  with a row of the other seed. A latent that only one seed finds says little about the model.
- A confirmation that does not use the autoencoder: a direction computed from the states alone,
  against which the decoder rows of the latents a reading names are compared.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from scipy.stats import rankdata
from torch import Tensor


class Variance:
    """Running sums for the fraction of unexplained variance of groups of states.

    Attributes:
        width: Width of the states.
    """

    def __init__(self, width: int) -> None:
        """Start empty sums for states of the given width."""
        self.width = width
        self.count: dict[str, int] = {}
        self.total: dict[str, Tensor] = {}
        self.squares: dict[str, float] = {}
        self.errors: dict[str, float] = {}

    def add(self, group: str, states: Tensor, rebuilt: Tensor) -> None:
        """Add states of one group, `(count, width)`, with their reconstruction."""
        states, rebuilt = states.double().cpu(), rebuilt.double().cpu()
        self.count[group] = self.count.get(group, 0) + len(states)
        self.total[group] = self.total.get(group, torch.zeros(self.width, dtype=torch.float64))
        self.total[group] += states.sum(dim=0)
        self.squares[group] = self.squares.get(group, 0.0) + float(states.pow(2).sum())
        self.errors[group] = self.errors.get(group, 0.0) + float((states - rebuilt).pow(2).sum())

    def unexplained(self) -> dict[str, float]:
        """Fraction of unexplained variance of every group, and the count of its states."""
        found = {}
        for group, count in self.count.items():
            mean = self.total[group] / count
            variance = self.squares[group] - count * float(mean.pow(2).sum())
            found[group] = self.errors[group] / variance if variance > 0 else float("nan")
        return found


def unexplained(states: Tensor, rebuilt: Tensor) -> float:
    """Fraction of the variance of states, `(count, width)`, a reconstruction leaves."""
    variance = Variance(states.shape[-1])
    variance.add("all", states, rebuilt)
    return variance.unexplained()["all"]


@torch.no_grad()
def best_match(first: Tensor, second: Tensor, chunk: int = 4096) -> Tensor:
    """For every decoder row of one autoencoder, the largest cosine with a row of another.

    Args:
        first: Decoder rows of shape `(latents, width)`.
        second: Decoder rows of the other autoencoder, of the same width.
        chunk: Rows compared at once.

    Returns:
        The largest cosine of every row of `first`, `(latents,)`, on the processor.
    """
    first = first.float() / first.float().norm(dim=-1, keepdim=True)
    second = second.float() / second.float().norm(dim=-1, keepdim=True)
    found = [
        (first[start : start + chunk] @ second.T).max(dim=-1).values
        for start in range(0, len(first), chunk)
    ]
    return torch.cat(found).cpu()


def replication(matches: Tensor, thresholds: Sequence[float] = (0.5, 0.7, 0.9)) -> dict[str, float]:
    """Share of latents whose best match in the other seed reaches each threshold."""
    found = {"median": float(matches.median())}
    for threshold in thresholds:
        found[f"over_{threshold}"] = float((matches >= threshold).float().mean())
    return found


def aurocs(activations: NDArray[np.floating], positive: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Area under the ROC curve of every latent for telling the positive items apart.

    Ties, which are most of a sparse latent, count as half, as in the Mann-Whitney statistic.

    Args:
        activations: Activations of shape `(items, latents)`.
        positive: Whether each item is positive.

    Returns:
        The area of every latent, one half for a latent that never fires.
    """
    ranks = rankdata(activations, axis=0)
    hits, misses = int(positive.sum()), int((~positive).sum())
    total = ranks[positive].sum(axis=0)
    return np.asarray((total - hits * (hits + 1) / 2) / (hits * misses), dtype=np.float64)


def cosines(directions: Tensor, rows: Tensor) -> Tensor:
    """Cosine of every direction, `(directions, width)`, with every row, `(rows, width)`."""
    directions = directions.float() / directions.float().norm(dim=-1, keepdim=True)
    rows = rows.float() / rows.float().norm(dim=-1, keepdim=True)
    return directions @ rows.T


def explained_by(direction: Tensor, rows: Tensor) -> float:
    """Share of the squared norm of a direction its least squares fit on some rows explains."""
    direction = direction.double()
    basis = rows.double().T
    solution = torch.linalg.lstsq(basis, direction[:, None]).solution
    fitted = (basis @ solution).flatten()
    return float(fitted.pow(2).sum() / direction.pow(2).sum())


def split_images(groups: NDArray[np.str_], seed: int = 0) -> NDArray[np.bool_]:
    """Whether each item falls in the first of two halves of the images, chosen at random."""
    images = sorted(set(groups.tolist()))
    chosen = set(np.random.default_rng(seed).permutation(images)[: len(images) // 2].tolist())
    return np.array([group in chosen for group in groups.tolist()], dtype=bool)


def separating_latents(
    activations: NDArray[np.floating],
    positive: NDArray[np.bool_],
    groups: NDArray[np.str_],
    *,
    top: int = 20,
    minimum: float = 0.01,
    seed: int = 0,
) -> pd.DataFrame:
    """Latents that best tell positive items apart on half the images, read on the other half.

    Args:
        activations: Activations of shape `(items, latents)`.
        positive: Whether each item is positive.
        groups: Image of each item. The two halves share no image.
        top: Latents to report.
        minimum: Share of the items of the first half a latent must fire on to be chosen.
        seed: Seed of the split.

    Returns:
        One row per chosen latent with its `latent`, its `selection_auroc` on the first half,
        its `confirmation_auroc` on the second, and the share of items it fires on in each.
    """
    first = split_images(groups, seed)
    chosen_on = activations[first]
    fires = (chosen_on > 0).mean(axis=0)
    selection = aurocs(chosen_on, positive[first])
    strength = np.where(fires >= minimum, np.abs(selection - 0.5), -1.0)
    latents = np.argsort(-strength)[:top]
    latents = latents[strength[latents] >= 0]
    confirmation = aurocs(activations[~first][:, latents], positive[~first])
    return pd.DataFrame(
        {
            "latent": latents,
            "selection_auroc": selection[latents],
            "confirmation_auroc": confirmation,
            "selection_fires": fires[latents],
            "confirmation_fires": (activations[~first][:, latents] > 0).mean(axis=0),
        }
    )
