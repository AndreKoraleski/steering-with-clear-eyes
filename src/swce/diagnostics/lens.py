"""Readings of the residual stream of a subject, with or without a steering method.

The logit lens (nostalgebraist, 2020) reads the residual stream after every block through the
final normalization and the unembedding of the model, as if the model stopped there. Applied at
the position that predicts the first token of the answer, it shows at which depth a model comes
to prefer yes or no, and where a steering method changes that.

The residual shift compares the stream under a method with the stream of the unmodified model on
the same input, block by block. It shows where an intervention takes effect and whether its
effect grows or fades toward the output.
"""

import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from torch import Tensor

from swce.models.subject import Subject, unembed
from swce.steering.steerer import Steerer

YES_WORDS = ("Yes", "yes")
"""Words whose first token counts as yes."""
NO_WORDS = ("No", "no")
"""Words whose first token counts as no."""


def observe(
    subject: Subject, image: Image.Image, prompt: str, steerer: Steerer | None = None
) -> Tensor:
    """Residual stream at the last prompt position after every decoder block.

    Args:
        subject: Model to observe.
        image: Image in RGB.
        prompt: Text of the user turn.
        steerer: Prepared method setting to steer the model with, or None for the unmodified
            model.

    Returns:
        A tensor of shape `(layers, width)` in single precision on the processor.
    """
    if steerer is None:
        return subject.residuals(image, prompt)
    with steerer.attach(subject):
        # A method whose intervention depends on the input computes it when notified.
        subject.notify(image, prompt)
        return subject.residuals(image, prompt)


@torch.no_grad()
def read(subject: Subject, states: Tensor) -> Tensor:
    """Logits the model would produce if every given state were its last one.

    Args:
        subject: Model whose final normalization and unembedding read the states.
        states: Residual states of shape `(..., width)`.

    Returns:
        Logits of shape `(..., vocabulary)` in single precision on the processor.
    """
    logits = unembed(subject, states)
    return logits.float().cpu()


def logit_lens(subject: Subject, states: Tensor, top: int = 3) -> pd.DataFrame:
    """Logit lens of the residual stream at the position predicting the answer.

    Args:
        subject: Model the states come from.
        states: Residual stream after every decoder block, of shape `(layers, width)`.
        top: Number of most likely tokens reported for every block.

    Returns:
        One row per block with its index `layer`, the probabilities `yes` and `no` summed over
        the spellings of each word, their log ratio `margin`, the `entropy` of the distribution
        in nats and the `top` tokens joined by spaces.
    """
    probabilities = torch.softmax(read(subject, states), dim=-1)
    yes = probabilities[:, subject.first_tokens(YES_WORDS)].sum(dim=-1)
    no = probabilities[:, subject.first_tokens(NO_WORDS)].sum(dim=-1)
    floor = 1e-12
    entropy = -(probabilities * torch.log(probabilities.clamp(min=floor))).sum(dim=-1)
    leading = probabilities.topk(top, dim=-1).indices
    return pd.DataFrame(
        {
            "layer": range(states.shape[0]),
            "yes": yes.tolist(),
            "no": no.tolist(),
            "margin": (torch.log(yes.clamp(min=floor)) - torch.log(no.clamp(min=floor))).tolist(),
            "entropy": entropy.tolist(),
            "top": [" ".join(subject.token_text(row.tolist())) for row in leading],
        }
    )


def residual_shift(steered: Tensor, unmodified: Tensor) -> pd.DataFrame:
    """Change a method makes to the residual stream, block by block.

    Args:
        steered: Residual stream under the method, of shape `(layers, width)`.
        unmodified: Residual stream of the unmodified model on the same input.

    Returns:
        One row per block with its index `layer`, the `cosine` similarity of the two streams, the
        `norm_ratio` of the steered stream to the unmodified one, and the `distance` between them
        relative to the norm of the unmodified one.
    """
    norm = unmodified.norm(dim=-1)
    return pd.DataFrame(
        {
            "layer": range(steered.shape[0]),
            "cosine": F.cosine_similarity(steered, unmodified, dim=-1).tolist(),
            "norm_ratio": (steered.norm(dim=-1) / norm).tolist(),
            "distance": ((steered - unmodified).norm(dim=-1) / norm).tolist(),
        }
    )
