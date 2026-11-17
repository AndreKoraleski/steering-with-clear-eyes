"""Template of a steering method. Copy it, rename it and fill it in.

Source: paper, venue and year, with the address of the official code and its commit.
Licence: licence of that code, or a statement that none is declared.
Settings: where every value below comes from, table or section of the paper or file of the code.
    A value the paper does not give for a model is the default of the official code, under a
    setting named `code-default` with `fallback=True`. Nothing is tuned here.

The method adds one fixed vector per decoder layer. Methods that gate, rescale or recompute their
direction implement their own edits with `swce.steering.hooks.edit_layers`.
"""

from contextlib import AbstractContextManager
from pathlib import Path
from typing import ClassVar

import torch

from swce.data.paths import artifacts_root
from swce.models.subject import Subject
from swce.steering.hooks import add_vector, edit_layers
from swce.steering.steerer import Steerer


class Template(Steerer):
    """Fixed additive steering at the strength its authors publish.

    Attributes:
        strength: Coefficient applied to every vector.
        vectors: File holding a mapping from decoder layer index to a vector of the residual
            width, or None for the one built for the subject under the artifacts folder.
    """

    name: ClassVar[str] = "template"
    description: ClassVar[str] = "Template of a fixed additive steering method."

    strength: float
    vectors: Path | None = None

    def attach(self, subject: Subject) -> AbstractContextManager[None]:
        """Add the scaled vectors to the decoder of the subject.

        Args:
            subject: Model to steer.

        Returns:
            A context inside which the vectors are added at every position of their layers.

        Raises:
            FileNotFoundError: If the vectors were not built, so that the unmodified model is
                never evaluated under the name of the method.
        """
        path = self.vectors or artifacts_root() / f"template_{subject.name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"build the vectors of {self.name} first, at {path}")
        loaded: dict[int, torch.Tensor] = torch.load(path, weights_only=True)
        edits = {layer: add_vector(vector, self.strength) for layer, vector in loaded.items()}
        return edit_layers(subject.decoder_layers(), edits)


def build() -> list[Steerer]:
    """Published settings of the method.

    Returns:
        One steerer per setting the paper reports, each named after the benchmark it was chosen
        for and declaring the data its vectors were fitted on.
    """
    fitted_on = ("coco-train2014",)
    return [
        Template(setting="chair", strength=0.1, fitted_on=fitted_on),
        Template(setting="pope", strength=0.5, fitted_on=fitted_on),
    ]
