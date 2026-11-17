"""Items of a task and the records an evaluation keeps of them."""

from pydantic import BaseModel, ConfigDict


class Item(BaseModel):
    """One question or image of a task.

    Attributes:
        id: Identifier of the item, unique within its task.
        group: Identifier of the image, shared by every item asked about it. Resampling draws
            whole groups, since answers about one image are not independent.
        prompt: Text given to the model with the image.
        image: File name of the image on its host, or its member name in `archive`.
        host: Key of the benchmark serving the image as a single file, or None.
        archive: Path under the data root of the pinned archive holding the image, or None.
        label: Expected answer in lower case, or None when the task has none per item.
        category: Subtask of the item, or None.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    group: str
    prompt: str
    image: str
    host: str | None = None
    archive: str | None = None
    label: str | None = None
    category: str | None = None


class Record(BaseModel):
    """Answer of one model under one steering method to one item.

    Attributes:
        id: Identifier of the item.
        group: Identifier of its image.
        label: Expected answer, or None.
        category: Subtask of the item, or None.
        text: Generated answer, verbatim.
        tokens: Number of generated tokens.
        yes: Probability of the token yes at the first generated position, or None when the model
            does not expose it.
        no: Probability of the token no at the first generated position, or None.
        lens_yes: Probability of yes read from the residual stream after every decoder block at
            the position that predicts the answer, or None when the lens was not recorded.
        lens_no: The same for no.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    group: str
    label: str | None = None
    category: str | None = None
    text: str
    tokens: int
    yes: float | None = None
    no: float | None = None
    lens_yes: tuple[float, ...] | None = None
    lens_no: tuple[float, ...] | None = None
