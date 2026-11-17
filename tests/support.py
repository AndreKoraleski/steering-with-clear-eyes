"""Stand ins shared by the tests. Nothing here touches the network or loads real weights."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor, nn

from swce.models.subject import Generation, Subject, unembed
from swce.registry.schema import BenchmarkCard, PinnedFile, Registry, TaskCard, TaskKind

QUESTIONS = [
    ("a.jpg", "Is there a dog in the image?", "yes"),
    ("a.jpg", "Is there a cat in the image?", "no"),
    ("b.jpg", "Is there a car in the image?", "yes"),
    ("b.jpg", "Is there a bus in the image?", "no"),
    ("c.jpg", "Is there a cup in the image?", "yes"),
    ("c.jpg", "Is there a fork in the image?", "no"),
]


class Block(nn.Module):
    """Residual block whose feed forward module is exposed, as in a transformer."""

    def __init__(self, width: int):
        super().__init__()
        self.mlp = nn.Linear(width, width)

    def forward(self, hidden: Tensor) -> tuple[Tensor, None]:
        return hidden + self.mlp(hidden), None


class ToySubject(Subject):
    """Three residual blocks standing in for a decoder, answering yes or no from their output."""

    name = "toy"
    description = "Three random residual blocks of width eight."

    def __init__(self, width: int = 8, vocabulary: int = 5):
        torch.manual_seed(0)
        self.width = width
        self.blocks = nn.ModuleList([Block(width) for _ in range(3)])
        self.head = nn.Linear(width, vocabulary, bias=False)
        self.calls = 0

    def states(self, prompt: str, image: bool = True) -> list[Tensor]:
        state = torch.full((1, 2, self.width), (len(prompt) % 7) / 7 + (0.5 if image else 0.0))
        states = []
        for block in self.blocks:
            state = block(state)[0]
            states.append(state)
        return states

    def hidden(self, prompt: str) -> Tensor:
        return self.states(prompt)[-1]

    def logits(self, prompt: str) -> Tensor:
        logits: Tensor = self.head(self.hidden(prompt))
        return logits

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int,
        settings: Mapping[str, object] | None = None,
    ) -> Generation:
        self.calls += 1
        self.notify(image, prompt)
        with torch.no_grad():
            states = self.states(prompt)
            yes = float(torch.sigmoid(states[-1].sum()))
            last = torch.stack([state[0, -1] for state in states])
            read = torch.softmax(unembed(self, last), dim=-1)
        return Generation(
            text="Yes" if yes > 0.5 else "No",
            tokens=1,
            yes=yes,
            no=1 - yes,
            lens_yes=tuple(read[:, 0].tolist()) if self.lens else None,
            lens_no=tuple(read[:, 1].tolist()) if self.lens else None,
        )

    def decoder_layers(self) -> Sequence[nn.Module]:
        return list(self.blocks)

    def decoder_mlps(self) -> Sequence[nn.Module]:
        return [block.get_submodule("mlp") for block in self.blocks]

    def encoder_mlps(self) -> Sequence[nn.Module]:
        return [block.get_submodule("mlp") for block in self.blocks]

    def unembedding(self) -> nn.Module:
        return self.head

    def final_norm(self) -> nn.Module:
        return nn.Identity()

    def first_tokens(self, words: Sequence[str]) -> list[int]:
        return [0] if words[0].lower() == "yes" else [1]

    def token_text(self, identifiers: Sequence[int]) -> list[str]:
        return [f"token{identifier}" for identifier in identifiers]

    def residuals(self, image: Image.Image | None, prompt: str, answer: str = "") -> Tensor:
        with torch.no_grad():
            states = self.states(prompt + answer, image is not None)
        return torch.stack([state[0, -1] for state in states])


def pin(path: Path, name: str) -> PinnedFile:
    """Pin of a local file served through a file address."""
    content = path.read_bytes()
    return PinnedFile(
        path=name, url=path.as_uri(), size=len(content), digest=hashlib.sha256(content).hexdigest()
    )


def tiny_registry(host: Path) -> Registry:
    """Registry of one POPE like task whose files and images live in a local folder."""
    host.mkdir(parents=True, exist_ok=True)
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        Image.new("RGB", (4, 4), "white").save(host / name)
    rows = [
        {"question_id": index + 1, "image": image, "text": text, "label": label}
        for index, (image, text, label) in enumerate(QUESTIONS)
    ]
    questions = host / "questions.json"
    questions.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    coco = BenchmarkCard(
        key="coco-2014",
        name="Images",
        reference="None",
        source="local",
        licence="None",
        terms="None",
        image_url=host.as_uri() + "/{name}",
        image_dir="images",
    )
    task = TaskCard(
        key="pope-tiny",
        kind=TaskKind.YES_NO,
        description="Six questions on three images.",
        loader="pope",
        scorer="pope",
        primary="f1",
        max_new_tokens=8,
        files=("tiny/questions.json",),
        requires=("coco-2014",),
    )
    pope = BenchmarkCard(
        key="tiny",
        name="Tiny",
        reference="None",
        source="local",
        licence="None",
        terms="None",
        files=(pin(questions, "tiny/questions.json"),),
        tasks=(task,),
    )
    return Registry(version=1, benchmarks=(coco, pope))
