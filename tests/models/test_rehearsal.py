"""Rehearsal of the generation path with the real processor of LLaVA-1.5.

The processor is downloaded, so the test runs only when `SWCE_NETWORK` is set.
"""

import os

import pytest
import torch
from PIL import Image

from swce.diagnostics import read
from swce.models import llava_15_rehearsal

pytestmark = pytest.mark.skipif(not os.environ.get("SWCE_NETWORK"), reason="needs the network")


def test_generation_takes_the_real_template_tokens_and_hooks() -> None:
    subject = llava_15_rehearsal()
    assert len(subject.decoder_layers()) == 32
    assert len(subject.encoder_mlps()) == 24
    image = Image.new("RGB", (640, 480), "white")
    template = subject.template("Is there a dog in the image?")
    assert template == "USER: <image>\nIs there a dog in the image? ASSISTANT:"
    assert subject.inputs(image, template)["input_ids"].shape[1] > 576
    assert subject.text_inputs(template)["input_ids"].shape[1] < 30
    spelled = subject.token_text(subject.first_tokens(("Yes", "yes")))
    assert {piece.strip() for piece in spelled} == {"Yes", "yes"}
    assert not set(subject.first_tokens(("Yes", "yes"))) & set(subject.first_tokens(("No", "no")))
    subject.lens = True
    answer = subject.generate(image, "Is there a dog in the image?", 4)
    assert answer.tokens <= 4
    assert answer.lens_yes is not None
    assert len(answer.lens_yes) == 32
    states = subject.residuals(image, "Is there a dog in the image?")
    probabilities = torch.softmax(read(subject, states[-1]), dim=-1)
    yes = probabilities[subject.first_tokens(("Yes", "yes"))].sum().item()
    assert answer.yes == pytest.approx(yes, rel=1e-3)
    assert answer.lens_yes[-1] == pytest.approx(yes, rel=1e-3)
    again = subject.generate(image, "Is there a dog in the image?", 4)
    assert again.text == answer.text
