from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from tiny import tiny_llava
from torch import Tensor

from swce.models.hf import HuggingFaceSubject
from swce.sae import harvest
from swce.sae.checks import (
    aurocs,
    best_match,
    explained_by,
    replication,
    separating_latents,
    unexplained,
)
from swce.sae.harvest import IMAGE, PROMPT, RESPONSE, Example, held_out, keep
from swce.sae.model import Bank, Trainer, load, save


def planted(
    count: int, width: int = 16, atoms: int = 32, active: int = 2, seed: int = 0
) -> tuple[Tensor, Tensor]:
    """States that are sums of a few unit atoms with positive coefficients."""
    generator = torch.Generator().manual_seed(seed)
    dictionary = torch.randn(atoms, width, generator=generator)
    dictionary /= dictionary.norm(dim=-1, keepdim=True)
    chosen = torch.rand(count, atoms, generator=generator).argsort(dim=-1)[:, :active]
    coefficients = torch.zeros(count, atoms).scatter(
        -1, chosen, 1 + torch.rand(count, active, generator=generator)
    )
    return coefficients @ dictionary, dictionary


def test_bank_learns_a_planted_dictionary() -> None:
    torch.manual_seed(0)
    states, dictionary = planted(20000)
    bank = Bank([0, 0], [0, 1], 16, latents=64, k=2)
    trainer = Trainer(bank, [0, 0], capacity=4096, batch=256)
    for _ in range(4):
        for start in range(0, len(states), 1000):
            trainer.add(states[start : start + 1000, None])
    trainer.finish()
    with torch.no_grad():
        rebuilt = bank(torch.stack([states[:2000], states[:2000]]))
    assert unexplained(states[:2000], rebuilt[0]) < 0.1
    assert unexplained(states[:2000], rebuilt[1]) < 0.1
    # Most planted atoms are found by a decoder row of each seed.
    assert (best_match(dictionary, bank.decoder[0].detach()) > 0.9).float().mean() > 0.7
    assert torch.allclose(bank.decoder.norm(dim=-1), torch.ones(2, 64), atol=1e-5)
    assert {row["block"] for row in trainer.history} == {0}


def test_one_autoencoder_matches_the_bank() -> None:
    states, _ = planted(50)
    bank = Bank([3, 5], [0, 1], 16, latents=32, k=4)
    bank.initialize(torch.stack([states, states]))
    with torch.no_grad():
        together = bank(torch.stack([states, states]))
        assert torch.allclose(bank.rebuild(1, states), together[1], atol=1e-5)
        assert (bank.latents_of(0, states) > 0).sum(dim=-1).max() <= 4


def test_saved_autoencoders_load_back(tmp_path: Path) -> None:
    states, _ = planted(50)
    bank = Bank([3, 5], [0, 1], 16, latents=32, k=4)
    bank.initialize(torch.stack([states, states]))
    paths = save(bank, tmp_path)
    assert [path.name for path in paths] == ["block3-seed0.pt", "block5-seed1.pt"]
    loaded = load(paths)
    assert loaded.blocks == [3, 5] and loaded.seeds == [0, 1] and loaded.k == 4
    with torch.no_grad():
        assert torch.allclose(loaded.rebuild(0, states), bank.rebuild(0, states), atol=1e-2)


def test_unexplained_is_zero_for_a_copy_and_one_for_the_mean() -> None:
    states = torch.randn(200, 8)
    assert unexplained(states, states) == pytest.approx(0.0)
    mean = states.mean(dim=0).expand_as(states)
    assert unexplained(states, mean) == pytest.approx(1.0)


def test_best_match_finds_a_permuted_copy() -> None:
    rows = torch.randn(40, 16)
    found = best_match(rows, rows[torch.randperm(40)] * 3)
    assert found == pytest.approx(torch.ones(40), abs=1e-5)
    assert replication(found)["over_0.9"] == 1.0


def test_aurocs_agree_with_scikit_learn_under_ties() -> None:
    generator = np.random.default_rng(0)
    activations = np.maximum(generator.normal(size=(300, 5)), 0)
    activations[:, 4] = 0
    positive = generator.random(300) < 0.4
    activations[positive, 1] += 1
    found = aurocs(activations, positive)
    expected = [roc_auc_score(positive, activations[:, j]) for j in range(4)]
    assert found[:4] == pytest.approx(expected)
    assert found[4] == pytest.approx(0.5)


def test_separating_latents_are_confirmed_on_other_images() -> None:
    generator = np.random.default_rng(0)
    images = np.repeat([f"image{i}" for i in range(100)], 4)
    positive = generator.random(400) < 0.5
    activations = np.maximum(generator.normal(size=(400, 50)), 0)
    activations[:, 7] = np.where(positive, 1.0 + generator.random(400), 0.0)
    found = separating_latents(activations, positive, images, top=3)
    assert isinstance(found, pd.DataFrame)
    assert found.latent.iloc[0] == 7
    assert found.confirmation_auroc.iloc[0] > 0.95


def test_explained_by_its_own_rows() -> None:
    rows = torch.randn(3, 16)
    assert explained_by(rows[0] * 2 - rows[2], rows) == pytest.approx(1.0)
    assert explained_by(torch.randn(16), rows[:1]) < 0.9


def test_held_out_images_are_a_tenth() -> None:
    names = [f"COCO_train2014_{index:012d}.jpg" for index in range(5000)]
    share = np.mean([held_out(name) for name in names])
    assert 0.08 < share < 0.12


def test_keep_drops_the_start_and_outliers_and_samples_images() -> None:
    states = torch.ones(10, 2, 4)
    states[5, 1] = 100.0
    tags = torch.tensor([PROMPT, IMAGE, IMAGE, IMAGE, IMAGE, PROMPT, PROMPT, RESPONSE, RESPONSE, 9])
    kept = keep(states, tags, image_tokens=2, seed=0).tolist()
    assert 0 not in kept and 5 not in kept
    assert len([index for index in kept if 1 <= index <= 4]) == 2
    assert {6, 7, 8, 9} <= set(kept)


def test_read_tags_every_position_of_a_tiny_llava(monkeypatch: pytest.MonkeyPatch) -> None:
    subject = tiny_llava()

    def inputs(_image: Image.Image, text: str) -> dict[str, Tensor]:
        identifiers = [1, 59, 59, 59, 59, 5, 6] + [7] * text.count("+")
        return {
            "input_ids": torch.tensor([identifiers]),
            "pixel_values": torch.zeros(1, 3, 28, 28),
        }

    monkeypatch.setattr(HuggingFaceSubject, "template", lambda self, prompt: "template")
    monkeypatch.setattr(subject, "inputs", inputs)
    monkeypatch.setattr(harvest, "open_image", lambda name: Image.new("RGB", (28, 28)))
    example = Example("image.jpg", "question", "+++", "description")
    reading = harvest.read(subject, example, [1, 3], image_tokens=2)
    assert reading.states.shape == (2 + 2 + 3, 2, 16)
    assert reading.tags.tolist() == [IMAGE, IMAGE, PROMPT, PROMPT, RESPONSE, RESPONSE, RESPONSE]
    assert reading.positions[-3:].tolist() == [7, 8, 9]
    assert reading.logits.shape[0] == 10
