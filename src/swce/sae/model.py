"""TopK sparse autoencoders of the residual stream, several trained together on one GPU.

A TopK autoencoder (Gao et al., Scaling and Evaluating Sparse Autoencoders, 2024) encodes a
state into many latents and keeps only the `k` largest, which fixes the sparsity instead of
tuning a penalty for it. The decoder rows have unit norm, the encoder starts as the transpose of
the decoder, and an auxiliary loss reconstructs the error from the latents that have stopped
firing, which keeps most of them alive. The settings below are those of that paper, with the
learning rate scaled to the number of latents as it prescribes, and are not tuned here.

Every autoencoder of a bank reads its own block and starts from its own seed. Their weights are
stacked, so that one batched product trains all of them at once.
"""

import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

LATENTS = 32768
"""Latents of every autoencoder."""
K = 64
"""Latents kept for every state."""
AUX_K = 512
"""Dead latents the auxiliary loss reconstructs the error from."""
AUX_WEIGHT = 1 / 32
"""Weight of the auxiliary loss."""
DEAD_AFTER = 1_000_000
"""Tokens after which a latent that has not fired counts as dead. The paper uses ten million on
a corpus a hundred times larger than this one."""
BATCH = 4096
"""States in every step."""


def learning_rate(latents: int) -> float:
    """Learning rate Gao et al. fit to the number of latents, 2e-4 at 16384."""
    return 2e-4 / math.sqrt(latents / 2**14)


class Bank(nn.Module):
    """TopK autoencoders of the same width, each reading one block.

    Attributes:
        blocks: Block each autoencoder reads.
        seeds: Seed each autoencoder started from.
        k: Latents kept for every state.
        encoder: Weights of shape `(autoencoders, width, latents)`.
        encoder_bias: Biases of shape `(autoencoders, latents)`.
        decoder: Unit rows of shape `(autoencoders, latents, width)`.
        bias: Bias subtracted before encoding and added after decoding, `(autoencoders, width)`.
        scale: Factor that brings the mean squared norm of the states of each block to their
            width, applied before encoding and undone after decoding, `(autoencoders,)`.
    """

    scale: Tensor

    def __init__(
        self,
        blocks: Sequence[int],
        seeds: Sequence[int],
        width: int,
        *,
        latents: int = LATENTS,
        k: int = K,
    ) -> None:
        """Start every autoencoder from its seed.

        Args:
            blocks: Block each autoencoder reads.
            seeds: Seed of each autoencoder, as many as blocks.
            width: Width of the residual stream.
            latents: Latents of every autoencoder.
            k: Latents kept for every state.
        """
        super().__init__()
        if len(blocks) != len(seeds):
            raise ValueError("every autoencoder needs one block and one seed")
        self.blocks = list(blocks)
        self.seeds = list(seeds)
        self.k = k
        rows = []
        for seed in seeds:
            generator = torch.Generator().manual_seed(seed)
            row = torch.randn(latents, width, generator=generator)
            rows.append(row / row.norm(dim=-1, keepdim=True))
        decoder = torch.stack(rows)
        self.decoder = nn.Parameter(decoder)
        self.encoder = nn.Parameter(decoder.transpose(1, 2).clone())
        self.encoder_bias = nn.Parameter(torch.zeros(len(seeds), latents))
        self.bias = nn.Parameter(torch.zeros(len(seeds), width))
        self.register_buffer("scale", torch.ones(len(seeds)))

    @property
    def latents(self) -> int:
        """Latents of every autoencoder."""
        return int(self.decoder.shape[1])

    @torch.no_grad()
    def initialize(self, states: Tensor) -> None:
        """Set the scale and the bias from a first sample of states.

        Args:
            states: States of shape `(autoencoders, count, width)`, in the units of the model.
        """
        states = states.float()
        squared = states.pow(2).sum(dim=-1).mean(dim=-1)
        self.scale.copy_(torch.sqrt(states.shape[-1] / squared))
        self.bias.copy_(states.mean(dim=1) * self.scale[:, None])

    def preactivations(self, states: Tensor) -> Tensor:
        """Latents before the top k is kept, of shape `(autoencoders, count, latents)`."""
        scaled = states.float() * self.scale[:, None, None]
        return torch.baddbmm(self.encoder_bias[:, None], scaled - self.bias[:, None], self.encoder)

    def encode(self, states: Tensor) -> Tensor:
        """Sparse latents of states, of shape `(autoencoders, count, latents)`.

        Args:
            states: States of shape `(autoencoders, count, width)`, in the units of the model.
        """
        return self.sparse(self.preactivations(states), self.k)

    @staticmethod
    def sparse(pre: Tensor, k: int) -> Tensor:
        """The `k` largest of every row, rectified, with every other entry zero."""
        values, indices = pre.topk(k, dim=-1)
        return torch.zeros_like(pre).scatter(-1, indices, values.relu())

    def decode(self, latents: Tensor) -> Tensor:
        """States rebuilt from latents, in the units of the model."""
        rebuilt = torch.baddbmm(self.bias[:, None], latents, self.decoder)
        return rebuilt / self.scale[:, None, None]

    def forward(self, states: Tensor) -> Tensor:
        """States rebuilt through the autoencoders, in the units of the model."""
        return self.decode(self.encode(states))

    def latents_of(self, index: int, states: Tensor) -> Tensor:
        """Sparse latents of one autoencoder for states of shape `(count, width)`."""
        scaled = states.float() * self.scale[index]
        pre = (scaled - self.bias[index]) @ self.encoder[index] + self.encoder_bias[index]
        return self.sparse(pre, self.k)

    def rebuild(self, index: int, states: Tensor) -> Tensor:
        """States of shape `(count, width)` rebuilt through one autoencoder, in model units."""
        latents = self.latents_of(index, states)
        return (latents @ self.decoder[index] + self.bias[index]) / self.scale[index]

    def loss(
        self, index: int, states: Tensor, dead: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Training loss of one autoencoder on a batch, in the scaled units.

        The autoencoders of a bank are trained one after the other on their batch, so that the
        memory of a step is that of one of them.

        Args:
            index: Autoencoder of the bank.
            states: States of shape `(count, width)`.
            dead: Whether each of its latents is dead, `(latents,)`, or None for none.

        Returns:
            The loss, the mean squared error and whether each latent fired, `(latents,)`.
        """
        scaled = states.float() * self.scale[index]
        pre = (scaled - self.bias[index]) @ self.encoder[index] + self.encoder_bias[index]
        latents = self.sparse(pre, self.k)
        error = scaled - (latents @ self.decoder[index] + self.bias[index])
        squared = error.pow(2).sum(dim=-1).mean()
        total = squared
        if dead is not None and bool(dead.any()):
            hidden = pre.masked_fill(~dead, -torch.inf)
            auxiliary = self.sparse(hidden, min(AUX_K, self.latents)) @ self.decoder[index]
            missed = (error.detach() - auxiliary).pow(2).sum(dim=-1).mean()
            total = total + AUX_WEIGHT * missed
        return total, squared.detach(), (latents > 0).any(dim=0)

    @torch.no_grad()
    def unit_rows(self) -> None:
        """Bring every decoder row back to unit norm."""
        self.decoder.div_(self.decoder.norm(dim=-1, keepdim=True).clamp_min(1e-12))

    @torch.no_grad()
    def drop_parallel_gradient(self) -> None:
        """Remove from the gradient of every decoder row its part along the row."""
        if self.decoder.grad is not None:
            along = (self.decoder.grad * self.decoder).sum(dim=-1, keepdim=True)
            self.decoder.grad.sub_(along * self.decoder)

    def single(self, index: int) -> dict[str, Any]:
        """Weights of one autoencoder, to save on their own."""
        return {
            "block": self.blocks[index],
            "seed": self.seeds[index],
            "k": self.k,
            "encoder": self.encoder[index].detach().cpu(),
            "encoder_bias": self.encoder_bias[index].detach().cpu(),
            "decoder": self.decoder[index].detach().cpu(),
            "bias": self.bias[index].detach().cpu(),
            "scale": float(self.scale[index]),
        }

    @classmethod
    def join(cls, saved: Sequence[dict[str, Any]]) -> "Bank":
        """Bank of autoencoders saved on their own with `single`."""
        latents, width = saved[0]["decoder"].shape
        bank = cls(
            [s["block"] for s in saved],
            [s["seed"] for s in saved],
            width,
            latents=latents,
            k=saved[0]["k"],
        )
        with torch.no_grad():
            for name in ("encoder", "encoder_bias", "decoder", "bias"):
                getattr(bank, name).copy_(torch.stack([s[name].float() for s in saved]))
            bank.scale.copy_(torch.tensor([s["scale"] for s in saved]))
        return bank


class Trainer:
    """Online training of a bank on states that arrive in passes over the corpus.

    States are gathered in a buffer on the device. Once it is full, it is shuffled, its first
    half is trained on in batches and its second half waits for the next states, so that the
    states of one sequence are spread over many batches.

    Attributes:
        bank: Autoencoders trained.
        reads: For every autoencoder, the index of its block among the blocks of the states.
        tokens: States trained on so far.
        idle: Tokens since every latent last fired, `(autoencoders, latents)`.
        history: Mean squared error, fraction of unexplained variance and dead fraction of every
            autoencoder, logged every hundred steps.
    """

    def __init__(
        self,
        bank: Bank,
        reads: Sequence[int],
        *,
        capacity: int = 2**18,
        batch: int = BATCH,
        seed: int = 0,
    ) -> None:
        """Prepare the training.

        Args:
            bank: Autoencoders to train, on the device they train on.
            reads: For every autoencoder, the index of its block among the blocks of the states.
            capacity: States the buffer holds.
            batch: States in every step.
            seed: Seed of the shuffling.
        """
        self.bank = bank
        self.reads = list(reads)
        self.capacity = capacity
        self.batch = batch
        self.device = bank.decoder.device
        self.optimizer = torch.optim.Adam(
            bank.parameters(), lr=learning_rate(bank.latents), betas=(0.9, 0.999), eps=6.25e-10
        )
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.buffer: Tensor | None = None
        self.filled = 0
        self.tokens = 0
        self.steps = 0
        self.idle = torch.zeros(len(bank.blocks), bank.latents, device=self.device)
        self.initialized = False
        self.history: list[dict[str, float]] = []

    def add(self, states: Tensor) -> None:
        """Add the states of one pass and train on the buffer whenever it fills.

        Args:
            states: States of shape `(tokens, blocks, width)`.
        """
        states = states.to(self.device)
        if self.buffer is None:
            shape = (self.capacity, *states.shape[1:])
            self.buffer = torch.empty(shape, dtype=torch.bfloat16, device=self.device)
        start = 0
        while start < len(states):
            taken = min(len(states) - start, self.capacity - self.filled)
            self.buffer[self.filled : self.filled + taken] = states[start : start + taken]
            self.filled += taken
            start += taken
            if self.filled == self.capacity:
                self._train(self.capacity // 2)

    def finish(self) -> None:
        """Train on what remains in the buffer."""
        if self.filled:
            self._train(self.filled)

    def _train(self, count: int) -> None:
        assert self.buffer is not None
        order = torch.randperm(self.filled, generator=self.generator, device=self.device)
        self.buffer[: self.filled] = self.buffer[order]
        if not self.initialized:
            self.bank.initialize(self._view(self.buffer[: min(self.filled, 2**14)]))
            self.initialized = True
        for start in range(0, count - self.batch + 1, self.batch):
            self.step(self._view(self.buffer[start : start + self.batch]))
        kept = self.filled - count
        self.buffer[:kept] = self.buffer[count : self.filled].clone()
        self.filled = kept

    def _view(self, states: Tensor) -> Tensor:
        return states[:, self.reads].transpose(0, 1)

    def step(self, states: Tensor) -> None:
        """One step on a batch of shape `(autoencoders, batch, width)`."""
        dead = self.idle >= DEAD_AFTER
        self.optimizer.zero_grad(set_to_none=True)
        errors, firings = [], []
        for index in range(len(self.bank.blocks)):
            total, squared, fired_one = self.bank.loss(index, states[index], dead[index])
            total.backward()  # type: ignore[no-untyped-call]
            errors.append(squared)
            firings.append(fired_one)
        squared, fired = torch.stack(errors), torch.stack(firings)
        self.bank.drop_parallel_gradient()
        self.optimizer.step()
        self.bank.unit_rows()
        count = states.shape[1]
        self.idle = torch.where(fired, torch.zeros_like(self.idle), self.idle + count)
        self.tokens += count
        self.steps += 1
        if self.steps % 100 == 0:
            for index, block in enumerate(self.bank.blocks):
                scaled = states[index].float() * self.bank.scale[index]
                variance = (scaled - scaled.mean(dim=0)).pow(2).sum(dim=-1).mean()
                self.history.append(
                    {
                        "step": self.steps,
                        "tokens": self.tokens,
                        "block": block,
                        "seed": self.bank.seeds[index],
                        "mse": float(squared[index]),
                        "unexplained": float(squared[index] / variance),
                        "dead": float(dead[index].float().mean()),
                    }
                )


def save(bank: Bank, folder: Path) -> list[Path]:
    """Save every autoencoder of a bank in half precision, as `block{B}-seed{S}.pt`."""
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for index in range(len(bank.blocks)):
        weights = {
            key: value.half() if isinstance(value, Tensor) else value
            for key, value in bank.single(index).items()
        }
        target = folder / f"block{bank.blocks[index]}-seed{bank.seeds[index]}.pt"
        torch.save(weights, target)
        written.append(target)
    return written


def load(paths: Iterable[Path]) -> Bank:
    """Bank of the autoencoders saved at the paths, in single precision on the processor."""
    return Bank.join([torch.load(path, map_location="cpu", weights_only=True) for path in paths])
