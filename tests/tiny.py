"""Real architectures with random weights and a few small layers, built without any download."""

from typing import Any

import torch
from torch import Tensor
from transformers import (
    CLIPVisionConfig,
    LlamaConfig,
    LlavaConfig,
    LlavaForConditionalGeneration,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
)

from swce.models.hf import HuggingFaceSubject

WIDTH = 16
"""Width of the residual stream of both tiny models."""


def tiny_llava() -> HuggingFaceSubject:
    """LLaVA with a three block CLIP encoder and a four block Llama decoder."""
    torch.manual_seed(0)
    vision = CLIPVisionConfig(
        hidden_size=WIDTH,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        image_size=28,
        patch_size=14,
    )
    text = LlamaConfig(
        hidden_size=WIDTH,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        vocab_size=60,
    )
    config = LlavaConfig(vision_config=vision, text_config=text, image_token_index=59)
    return HuggingFaceSubject(
        LlavaForConditionalGeneration(config),
        None,
        name="tiny-llava",
        description="Random weights.",
        decoder=("language_model", "layers"),
        encoder=("vision_tower", "layers"),
        image_span="<image>",
        fixed_grid=True,
    )


def tiny_qwen() -> HuggingFaceSubject:
    """Qwen3-VL with a three block vision encoder and a four block decoder."""
    torch.manual_seed(0)
    config = Qwen3VLConfig(
        vision_config={
            "depth": 3,
            "hidden_size": WIDTH,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": WIDTH,
            "patch_size": 14,
            "deepstack_visual_indexes": [1],
            "num_position_embeddings": 16,
        },
        text_config={
            "hidden_size": WIDTH,
            "intermediate_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 60,
        },
        image_token_id=59,
        video_token_id=58,
        vision_start_token_id=57,
        vision_end_token_id=56,
    )
    return HuggingFaceSubject(
        Qwen3VLForConditionalGeneration(config),
        None,
        name="tiny-qwen",
        description="Random weights.",
        decoder=("language_model", "layers"),
        encoder=("visual", "blocks"),
        image_span="<|vision_start|><|image_pad|><|vision_end|>",
        fixed_grid=False,
    )


def inputs(subject: HuggingFaceSubject) -> dict[str, Tensor]:
    """A fixed prompt with one small image, in the precision of the model."""
    dtype = subject.dtype
    if subject.name == "tiny-llava":
        return {
            "input_ids": torch.tensor([[1, 59, 59, 59, 59, 5, 6]]),
            "pixel_values": torch.zeros(1, 3, 28, 28, dtype=dtype),
        }
    tokens = torch.tensor([[1, 57, 59, 56, 5, 6]])
    return {
        "input_ids": tokens,
        "attention_mask": torch.ones_like(tokens),
        "pixel_values": torch.zeros(4, 3 * 2 * 14 * 14, dtype=dtype),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
        "mm_token_type_ids": torch.tensor([[0, 0, 1, 0, 0, 0]]),
    }


def outputs(subject: HuggingFaceSubject) -> Any:
    """Output of the model on the fixed prompt."""
    with torch.no_grad():
        return subject.model(**inputs(subject))


def logits(subject: HuggingFaceSubject) -> Tensor:
    """Logits of a fixed prompt with one small image, in the precision of the model."""
    found: Tensor = outputs(subject).logits
    return found
