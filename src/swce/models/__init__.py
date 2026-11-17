"""Subjects, the frozen vision language models under evaluation."""

from swce.models.hf import (
    HuggingFaceSubject,
    llava_15,
    llava_15_rehearsal,
    llava_onevision_15_8b,
    qwen3_vl_8b,
)
from swce.models.subject import Generation, Subject

__all__ = [
    "Generation",
    "HuggingFaceSubject",
    "Subject",
    "llava_15",
    "llava_15_rehearsal",
    "llava_onevision_15_8b",
    "qwen3_vl_8b",
]
