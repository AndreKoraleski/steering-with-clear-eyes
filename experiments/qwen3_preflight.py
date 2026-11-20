"""Validate the Qwen environment and compare 40 answers with the official checkpoint.

The reference uses native Transformers directly, without Subject, steering or evaluation
generation. It uses the pinned official checkpoint and processor, the same benchmark prompts,
the default pixel budget, bfloat16 and the checkpoint's sampling settings with seed zero.
The reference and SWCE models are loaded sequentially, never together on the GPU.
This is an adapter-equivalence check, not an independent implementation of Transformers.
"""

import argparse
import gc
import json
import platform
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from swce.models.hf import (
    QWEN3_VL,
    qwen3_vl_8b,
    require_every_weight,
    require_finite_buffers,
)
from swce.models.runtime import attention_settings
from swce.tasks.schema import Item
from swce.tasks.task import Task

QUESTIONS = 40
"""Leading questions to compare, not a limit over images."""


def write_json(path: Path, value: object) -> None:
    """Write a readable report without accepting NaN or infinity.

    Args:
        path: Destination in the private preflight folder.
        value: JSON-compatible report.
    """
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def check_sampling(settings: Mapping[str, Any]) -> None:
    """Require the checkpoint settings fixed by issue #3, without overriding them.

    Args:
        settings: The generation configuration as a dictionary.

    Raises:
        RuntimeError: If any required setting is different.
    """
    expected = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20}
    changed = {
        key: settings.get(key) for key, value in expected.items() if settings.get(key) != value
    }
    if changed:
        raise RuntimeError(f"Unexpected checkpoint generation settings: {changed}")


def checked_environment() -> dict[str, object]:
    """Read software versions and require one functioning H100 with the locked CUDA build.

    Returns:
        Environment fields that must remain identical when resuming this run.

    Raises:
        RuntimeError: If CUDA, bfloat16 or the expected accelerator/build is unavailable.
    """
    import torchvision  # type: ignore[import-untyped]
    import transformers

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one Slurm-allocated CUDA GPU must be visible; no CPU fallback")
    gpu = torch.cuda.get_device_name(0)
    if "H100" not in gpu or not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"Expected an H100 supporting bfloat16, found {gpu}")
    if torch.version.cuda != "13.0":
        raise RuntimeError(f"Expected locked CUDA 13.0 build, found {torch.version.cuda}")
    # A real operation detects incompatible drivers instead of trusting device enumeration.
    sample = torch.ones((16, 16), device="cuda", dtype=torch.bfloat16)
    if not torch.isfinite(sample @ sample).all().item():
        raise RuntimeError("The CUDA bfloat16 check produced non-finite values")
    torch.cuda.synchronize()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu,
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "checkpoint": QWEN3_VL[0],
        "revision": QWEN3_VL[1],
        "dtype": "torch.bfloat16",
        "seed": 0,
        "attention": attention_settings(),
    }


def answer_row(item: Item, text: str, tokens: int) -> dict[str, object]:
    """Describe one answer with the identity of the original question.

    Args:
        item: Original POPE question.
        text: Decoded answer.
        tokens: Number of generated tokens, including any special tokens.

    Returns:
        Fields compared between the native checkpoint and the SWCE adapter.
    """
    return {
        "id": item.id,
        "image": item.image,
        "prompt": item.prompt,
        "text": text,
        "tokens": tokens,
    }


@torch.inference_mode()
def reference_answers(
    task: Task, items: Sequence[Item]
) -> tuple[list[dict[str, object]], dict[str, Any]]:
    """Generate native checkpoint answers without using the SWCE model wrapper.

    Args:
        task: POPE task, used only to obtain the benchmark images and token budget.
        items: The leading questions in file order.

    Returns:
        Reference answers and the complete unmodified generation configuration.
    """
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    # The native reference bypasses the Subject constructor; use the same backend policy.
    torch.backends.cuda.enable_cudnn_sdp(False)
    checkpoint, revision = QWEN3_VL
    processor = AutoProcessor.from_pretrained(checkpoint, revision=revision)  # type: ignore[no-untyped-call]
    model: Any
    loading: dict[str, Any]
    model, loading = Qwen3VLForConditionalGeneration.from_pretrained(  # type: ignore[misc]
        checkpoint, revision=revision, dtype=torch.bfloat16, output_loading_info=True
    )
    require_every_weight(loading, checkpoint)
    require_finite_buffers(model, checkpoint)
    model = model.to("cuda").eval()
    settings: dict[str, Any] = model.generation_config.to_dict()
    check_sampling(settings)
    rows = []
    for index, item in enumerate(items, start=1):
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": item.prompt}],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        with task.image(item) as image:
            encoded = processor(images=[image], text=[text], return_tensors="pt")
        inputs = {
            key: value.to("cuda", dtype=torch.bfloat16 if value.is_floating_point() else None)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }
        torch.manual_seed(0)
        output = model.generate(
            **inputs,
            max_new_tokens=task.card.max_new_tokens,
            output_logits=True,
            return_dict_in_generate=True,
        )
        tokens = output.sequences[0, inputs["input_ids"].shape[1] :]
        answer = processor.decode(tokens, skip_special_tokens=True).strip()
        rows.append(answer_row(item, answer, int(tokens.numel())))
        print(f"Official checkpoint: {index}/{len(items)}", flush=True)
    return rows, settings


def subject_answers(
    task: Task, items: Sequence[Item]
) -> tuple[list[dict[str, object]], dict[str, Any]]:
    """Generate answers with the exact factory used by the main experiment.

    Args:
        task: POPE task supplying images and the token budget.
        items: The same leading questions as the reference.

    Returns:
        SWCE answers and the checkpoint generation configuration.

    Raises:
        RuntimeError: If the factory changes the device, dtype or seed.
    """
    subject = qwen3_vl_8b(device="cuda")
    if subject.device.type != "cuda" or subject.dtype != torch.bfloat16 or subject.seed != 0:
        raise RuntimeError("The subject must use CUDA, bfloat16 and seed zero")
    generation = getattr(subject.model, "generation_config", None)
    if generation is None:
        raise RuntimeError("The subject has no checkpoint generation configuration")
    settings: dict[str, Any] = generation.to_dict()
    check_sampling(settings)
    rows = []
    for index, item in enumerate(items, start=1):
        with task.image(item) as image:
            answer = subject.generate(image, item.prompt, task.card.max_new_tokens)
        rows.append(answer_row(item, answer.text, answer.tokens))
        print(f"SWCE: {index}/{len(items)}", flush=True)
    return rows, settings


def compare_answers(
    reference: Sequence[dict[str, object]], actual: Sequence[dict[str, object]]
) -> list[str]:
    """Check the complete sample and return every mismatching question identifier.

    Args:
        reference: Native checkpoint records in the original order.
        actual: SWCE records in the same order.

    Returns:
        Question identifiers whose identity, prompt, text or token count differs.

    Raises:
        ValueError: If either run did not answer exactly forty distinct questions.
    """
    for rows in (reference, actual):
        if len(rows) != QUESTIONS or len({str(row["id"]) for row in rows}) != QUESTIONS:
            raise ValueError(f"Expected exactly {QUESTIONS} distinct POPE questions in each run")
    return [
        str(expected["id"])
        for expected, found in zip(reference, actual, strict=True)
        if expected != found
    ]


def main(argv: list[str] | None = None) -> int:
    """Run the preflight and save evidence before allowing the long experiment.

    Args:
        argv: CLI arguments, or None for the process arguments.

    Returns:
        Zero only after the environment, strict loads and all forty answers match.

    Raises:
        RuntimeError: If the environment changed or the reference comparison fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args(argv)
    args.results.mkdir(parents=True, exist_ok=True)
    report_path = args.results / "comparison.json"
    write_json(report_path, {"status": "started", "questions": QUESTIONS})
    # Snapshot the effective policy before either path loads its model.
    torch.backends.cuda.enable_cudnn_sdp(False)
    environment = checked_environment()
    environment_path = args.results / "environment.json"
    if environment_path.exists():
        previous = json.loads(environment_path.read_text(encoding="utf-8"))
        if previous != environment:
            raise RuntimeError("Resume environment differs; use a new SWCE_RUN_ID")
    write_json(environment_path, environment)
    initial_attention = attention_settings()
    task = Task("pope-adversarial", args.data)
    items = task.items()[:QUESTIONS]
    if len(items) != QUESTIONS:
        raise RuntimeError(f"Task does not contain the required {QUESTIONS} questions")
    reference, reference_settings = reference_answers(task, items)
    reference_attention = attention_settings()
    write_json(args.results / "reference.json", reference)
    write_json(args.results / "generation.json", reference_settings)
    gc.collect()
    torch.cuda.empty_cache()
    actual, actual_settings = subject_answers(task, items)
    actual_attention = attention_settings()
    write_json(args.results / "swce.json", actual)
    mismatches = compare_answers(reference, actual)
    same_settings = reference_settings == actual_settings
    same_attention = initial_attention == reference_attention == actual_attention
    passed = not mismatches and same_settings and same_attention
    write_json(
        report_path,
        {
            "status": "passed" if passed else "failed",
            "questions": QUESTIONS,
            "task_fingerprint": task.fingerprint,
            "mismatching_ids": mismatches,
            "same_generation_settings": same_settings,
            "same_attention_settings": same_attention,
            "attention": {
                "initial": initial_attention,
                "reference": reference_attention,
                "swce": actual_attention,
            },
            "environment": environment,
        },
    )
    if not passed:
        raise RuntimeError(f"Reference comparison failed; inspect {report_path}")
    print(f"Preflight passed: {QUESTIONS}/{QUESTIONS} answers match. Evidence: {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
