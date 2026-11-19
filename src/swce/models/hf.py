"""Vision language models read through the Hugging Face `transformers` library.

Every checkpoint is pinned by revision and loaded without quantization, in the precision its
authors use. Generation uses the settings shipped with the checkpoint, unchanged. Some
checkpoints sample by default, so the random state is seeded before every generation, which makes
an answer repeatable on one machine whatever the order of the items. Weights are downloaded by
the library into its own cache and never enter git.

Module paths differ between architectures and changed across versions of the library, so the
blocks of the decoder and of the vision encoder are found by name.
"""

import inspect
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn

from swce.models.subject import Generation, Subject, unembed

YES_WORDS = ("Yes", "yes")
"""Words whose first token counts as yes at the first generated position."""
NO_WORDS = ("No", "no")
"""Words whose first token counts as no at the first generated position."""


def module_list(model: nn.Module, *fragments: str) -> nn.ModuleList:
    """Longest list of blocks whose name contains every fragment.

    Args:
        model: Model to search.
        *fragments: Substrings the name of the list must contain.

    Returns:
        The longest matching `nn.ModuleList`.

    Raises:
        ValueError: If no list matches.
    """
    found = [
        module
        for name, module in model.named_modules()
        if isinstance(module, nn.ModuleList) and all(fragment in name for fragment in fragments)
    ]
    if not found:
        raise ValueError(f"no list of blocks named like {fragments}")
    return max(found, key=len)


class HuggingFaceSubject(Subject):
    """Vision language model of the `transformers` library with its default prompt template.

    Attributes:
        model: The model, in evaluation mode.
        processor: Its processor, or None for a model built without one in tests.
        image_span: Text standing for the image in the templated prompt.
        fixed_grid: Whether every image yields the same number of image tokens. Methods that fit
            one direction per image token need it.
        seed: Seed of the random state set before every generation.
    """

    def __init__(
        self,
        model: nn.Module,
        processor: Any,
        *,
        name: str,
        description: str,
        decoder: tuple[str, ...],
        encoder: tuple[str, ...],
        image_span: str,
        fixed_grid: bool,
        seed: int = 0,
    ):
        self.model = model.eval()
        self.processor = processor
        self.name = name
        self.description = description
        self.image_span = image_span
        self.fixed_grid = fixed_grid
        self.seed = seed
        self._decoder = module_list(model, *decoder)
        self._encoder = module_list(model, *encoder)

    @property
    def device(self) -> torch.device:
        """Device of the model."""
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Floating point type of the model."""
        return next(self.model.parameters()).dtype

    def settings(self) -> dict[str, object]:
        """Precision, seed, library version and generation settings of the checkpoint.

        Returns:
            The settings as values that can be written as JSON.
        """
        import transformers

        generation = getattr(self.model, "generation_config", None)
        return {
            "transformers": transformers.__version__,
            "dtype": str(self.dtype),
            "seed": self.seed,
            "generation": generation.to_dict() if generation is not None else {},
        }

    def decoder_layers(self) -> Sequence[nn.Module]:
        """Blocks of the language decoder.

        Returns:
            The decoder blocks, whose outputs start with the residual stream.
        """
        return list(self._decoder)

    def decoder_mlps(self) -> Sequence[nn.Module]:
        """Feed forward module of every decoder block.

        Returns:
            The `mlp` module of every block.
        """
        return [block.get_submodule("mlp") for block in self._decoder]

    def encoder_mlps(self) -> Sequence[nn.Module]:
        """Feed forward module of every vision encoder block.

        Returns:
            The `mlp` module of every block.
        """
        return [block.get_submodule("mlp") for block in self._encoder]

    def unembedding(self) -> nn.Module:
        """Linear head mapping the final hidden state to the vocabulary.

        Returns:
            The output embedding of the model.
        """
        head: nn.Module = self.model.get_output_embeddings()  # type: ignore[operator]
        return head

    def template(self, prompt: str) -> str:
        """Prompt placed in the conversation template of the model, with its image.

        Args:
            prompt: Text of the user turn.

        Returns:
            The text given to the processor, ending where the answer starts.
        """
        content = [{"type": "image"}, {"type": "text", "text": prompt}]
        conversation = [{"role": "user", "content": content}]
        text: str = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        return text

    def accepts(self, key: str) -> bool:
        """Whether the forward pass of the model takes an argument.

        A processor of a recent version of the library can return inputs that model code written
        for an earlier version does not take, such as the `mm_token_type_ids` of the Qwen2.5-VL
        processor given to the code of LLaVA-OneVision-1.5. Such code computes what it needs from
        the other inputs, as it did under the version it was written for.

        Args:
            key: Name of the argument.

        Returns:
            True when the forward pass names it or takes any keyword argument.
        """
        parameters = inspect.signature(self.model.forward).parameters
        return key in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        )

    def moved(self, encoded: Any) -> dict[str, Tensor]:
        """Tensors of a processor output on the device and in the type of the model.

        Args:
            encoded: Output of the processor or of its tokenizer.

        Returns:
            Its tensors, moved, without those the forward pass of the model does not take.
        """
        return {
            key: value.to(self.device, self.dtype if value.is_floating_point() else None)
            for key, value in encoded.items()
            if isinstance(value, Tensor) and self.accepts(key)
        }

    def inputs(self, image: Image.Image, text: str) -> dict[str, Tensor]:
        """Tensors of an image and a templated text.

        Args:
            image: Image in RGB.
            text: Templated text holding the image span.

        Returns:
            The keyword arguments of the model.
        """
        return self.moved(self.processor(images=[image], text=[text], return_tensors="pt"))

    def text_inputs(self, text: str) -> dict[str, Tensor]:
        """Tensors of a templated text with its image span removed.

        The text before the span and the text after it are tokenized apart and joined, after the
        token that starts a sequence when the tokenizer has one. This is how VISTA builds the
        prompt it runs without the image.

        Args:
            text: Templated text holding the image span.

        Returns:
            The token identifiers and their attention mask.

        Raises:
            ValueError: If the text does not hold the image span exactly once.
        """
        chunks = text.split(self.image_span)
        if len(chunks) != 2:
            raise ValueError(f"expected one image span {self.image_span!r} in the prompt")
        tokenizer = self.processor.tokenizer
        tokens = [
            token
            for chunk in chunks
            for token in tokenizer(chunk, add_special_tokens=False).input_ids
        ]
        starts_with = tokenizer(text).input_ids[:1]
        if tokenizer.bos_token_id is not None and starts_with == [tokenizer.bos_token_id]:
            tokens = [tokenizer.bos_token_id, *tokens]
        identifiers = torch.tensor([tokens], device=self.device)
        return {"input_ids": identifiers, "attention_mask": torch.ones_like(identifiers)}

    def final_norm(self) -> nn.Module:
        """Normalization the decoder applies before its unembedding.

        Returns:
            The module named `norm` directly under the language model.

        Raises:
            ValueError: If the language model has no such module.
        """
        found: list[nn.Module] = [
            module
            for name, module in self.model.named_modules()
            if name.endswith("language_model.norm")
        ]
        if not found:
            raise ValueError("no final normalization under the language model")
        return found[0]

    def token_text(self, identifiers: Sequence[int]) -> list[str]:
        """Text of tokens of the vocabulary.

        Args:
            identifiers: Identifiers of the tokens.

        Returns:
            Each token decoded on its own.
        """
        return [self.processor.tokenizer.decode([identifier]) for identifier in identifiers]

    def first_tokens(self, words: Sequence[str]) -> list[int]:
        """Identifiers of the tokens that are one of the words, written with or without a space.

        Some tokenizers start a word written after a space with a token holding the space alone,
        which is not the word. Each spelling therefore contributes its first token that is not
        blank, and only when that token is the whole word.

        Args:
            words: Words, such as the spellings of yes.

        Returns:
            The identifiers, sorted and without repetition.

        Raises:
            ValueError: If the vocabulary holds none of the words as one token.
        """
        tokenizer = self.processor.tokenizer
        found = set()
        for word in words:
            for spelling in (word, f" {word}"):
                for identifier in tokenizer.encode(spelling, add_special_tokens=False):
                    piece = tokenizer.decode([identifier]).strip()
                    if piece:
                        if piece == word:
                            found.add(identifier)
                        break
        if not found:
            raise ValueError(f"no token of the vocabulary is one of {tuple(words)}")
        return sorted(found)

    @torch.inference_mode()
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int,
        settings: Mapping[str, object] | None = None,
    ) -> Generation:
        """Answer one image and prompt with the generation settings of the checkpoint.

        Args:
            image: Image in RGB.
            prompt: Text of the user turn.
            max_new_tokens: Largest number of tokens to generate.
            settings: Generation settings replacing those of the checkpoint.

        Returns:
            The answer, with the probabilities of yes and no at its first position, each summed
            over the capitalized and lower case spellings. They are read from the logits before
            any temperature or truncation of the checkpoint settings.
        """
        self.notify(image, prompt)
        inputs = self.inputs(image, self.template(prompt))
        yes, no = self.first_tokens(YES_WORDS), self.first_tokens(NO_WORDS)
        captured: dict[int, Tensor] = {}

        def capture(index: int) -> Any:
            def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
                # Only the first pass of a generation runs over the whole prompt.
                if index not in captured:
                    state = output[0] if isinstance(output, tuple) else output
                    captured[index] = state[0, -1]

            return hook

        blocks = enumerate(self._decoder) if self.lens else ()
        handles = [block.register_forward_hook(capture(index)) for index, block in blocks]
        torch.manual_seed(self.seed)
        try:
            output = self.model.generate(  # type: ignore[operator]
                **inputs,
                **(settings or {}),
                max_new_tokens=max_new_tokens,
                output_logits=True,
                return_dict_in_generate=True,
            )
        finally:
            for handle in handles:
                handle.remove()
        generated = output.sequences[0, inputs["input_ids"].shape[1] :]
        probabilities = torch.softmax(output.logits[0][0].float(), dim=-1)
        lens_yes = lens_no = None
        if captured:
            states = torch.stack([captured[index] for index in range(len(self._decoder))])
            read = torch.softmax(unembed(self, states).float(), dim=-1)
            lens_yes = tuple(read[:, yes].sum(dim=-1).tolist())
            lens_no = tuple(read[:, no].sum(dim=-1).tolist())
        return Generation(
            text=self.processor.decode(generated, skip_special_tokens=True).strip(),
            tokens=int(generated.numel()),
            yes=float(probabilities[yes].sum()),
            no=float(probabilities[no].sum()),
            lens_yes=lens_yes,
            lens_no=lens_no,
        )

    def hidden_states(self, image: Image.Image | None, text: str) -> Tensor:
        """Hidden states at the last position, from the embeddings to the last decoder block.

        Args:
            image: Image in RGB, or None to run the text with its image span removed.
            text: Templated text holding the image span, followed by any beginning of an answer.

        Returns:
            A tensor of shape `(layers + 1, width)` in single precision on the processor.
        """
        inputs = self.text_inputs(text) if image is None else self.inputs(image, text)
        return self.all_states(inputs)[:, -1]

    @torch.inference_mode()
    def all_states(self, inputs: Mapping[str, Tensor]) -> Tensor:
        """Hidden states at every position, from the embeddings to the last decoder block.

        Args:
            inputs: Keyword arguments of the model for one sequence.

        Returns:
            A tensor of shape `(layers + 1, positions, width)` in single precision on the
            processor. As in the library, its last entry is the state after the final
            normalization and not the output of the last block. The official code of VTI, Nullu
            and REVIS fits on these states, and so do their ports.
        """
        output = self.model(**inputs, output_hidden_states=True)
        return torch.stack([state[0] for state in output.hidden_states]).float().cpu()

    def residuals(self, image: Image.Image | None, prompt: str, answer: str = "") -> Tensor:
        """Residual stream at the last position after every decoder block.

        Args:
            image: Image in RGB, or None to run the prompt with the image span removed.
            prompt: Text of the user turn.
            answer: Text appended right after the template.

        Returns:
            A tensor of shape `(layers, width)` in single precision on the processor.
        """
        text = self.template(prompt) + answer
        return self.block_outputs(
            self.text_inputs(text) if image is None else self.inputs(image, text)
        )

    @torch.inference_mode()
    def block_outputs(self, inputs: Mapping[str, Tensor]) -> Tensor:
        """Output of every decoder block at the last position, read from the blocks themselves.

        The hidden states the library returns end with the state after the final normalization,
        which is not the output of the last block. Reading the blocks gives the residual stream
        itself, as the methods that hook the blocks see it, including what steering hooks
        attached earlier changed.

        Args:
            inputs: Keyword arguments of the model for one sequence.

        Returns:
            A tensor of shape `(layers, width)` in single precision on the processor.
        """
        captured: dict[int, Tensor] = {}

        def capture(index: int) -> Any:
            def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
                state = output[0] if isinstance(output, tuple) else output
                captured[index] = state[0, -1].float().cpu()

            return hook

        handles = [
            block.register_forward_hook(capture(index)) for index, block in enumerate(self._decoder)
        ]
        try:
            self.model(**inputs)
        finally:
            for handle in handles:
                handle.remove()
        return torch.stack([captured[index] for index in range(len(self._decoder))])


def _device(device: str | None) -> str:
    """Requested device, or the accelerator when one is present."""
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


LLAVA_15 = ("llava-hf/llava-1.5-7b-hf", "b234b804b114d9e37bb655e11cbbb5f5e971b7a9")
"""Checkpoint and revision of LLaVA-1.5-7B, under the Llama 2 Community Licence."""
QWEN3_VL = ("Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
"""Checkpoint and revision of Qwen3-VL-8B, under the Apache 2.0 licence."""
ONEVISION_15 = (
    "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
    "e137910552d616eb0f7305147ca054311a90c542",
)
"""Checkpoint and revision of LLaVA-OneVision-1.5-8B, under the Apache 2.0 licence."""


def llava_15(device: str | None = None) -> HuggingFaceSubject:
    """LLaVA-1.5-7B (Liu et al., 2024) in half precision.

    Args:
        device: Device to load the model on, or None for the accelerator when one is present.

    Returns:
        The subject, named `llava-1.5-7b`.
    """
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    checkpoint, revision = LLAVA_15
    processor = AutoProcessor.from_pretrained(checkpoint, revision=revision)  # type: ignore[no-untyped-call]
    model: nn.Module = LlavaForConditionalGeneration.from_pretrained(
        checkpoint, revision=revision, dtype=torch.float16
    )
    return HuggingFaceSubject(
        model.to(_device(device)),
        processor,
        name="llava-1.5-7b",
        description=f"{checkpoint} at revision {revision[:8]}, half precision, default generation.",
        decoder=("language_model", "layers"),
        encoder=("vision_tower", "layers"),
        image_span="<image>",
        fixed_grid=True,
    )


def qwen3_vl_8b(device: str | None = None) -> HuggingFaceSubject:
    """Qwen3-VL-8B-Instruct in bfloat16, with the default pixel budget of its processor.

    Args:
        device: Device to load the model on, or None for the accelerator when one is present.

    Returns:
        The subject, named `qwen3-vl-8b`.
    """
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    checkpoint, revision = QWEN3_VL
    processor = AutoProcessor.from_pretrained(checkpoint, revision=revision)  # type: ignore[no-untyped-call]
    model: nn.Module = Qwen3VLForConditionalGeneration.from_pretrained(
        checkpoint, revision=revision, dtype=torch.bfloat16
    )
    return HuggingFaceSubject(
        model.to(_device(device)),
        processor,
        name="qwen3-vl-8b",
        description=f"{checkpoint} at revision {revision[:8]}, bfloat16, default generation.",
        decoder=("language_model", "layers"),
        encoder=("visual", "blocks"),
        image_span="<|vision_start|><|image_pad|><|vision_end|>",
        fixed_grid=False,
    )


def default_rope_parameters(
    config: Any, device: torch.device | str | None = None, *_args: Any, **_kwargs: Any
) -> tuple[Tensor, float]:
    """Frequencies of the original rotary embedding, as `transformers` 4.53 computes them.

    Args:
        config: Configuration holding `rope_theta`, the width of the heads or the hidden size and
            the number of heads, and optionally `partial_rotary_factor`.
        device: Device of the frequencies.
        *_args: Ignored, accepted like the other initializations of the library.
        **_kwargs: Ignored, accepted like the other initializations of the library.

    Returns:
        The inverse frequencies and an attention scaling of one.
    """
    partial = getattr(config, "partial_rotary_factor", 1.0)
    head = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head * partial)
    steps = torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
    return 1.0 / (config.rope_theta ** (steps / dim)), 1.0


@contextmanager
def onevision_code() -> Iterator[None]:
    """Make the model code of LLaVA-OneVision-1.5 buildable under the pinned `transformers`.

    The code reads the original rotary embedding as the entry `default` of the initializations of
    the library when it builds its rotary embedding, which `transformers` 4.53 held and current
    versions do not, since every model now computes it itself. The entry is added with the
    computation of 4.53 while the model is built, and removed afterwards, since other models of the
    library treat its presence differently. The forward pass reads the table only for dynamic
    embeddings, which this checkpoint does not use.

    Yields:
        Nothing. The entry exists inside the context.
    """
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    added = "default" not in ROPE_INIT_FUNCTIONS
    if added:
        ROPE_INIT_FUNCTIONS["default"] = default_rope_parameters
    try:
        yield
    finally:
        if added:
            del ROPE_INIT_FUNCTIONS["default"]


def require_every_weight(loading: Mapping[str, Any], checkpoint: str) -> None:
    """Refuse a model whose weights were not all read from its checkpoint.

    The library loads a model whose weights are missing from the checkpoint, or named otherwise,
    by initializing them at random and only logging it. Such a model answers, and its answers
    mean nothing, so it is never evaluated.

    Args:
        loading: Loading information returned by `from_pretrained` with `output_loading_info`.
        checkpoint: Name of the checkpoint, for the message.

    Raises:
        RuntimeError: If a weight of the model was missing, a weight of the checkpoint unused or a
            shape different.
    """
    problems = {
        kind: sorted(map(str, found))[:5]
        for kind in ("missing_keys", "unexpected_keys", "mismatched_keys")
        if (found := loading.get(kind))
    }
    if problems:
        raise RuntimeError(f"{checkpoint} did not load as its checkpoint stores it: {problems}")


def onevision_buffers(model: nn.Module) -> None:
    """Compute again the rotary frequencies of LLaVA-OneVision-1.5 after its weights are loaded.

    Current versions of the library build a model on the meta device and fill only what the
    checkpoint holds. The rotary embeddings of the language model and of the vision encoder keep
    their frequencies in buffers that are not saved, which their constructors computed under 4.53
    and which are left uninitialized now. Each is built again by its own class, with the arguments
    the model code gives it, and its frequencies are copied into the loaded model.

    Args:
        model: The loaded model, changed in place.
    """
    with onevision_code():
        for module in model.modules():
            kind = type(module).__name__
            if kind not in ("LLaVAOneVision1_5_RotaryEmbedding", "RiceRotaryEmbedding"):
                continue
            current = module.get_buffer("inv_freq")
            build: Any = type(module)
            if kind == "RiceRotaryEmbedding":
                fresh = build(2 * current.numel())
            else:
                fresh = build(config=module.config)
                module.attention_scaling = fresh.attention_scaling
            module.register_buffer("inv_freq", fresh.inv_freq.to(current.device), persistent=False)
            if hasattr(module, "original_inv_freq"):
                module.original_inv_freq = module.get_buffer("inv_freq")


def require_finite_buffers(model: nn.Module, checkpoint: str) -> None:
    """Refuse a model holding a floating buffer that is not finite.

    Args:
        model: The loaded model.
        checkpoint: Name of the checkpoint, for the message.

    Raises:
        RuntimeError: If a floating buffer holds a value that is not finite.
    """
    broken = [
        name
        for name, buffer in model.named_buffers()
        if buffer.is_floating_point() and not torch.isfinite(buffer).all()
    ]
    if broken:
        raise RuntimeError(f"{checkpoint} holds buffers that are not finite: {broken[:5]}")


def onevision_config(config: Any) -> Any:
    """Configuration of LLaVA-OneVision-1.5 completed with what its model code expects.

    The model code shipped with the checkpoint was written for `transformers` 4.53, whose
    configurations always held `pad_token_id`, None unless set. Current versions only define it on
    the configurations that declare it, and the configuration of the language model of this
    checkpoint does not. The model code reads it as the padding index of its embedding, so it is
    set to None, the value it had under the version the code was written for.

    Args:
        config: Configuration read from the checkpoint with its own code.

    Returns:
        The same configuration, completed.
    """
    text = config.text_config
    if "pad_token_id" not in text.__dict__:
        text.pad_token_id = None
    return config


def llava_onevision_15_8b(device: str | None = None) -> HuggingFaceSubject:
    """LLaVA-OneVision-1.5-8B-Instruct in bfloat16, with the default pixel budget of its processor.

    The checkpoint ships its own model code, which the library runs when loading it. The code is
    pinned by the revision of the checkpoint and was written for `transformers` 4.53. It is run
    unchanged, and four differences of the pinned version are bridged around it, each restoring
    what 4.53 did: the renaming of the weights the code declares in
    `_checkpoint_conversion_mapping`, which current versions ignore and take as `key_mapping`, the
    padding index of the language configuration, by `onevision_config`, the original rotary
    embedding, by `onevision_code`, and the `mm_token_type_ids` the processor now returns, which
    `HuggingFaceSubject.moved` leaves out since the forward pass does not take it. The processor
    needs `torchvision`. Loading fails unless every weight is read from the checkpoint.

    Args:
        device: Device to load the model on, or None for the accelerator when one is present.

    Returns:
        The subject, named `llava-onevision-1.5-8b`.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    checkpoint, revision = ONEVISION_15
    processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
        checkpoint, revision=revision, trust_remote_code=True
    )
    config = onevision_config(
        AutoConfig.from_pretrained(checkpoint, revision=revision, trust_remote_code=True)
    )
    code: Any = get_class_from_dynamic_module(
        config.auto_map["AutoModelForCausalLM"], checkpoint, revision=revision
    )
    # Fetched ahead of the automatic class, the class would not be marked as remote code, and
    # generation creates `cache_position`, which the code reads, only for remote code.
    code.register_for_auto_class("AutoModelForCausalLM")
    model: nn.Module
    loading: dict[str, Any]
    with onevision_code():
        model, loading = AutoModelForCausalLM.from_pretrained(  # type: ignore[misc]
            checkpoint,
            revision=revision,
            config=config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            key_mapping=code._checkpoint_conversion_mapping,
            output_loading_info=True,
        )
    require_every_weight(loading, checkpoint)
    onevision_buffers(model)
    require_finite_buffers(model, checkpoint)
    return HuggingFaceSubject(
        model.to(_device(device)),
        processor,
        name="llava-onevision-1.5-8b",
        description=f"{checkpoint} at revision {revision[:8]}, bfloat16, default generation.",
        decoder=("language_model", "layers"),
        encoder=("visual", "blocks"),
        image_span="<|vision_start|><|image_pad|><|vision_end|>",
        fixed_grid=False,
    )


def llava_15_rehearsal() -> HuggingFaceSubject:
    """The processor of LLaVA-1.5 with a small random model of its architecture, on the CPU.

    The model has the vocabulary, the image resolution and the numbers of blocks of LLaVA-1.5,
    with a width of sixteen and random weights, so it runs in seconds without an accelerator and
    answers nonsense. It exists to rehearse an evaluation end to end before any real run, since
    the prompt template, the token lookup, the hooks and the recipes of the methods all take the
    paths they take on the real model. Only the processor is downloaded.

    Returns:
        The subject, named `llava-1.5-rehearsal`, whose results never mix with real ones.
    """
    from transformers import (
        AutoProcessor,
        CLIPVisionConfig,
        LlamaConfig,
        LlavaConfig,
        LlavaForConditionalGeneration,
    )

    checkpoint, revision = LLAVA_15
    processor = AutoProcessor.from_pretrained(checkpoint, revision=revision)  # type: ignore[no-untyped-call]
    tokenizer = processor.tokenizer
    vision = CLIPVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=24,
        num_attention_heads=2,
        image_size=336,
        patch_size=14,
    )
    language = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=32,
        num_attention_heads=2,
        vocab_size=len(tokenizer),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    config = LlavaConfig(
        vision_config=vision,
        text_config=language,
        image_token_index=tokenizer.convert_tokens_to_ids("<image>"),
    )
    torch.manual_seed(0)
    model: nn.Module = LlavaForConditionalGeneration(config)
    return HuggingFaceSubject(
        model,
        processor,
        name="llava-1.5-rehearsal",
        description=f"Random weights of width 16 behind the processor of {checkpoint}.",
        decoder=("language_model", "layers"),
        encoder=("vision_tower", "layers"),
        image_span="<image>",
        fixed_grid=True,
    )
