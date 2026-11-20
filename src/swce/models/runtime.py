"""Read-only snapshots of process-wide attention flags for preflight provenance."""

import torch


def attention_settings() -> dict[str, bool]:
    """Read effective backend flags without initializing CUDA or claiming kernel dispatch.

    Returns:
        Selectable SDPA backends, general cuDNN state and deterministic-algorithm policy.
    """
    # These public flag getters lack annotations in the locked PyTorch release.
    cuda = torch.backends.cuda
    return {
        "cudnn_sdp_enabled": cuda.cudnn_sdp_enabled(),  # type: ignore[no-untyped-call]
        "flash_sdp_enabled": cuda.flash_sdp_enabled(),  # type: ignore[no-untyped-call]
        "mem_efficient_sdp_enabled": cuda.mem_efficient_sdp_enabled(),  # type: ignore[no-untyped-call]
        "math_sdp_enabled": cuda.math_sdp_enabled(),  # type: ignore[no-untyped-call]
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
