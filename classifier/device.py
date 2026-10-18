"""Explicit CPU/CUDA selection with a small runtime compatibility check."""


def select_device(requested="cuda"):
    """Return a torch.device; never silently replace requested CUDA with CPU."""
    if requested not in ("cuda", "cpu"):
        raise ValueError("Device must be 'cuda' or 'cpu'.")

    import torch

    if requested == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Check the NVIDIA driver, "
            "container GPU access, and CUDA-enabled PyTorch installation. "
            "Use --device cpu only if you explicitly want CPU training."
        )

    # Availability alone does not prove this build has kernels for the GPU.
    try:
        device = torch.device("cuda")
        with torch.no_grad():
            probe = torch.ones((2, 2), device=device)
            result = probe @ probe
            torch.cuda.synchronize()
            if result[0, 0].item() != 2.0:
                raise RuntimeError("CUDA compatibility probe returned an invalid result.")
        return device
    except (RuntimeError, AssertionError) as error:
        raise RuntimeError(
            "CUDA is visible, but a small GPU operation failed. Install a "
            "PyTorch CUDA build that supports this GPU's compute capability; "
            "do not assume the newest CUDA build supports older cards. "
            "Original error: {}".format(error)
        ) from error
