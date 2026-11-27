"""Jeveration: System One Models Now Generate. Inference code of the released checkpoint."""
import os

# AMD GPUs (ROCm): PyTorch's flash / memory-efficient attention (AOTriton) is still flagged experimental for RDNA 3.5 / 4 and
# falls back to the slow math path without this; it must be set before torch is imported. No effect on NVIDIA.
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

from .pipeline import JeverationPipeline

__all__ = ["JeverationPipeline"]
