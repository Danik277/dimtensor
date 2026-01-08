"""PyTorch integration for dimtensor.

Provides DimTensor, a torch.Tensor wrapper with physical unit tracking.
Supports autograd, GPU acceleration, and neural network compatibility.

Example:
    >>> import torch
    >>> from dimtensor.torch import DimTensor
    >>> from dimtensor import units
    >>>
    >>> velocity = DimTensor(torch.randn(32, 3), units.m / units.s)
    >>> velocity.requires_grad_(True)
    >>> energy = 0.5 * mass * velocity**2
    >>> energy.sum().backward()
"""

from .dimtensor import DimTensor

__all__ = ["DimTensor"]
