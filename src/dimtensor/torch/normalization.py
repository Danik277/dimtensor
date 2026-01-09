"""Dimension-aware normalization layers.

Provides batch normalization and layer normalization that preserve
physical dimensions through the normalization process.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..core.dimensions import Dimension, DIMENSIONLESS
from ..core.units import Unit
from .dimtensor import DimTensor


class DimBatchNorm1d(nn.Module):
    """Batch normalization over 2D/3D input preserving dimensions.

    Normalizes the input to zero mean and unit variance while preserving
    the physical dimension. The running mean and variance are stored in
    the physical units of the input.

    Args:
        num_features: Number of features/channels.
        dimension: Physical dimension of the input.
        eps: Value added for numerical stability.
        momentum: Running stats momentum.
        affine: If True, has learnable affine parameters.
        track_running_stats: If True, tracks running mean and variance.

    Examples:
        >>> from dimtensor.torch import DimTensor, DimBatchNorm1d
        >>> from dimtensor import units, Dimension
        >>>
        >>> bn = DimBatchNorm1d(num_features=10, dimension=Dimension(L=1))
        >>> x = DimTensor(torch.randn(32, 10), units.m)
        >>> y = bn(x)  # Normalized, still in meters
    """

    def __init__(
        self,
        num_features: int,
        dimension: Dimension = DIMENSIONLESS,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self._unit = Unit(str(dimension), dimension, 1.0)

        self.bn = nn.BatchNorm1d(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: DimTensor | Tensor) -> DimTensor:
        """Apply batch normalization.

        Args:
            x: Input tensor (N, C) or (N, C, L).

        Returns:
            Normalized DimTensor with same dimension.
        """
        if isinstance(x, DimTensor):
            # Validate dimension
            if x.dimension != self.dimension:
                from ..errors import DimensionError

                raise DimensionError(
                    f"Expected dimension {self.dimension}, got {x.dimension}"
                )
            tensor = x.data
            unit = x.unit
        else:
            tensor = x
            unit = self._unit

        result = self.bn(tensor)
        return DimTensor._from_tensor_and_unit(result, unit)


class DimBatchNorm2d(nn.Module):
    """Batch normalization over 4D input preserving dimensions.

    Same as DimBatchNorm1d but for 4D input (N, C, H, W).

    Args:
        num_features: Number of features/channels.
        dimension: Physical dimension of the input.
        eps: Value added for numerical stability.
        momentum: Running stats momentum.
        affine: If True, has learnable affine parameters.
        track_running_stats: If True, tracks running mean and variance.
    """

    def __init__(
        self,
        num_features: int,
        dimension: Dimension = DIMENSIONLESS,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self._unit = Unit(str(dimension), dimension, 1.0)

        self.bn = nn.BatchNorm2d(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: DimTensor | Tensor) -> DimTensor:
        """Apply batch normalization.

        Args:
            x: Input tensor (N, C, H, W).

        Returns:
            Normalized DimTensor with same dimension.
        """
        if isinstance(x, DimTensor):
            if x.dimension != self.dimension:
                from ..errors import DimensionError

                raise DimensionError(
                    f"Expected dimension {self.dimension}, got {x.dimension}"
                )
            tensor = x.data
            unit = x.unit
        else:
            tensor = x
            unit = self._unit

        result = self.bn(tensor)
        return DimTensor._from_tensor_and_unit(result, unit)


class DimLayerNorm(nn.Module):
    """Layer normalization preserving physical dimensions.

    Normalizes across the specified dimensions while preserving the
    physical dimension of the input.

    Args:
        normalized_shape: Input shape from expected input.
        dimension: Physical dimension of the input.
        eps: Value added for numerical stability.
        elementwise_affine: If True, has learnable affine parameters.

    Examples:
        >>> from dimtensor.torch import DimTensor, DimLayerNorm
        >>> from dimtensor import units, Dimension
        >>>
        >>> ln = DimLayerNorm(normalized_shape=10, dimension=Dimension(M=1, L=2, T=-2))
        >>> x = DimTensor(torch.randn(32, 10), units.J)  # Energy
        >>> y = ln(x)  # Normalized, still in Joules
    """

    def __init__(
        self,
        normalized_shape: int | list[int],
        dimension: Dimension = DIMENSIONLESS,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self._unit = Unit(str(dimension), dimension, 1.0)

        self.ln = nn.LayerNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: DimTensor | Tensor) -> DimTensor:
        """Apply layer normalization.

        Args:
            x: Input tensor.

        Returns:
            Normalized DimTensor with same dimension.
        """
        if isinstance(x, DimTensor):
            if x.dimension != self.dimension:
                from ..errors import DimensionError

                raise DimensionError(
                    f"Expected dimension {self.dimension}, got {x.dimension}"
                )
            tensor = x.data
            unit = x.unit
        else:
            tensor = x
            unit = self._unit

        result = self.ln(tensor)
        return DimTensor._from_tensor_and_unit(result, unit)


class DimInstanceNorm1d(nn.Module):
    """Instance normalization for 3D input preserving dimensions.

    Normalizes each instance independently.

    Args:
        num_features: Number of features/channels.
        dimension: Physical dimension of the input.
        eps: Value added for numerical stability.
        momentum: Running stats momentum.
        affine: If True, has learnable affine parameters.
        track_running_stats: If True, tracks running mean and variance.
    """

    def __init__(
        self,
        num_features: int,
        dimension: Dimension = DIMENSIONLESS,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self._unit = Unit(str(dimension), dimension, 1.0)

        self.norm = nn.InstanceNorm1d(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: DimTensor | Tensor) -> DimTensor:
        """Apply instance normalization."""
        if isinstance(x, DimTensor):
            if x.dimension != self.dimension:
                from ..errors import DimensionError

                raise DimensionError(
                    f"Expected dimension {self.dimension}, got {x.dimension}"
                )
            tensor = x.data
            unit = x.unit
        else:
            tensor = x
            unit = self._unit

        result = self.norm(tensor)
        return DimTensor._from_tensor_and_unit(result, unit)


class DimInstanceNorm2d(nn.Module):
    """Instance normalization for 4D input preserving dimensions.

    Args:
        num_features: Number of features/channels.
        dimension: Physical dimension of the input.
        eps: Value added for numerical stability.
        momentum: Running stats momentum.
        affine: If True, has learnable affine parameters.
        track_running_stats: If True, tracks running mean and variance.
    """

    def __init__(
        self,
        num_features: int,
        dimension: Dimension = DIMENSIONLESS,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self._unit = Unit(str(dimension), dimension, 1.0)

        self.norm = nn.InstanceNorm2d(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: DimTensor | Tensor) -> DimTensor:
        """Apply instance normalization."""
        if isinstance(x, DimTensor):
            if x.dimension != self.dimension:
                from ..errors import DimensionError

                raise DimensionError(
                    f"Expected dimension {self.dimension}, got {x.dimension}"
                )
            tensor = x.data
            unit = x.unit
        else:
            tensor = x
            unit = self._unit

        result = self.norm(tensor)
        return DimTensor._from_tensor_and_unit(result, unit)
