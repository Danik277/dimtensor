"""Dimensional scaler for physics-informed machine learning.

Provides automatic non-dimensionalization of physical quantities
for neural network training, with inverse transformation to
recover physical units on output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from ..core.dimensions import Dimension
from ..core.dimarray import DimArray
from ..core.units import Unit, dimensionless
from ..errors import DimensionError
from .dimtensor import DimTensor


@dataclass
class ScaleInfo:
    """Information about scaling for a single dimension.

    Attributes:
        dimension: Physical dimension being scaled.
        scale: Scale factor (characteristic value).
        offset: Offset (for centering).
        unit: Original unit used.
    """

    dimension: Dimension
    scale: float
    offset: float
    unit: Unit


class DimScaler:
    """Non-dimensionalizes physical quantities for neural network training.

    Neural networks train best with inputs in range [-1, 1] or [0, 1].
    Physical quantities often span many orders of magnitude and have
    arbitrary units. DimScaler automatically scales physical data to
    dimensionless values and inverts the transformation on outputs.

    Three scaling methods are available:
    - 'characteristic': Divide by characteristic (max absolute) value
    - 'standard': Subtract mean, divide by std (z-score)
    - 'minmax': Scale to [0, 1] range

    Examples:
        >>> from dimtensor import DimArray, units
        >>> from dimtensor.torch import DimScaler
        >>>
        >>> # Create scaler
        >>> scaler = DimScaler(method='characteristic')
        >>>
        >>> # Fit on training data
        >>> velocities = DimArray([10, 100, 1000], units.m / units.s)
        >>> temperatures = DimArray([300, 400, 500], units.K)
        >>> scaler.fit(velocities, temperatures)
        >>>
        >>> # Transform to dimensionless tensors
        >>> v_scaled = scaler.transform(velocities)  # torch.Tensor in [-1, 1]
        >>> T_scaled = scaler.transform(temperatures)
        >>>
        >>> # Train neural network with scaled data...
        >>>
        >>> # Inverse transform predictions back to physical units
        >>> v_pred = scaler.inverse_transform(output_tensor, units.m / units.s)
    """

    def __init__(
        self,
        method: Literal["characteristic", "standard", "minmax"] = "characteristic",
    ) -> None:
        """Initialize scaler.

        Args:
            method: Scaling method to use:
                - 'characteristic': Divide by max absolute value
                - 'standard': z-score normalization (mean=0, std=1)
                - 'minmax': Scale to [0, 1]
        """
        self.method = method
        self._scales: dict[Dimension, ScaleInfo] = {}
        self._fitted = False

    def fit(self, *arrays: DimArray | DimTensor) -> "DimScaler":
        """Learn scaling parameters from data.

        Args:
            *arrays: DimArray or DimTensor instances to learn scales from.
                     Each unique dimension will have its own scale.

        Returns:
            self for chaining.

        Raises:
            ValueError: If arrays is empty.
        """
        if not arrays:
            raise ValueError("At least one array required for fitting")

        for arr in arrays:
            dim = arr.unit.dimension

            # Get values as numpy for statistics
            if isinstance(arr, DimTensor):
                values = arr.data.detach().cpu().numpy()
            else:
                values = np.asarray(arr.data)

            # Compute scaling parameters
            if self.method == "characteristic":
                scale = float(np.max(np.abs(values)))
                if scale == 0:
                    scale = 1.0
                offset = 0.0
            elif self.method == "standard":
                offset = float(np.mean(values))
                scale = float(np.std(values))
                if scale == 0:
                    scale = 1.0
            elif self.method == "minmax":
                min_val = float(np.min(values))
                max_val = float(np.max(values))
                offset = min_val
                scale = max_val - min_val
                if scale == 0:
                    scale = 1.0
            else:
                raise ValueError(f"Unknown method: {self.method}")

            # Store or update scale info
            if dim not in self._scales:
                self._scales[dim] = ScaleInfo(
                    dimension=dim, scale=scale, offset=offset, unit=arr.unit
                )
            else:
                # Update with combined statistics (take max scale)
                existing = self._scales[dim]
                self._scales[dim] = ScaleInfo(
                    dimension=dim,
                    scale=max(existing.scale, scale),
                    offset=(existing.offset + offset) / 2,
                    unit=arr.unit,
                )

        self._fitted = True
        return self

    def transform(self, arr: DimArray | DimTensor) -> Tensor:
        """Transform physical array to dimensionless tensor.

        Args:
            arr: DimArray or DimTensor with physical units.

        Returns:
            Dimensionless torch.Tensor ready for neural network.

        Raises:
            RuntimeError: If not fitted.
            DimensionError: If dimension not seen during fit.
        """
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        dim = arr.unit.dimension

        if dim not in self._scales:
            raise DimensionError(
                f"Dimension {dim} not seen during fitting. "
                f"Known dimensions: {list(self._scales.keys())}"
            )

        scale_info = self._scales[dim]

        # Get values
        if isinstance(arr, DimTensor):
            values = arr.data
        else:
            values = torch.tensor(arr.data, dtype=torch.float32)

        # Apply scaling
        if self.method == "characteristic":
            return values / scale_info.scale
        elif self.method == "standard":
            return (values - scale_info.offset) / scale_info.scale
        elif self.method == "minmax":
            return (values - scale_info.offset) / scale_info.scale
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def inverse_transform(
        self, tensor: Tensor, unit: Unit, infer_dimension: bool = True
    ) -> DimTensor:
        """Transform dimensionless tensor back to physical units.

        Args:
            tensor: Dimensionless tensor from neural network output.
            unit: Target unit for the output.
            infer_dimension: If True, infer scale from unit's dimension.
                            If False, use unit directly without scaling.

        Returns:
            DimTensor with physical units.

        Raises:
            RuntimeError: If not fitted.
            DimensionError: If dimension not known.
        """
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        dim = unit.dimension

        if infer_dimension and dim in self._scales:
            scale_info = self._scales[dim]

            # Inverse scaling
            if self.method == "characteristic":
                values = tensor * scale_info.scale
            elif self.method == "standard":
                values = tensor * scale_info.scale + scale_info.offset
            elif self.method == "minmax":
                values = tensor * scale_info.scale + scale_info.offset
            else:
                raise ValueError(f"Unknown method: {self.method}")

            # Convert to target unit if different from fitted unit
            if unit != scale_info.unit and scale_info.unit.is_compatible(unit):
                factor = scale_info.unit.conversion_factor(unit)
                values = values * factor

            return DimTensor._from_tensor_and_unit(values, unit)
        else:
            # No scaling, just assign unit
            return DimTensor._from_tensor_and_unit(tensor.clone(), unit)

    def get_scale(self, dimension: Dimension) -> float:
        """Get the scale factor for a dimension.

        Args:
            dimension: Physical dimension to query.

        Returns:
            Scale factor.
        """
        if dimension not in self._scales:
            raise DimensionError(f"Dimension {dimension} not in scaler")
        return self._scales[dimension].scale

    def get_offset(self, dimension: Dimension) -> float:
        """Get the offset for a dimension.

        Args:
            dimension: Physical dimension to query.

        Returns:
            Offset value.
        """
        if dimension not in self._scales:
            raise DimensionError(f"Dimension {dimension} not in scaler")
        return self._scales[dimension].offset

    @property
    def dimensions(self) -> list[Dimension]:
        """List of dimensions this scaler knows about."""
        return list(self._scales.keys())

    def __repr__(self) -> str:
        if not self._fitted:
            return f"DimScaler(method='{self.method}', fitted=False)"

        dims = ", ".join(str(d) for d in self._scales.keys())
        return f"DimScaler(method='{self.method}', dimensions=[{dims}])"


class MultiScaler:
    """Manages multiple DimScalers for complex physics problems.

    For problems with many physical quantities (position, velocity,
    temperature, pressure, etc.), MultiScaler provides a convenient
    interface to manage scaling for all of them.

    Examples:
        >>> from dimtensor import DimArray, units
        >>> from dimtensor.torch import MultiScaler
        >>>
        >>> scaler = MultiScaler()
        >>> scaler.add('position', position_data)
        >>> scaler.add('velocity', velocity_data)
        >>> scaler.add('temperature', temperature_data)
        >>>
        >>> # Transform
        >>> x_scaled = scaler.transform('position', x)
        >>> v_scaled = scaler.transform('velocity', v)
        >>>
        >>> # Inverse transform
        >>> x_pred = scaler.inverse_transform('position', output)
    """

    def __init__(
        self, method: Literal["characteristic", "standard", "minmax"] = "characteristic"
    ) -> None:
        self.method = method
        self._scalers: dict[str, DimScaler] = {}

    def add(self, name: str, *arrays: DimArray | DimTensor) -> "MultiScaler":
        """Add a named quantity and fit its scaler.

        Args:
            name: Name for this quantity.
            *arrays: Arrays to fit the scaler on.

        Returns:
            self for chaining.
        """
        scaler = DimScaler(method=self.method)
        scaler.fit(*arrays)
        self._scalers[name] = scaler
        return self

    def transform(self, name: str, arr: DimArray | DimTensor) -> Tensor:
        """Transform a named quantity.

        Args:
            name: Name of the quantity.
            arr: Array to transform.

        Returns:
            Dimensionless tensor.
        """
        if name not in self._scalers:
            raise KeyError(f"Unknown quantity: {name}")
        return self._scalers[name].transform(arr)

    def inverse_transform(self, name: str, tensor: Tensor, unit: Unit) -> DimTensor:
        """Inverse transform a named quantity.

        Args:
            name: Name of the quantity.
            tensor: Tensor to transform.
            unit: Target unit.

        Returns:
            DimTensor with units.
        """
        if name not in self._scalers:
            raise KeyError(f"Unknown quantity: {name}")
        return self._scalers[name].inverse_transform(tensor, unit)

    def get_scaler(self, name: str) -> DimScaler:
        """Get the scaler for a named quantity."""
        if name not in self._scalers:
            raise KeyError(f"Unknown quantity: {name}")
        return self._scalers[name]

    @property
    def quantities(self) -> list[str]:
        """List of registered quantity names."""
        return list(self._scalers.keys())

    def __repr__(self) -> str:
        return f"MultiScaler(quantities={self.quantities})"
