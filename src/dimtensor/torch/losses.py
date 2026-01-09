"""Dimension-aware loss functions.

Provides loss functions that check dimensional compatibility between
predictions and targets, preventing training with mismatched units.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from ..core.dimensions import Dimension, DIMENSIONLESS
from ..core.units import Unit
from ..errors import DimensionError
from .dimtensor import DimTensor


class DimMSELoss(nn.Module):
    """Mean squared error loss with dimensional checking.

    Computes MSE loss between predictions and targets, ensuring they have
    compatible physical dimensions. The resulting loss has dimension of
    (input_dimension)^2.

    Args:
        reduction: Specifies the reduction to apply:
            'none': no reduction
            'mean': mean of all elements
            'sum': sum of all elements
        check_units: If True, raise DimensionError on dimension mismatch.
                    If False, skip checking (for raw tensors).

    Examples:
        >>> from dimtensor.torch import DimTensor, DimMSELoss
        >>> from dimtensor import units
        >>>
        >>> loss_fn = DimMSELoss()
        >>> pred = DimTensor(torch.tensor([1.0, 2.0, 3.0]), units.m)
        >>> target = DimTensor(torch.tensor([1.1, 1.9, 3.2]), units.m)
        >>> loss = loss_fn(pred, target)
        >>> print(loss.unit)  # m^2 (squared error)

    Raises:
        DimensionError: If pred and target have incompatible dimensions.
    """

    def __init__(
        self,
        reduction: Literal["none", "mean", "sum"] = "mean",
        check_units: bool = True,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.check_units = check_units
        self._mse = nn.MSELoss(reduction=reduction)

    def forward(
        self, pred: DimTensor | Tensor, target: DimTensor | Tensor
    ) -> DimTensor | Tensor:
        """Compute MSE loss.

        Args:
            pred: Predicted values.
            target: Target values.

        Returns:
            DimTensor if inputs are DimTensor, else raw Tensor.
        """
        # Extract tensors and check dimensions
        if isinstance(pred, DimTensor) and isinstance(target, DimTensor):
            if self.check_units and pred.dimension != target.dimension:
                raise DimensionError(
                    f"Cannot compute loss between {pred.unit} and {target.unit}"
                )

            loss = self._mse(pred.data, target.data)

            # Loss has dimension of input^2
            squared_dim = pred.dimension * pred.dimension
            return DimTensor._from_tensor_and_unit(
                loss, Unit(str(squared_dim), squared_dim, 1.0)
            )

        # Raw tensors - just compute MSE
        pred_t = pred.data if isinstance(pred, DimTensor) else pred
        target_t = target.data if isinstance(target, DimTensor) else target
        result: Tensor = self._mse(pred_t, target_t)
        return result


class DimL1Loss(nn.Module):
    """Mean absolute error loss with dimensional checking.

    Computes L1 (MAE) loss between predictions and targets, ensuring they
    have compatible physical dimensions.

    Args:
        reduction: Specifies the reduction to apply.
        check_units: If True, raise DimensionError on dimension mismatch.

    Examples:
        >>> from dimtensor.torch import DimTensor, DimL1Loss
        >>> from dimtensor import units
        >>>
        >>> loss_fn = DimL1Loss()
        >>> pred = DimTensor(torch.tensor([1.0, 2.0, 3.0]), units.m / units.s)
        >>> target = DimTensor(torch.tensor([1.1, 1.9, 3.2]), units.m / units.s)
        >>> loss = loss_fn(pred, target)  # Has dimension m/s
    """

    def __init__(
        self,
        reduction: Literal["none", "mean", "sum"] = "mean",
        check_units: bool = True,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.check_units = check_units
        self._l1 = nn.L1Loss(reduction=reduction)

    def forward(
        self, pred: DimTensor | Tensor, target: DimTensor | Tensor
    ) -> DimTensor | Tensor:
        """Compute L1 loss."""
        if isinstance(pred, DimTensor) and isinstance(target, DimTensor):
            if self.check_units and pred.dimension != target.dimension:
                raise DimensionError(
                    f"Cannot compute loss between {pred.unit} and {target.unit}"
                )

            loss = self._l1(pred.data, target.data)

            # L1 loss preserves dimension
            return DimTensor._from_tensor_and_unit(loss, pred.unit)

        pred_t = pred.data if isinstance(pred, DimTensor) else pred
        target_t = target.data if isinstance(target, DimTensor) else target
        result: Tensor = self._l1(pred_t, target_t)
        return result


class DimHuberLoss(nn.Module):
    """Huber loss (smooth L1) with dimensional checking.

    A smooth approximation of L1 loss that is quadratic for small errors
    and linear for large errors.

    Args:
        delta: Threshold at which to change from quadratic to linear.
               Note: delta should be dimensionless or match input dimension.
        reduction: Specifies the reduction to apply.
        check_units: If True, raise DimensionError on dimension mismatch.
    """

    def __init__(
        self,
        delta: float = 1.0,
        reduction: Literal["none", "mean", "sum"] = "mean",
        check_units: bool = True,
    ) -> None:
        super().__init__()
        self.delta = delta
        self.reduction = reduction
        self.check_units = check_units
        self._huber = nn.HuberLoss(reduction=reduction, delta=delta)

    def forward(
        self, pred: DimTensor | Tensor, target: DimTensor | Tensor
    ) -> DimTensor | Tensor:
        """Compute Huber loss."""
        if isinstance(pred, DimTensor) and isinstance(target, DimTensor):
            if self.check_units and pred.dimension != target.dimension:
                raise DimensionError(
                    f"Cannot compute loss between {pred.unit} and {target.unit}"
                )

            loss = self._huber(pred.data, target.data)

            # Huber loss has mixed dimension (quadratic and linear parts)
            # For simplicity, we report it as having input dimension
            return DimTensor._from_tensor_and_unit(loss, pred.unit)

        pred_t = pred.data if isinstance(pred, DimTensor) else pred
        target_t = target.data if isinstance(target, DimTensor) else target
        result: Tensor = self._huber(pred_t, target_t)
        return result


class PhysicsLoss(nn.Module):
    """Loss function for conservation law enforcement.

    Computes a loss term that penalizes violations of conservation laws.
    The loss is the relative change in the conserved quantity.

    Args:
        rtol: Relative tolerance for conservation.
        reduction: How to reduce over the batch.

    Examples:
        >>> from dimtensor.torch import DimTensor, PhysicsLoss
        >>> from dimtensor import units
        >>>
        >>> # Penalize energy non-conservation
        >>> physics_loss = PhysicsLoss()
        >>>
        >>> E_initial = DimTensor(torch.tensor([100.0, 200.0]), units.J)
        >>> E_final = DimTensor(torch.tensor([99.0, 198.0]), units.J)
        >>> loss = physics_loss(E_initial, E_final)
        >>> # Loss penalizes the 1% change in energy
    """

    def __init__(
        self,
        rtol: float = 1e-6,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> None:
        super().__init__()
        self.rtol = rtol
        self.reduction = reduction

    def forward(
        self, initial: DimTensor | Tensor, final: DimTensor | Tensor
    ) -> Tensor:
        """Compute conservation loss.

        Args:
            initial: Initial value of conserved quantity.
            final: Final value of conserved quantity.

        Returns:
            Dimensionless loss (relative error squared).
        """
        # Check dimensions match
        if isinstance(initial, DimTensor) and isinstance(final, DimTensor):
            if initial.dimension != final.dimension:
                raise DimensionError(
                    f"Conservation check requires same dimension: "
                    f"{initial.dimension} vs {final.dimension}"
                )
            initial_t = initial.data
            final_t = final.data
        else:
            initial_t = initial.data if isinstance(initial, DimTensor) else initial
            final_t = final.data if isinstance(final, DimTensor) else final

        # Relative error: |final - initial| / (|initial| + eps)
        eps = 1e-8
        diff = torch.abs(final_t - initial_t)
        scale = torch.abs(initial_t) + eps
        relative_error = diff / scale

        # Quadratic penalty for exceeding tolerance
        excess: Tensor = torch.relu(relative_error - self.rtol)
        loss: Tensor = excess**2

        if self.reduction == "mean":
            result: Tensor = loss.mean()
            return result
        elif self.reduction == "sum":
            result = loss.sum()
            return result
        return loss


class CompositeLoss(nn.Module):
    """Combine multiple loss terms with dimensional awareness.

    Allows combining data fidelity loss with physics loss terms,
    handling the different dimensions properly.

    Args:
        data_loss: Loss function for data fidelity.
        physics_losses: Dict of {name: (loss_fn, weight)}.
        normalize: If True, normalize weights to sum to 1.

    Examples:
        >>> from dimtensor.torch import DimMSELoss, PhysicsLoss, CompositeLoss
        >>>
        >>> composite = CompositeLoss(
        ...     data_loss=DimMSELoss(),
        ...     physics_losses={
        ...         'energy': (PhysicsLoss(), 0.1),
        ...         'momentum': (PhysicsLoss(), 0.1),
        ...     }
        ... )
    """

    def __init__(
        self,
        data_loss: nn.Module,
        physics_losses: dict[str, tuple[nn.Module, float]] | None = None,
        normalize: bool = False,
    ) -> None:
        super().__init__()
        self.data_loss = data_loss
        self.physics_losses = nn.ModuleDict()
        self.physics_weights: dict[str, float] = {}

        if physics_losses:
            total_weight = 1.0 + sum(w for _, w in physics_losses.values())
            for name, (loss_fn, weight) in physics_losses.items():
                self.physics_losses[name] = loss_fn
                self.physics_weights[name] = (
                    weight / total_weight if normalize else weight
                )

    def forward(
        self,
        pred: DimTensor | Tensor,
        target: DimTensor | Tensor,
        physics_terms: dict[str, tuple[DimTensor | Tensor, DimTensor | Tensor]]
        | None = None,
    ) -> Tensor:
        """Compute composite loss.

        Args:
            pred: Predicted values.
            target: Target values.
            physics_terms: Dict of {loss_name: (initial, final)} for physics losses.

        Returns:
            Total loss (dimensionless scalar).
        """
        # Data loss - extract magnitude
        data = self.data_loss(pred, target)
        data_t: Tensor
        if isinstance(data, DimTensor):
            data_t = data.data
        else:
            data_t = data

        total: Tensor = data_t

        # Add physics losses
        if physics_terms:
            for name, (initial, final) in physics_terms.items():
                if name in self.physics_losses:
                    loss_fn = self.physics_losses[name]
                    weight = self.physics_weights[name]
                    physics_result = loss_fn(initial, final)
                    physics_t: Tensor
                    if isinstance(physics_result, DimTensor):
                        physics_t = physics_result.data
                    else:
                        physics_t = physics_result
                    total = total + weight * physics_t

        return total
