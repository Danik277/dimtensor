"""Module-level array functions for DimArray.

Provides NumPy-style functions that operate on DimArrays while
maintaining dimensional correctness.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .core.dimarray import DimArray
from .core.units import Unit
from .errors import DimensionError


def _check_same_dimension(arrays: Sequence[DimArray], operation: str) -> Unit:
    """Verify all arrays have same dimension, return the first unit.

    Args:
        arrays: Sequence of DimArrays to check.
        operation: Name of the operation (for error messages).

    Returns:
        The unit of the first array.

    Raises:
        ValueError: If arrays sequence is empty.
        DimensionError: If arrays have incompatible dimensions.
    """
    if not arrays:
        raise ValueError(f"Need at least one array for {operation}")

    first_unit = arrays[0]._unit
    first_dim = first_unit.dimension

    for arr in arrays[1:]:
        if arr._unit.dimension != first_dim:
            raise DimensionError.incompatible(
                first_dim, arr._unit.dimension, operation
            )

    return first_unit


def concatenate(
    arrays: Sequence[DimArray],
    axis: int = 0,
) -> DimArray:
    """Join a sequence of DimArrays along an existing axis.

    All arrays must have the same unit dimension. Arrays with compatible
    units (e.g., km and m) will be converted to the first array's unit.

    Args:
        arrays: Sequence of DimArrays to concatenate.
        axis: The axis along which to concatenate (default: 0).

    Returns:
        Concatenated DimArray with the same unit as the first input array.

    Raises:
        DimensionError: If arrays have incompatible dimensions.
        ValueError: If arrays sequence is empty.

    Examples:
        >>> a = DimArray([1.0, 2.0], units.m)
        >>> b = DimArray([3.0, 4.0], units.m)
        >>> concatenate([a, b])
        DimArray([1. 2. 3. 4.], unit='m')
    """
    unit = _check_same_dimension(arrays, "concatenate")

    # Convert all arrays to the same unit before concatenating
    converted = [arr.to(unit) for arr in arrays]
    raw_arrays = [arr._data for arr in converted]

    result = np.concatenate(raw_arrays, axis=axis)
    return DimArray._from_data_and_unit(result, unit)


def stack(
    arrays: Sequence[DimArray],
    axis: int = 0,
) -> DimArray:
    """Stack a sequence of DimArrays along a new axis.

    All arrays must have the same unit dimension. Arrays with compatible
    units (e.g., km and m) will be converted to the first array's unit.

    Args:
        arrays: Sequence of DimArrays to stack.
        axis: The axis along which to stack (default: 0).

    Returns:
        Stacked DimArray with the same unit as the first input array.

    Raises:
        DimensionError: If arrays have incompatible dimensions.
        ValueError: If arrays sequence is empty.

    Examples:
        >>> a = DimArray([1.0, 2.0], units.m)
        >>> b = DimArray([3.0, 4.0], units.m)
        >>> stack([a, b])
        DimArray([[1. 2.] [3. 4.]], unit='m')
    """
    unit = _check_same_dimension(arrays, "stack")

    # Convert all arrays to the same unit before stacking
    converted = [arr.to(unit) for arr in arrays]
    raw_arrays = [arr._data for arr in converted]

    result = np.stack(raw_arrays, axis=axis)
    return DimArray._from_data_and_unit(result, unit)


def split(
    array: DimArray,
    indices_or_sections: int | Sequence[int],
    axis: int = 0,
) -> list[DimArray]:
    """Split a DimArray into sub-arrays.

    Args:
        array: The DimArray to split.
        indices_or_sections: If an integer N, split into N equal parts.
            If a sequence of integers, split at those indices.
        axis: The axis along which to split (default: 0).

    Returns:
        List of DimArrays, all with the same unit as the input.

    Examples:
        >>> arr = DimArray([1.0, 2.0, 3.0, 4.0], units.m)
        >>> parts = split(arr, 2)
        >>> len(parts)
        2
    """
    raw_splits = np.split(array._data, indices_or_sections, axis=axis)
    return [
        DimArray._from_data_and_unit(sub, array._unit)
        for sub in raw_splits
    ]


def dot(a: DimArray, b: DimArray) -> DimArray:
    """Dot product of two DimArrays.

    Dimensions multiply: if a has dimension D1 and b has dimension D2,
    the result has dimension D1 * D2.

    Args:
        a: First array.
        b: Second array.

    Returns:
        Dot product with multiplied dimensions.

    Examples:
        >>> length = DimArray([1.0, 2.0, 3.0], units.m)
        >>> force = DimArray([4.0, 5.0, 6.0], units.N)
        >>> work = dot(length, force)  # Result has dimension of energy (J)
    """
    result = np.dot(a._data, b._data)
    new_unit = a._unit * b._unit

    # Ensure result is at least 1D for API consistency
    if np.isscalar(result):
        result = np.array([result])

    return DimArray._from_data_and_unit(result, new_unit)


def matmul(a: DimArray, b: DimArray) -> DimArray:
    """Matrix multiplication of two DimArrays.

    Dimensions multiply: if a has dimension D1 and b has dimension D2,
    the result has dimension D1 * D2.

    Args:
        a: First array (must be at least 1D).
        b: Second array (must be at least 1D).

    Returns:
        Matrix product with multiplied dimensions.

    Examples:
        >>> A = DimArray([[1, 2], [3, 4]], units.m)
        >>> B = DimArray([[5, 6], [7, 8]], units.s)
        >>> C = matmul(A, B)  # Result has dimension m*s
    """
    result = np.matmul(a._data, b._data)
    new_unit = a._unit * b._unit

    # Ensure result is at least 1D for API consistency
    if np.isscalar(result):
        result = np.array([result])

    return DimArray._from_data_and_unit(result, new_unit)


def norm(
    array: DimArray,
    ord: float | None = None,
    axis: int | None = None,
    keepdims: bool = False,
) -> DimArray:
    """Compute the norm of a DimArray.

    The result preserves the original unit (norm of meters is meters).

    Args:
        array: Input array.
        ord: Order of the norm (see numpy.linalg.norm).
            None = 2-norm for vectors, Frobenius for matrices.
            Other values: 1, 2, inf, -inf, etc.
        axis: Axis along which to compute the norm.
            If None, computes norm of flattened array.
        keepdims: If True, the reduced axes are kept with size 1.

    Returns:
        Norm with the same unit as the input array.

    Examples:
        >>> v = DimArray([3.0, 4.0], units.m)
        >>> norm(v)  # 5.0 m
        DimArray([5.], unit='m')
    """
    result = np.linalg.norm(array._data, ord=ord, axis=axis, keepdims=keepdims)

    # Ensure result is at least 1D for API consistency
    result_arr = np.atleast_1d(result)

    return DimArray._from_data_and_unit(result_arr, array._unit)
