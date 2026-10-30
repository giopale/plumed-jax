"""Weighted RMSD and smooth path CVs, with differentiable optimal alignment."""

from __future__ import annotations

from collections.abc import Sequence
from jax import Array
from jax.typing import ArrayLike
from .geometry import PreparedCell
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from .geometry import unwrap_ordered


@jax.custom_jvp
def _principal_quaternion(matrix):
    return jnp.linalg.eigh(matrix)[1][:, -1]


@_principal_quaternion.defjvp
def _quaternion_jvp(primals, tangents):
    (matrix,), (tangent,) = primals, tangents
    q = _principal_quaternion(matrix)
    eigenvalue = q @ matrix @ q
    deigenvalue = q @ tangent @ q
    # Only the principal eigenvalue must be isolated. Repeated smaller
    # eigenvalues (e.g. a symmetric tetrahedron) do not cause division by zero.
    dq = jnp.linalg.solve(
        eigenvalue * jnp.eye(4) - matrix + jnp.outer(q, q),
        tangent @ q - deigenvalue * q,
    )
    return q, dq


def _rotation(cov):
    a, b, c = cov[0]
    d, e, f = cov[1]
    g, h, i = cov[2]
    matrix = jnp.array(
        [
            [a + e + i, f - h, g - c, b - d],
            [f - h, a - e - i, b + d, c + g],
            [g - c, b + d, -a + e - i, f + h],
            [b - d, c + g, f + h, -a - e + i],
        ]
    )
    w, x, y, z = _principal_quaternion(matrix)
    column = jnp.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return column.T


def rmsd(
    positions: Array,
    reference: ArrayLike,
    indices: Sequence[int] | None = None,
    *,
    alignment_weights: ArrayLike | None = None,
    displacement_weights: ArrayLike | None = None,
    optimal: bool = True,
    squared: bool = False,
    cell: ArrayLike | PreparedCell | None = None,
) -> Array:
    """Return scalar root-mean-square displacement after weighted alignment.

    positions has shape (N, 3); static zero-based indices select M atoms
    (or all atoms when None). reference has shape (M, 3) in the same length
    unit. alignment_weights and displacement_weights are (M,) nonnegative
    arrays with positive sums; defaults are equal alignment weights and
    the same displacement weights. These weight preconditions are not checked.
    optimal=True rotates after removing translation; False only removes
    translation. squared=True returns length squared instead of length.
    cell rebuilds selected atoms in their order. Fix flags and selection
    before JIT. Mismatched coordinate/reference shapes raise ValueError.
    Unsquared RMSD at zero and nonunique optimal rotations are not smooth.
    """
    points = positions if indices is None else positions[jnp.asarray(indices)]
    points = unwrap_ordered(points, cell)
    target = jnp.asarray(reference)
    if points.shape != target.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Reference shape must match the selected coordinates")
    alignment = (
        jnp.ones(len(points), dtype=points.dtype)
        if alignment_weights is None
        else jnp.asarray(alignment_weights)
    )
    weights = (
        alignment if displacement_weights is None else jnp.asarray(displacement_weights)
    )
    alignment = alignment / jnp.sum(alignment)
    weights = weights / jnp.sum(weights)
    centered_points = points - jnp.sum(points * alignment[:, None], axis=0)
    centered_target = target - jnp.sum(target * alignment[:, None], axis=0)
    if optimal:
        centered_points = centered_points @ _rotation(
            centered_points.T @ (alignment[:, None] * centered_target)
        )
    msd = jnp.sum(weights * jnp.sum((centered_points - centered_target) ** 2, axis=1))
    return msd if squared else jnp.sqrt(msd)


def path_msd(
    positions: Array,
    references: ArrayLike,
    lam: float,
    indices: Sequence[int] | None = None,
    *,
    cell: ArrayLike | PreparedCell | None = None,
    alignment_weights: ArrayLike | None = None,
    displacement_weights: ArrayLike | None = None,
) -> tuple[Array, Array]:
    """Return (s, z), the smooth frame label and squared-length path distance.

    positions has shape (N, 3); references has shape (F, M, 3). Static
    zero-based indices select M atoms, or None selects all atoms. lam is
    a positive static smoothing constant in inverse length squared
    (ValueError otherwise). s averages frame labels 1..F and is dimensionless;
    z uses length squared and can be negative. Each frame uses optimal
    squared RMSD; weights, cell, and their preconditions are as in rmsd.
    References and selected coordinates must use the same length unit.
    """
    if lam <= 0:
        raise ValueError("Path smoothing lambda must be positive")
    d = jax.vmap(
        lambda ref: rmsd(
            positions,
            ref,
            indices,
            optimal=True,
            squared=True,
            cell=cell,
            alignment_weights=alignment_weights,
            displacement_weights=displacement_weights,
        )
    )(jnp.asarray(references))
    logs = -lam * d
    normalization = logsumexp(logs)
    probabilities = jnp.exp(logs - normalization)
    return jnp.sum((jnp.arange(len(d)) + 1) * probabilities), -normalization / lam
