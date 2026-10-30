"""Structural order and contact matrices, with fixed atom selections."""

from __future__ import annotations

from typing import Any

from collections.abc import Sequence
from jax import Array
from jax.typing import ArrayLike
from .geometry import PreparedCell
import jax.numpy as jnp
from math import sqrt
from .geometry import displacement
from .switches import switching


def _indices(values):
    indices = jnp.asarray(values)
    if indices.ndim != 1:
        raise ValueError("Atom groups must be one-dimensional")
    if indices.size == 0:
        return jnp.empty((0,), dtype=jnp.int32)
    if not jnp.issubdtype(indices.dtype, jnp.integer):
        raise ValueError("Atom indices must be integers")
    return indices


def contact_matrix(
    positions: Array,
    group_a: Sequence[int] | ArrayLike,
    group_b: Sequence[int] | ArrayLike | None = None,
    *,
    cell: ArrayLike | PreparedCell | None = None,
    **switch: Any,
) -> Array:
    """Return dimensionless contact weights with shape (len(group_a), len(group_b)).

    positions has shape (N, 3); groups are static zero-based integer
    selections. group_b=None uses group_a. Self pairs are zero; empty groups
    produce empty matrix axes. cell optionally applies periodic wrapping.
    switch keywords are passed to switching (including r0, kind, dmax).
    Invalid group rank or noninteger indices raise ValueError. Indices
    must be in bounds; no distance neighbor list is constructed.
    """
    a = _indices(group_a)
    b = a if group_b is None else _indices(group_b)
    valid = a[:, None] != b[None, :]
    vectors = displacement(positions[a, None, :], positions[b][None, :, :], cell)
    # A finite safe vector prevents the undefined norm gradient on self pairs.
    safe = jnp.where(valid[..., None], vectors, jnp.array([1.0, 0.0, 0.0]))
    values = switching(jnp.linalg.norm(safe, axis=-1), **switch)
    return jnp.where(valid, values, 0.0)


def _harmonic_norm_squared(unit, weights, l, denominator):
    """Addition-theorem norm without the neighbor-by-neighbor angle matrix.

    R_lm=sqrt((l-m)!/(l+m)!)*P_l^m(z)*exp(i*m*phi). Cartesian
    recurrences avoid a polar angle or division by sin(theta) at the poles.
    Q_l^2=|<R_l0>|^2+2*sum_{m=1}^l |<R_lm>|^2.
    """
    xy = unit[..., 0] + 1j * unit[..., 1]
    z = unit[..., 2]
    diagonal = jnp.ones_like(xy)
    total = jnp.zeros_like(denominator)
    for m in range(l + 1):
        previous = jnp.zeros_like(diagonal)
        polynomial = diagonal
        for degree in range(m + 1, l + 1):
            scale = degree * degree - m * m
            previous, polynomial = (
                polynomial,
                (
                    (2 * degree - 1) / sqrt(scale) * z * polynomial
                    - sqrt(((degree - 1) ** 2 - m * m) / scale) * previous
                ),
            )
        moment = jnp.sum(weights * polynomial, axis=-1) / denominator
        total = total + (1 if m == 0 else 2) * (
            jnp.real(moment) ** 2 + jnp.imag(moment) ** 2
        )
        diagonal = -sqrt((2 * m + 1) / (2 * m + 2)) * xy * diagonal
    return total


def steinhardt(
    positions: Array,
    centers: Sequence[int] | ArrayLike,
    neighbors: Sequence[int] | ArrayLike,
    *,
    l: int = 6,
    cell: ArrayLike | PreparedCell | None = None,
    normalization: str = "conventional",
    method: str = "pairwise",
    **switch: Any,
) -> Array:
    """Return one dimensionless Steinhardt Q_l value per center.

    positions has shape (N, 3); centers/neighbors are static zero-based
    integer selections. l is a nonnegative static integer. switch options
    are passed to switching; cell optionally applies periodic wrapping.
    normalization='conventional' includes sqrt(4*pi/(2*l+1)); 'plumed'
    returns the unscaled harmonic norm. method='pairwise' uses an angular
    matrix; 'harmonics' uses linear neighbor storage with different summation
    rounding. Fix l, choices, and selections before JIT. Invalid choices or
    group types raise ValueError. Self pairs are excluded and empty weighted
    neighborhoods return zero. Coincident distinct atoms and nonempty Q_l=0
    have undefined norm derivatives; indices must be in bounds.
    """
    if not isinstance(l, int) or l < 0:
        raise ValueError("l must be a nonnegative static integer")
    if normalization not in ("conventional", "plumed"):
        raise ValueError("Unknown order normalization")
    if method not in ("pairwise", "harmonics"):
        raise ValueError("Unknown order method")
    a = _indices(centers)
    b = _indices(neighbors)
    valid = a[:, None] != b[None, :]
    vectors = displacement(positions[a, None, :], positions[b][None, :, :], cell)
    vectors = jnp.where(valid[..., None], vectors, jnp.array([1.0, 0.0, 0.0]))
    distances = jnp.linalg.norm(vectors, axis=-1)
    weights = jnp.where(valid, switching(distances, **switch), 0.0)
    u = vectors / distances[..., None]
    if method == "pairwise":
        cosine = jnp.clip(jnp.einsum("cij,ckj->cik", u, u), -1.0, 1.0)
        previous = jnp.ones_like(cosine)
        polynomial = cosine if l else previous
        for degree in range(2, l + 1):
            previous, polynomial = (
                polynomial,
                ((2 * degree - 1) * cosine * polynomial - (degree - 1) * previous)
                / degree,
            )
        norm = jnp.sum(weights, axis=-1)
        denominator = jnp.where(norm > 0, norm, 1.0)
        q2 = jnp.einsum("ci,ck,cik->c", weights, weights, polynomial) / denominator**2
    else:
        norm = jnp.sum(weights, axis=-1)
        denominator = jnp.where(norm > 0, norm, 1.0)
        q2 = _harmonic_norm_squared(u, weights, l, denominator)
    # Avoid sqrt(0)'s derivative for an explicitly empty neighborhood.
    value = jnp.where(
        norm > 0, jnp.sqrt(jnp.maximum(jnp.where(norm > 0, q2, 1.0), 0.0)), 0.0
    )
    return (
        value
        if normalization == "conventional"
        else value * jnp.sqrt((2 * l + 1) / (4 * jnp.pi))
    )


def tetrahedral_order(
    positions: Array,
    center: int,
    neighbors: Sequence[int],
    *,
    cell: ArrayLike | PreparedCell | None = None,
) -> Array:
    """Return the dimensionless angular order of exactly four neighbors.

    positions has shape (N, 3); center and neighbors are static zero-based
    indices. cell optionally applies periodic wrapping. A perfect tetrahedron
    returns one. A neighbor count other than four raises ValueError. Indices
    must be in bounds and distinct from the center; coincident atoms have
    undefined derivatives. No nearest-neighbor selection is performed.
    """
    if len(neighbors) != 4:
        raise ValueError("Exactly four neighbors are required")
    vectors = displacement(positions[center], positions[jnp.asarray(neighbors)], cell)
    unit = vectors / jnp.linalg.norm(vectors, axis=-1)[:, None]
    dot = unit @ unit.T
    i, j = jnp.triu_indices(4, 1)
    return 1 - 3 / 8 * jnp.sum((dot[i, j] + 1 / 3) ** 2)
