"""Geometry shared by CVs.

Cell vectors are rows of a (3, 3) array. A QR sphere search finds the nearest lattice image without a fixed image shell.
"""

from __future__ import annotations

from jax import Array
from jax.typing import ArrayLike

import jax
import jax.numpy as jnp
from typing import NamedTuple


class PreparedCell(NamedTuple):
    """Original cell plus reusable discrete image-search factors.

    cell is the live (3, 3) row-vector array; diagonal is the (3,) row-norm
    squared vector; orthogonal is a scalar boolean. inverse and q/r are
    (3, 3) inverse/QR factors of the detached search cell; orthogonal cells
    use unused identity placeholders. Construct with prepare_cell and
    rebuild when the cell changes. This tuple contains arrays, not a cache.
    """

    cell: Array
    diagonal: Array
    orthogonal: Array
    inverse: Array
    q: Array
    r: Array


def _positive_diagonal_qr(matrix):
    """Fix QR sign ambiguity so lattice-search bounds use positive divisors."""
    q, r = jnp.linalg.qr(matrix)
    signs = jnp.where(jnp.diag(r) < 0, -1.0, 1.0)
    return q * signs[None, :], r * signs[:, None]


def prepare_cell(cell: ArrayLike) -> PreparedCell:
    """Prepare reusable closest-image search factors for one cell evaluation.

    cell has shape (3, 3) with lattice vectors as rows, in coordinate length
    units. Returns PreparedCell, accepted wherever the CV API accepts cell.
    A shape mismatch raises ValueError; the cell must be finite and nonsingular.
    Rebuild whenever the cell changes. Prepare inside the differentiated
    energy for cell gradients: the original cell retains derivatives while
    search factors do not. Orthogonal cells avoid inversion and QR.
    """
    cell = jnp.asarray(cell)
    if cell.shape != (3, 3):
        raise ValueError("Cell must have shape (3,3)")
    search_cell = jax.lax.stop_gradient(cell)
    gram = search_cell @ search_cell.T
    diagonal = jnp.diag(gram)
    orthogonal = jnp.all(gram - jnp.diag(diagonal) == 0)

    def factor_cell(matrix):
        inverse = jnp.linalg.inv(matrix)
        q, r = _positive_diagonal_qr(matrix.T)
        return inverse, q, r

    # Orthogonal displacements never use these three placeholder factors.
    inverse, q, r = jax.lax.cond(
        orthogonal,
        lambda matrix: (jnp.eye(3, dtype=matrix.dtype),) * 3,
        factor_cell,
        search_cell,
    )
    return PreparedCell(cell, diagonal, orthogonal, inverse, q, r)


def _closest_shift(delta, cell, prepared=None):
    """Exact 3D closest-lattice-vector search by QR sphere enumeration."""
    # Search is discrete; gradients follow the selected image, not the search.
    delta = jax.lax.stop_gradient(delta)
    cell = jax.lax.stop_gradient(cell)
    inverse = jnp.linalg.inv(cell) if prepared is None else prepared.inverse
    fractional = delta @ inverse
    base = jnp.round(fractional)
    residual = delta - base @ cell
    if prepared is None:
        q, r = _positive_diagonal_qr(cell.T)
    else:
        q, r = prepared.q, prepared.r
    y = residual @ q
    initial = jnp.dot(y, y)
    radius = jnp.sqrt(initial)
    integer = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32
    low = jnp.ceil((y[2] - radius) / r[2, 2]).astype(integer)
    high = jnp.floor((y[2] + radius) / r[2, 2]).astype(integer)

    def outer(s):
        n2, best, shift = s
        e2 = (y[2] - r[2, 2] * n2) ** 2
        radius = jnp.sqrt(jnp.maximum(best - e2, 0.0))
        center1 = y[1] - r[1, 2] * n2
        low1 = jnp.ceil((center1 - radius) / r[1, 1]).astype(integer)
        high1 = jnp.floor((center1 + radius) / r[1, 1]).astype(integer)

        def inner(t):
            n1, cost, n = t
            n0 = jnp.round((y[0] - r[0, 1] * n1 - r[0, 2] * n2) / r[0, 0])
            candidate = jnp.array([n0, n1, n2], dtype=delta.dtype)
            error = y - r @ candidate
            new_cost = jnp.dot(error, error)
            improve = new_cost < cost
            return (
                n1 + 1,
                jnp.where(improve, new_cost, cost),
                jnp.where(improve, candidate, n),
            )

        _, best, shift = jax.lax.while_loop(
            lambda t: t[0] <= high1, inner, (low1, best, shift)
        )
        return n2 + 1, best, shift

    _, _, offset = jax.lax.while_loop(
        lambda s: s[0] <= high, outer, (low, initial, jnp.zeros(3, dtype=delta.dtype))
    )
    return jax.lax.stop_gradient(base + offset)


def displacement(
    a: ArrayLike, b: ArrayLike, cell: ArrayLike | PreparedCell | None = None
) -> Array:
    """Return the closest-image vector b-a, with shape (..., 3).

    a and b must broadcast to (..., 3) in the same length unit. cell is
    None (no wrapping), a nonsingular row-vector (3, 3) cell, or PreparedCell.
    With a cell, invalid cell/vector shapes raise ValueError. Piecewise
    coordinate and cell gradients follow the selected image; image ties
    are not smooth. General cells use an exact variable-time lattice search
    under JIT. Ill-conditioned cells can be slow or exceed representable
    integer/floating-point ranges; no conditioning check is performed.
    """
    delta = jnp.asarray(b) - jnp.asarray(a)
    if cell is None:
        return delta
    prepared = cell if isinstance(cell, PreparedCell) else None
    cell = jnp.asarray(cell.cell if prepared is not None else cell)
    if cell.shape != (3, 3):
        raise ValueError("Cell must have shape (3,3)")
    if delta.shape[-1] != 3:
        raise ValueError("Coordinates must have three components")
    flat = delta.reshape((-1, 3))
    discrete_cell = jax.lax.stop_gradient(cell)
    discrete_flat = jax.lax.stop_gradient(flat)
    if prepared is None:
        gram = discrete_cell @ discrete_cell.T
        diagonal = jnp.diag(gram)
        orthogonal = jnp.all(gram - jnp.diag(diagonal) == 0)
    else:
        diagonal, orthogonal = prepared.diagonal, prepared.orthogonal
    # Orthogonal vectors have an exact componentwise closest image. Avoid
    # GPU QR/variable-loop overhead for the usual orthorhombic MD boxes.
    shifts = jax.lax.cond(
        orthogonal,
        lambda d: jnp.round((d @ discrete_cell.T) / diagonal),
        lambda d: jax.vmap(lambda v: _closest_shift(v, discrete_cell, prepared))(d),
        discrete_flat,
    ).reshape(delta.shape)
    shifts = jax.lax.stop_gradient(shifts)
    return delta - shifts @ cell


def unwrap_ordered(
    points: ArrayLike, cell: ArrayLike | PreparedCell | None = None
) -> Array:
    """Rebuild an ordered (M, 3) molecule from successive closest-image links.

    Returns an (M, 3) array in the input length unit, anchored at the first
    point. cell=None returns the points unchanged; otherwise cell follows
    displacement's conventions. Point order must connect the intended molecule.
    """
    points = jnp.asarray(points)
    if cell is None:
        return points
    links = displacement(points[:-1], points[1:], cell)
    return jnp.concatenate((points[:1], points[:1] + jnp.cumsum(links, axis=0)), axis=0)
