"""First-stage scalar CVs.

Atom indices are zero-based. `cell=None` disables PBC. For coordination,
`r0` and `d0` use the same length unit as positions. The rational switch
supports positive integers `m > n` (PLUMED defaults to n=6, m=12); additional
switches are provided by plumed_jax.switches.
"""

from __future__ import annotations

from collections.abc import Sequence
from jax import Array
from jax.typing import ArrayLike
from .geometry import PreparedCell

import jax.numpy as jnp

from .geometry import displacement, unwrap_ordered


def distance(
    positions: Array, i: int, j: int, cell: ArrayLike | PreparedCell | None = None
) -> Array:
    """Return the closest-image distance between atoms i and j.

    positions is an (N, 3) coordinate array; i and j are static zero-based
    indices. cell is None or a (3, 3) row-vector cell (also PreparedCell).
    Returns a scalar in the coordinate length unit. Indices must be in
    bounds; this direct function does not check them. The gradient at
    coincident atoms is undefined.
    """
    return jnp.linalg.norm(displacement(positions[i], positions[j], cell))


def position(
    positions: Array, i: int, cell: ArrayLike | PreparedCell | None = None
) -> Array:
    """Return atom i's closest-image position relative to the origin.

    positions has shape (N, 3); i is a static zero-based index. cell is
    None or a (3, 3) row-vector cell (also PreparedCell). Returns a (3,)
    vector in the coordinate length unit. i must be in bounds.
    """
    return displacement(jnp.zeros(3, dtype=positions.dtype), positions[i], cell)


def angle(
    positions: Array,
    i: int,
    j: int,
    k: int,
    cell: ArrayLike | PreparedCell | None = None,
) -> Array:
    """Return angle i-j-k as a scalar in [0, pi] radians.

    positions has shape (N, 3); i, j, k are static zero-based indices.
    cell controls closest-image bond vectors, or None disables wrapping.
    Indices must be in bounds. Coincident atoms and collinear bond vectors
    have undefined derivatives.
    """
    a = displacement(positions[j], positions[i], cell)
    b = displacement(positions[j], positions[k], cell)
    return _angle_from_vectors(a, b)


def _angle_from_vectors(a, b):
    return jnp.arctan2(jnp.linalg.norm(jnp.cross(a, b)), jnp.dot(a, b))


def torsion(
    positions: Array,
    i: int,
    j: int,
    k: int,
    l: int,
    cell: ArrayLike | PreparedCell | None = None,
) -> Array:
    """Return the signed i-j-k-l dihedral in [-pi, pi] radians.

    positions has shape (N, 3); indices are static and zero-based. cell
    controls closest-image bond vectors, or None disables wrapping.
    Indices must be in bounds. Zero bonds or collinear adjacent bonds have
    undefined derivatives; use a 2*pi CV period for periodic biases.
    """
    b1 = displacement(positions[i], positions[j], cell)
    b2 = displacement(positions[j], positions[k], cell)
    b3 = displacement(positions[k], positions[l], cell)
    return _torsion_from_vectors(b1, b2, b3)


def _torsion_from_vectors(b1, b2, b3):
    n1 = jnp.cross(b1, b2)
    n2 = jnp.cross(b2, b3)
    return jnp.arctan2(
        jnp.dot(jnp.cross(n1, n2), b2 / jnp.linalg.norm(b2)), jnp.dot(n1, n2)
    )


def center_of_mass(
    positions: Array,
    indices: Sequence[int],
    masses: ArrayLike | None,
    cell: ArrayLike | PreparedCell | None = None,
) -> Array:
    """Return the mass-weighted center of an ordered atom selection.

    positions has shape (N, 3); indices is a static zero-based selection
    of length M; masses has shape (N,). Returns a (3,) position in the
    coordinate length unit. Under PBC, atoms are rebuilt in selection
    order using cell; order must connect the intended molecule. Masses
    must be nonnegative with positive selected sum; inputs are not checked.
    """
    points = unwrap_ordered(positions[jnp.asarray(indices)], cell)
    weights = jnp.asarray(masses)[jnp.asarray(indices)]
    return jnp.sum(points * weights[:, None], axis=0) / jnp.sum(weights)


def _rational_switch(r, r0, d0=0.0, n=6, m=12, stretch=True):
    if n <= 0 or m <= n:
        raise ValueError("Rational switch needs positive integers m > n")

    def unscaled(x):
        if m % n == 0:
            y = x**n
            return 1.0 / sum(y**power for power in range(m // n))
        # Cancel the common (1-x) factor. This remains finite at x=1, where
        # the rational switch has the exact limit n/m.
        return sum(x**power for power in range(n)) / sum(x**power for power in range(m))

    x = jnp.maximum((r - d0) / r0, 0.0)
    raw = unscaled(x)
    if not stretch:
        return raw
    # PLUMED 2.10.1's simplified COORDINATION syntax chooses a finite range
    # from 1e-5 and shifts/scales the rational function to reach zero there.
    dmax = d0 + r0 * 1e-5 ** (1.0 / (n - m))
    at_max = unscaled((dmax - d0) / r0)
    return jnp.where(r <= dmax, (raw - at_max) / (1.0 - at_max), 0.0)


def contact_values(
    positions: Array,
    pairs: ArrayLike,
    r0: float,
    *,
    d0: float = 0.0,
    n: int = 6,
    m: int = 12,
    stretch: bool = True,
    cell: ArrayLike | PreparedCell | None = None,
) -> Array:
    """Return one rational contact weight per explicit atom pair.

    positions has shape (N, 3); pairs has shape (P, 2) with zero-based
    indices. Returns a dimensionless (P,) array. r0 must be positive;
    r0 and d0 use coordinate length units. Static positive integer
    exponents require m > n (ValueError otherwise). stretch=True shifts
    and scales the finite cutoff to zero. Select pairs and fix exponents
    and stretch before JIT; cell optionally applies periodic wrapping.
    """
    pairs = jnp.asarray(pairs)
    vectors = displacement(positions[pairs[:, 0]], positions[pairs[:, 1]], cell)
    distances = jnp.linalg.norm(vectors, axis=-1)
    return _rational_switch(distances, r0, d0=d0, n=n, m=m, stretch=stretch)


def coordination_number(
    positions: Array,
    group_a: Sequence[int],
    group_b: Sequence[int] | None = None,
    *,
    r0: float,
    d0: float = 0.0,
    n: int = 6,
    m: int = 12,
    stretch: bool = True,
    cell: ArrayLike | PreparedCell | None = None,
) -> Array:
    """Return a scalar sum of rational contacts between static groups.

    positions has shape (N, 3); groups contain zero-based indices. With
    no group_b, each unordered pair in group_a is counted once; otherwise
    all cross-group pairs except equal indices are included. Empty pairs
    return zero. Output is dimensionless. r0 must be positive; r0/d0 use
    length units. m > n must be positive static integers. Fix groups and
    stretch before JIT; cell optionally applies periodic wrapping.
    """
    a = tuple(int(index) for index in group_a)
    if group_b is None:
        pair_list = [(a[i], a[j]) for i in range(len(a)) for j in range(i + 1, len(a))]
    else:
        b = tuple(int(index) for index in group_b)
        pair_list = [(i, j) for i in a for j in b if i != j]
    if not pair_list:
        return jnp.asarray(0.0, dtype=positions.dtype)
    pairs = jnp.asarray(pair_list)
    values = contact_values(
        positions, pairs, r0, d0=d0, n=n, m=m, stretch=stretch, cell=cell
    )
    return jnp.sum(values)


def radius_of_gyration(
    positions: Array,
    indices: Sequence[int],
    *,
    masses: ArrayLike | None = None,
    cell: ArrayLike | PreparedCell | None = None,
) -> Array:
    """Return the scalar radius of an ordered atom selection.

    positions has shape (N, 3); indices is a static zero-based selection.
    Output uses coordinate length units. masses=None gives equal weights;
    otherwise masses has shape (N,) and must be nonnegative with positive
    selected sum. The selection must be nonempty and in bounds. cell
    rebuilds points in selection order under PBC. Zero radius has an
    undefined derivative; these preconditions are not checked here.
    """
    points = unwrap_ordered(positions[jnp.asarray(indices)], cell)
    weights = (
        jnp.ones((len(indices),), dtype=points.dtype)
        if masses is None
        else jnp.asarray(masses)[jnp.asarray(indices)]
    )
    center = jnp.sum(points * weights[:, None], axis=0) / jnp.sum(weights)
    return jnp.sqrt(
        jnp.sum(weights * jnp.sum((points - center) ** 2, axis=-1)) / jnp.sum(weights)
    )


def proton_transfer(
    positions: Array,
    donor_a: int,
    proton: int,
    donor_b: int,
    cell: ArrayLike | PreparedCell | None = None,
) -> Array:
    """Return distance(donor_a, proton) minus distance(donor_b, proton).

    positions has shape (N, 3); all three indices are static and zero-based.
    Returns a scalar in coordinate length units, negative nearer donor_a
    and positive nearer donor_b. cell optionally applies periodic wrapping.
    Indices must be in bounds; coincident-atom derivatives are undefined.
    """
    return distance(positions, donor_a, proton, cell) - distance(
        positions, donor_b, proton, cell
    )
