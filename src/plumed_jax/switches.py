"""Contact switches from PLUMED's published mathematical definitions."""

from __future__ import annotations

from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp
from .cvs import _rational_switch


def switching(
    distance: ArrayLike,
    *,
    kind: str = "RATIONAL",
    r0: float = 1.0,
    d0: float = 0.0,
    dmax: float | None = None,
    n: int = 6,
    m: int = 12,
    a: float = 6.0,
    b: float = 12.0,
    stretch: bool = False,
) -> Array:
    """Return dimensionless contact weights with the shape of distance.

    kind is RATIONAL, EXP, GAUSSIAN, TANH, SMAP, COSINUS, or CUBIC.
    r0 > 0 and d0/dmax use the same length unit as distance; dmax must
    exceed d0 when supplied. n/m are rational exponents; a/b are positive
    SMAP parameters. CUBIC requires dmax. Values at/above dmax are zero;
    stretch=True also shifts/scales the function to zero at dmax.
    Fix kind, scales, exponents, and stretch before JIT. Invalid choices
    raise ValueError. Finite input distances are the caller's responsibility.
    """
    if r0 <= 0 or (dmax is not None and dmax <= d0):
        raise ValueError("Invalid switching length scale")
    kind = kind.upper()

    def raw(r):
        x = jnp.maximum((r - d0) / r0, 0.0)
        if kind == "RATIONAL":
            return _rational_switch(r, r0, d0, n, m, False)
        if kind == "EXP":
            return jnp.exp(-x)
        if kind == "GAUSSIAN":
            return jnp.exp(-0.5 * x * x)
        if kind == "TANH":
            return 1 - jnp.tanh(x)
        if kind == "SMAP":
            if a <= 0 or b <= 0:
                raise ValueError("SMAP a and b must be positive")
            return (1 + (2 ** (a / b) - 1) * x**a) ** (-b / a)
        if kind == "COSINUS":
            return 0.5 * (1 + jnp.cos(jnp.pi * jnp.minimum(x, 1.0)))
        if kind == "CUBIC":
            if dmax is None:
                raise ValueError("CUBIC requires dmax")
            y = jnp.clip((r - d0) / (dmax - d0), 0.0, 1.0)
            return (y - 1) ** 2 * (1 + 2 * y)
        raise ValueError(f"Unknown switch {kind}")

    value = raw(jnp.asarray(distance))
    if dmax is not None:
        if stretch:
            end = raw(jnp.asarray(dmax))
            value = (value - end) / (1 - end)
        value = jnp.where(distance < dmax, value, 0.0)
    elif stretch:
        raise ValueError("Stretching requires an explicit cutoff")
    return value
