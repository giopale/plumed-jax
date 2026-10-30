"""Time-dependent normalization for metadynamics histogram reweighting."""

from __future__ import annotations

from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np


def metadynamics_offset(
    bias_grid: ArrayLike, *, kT: float, bias_factor: float | None = None
) -> Array:
    """Return c(t), in the same energy units as bias_grid and kT.

    bias_grid is V(s,t) on a uniform, finite grid over the biased coordinate
    domain. Equal grid-point quadrature matches PLUMED CALC_RCT. For finite
    gamma=bias_factor, c=kT*log[integral exp(gamma*V/((gamma-1)*kT)) /
    integral exp(V/((gamma-1)*kT))]. None uses the ordinary gamma->infinity
    limit; the denominator then equals the finite domain's volume.

    Weights are exp((instantaneous_bias-c)/kT). Both terms must correspond
    to the same hill history. This estimator assumes slow enough bias evolution
    for local equilibration. Its ordinary limit depends on a finite domain.
    kT and bias_factor are configuration constants for a compiled call.
    """
    if not np.isfinite(kT) or kT <= 0:
        raise ValueError("kT must be a finite positive energy")
    if bias_factor is not None and (not np.isfinite(bias_factor) or bias_factor <= 1):
        raise ValueError("bias_factor must be finite and greater than one")
    bias_grid = jnp.asarray(bias_grid)
    if bias_grid.ndim != 1 or bias_grid.size < 2:
        raise ValueError("At least two uniform grid-point bias values are required")
    a = 1.0 if bias_factor is None else bias_factor / (bias_factor - 1.0)
    maximum = jnp.max(bias_grid)
    centered = (bias_grid - maximum) / kT
    return maximum + kT * (logsumexp(a * centered) - logsumexp((a - 1.0) * centered))
