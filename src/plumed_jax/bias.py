"""Scalar bias energies; energy units are chosen consistently by the caller."""

from __future__ import annotations

from collections.abc import Callable
from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp


def periodic_difference(
    value: ArrayLike, reference: ArrayLike, periods: ArrayLike = 0.0
) -> Array:
    """Return value-reference, wrapped into [-period/2, period/2).

    value, reference, and periods are broadcast-compatible scalars/arrays.
    The output has their broadcast shape and the same units as value.
    Positive periods wrap components; zero denotes nonperiodic components.
    Negative periods also follow the nonperiodic branch. Periodic boundary
    ties are not differentiable.
    """
    delta = jnp.asarray(value) - jnp.asarray(reference)
    periods = jnp.asarray(periods)
    safe = jnp.where(periods > 0, periods, 1.0)
    return jnp.where(periods > 0, delta - safe * jnp.floor(delta / safe + 0.5), delta)


def harmonic_restraint(
    values: ArrayLike,
    centers: ArrayLike,
    kappa: ArrayLike,
    *,
    slopes: ArrayLike = 0.0,
    periods: ArrayLike = 0.0,
) -> Array:
    """Return sum(0.5*kappa*delta**2 + slopes*delta) as scalar energy.

    delta is periodic_difference(values, centers, periods). Inputs must
    broadcast componentwise: centers/periods use CV units, kappa uses
    energy/CV-unit squared, and slopes uses energy/CV-unit. A scalar is
    one CV; arrays represent several components. Units must be consistent;
    no parameter validation or conversion is performed.
    """
    delta = periodic_difference(values, centers, periods)
    return jnp.sum(0.5 * jnp.asarray(kappa) * delta**2 + jnp.asarray(slopes) * delta)


def wall(
    values: ArrayLike,
    centers: ArrayLike,
    kappa: ArrayLike,
    *,
    upper: bool = True,
    scale: ArrayLike = 1.0,
    exponent: ArrayLike = 2,
    offset: ArrayLike = 0.0,
) -> Array:
    """Return the scalar energy of an upper or lower power-law wall.

    For upper=True the signed distance is values-centers; otherwise it is
    centers-values. Energy is sum(kappa*max((distance+offset)/scale, 0)**exponent).
    Inputs broadcast componentwise. centers, offset, and positive scale
    use CV units; kappa uses energy units and exponent is dimensionless.
    Fix upper before JIT. This wall does not wrap periodic CVs or validate
    parameter values.
    """
    delta = (jnp.asarray(values) - jnp.asarray(centers)) * (1 if upper else -1)
    return jnp.sum(
        jnp.asarray(kappa) * jnp.maximum((delta + offset) / scale, 0.0) ** exponent
    )


def make_biased_energy(
    energy_fn: Callable, cv_fn: Callable, bias_fn: Callable
) -> Callable:
    """Build a scalar total-energy function from three callables.

    energy_fn takes (positions, *args, **kwargs); cv_fn takes positions;
    bias_fn takes CV values, or (CV values, bias_state) when state is supplied.
    The returned callable accepts (positions, *args, bias_state=None, **kwargs).
    Only the base potential receives args/kwargs. Bind atom selections and
    cell in cv_fn; for a changing cell, compose the energy explicitly.
    Both energies must use the same units. JAX differentiates their sum;
    bias state is read, never updated, by this wrapper.
    """

    def energy(positions, *args, bias_state=None, **kwargs):
        values = cv_fn(positions)
        bias = bias_fn(values) if bias_state is None else bias_fn(values, bias_state)
        return energy_fn(positions, *args, **kwargs) + bias

    return energy
