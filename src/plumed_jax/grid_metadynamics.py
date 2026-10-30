"""Fixed-grid metadynamics with differentiable cubic Hermite interpolation.

Energies depend only on neighboring grid vertices, not on the number of hills.
Deposition updates grid values and their first coordinate derivatives. Mixed
derivatives in the tensor interpolation use a local separable-function closure:
the derivative for an axis subset is the product of its first derivatives,
divided by the node value to the power subset_size-1. This needs no extra grid
arrays and reduces to ordinary cubic Hermite interpolation in one dimension.
"""

from __future__ import annotations

from jax import Array
from jax.typing import ArrayLike
from numpy.typing import DTypeLike
from dataclasses import dataclass
from itertools import product
from math import ceil, prod, sqrt
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .bias import periodic_difference
from .metadynamics import Metadynamics, _walker_hills
from .reweighting import metadynamics_offset


class GridState(NamedTuple):
    """Caller-owned fixed-grid bias state.

    values: grid_shape in energy units; derivatives: grid_shape+(CVs,) in
    energy per respective CV unit. count/step are scalar int32 deposit/step
    counters; overflow is a sticky scalar boolean. last_height is the last
    update's deposited energy (sum of walker heights for a batch, zero when
    no hill was deposited).
    """

    values: Array
    derivatives: Array
    count: Array
    step: Array
    overflow: Array
    last_height: Array


@dataclass(frozen=True)
class GridMetadynamics:
    """Fixed one-to-three-dimensional metadynamics grid configuration.

    sigma/height/pace/periods/bias_factor/kT/cutoff follow Metadynamics.
    grid_min/grid_max give finite increasing bounds in each CV's units;
    grid_bin gives integer interval counts >=2. Periodic bounds must span
    one period. spline_floor is a nonnegative energy threshold for suppressing
    near-zero vertex slopes, matching PLUMED interpolation. Invalid settings
    raise ValueError. Configuration is frozen; state changes separately.
    Grid shape, bounds, and algorithms must be fixed before JIT.
    """

    sigma: tuple[float, ...]
    height: float
    pace: int
    grid_min: tuple[float, ...]
    grid_max: tuple[float, ...]
    grid_bin: tuple[int, ...]
    periods: tuple[float, ...] = ()
    bias_factor: float | None = None
    kT: float | None = None
    cutoff: float | None = 6.25
    spline_floor: float = 1e-7

    def __post_init__(self):
        Metadynamics(
            self.sigma,
            self.height,
            self.pace,
            1,
            self.periods,
            self.bias_factor,
            self.kT,
            self.cutoff,
        )
        n = len(self.sigma)
        if not np.isfinite(self.spline_floor) or self.spline_floor < 0:
            raise ValueError("spline_floor must be finite and nonnegative")
        if n > 3 or any(
            len(v) != n for v in (self.grid_min, self.grid_max, self.grid_bin)
        ):
            raise ValueError(
                "Grid needs one to three dimensions and one bound/bin count per CV"
            )
        for lo, hi, bins, period in zip(
            self.grid_min, self.grid_max, self.grid_bin, self.periods or (0.0,) * n
        ):
            if (
                not np.isfinite(lo)
                or not np.isfinite(hi)
                or hi <= lo
                or type(bins) is not int
                or bins < 2
            ):
                raise ValueError(
                    "Grid bounds must increase; bin counts must be integers >=2"
                )
            if period > 0 and not np.isclose(hi - lo, period, rtol=1e-12, atol=1e-12):
                raise ValueError("A periodic grid must cover exactly one period")

    @property
    def grid_shape(self) -> tuple[int, ...]:
        """Number of vertices per CV axis: bins for periodic axes, bins+1 otherwise."""
        return tuple(
            b if p > 0 else b + 1
            for b, p in zip(self.grid_bin, self.periods or (0.0,) * len(self.sigma))
        )

    @property
    def spacing(self) -> tuple[float, ...]:
        """Grid spacings, one per CV, in the corresponding CV units."""
        return tuple(
            (hi - lo) / b
            for lo, hi, b in zip(self.grid_min, self.grid_max, self.grid_bin)
        )

    def init(self, dtype: DTypeLike = jnp.float64) -> GridState:
        """Return empty GridState arrays with this configuration's fixed shape.

        values has grid_shape; derivatives has grid_shape+(number_of_CVs,).
        count/step are int32, overflow is boolean, and last_height is scalar.
        Only float32/float64 are accepted; float64 requires jax_enable_x64=True.
        Invalid dtype raises ValueError. Choose dtype/configuration before JIT.
        """
        if jnp.dtype(dtype) == jnp.float64 and not jax.config.jax_enable_x64:
            raise ValueError("Enable jax_enable_x64 before requesting float64 state")
        if jnp.dtype(dtype) not in (jnp.float32, jnp.float64):
            raise ValueError("Grid state requires float32 or float64")
        return GridState(
            jnp.zeros(self.grid_shape, dtype),
            jnp.zeros(self.grid_shape + (len(self.sigma),), dtype),
            jnp.int32(0),
            jnp.int32(0),
            jnp.array(False),
            jnp.asarray(0.0, dtype),
        )

    def _coordinate(self, values):
        values = jnp.atleast_1d(values)
        if values.shape != (len(self.sigma),):
            raise ValueError("CV shape does not match grid dimensions")
        lo, hi = (
            jnp.asarray(self.grid_min, dtype=values.dtype),
            jnp.asarray(self.grid_max, dtype=values.dtype),
        )
        periodic = jnp.asarray(self.periods or (0.0,) * len(self.sigma)) > 0
        good = jnp.all(
            jnp.isfinite(values) & (periodic | ((values >= lo) & (values <= hi)))
        )
        # Sanitize invalid coordinates before converting to integer indices.
        safe = jnp.where(jnp.isfinite(values), values, lo)
        # Strict comparisons preserve the one-sided derivative at endpoints;
        # jnp.clip would halve it exactly at a bound.
        bounded = jnp.where(safe < lo, lo, jnp.where(safe > hi, hi, safe))
        safe = jnp.where(periodic, lo + jnp.mod(safe - lo, hi - lo), bounded)
        return safe, good

    def energy(self, values: ArrayLike, state: GridState) -> Array:
        """Return scalar bias energy interpolated at one CV vector.

        values has shape (number_of_CVs,), or is scalar for one CV, and is
        converted to the state's floating dtype. Each component uses its
        sigma/grid units. The supplied state must match this configuration.
        CV shape mismatches raise ValueError. Nonfinite coordinates or points
        outside nonperiodic bounds return NaN; endpoints are included and
        periodic axes wrap. Forces are derivatives of this interpolated energy;
        interpolation approximates the direct hill sum. No hill is deposited.
        """
        values, good = self._coordinate(jnp.asarray(values, dtype=state.values.dtype))
        dimension = len(self.sigma)
        t = (values - jnp.asarray(self.grid_min, dtype=values.dtype)) / jnp.asarray(
            self.spacing, dtype=values.dtype
        )
        cell = jnp.minimum(
            jnp.floor(t).astype(jnp.int32), jnp.asarray(self.grid_bin) - 1
        )
        fraction = t - cell
        corners = jnp.asarray(tuple(product((0, 1), repeat=dimension)), dtype=jnp.int32)
        index = cell[None, :] + corners
        periodic = jnp.asarray(self.periods or (0.0,) * dimension) > 0
        index = jnp.where(periodic, index % jnp.asarray(self.grid_shape), index)
        node_values = state.values[tuple(index[:, d] for d in range(dimension))]
        node_derivatives = state.derivatives[
            tuple(index[:, d] for d in range(dimension))
        ]
        # PLUMED's spline suppresses slopes at |node value| < 1e-7. Verified
        # with independently specified EXTERNAL grids, including threshold ties.
        node_derivatives = jnp.where(
            (jnp.abs(node_values) >= self.spline_floor)[:, None], node_derivatives, 0.0
        )
        t2, t3 = fraction**2, fraction**3
        p0, p1 = 1 - 3 * t2 + 2 * t3, 3 * t2 - 2 * t3
        q0, q1 = fraction - 2 * t2 + t3, -t2 + t3
        value_basis = jnp.where(corners == 0, p0, p1)
        derivative_basis = jnp.where(corners == 0, q0, q1)
        result = jnp.sum(node_values * jnp.prod(value_basis, axis=1))
        for d in range(dimension):
            others = tuple(k for k in range(dimension) if k != d)
            weight = jnp.prod(
                value_basis[:, jnp.asarray(others, dtype=jnp.int32)], axis=1
            )
            result = result + jnp.sum(
                node_derivatives[:, d]
                * self.spacing[d]
                * derivative_basis[:, d]
                * weight
            )
        # Higher mixed Hermite terms inferred from local factorization. At empty
        # grid vertices both value and derivatives are zero; avoid 0/0 there.
        denominator = jnp.where(
            (jnp.abs(node_values) >= self.spline_floor) & (node_values != 0),
            node_values,
            1.0,
        )
        for subset in product((0, 1), repeat=dimension):
            axes = tuple(d for d, enabled in enumerate(subset) if enabled)
            if len(axes) < 2:
                continue
            term = jnp.ones_like(node_values)
            for d in range(dimension):
                term = term * (
                    node_derivatives[:, d] * self.spacing[d] * derivative_basis[:, d]
                    if subset[d]
                    else value_basis[:, d]
                )
            result = result + jnp.sum(term / denominator ** (len(axes) - 1))
        return jnp.where(good, result, jnp.asarray(jnp.nan, dtype=result.dtype))

    def update(
        self,
        values: ArrayLike,
        state: GridState,
        *,
        step: ArrayLike | None = None,
        bias_energy: ArrayLike | None = None,
        dense: bool = False,
    ) -> GridState:
        """Return the next GridState after a completed force evaluation.

        values has shape (number_of_CVs,), or is scalar for one CV. step is
        an increasing integer scalar; None advances state.step by one. Deposits
        occur at multiples of pace. bias_energy optionally reuses this bias's
        scalar energy at these values and the pre-deposition state.
        Fix dense before JIT: True keeps full-grid deposition; False permits
        finite-support patches when appropriate. Invalid coordinates, step
        order, heights, or counter exhaustion set sticky overflow; check it
        on the host. Invalid CV/energy shapes raise ValueError. last_height is
        zero without deposition. Deposited coordinates/heights are detached.
        """
        coordinates, inside = self._coordinate(
            jnp.asarray(values, dtype=state.values.dtype)
        )
        new_step = (
            state.step + 1 if step is None else jnp.asarray(step, dtype=jnp.int32)
        )
        ordered = new_step > state.step
        due = new_step % self.pace == 0
        height = jnp.asarray(self.height, dtype=state.values.dtype)
        if self.bias_factor is not None:
            energy = (
                self.energy(coordinates, state)
                if bias_energy is None
                else jnp.asarray(bias_energy)
            )
            if energy.shape != ():
                raise ValueError("bias_energy must be a scalar")
            height = height * jnp.exp(-energy / (self.kT * (self.bias_factor - 1)))
        valid = (
            inside
            & ordered
            & jnp.isfinite(height)
            & (state.count < jnp.iinfo(jnp.int32).max)
        )

        state = state._replace(last_height=jnp.asarray(0.0, state.values.dtype))
        result = jax.lax.cond(
            due & valid & ~state.overflow,
            lambda s: self._deposit(coordinates, height, s, dense=dense,
                                    cached_bias=bias_energy is not None),
            lambda s: s, state
        )
        return result._replace(step=new_step, overflow=state.overflow | ~valid)

    def update_walkers(
        self,
        values: ArrayLike,
        state: GridState,
        *,
        step: ArrayLike | None = None,
        bias_energies: ArrayLike | None = None,
        dense: bool = False,
    ) -> GridState:
        """Synchronously deposit all walkers into one caller-owned bias grid.

        values has fixed shape (walkers, number_of_CVs); bias_energies optionally
        supplies (walkers,) energies from the same pre-deposition grid.
        All heights use that old grid, then hills accumulate in walker order.
        step advances once; count counts hills, last_height sums applied heights.
        height is per walker without automatic division. Invalid batches,
        step order or exhausted int32 counters set sticky overflow and deposit
        nothing. Fix walker count and dense before JIT. Coordinates/heights
        are detached. One walker uses the existing update path unchanged.
        """
        values, heights, new_step = _walker_hills(
            self, values, state, step, bias_energies, state.values.dtype
        )
        walkers = values.shape[0]
        if walkers == 1:
            return self.update(values[0], state, step=step, dense=dense,
                bias_energy=None if bias_energies is None else jnp.asarray(bias_energies)[0])
        coordinates, inside = jax.vmap(self._coordinate)(
            jnp.asarray(values, dtype=state.values.dtype)
        )
        ordered = new_step > state.step
        due = new_step % self.pace == 0
        valid = (jnp.all(inside) & ordered & jnp.all(jnp.isfinite(heights))
                 & (state.count <= jnp.iinfo(jnp.int32).max - walkers))

        def deposit(s):
            def add(s, hill):
                return self._deposit(*hill, s, dense=dense, cached_bias=True), None
            s, _ = jax.lax.scan(add, s, (coordinates, heights))
            return s._replace(last_height=jax.lax.stop_gradient(jnp.sum(heights)))

        state = state._replace(last_height=jnp.asarray(0., state.values.dtype))
        updated = jax.lax.cond(due & valid & ~state.overflow, deposit, lambda s: s, state)
        return updated._replace(step=new_step, overflow=state.overflow | ~valid)

    def _deposit(self, coordinates, height, state, *, dense, cached_bias):
        """Add one tempered hill; callers validate inputs and advance the step."""
        # A conservative rectangular stencil encloses the finite Gaussian's
        # entire support. Shapes stay fixed. Fall back to the dense calculation
        # when the stencil is not substantially smaller, or the tail is infinite.
        half = (
            None
            if self.cutoff is None
            else tuple(
                ceil(min(size, sqrt(2 * self.cutoff) * sigma / dx))
                for size, sigma, dx in zip(self.grid_shape, self.sigma, self.spacing)
            )
        )
        span = (
            self.grid_shape
            if half is None
            else tuple(
                min(size, 2 * radius + 2) for size, radius in zip(self.grid_shape, half)
            )
        )
        # The uncached WT path can force costly full-state copies when combined
        # with reverse-mode forces. Keep the dense path in that case. Replay
        # and bias-and-update APIs supply the already computed scalar bias.
        local = (
            not dense
            and half is not None
            and prod(span) * 2 < prod(self.grid_shape)
            and (self.bias_factor is None or cached_bias)
        )
        periodic = self.periods or (0.0,) * len(self.sigma)

        if local:
            cell = jnp.floor(
                (coordinates - jnp.asarray(self.grid_min, dtype=coordinates.dtype))
                / jnp.asarray(self.spacing, dtype=coordinates.dtype)
            ).astype(jnp.int32)
            starts = []
            indices = []
            for d, (size, width, radius, period) in enumerate(
                zip(self.grid_shape, span, half, periodic)
            ):
                start = cell[d] - radius
                if width == size:
                    start = jnp.int32(0)
                elif period == 0:
                    start = jnp.clip(start, 0, size - width)
                index = jnp.arange(width, dtype=jnp.int32) + start
                if period > 0:
                    index = index % size
                starts.append(start)
                indices.append(index)
            axes = [
                jnp.asarray(lo, dtype=state.values.dtype)
                + index.astype(state.values.dtype) * dx
                for lo, index, dx in zip(self.grid_min, indices, self.spacing)
            ]
        else:
            axes = [
                jnp.asarray(lo, dtype=state.values.dtype)
                + jnp.arange(size, dtype=state.values.dtype) * dx
                for lo, size, dx in zip(
                    self.grid_min, self.grid_shape, self.spacing
                )
            ]
        nodes = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)
        center = jax.lax.stop_gradient(coordinates)
        applied = jax.lax.stop_gradient(height)
        delta = periodic_difference(
            nodes, center, jnp.asarray(self.periods or 0.0, dtype=nodes.dtype)
        )
        sigma = jnp.asarray(self.sigma, dtype=nodes.dtype)
        exponent = 0.5 * jnp.sum((delta / sigma) ** 2, axis=-1)
        gaussian = jnp.exp(-exponent)
        if self.cutoff is None:
            kernel, scale = gaussian, 1.0
            active = jnp.ones_like(exponent, dtype=bool)
        else:
            tail = jnp.exp(jnp.asarray(-self.cutoff, dtype=nodes.dtype))
            scale = 1.0 - tail
            active = exponent < self.cutoff
            kernel = (gaussian - tail) / scale
        increment = jnp.where(active, applied * kernel, 0.0)
        derivative = jnp.where(
            active[..., None],
            -applied * gaussian[..., None] / scale * delta / sigma**2,
            0.0,
        )
        if not local:
            values, derivatives = state.values + increment, state.derivatives + derivative
        elif not any(periodic):
            # Contiguous nonperiodic patches permit buffer reuse, avoiding
            # full-grid elementwise operations and scatter index arrays.
            values = jax.lax.dynamic_update_slice(
                state.values,
                jax.lax.dynamic_slice(state.values, tuple(starts), span) + increment,
                tuple(starts),
            )
            derivatives = jax.lax.dynamic_update_slice(
                state.derivatives,
                jax.lax.dynamic_slice(
                    state.derivatives,
                    tuple(starts) + (jnp.int32(0),),
                    span + (len(self.sigma),),
                )
                + derivative,
                tuple(starts) + (jnp.int32(0),),
            )
        else:
            # Modular axes select each grid vertex at most once, including
            # seam-crossing stencils. Unique indices avoid atomic reduction.
            mesh = jnp.meshgrid(*indices, indexing="ij")
            values = state.values.at[tuple(mesh)].add(
                increment, unique_indices=True, mode="promise_in_bounds"
            )
            derivatives = state.derivatives.at[tuple(mesh)].add(
                derivative, unique_indices=True, mode="promise_in_bounds"
            )
        return state._replace(
            values=values,
            derivatives=derivatives,
            count=state.count + 1,
            last_height=applied,
        )

    def replay(
        self,
        values: ArrayLike,
        state: GridState | None = None,
        *,
        dense: bool = False,
        reuse_bias: bool = True,
    ) -> tuple[GridState, Array]:
        """Return (final_state, pre_deposition_energies) for a CV trajectory.

        values has shape (steps, number_of_CVs); energies has shape (steps,).
        state=None initializes using the trajectory dtype; otherwise continues
        that state. Fix dense and reuse_bias before JIT. True reuse_bias supplies
        each evaluated bias to the update; False keeps uncached deposition.
        Check final_state.overflow outside the compiled loop.
        """
        state = self.init(jnp.asarray(values).dtype) if state is None else state

        def advance(s, v):
            energy = self.energy(v, s)
            return self.update(
                v, s, bias_energy=energy if reuse_bias else None, dense=dense
            ), energy

        return jax.lax.scan(advance, state, values)

    def free_energy(self, values: ArrayLike, state: GridState) -> Array:
        """Return the scalar asymptotic negative-bias free-energy estimate.

        values/state follow energy. Ordinary sampling uses -bias; well-tempered
        sampling multiplies by bias_factor/(bias_factor-1). Output uses energy
        units with an arbitrary additive constant. Grid interpolation error and
        finite-history sampling error remain; convergence is not guaranteed.
        """
        factor = (
            1.0
            if self.bias_factor is None
            else self.bias_factor / (self.bias_factor - 1)
        )
        return -factor * self.energy(values, state)

    def offset(self, state: GridState) -> Array:
        """Return scalar c(t) from the supplied grid/history in energy units.

        kT must be supplied even for ordinary metadynamics (ValueError otherwise).
        The flattened bias grid uses equal-point quadrature. Use this offset
        with the instantaneous bias from the same history for reweighting.
        """
        if self.kT is None:
            raise ValueError("Supply kT to compute c(t)")
        return metadynamics_offset(
            state.values.reshape(-1), kT=self.kT, bias_factor=self.bias_factor
        )
