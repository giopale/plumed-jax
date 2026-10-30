"""Fixed-capacity Gaussian metadynamics with explicit, restartable JAX state.

Bias evaluation precedes deposition at a completed MD step. Step zero does
not deposit. Overflow is sticky and must be checked by the host; histories
are never silently overwritten. Heights store the actual applied bias, not
PLUMED's rescaled well-tempered HILLS file height.
"""

from __future__ import annotations

from jax import Array
from jax.typing import ArrayLike
from numpy.typing import DTypeLike
from dataclasses import dataclass
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from .bias import periodic_difference


def _walker_hills(config, values, state, step, bias_energies, dtype):
    """Validate batch shapes and temper all hills against the same old state."""
    values = jnp.asarray(values)
    if values.ndim != 2 or values.shape[1] != len(config.sigma) or not values.shape[0]:
        raise ValueError("Walker CVs must have shape (walkers, number_of_CVs)")
    new_step = state.step + 1 if step is None else jnp.asarray(step, dtype=jnp.int32)
    if new_step.shape != ():
        raise ValueError("step must be a scalar")
    if bias_energies is not None:
        bias_energies = jnp.asarray(bias_energies)
        if bias_energies.shape != (values.shape[0],):
            raise ValueError("bias_energies must have one scalar per walker")
    heights = jnp.full((values.shape[0],), config.height, dtype=dtype)
    if config.bias_factor is not None:
        if bias_energies is None:
            bias_energies = jax.vmap(config.energy, in_axes=(0, None))(values, state)
        bias_energies = jnp.asarray(bias_energies, dtype=dtype)
        heights = heights * jnp.exp(-bias_energies / (config.kT * (config.bias_factor - 1)))
    return values, heights, new_step


class HillState(NamedTuple):
    """Caller-owned fixed-capacity hill history.

    centers: (capacity, CVs) in CV units; heights: (capacity,) in energy
    units. count is the active hill count, step the last completed step
    (both scalar int32); overflow is a sticky scalar boolean failure flag.
    """

    centers: Array
    heights: Array
    count: Array
    step: Array
    overflow: Array


@dataclass(frozen=True)
class Metadynamics:
    """Fixed configuration for direct Gaussian metadynamics.

    sigma gives positive widths in each CV's units; height is deposited
    energy; pace is the positive step interval; capacity is allocated hills.
    periods is empty for all nonperiodic CVs or gives a nonnegative period
    per CV. bias_factor=None selects ordinary sampling; otherwise it must
    exceed one and kT must be positive in energy units. cutoff is the
    positive Gaussian exponent cutoff (6.25), or None for infinite tails.
    Invalid configuration raises ValueError. Configuration is frozen; keep
    changing HillState separately. Fix this configuration before JIT.
    """

    sigma: tuple[float, ...]
    height: float
    pace: int
    capacity: int
    periods: tuple[float, ...] = ()
    bias_factor: float | None = None
    kT: float | None = None
    cutoff: float | None = 6.25

    def __post_init__(self):
        if (
            not self.sigma
            or any(not np.isfinite(s) or s <= 0 for s in self.sigma)
            or not np.isfinite(self.height)
            or self.height < 0
            or self.pace < 1
            or self.capacity < 1
            or not isinstance(self.pace, int)
            or not isinstance(self.capacity, int)
        ):
            raise ValueError("Invalid hill width/height/pace/capacity")
        if self.periods and (
            len(self.periods) != len(self.sigma)
            or any(not np.isfinite(p) or p < 0 for p in self.periods)
        ):
            raise ValueError("One nonnegative period per CV is required")
        if self.bias_factor is not None and (
            self.bias_factor <= 1
            or not np.isfinite(self.bias_factor)
            or self.kT is None
            or not np.isfinite(self.kT)
            or self.kT <= 0
        ):
            raise ValueError(
                "Well-tempered metadynamics requires bias_factor>1 and kT>0"
            )
        if self.cutoff is not None and (
            not np.isfinite(self.cutoff) or self.cutoff <= 0
        ):
            raise ValueError("Invalid Gaussian cutoff")

    def init(self, dtype: DTypeLike = jnp.float64) -> HillState:
        """Return an empty HillState with fixed-capacity arrays.

        centers has shape (capacity, number_of_CVs); heights has shape (capacity,).
        Floating arrays use dtype; counters are int32 and overflow is boolean.
        Float64 requires jax_enable_x64=True (ValueError otherwise).
        Choose configuration and dtype before JIT.
        """
        if jnp.dtype(dtype) == jnp.float64 and not jax.config.jax_enable_x64:
            raise ValueError("Enable jax_enable_x64 before requesting float64 state")
        return HillState(
            jnp.zeros((self.capacity, len(self.sigma)), dtype),
            jnp.zeros(self.capacity, dtype),
            jnp.int32(0),
            jnp.int32(0),
            jnp.array(False),
        )

    def energy(self, values: ArrayLike, state: HillState) -> Array:
        """Return scalar bias energy at one CV vector using the supplied history.

        values has shape (number_of_CVs,), or is scalar for a single CV.
        state must come from this configuration. Output uses height/kT energy
        units; each CV component uses the unit of its sigma and period.
        A CV shape mismatch raises ValueError. Every allocated hill slot is
        visited, including unused slots; this call does not deposit a hill.
        """
        values = jnp.atleast_1d(values)
        if values.shape != (len(self.sigma),):
            raise ValueError("CV shape does not match sigma")
        delta = periodic_difference(values, state.centers, self.periods or 0.0)
        exponent = 0.5 * jnp.sum((delta / jnp.asarray(self.sigma)) ** 2, axis=-1)
        active = jnp.arange(self.capacity) < state.count
        kernel = jnp.exp(-exponent)
        if self.cutoff is not None:
            active = active & (exponent < self.cutoff)
            # PLUMED's default stretched Gaussian is height at its center and
            # zero at the finite cutoff. None selects an untruncated Gaussian.
            tail = jnp.exp(-self.cutoff)
            kernel = (kernel - tail) / (1 - tail)
        return jnp.sum(jnp.where(active, state.heights * kernel, 0.0))

    def update(
        self,
        values: ArrayLike,
        state: HillState,
        *,
        step: ArrayLike | None = None,
        bias_energy: ArrayLike | None = None,
    ) -> HillState:
        """Return the next HillState after a completed force evaluation.

        values has shape (number_of_CVs,), or is scalar for one CV. step is
        an increasing integer scalar; None advances state.step by one. Deposits
        occur at multiples of pace. bias_energy may reuse this bias's scalar
        energy at these values and the pre-deposition state, never total energy.
        Invalid order, invalid deposits, or capacity exhaustion set sticky
        overflow; inspect it on the host. CV/scalar shape mismatches raise
        ValueError. Deposited coordinates/heights are detached from gradients;
        state remains caller-owned.
        """
        new_step = (
            state.step + 1 if step is None else jnp.asarray(step, dtype=jnp.int32)
        )
        ordered = new_step > state.step
        due = ordered & (new_step % self.pace == 0)
        values = jnp.atleast_1d(values)
        if values.shape != (len(self.sigma),):
            raise ValueError("CV shape does not match sigma")
        height = jnp.asarray(self.height, dtype=state.heights.dtype)
        if self.bias_factor is not None:
            energy = (
                self.energy(values, state)
                if bias_energy is None
                else jnp.asarray(bias_energy)
            )
            if energy.shape != ():
                raise ValueError("bias_energy must be a scalar")
            height = height * jnp.exp(-energy / (self.kT * (self.bias_factor - 1)))
        good = jnp.all(jnp.isfinite(values)) & jnp.isfinite(height)
        overflow = (
            state.overflow | ~ordered | (due & ((state.count >= self.capacity) | ~good))
        )

        def deposit(s):
            return s._replace(
                centers=s.centers.at[s.count].set(jax.lax.stop_gradient(values)),
                heights=s.heights.at[s.count].set(jax.lax.stop_gradient(height)),
                count=s.count + 1,
            )

        state = jax.lax.cond(
            due & good & (state.count < self.capacity), deposit, lambda s: s, state
        )
        return state._replace(step=new_step, overflow=overflow)

    def update_walkers(
        self,
        values: ArrayLike,
        state: HillState,
        *,
        step: ArrayLike | None = None,
        bias_energies: ArrayLike | None = None,
    ) -> HillState:
        """Synchronously deposit one hill per walker into a shared history.

        values has fixed shape (walkers, number_of_CVs). All well-tempered
        heights use the same pre-deposition state; optional bias_energies has
        shape (walkers,) and must come from that state and this bias only.
        step advances once, count advances by walkers at multiples of pace.
        height is per walker, with no automatic division by walker count.
        Invalid due batches or insufficient capacity deposit nothing and set
        sticky overflow. Coordinates/heights are detached. For one walker,
        the existing update path is used unchanged. No replica is exchanged.
        """
        values, heights, new_step = _walker_hills(
            self, values, state, step, bias_energies, state.heights.dtype
        )
        walkers = values.shape[0]
        if walkers == 1:
            return self.update(values[0], state, step=step,
                bias_energy=None if bias_energies is None else jnp.asarray(bias_energies)[0])
        ordered = new_step > state.step
        due = new_step % self.pace == 0
        good = jnp.all(jnp.isfinite(values)) & jnp.all(jnp.isfinite(heights))
        fits = state.count <= self.capacity - walkers

        def deposit(s):
            indices = s.count + jnp.arange(walkers, dtype=jnp.int32)
            return s._replace(
                centers=s.centers.at[indices].set(jax.lax.stop_gradient(values)),
                heights=s.heights.at[indices].set(jax.lax.stop_gradient(heights)),
                count=s.count + walkers,
            )

        updated = jax.lax.cond(
            ordered & due & good & fits & ~state.overflow, deposit, lambda s: s, state
        )
        return updated._replace(step=new_step,
            overflow=state.overflow | ~ordered | (due & (~good | ~fits)))

    def replay(
        self,
        values: ArrayLike,
        state: HillState | None = None,
        *,
        reuse_bias: bool = True,
    ) -> tuple[HillState, Array]:
        """Return (final_state, pre_deposition_energies) for a CV trajectory.

        values has shape (steps, number_of_CVs); energies has shape (steps,).
        state=None initializes history using the trajectory dtype; otherwise
        continues its step counter/history. Fix reuse_bias before JIT: True
        reuses each evaluated bias in well-tempered deposition, False preserves
        the uncached calculation. Check final_state.overflow outside JIT.
        """
        state = self.init(jnp.asarray(values).dtype) if state is None else state

        def step(s, v):
            energy = self.energy(v, s)
            return self.update(v, s, bias_energy=energy if reuse_bias else None), energy

        return jax.lax.scan(step, state, values)

    def free_energy(self, values: ArrayLike, state: HillState) -> Array:
        """Return the scalar asymptotic negative-bias free-energy estimate.

        values/state follow energy. Ordinary sampling uses -bias; well-tempered
        sampling multiplies it by bias_factor/(bias_factor-1). Output uses
        energy units and has an arbitrary additive constant. A finite history
        need not represent a converged free energy.
        """
        factor = (
            1.0
            if self.bias_factor is None
            else self.bias_factor / (self.bias_factor - 1)
        )
        return -factor * self.energy(values, state)
