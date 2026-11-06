"""OPES_METAD: on-the-fly probability enhanced sampling with explicit JAX state.

The bias is V(s) = (1-1/gamma) kT log(P(s)/Z + epsilon), where P is a
weighted kernel density estimate built from compressed, truncated Gaussian
kernels. This module follows PLUMED 2.10's OPES_METAD (Invernizzi and
Parrinello, J. Phys. Chem. Lett. 11, 2731 (2020)), including kernel
compression, recursive merging, sigma shrinking, adaptive sigma, NO_ZED,
CALC_WORK, EXTRA_BIAS weights and the synchronous WALKERS_MPI update.

As for the other biases in this package, a completed step evaluates the bias
with the old state, then deposits, then refreshes cached forces. The initial
state corresponds to PLUMED's (skipped) first step zero; the first update is
completed step one. The kernel list has a fixed capacity; an update that would
exceed it deposits nothing and sets the sticky overflow flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, sqrt
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from numpy.typing import DTypeLike

from .bias import periodic_difference

_SIGMA_FLOOR = 1e-6  # PLUMED's fallback minimum for adaptive sigma


class OPESState(NamedTuple):
    """Caller-owned OPES_METAD state.

    centers/sigmas: (capacity, CVs) compressed kernels in CV units; heights:
    (capacity,) dimensionless kernel heights. count is the number of active
    kernels, step the last completed step and counter PLUMED's sample counter
    (1 + number of deposited samples); all are scalar int32. overflow is a
    sticky scalar boolean failure flag. sum_weights/sum_weights2 accumulate
    the reweighting factors exp(V/kT) and their squares; zed is Z_n. work has
    shape (walkers,) in energy units (zero unless calc_work). The adaptive
    sigma fields hold per-walker (walkers, CVs) sigma0, running average and
    M2 of the CVs, plus the shared adaptive_counter and a per-walker flag
    recording that PLUMED's 1e-6 sigma floor has been switched on.
    """

    centers: Array
    sigmas: Array
    heights: Array
    count: Array
    step: Array
    overflow: Array
    counter: Array
    sum_weights: Array
    sum_weights2: Array
    zed: Array
    work: Array
    adaptive_counter: Array
    sigma0: Array
    av_cv: Array
    av_m2: Array
    sigma_floor: Array


def _positive(x):
    return x is not None and isfinite(x) and x > 0


@dataclass(frozen=True)
class OPESMetad:
    """Fixed OPES_METAD configuration (PLUMED keyword in parentheses).

    sigma: initial kernel widths in CV units (SIGMA), or the string
        "adaptive" for SIGMA=ADAPTIVE; then dimension gives the CV count.
    barrier: expected free energy barrier in energy units (BARRIER).
    pace: deposition interval in completed steps (PACE).
    capacity: allocated number of compressed kernels.
    kT: positive thermal energy in the same energy units (from TEMP).
    periods: empty for nonperiodic CVs, otherwise one nonnegative period per
        CV (zero for a nonperiodic component). periodic_min optionally gives
        each periodic domain's lower bound (default -period/2); it only
        affects where merged centers are wrapped.
    bias_factor: gamma (BIASFACTOR); None uses barrier/kT, and float("inf")
        selects a uniform target.
    epsilon (EPSILON), kernel_cutoff (KERNEL_CUTOFF, in units of sigma):
        None uses PLUMED's barrier-derived defaults.
    compression_threshold: merge distance in units of sigma; 0 disables
        compression (COMPRESSION_THRESHOLD).
    sigma_min: optional lower bound per CV (SIGMA_MIN).
    adaptive_sigma_stride: steps for measuring adaptive sigma; None uses
        10*pace (ADAPTIVE_SIGMA_STRIDE).
    fixed_sigma (FIXED_SIGMA), recursive_merge (not RECURSIVE_MERGE_OFF),
    no_zed (NO_ZED), calc_work (CALC_WORK): PLUMED flags.
    walkers: number of synchronous walkers sharing this bias (WALKERS_MPI
        replica count); fixed before JIT.
    exact_gradient: False (default) reproduces PLUMED's force exactly. PLUMED
        takes the derivative of each truncated kernel h(exp(-n2/2)-tail) as
        -(s-c)/sigma**2 * h(exp(-n2/2)-tail), i.e. it keeps the cutoff shift
        inside the derivative, so its force is not the exact gradient of its
        bias (the difference vanishes as cutoff -> inf). True uses the exact,
        conservative autodiff gradient. Energies are identical either way.

    Invalid configuration raises ValueError. NLIST, EXCLUDED_REGION,
    OPES_METAD_EXPLORE and STATE/KERNELS text files are not implemented;
    use plumed_jax.checkpoint for restarts.
    """

    sigma: tuple[float, ...] | str
    barrier: float
    pace: int
    capacity: int
    kT: float
    periods: tuple[float, ...] = ()
    periodic_min: tuple[float, ...] = ()
    bias_factor: float | None = None
    epsilon: float | None = None
    kernel_cutoff: float | None = None
    compression_threshold: float = 1.0
    sigma_min: tuple[float, ...] = ()
    adaptive_sigma_stride: int | None = None
    fixed_sigma: bool = False
    recursive_merge: bool = True
    no_zed: bool = False
    calc_work: bool = False
    walkers: int = 1
    dimension: int | None = None
    exact_gradient: bool = False

    def __post_init__(self):
        if isinstance(self.sigma, str):
            if self.sigma.lower() != "adaptive":
                raise ValueError('sigma must be numeric or "adaptive"')
            if type(self.dimension) is not int or self.dimension < 1:
                raise ValueError("Adaptive sigma requires a positive integer dimension")
        else:
            if not self.sigma or any(not _positive(s) for s in self.sigma):
                raise ValueError("SIGMA must be positive and finite")
            if self.dimension is not None and self.dimension != len(self.sigma):
                raise ValueError("dimension does not match sigma")
        n = self.ncv
        for name in ("pace", "capacity", "walkers"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not _positive(self.kT):
            raise ValueError("kT must be positive and finite")
        if not isfinite(self.barrier) or self.barrier < 0:
            raise ValueError("BARRIER must be finite and nonnegative")
        if self.periods and (
            len(self.periods) != n
            or any(not isfinite(p) or p < 0 for p in self.periods)
        ):
            raise ValueError("One nonnegative period per CV is required")
        if self.periodic_min and (
            len(self.periodic_min) != n
            or any(not isfinite(p) for p in self.periodic_min)
        ):
            raise ValueError("periodic_min needs one finite value per CV")
        gamma = self.gamma
        if np.isnan(gamma) or gamma <= 1:
            raise ValueError("BIASFACTOR must be greater than one (inf for uniform)")
        if self.adaptive and np.isinf(gamma):
            raise ValueError("Adaptive sigma is incompatible with BIASFACTOR=inf")
        if self.adaptive_sigma_stride is not None:
            if not self.adaptive:
                raise ValueError("ADAPTIVE_SIGMA_STRIDE requires adaptive sigma")
            if (
                type(self.adaptive_sigma_stride) is not int
                or self.adaptive_sigma_stride < self.pace
            ):
                raise ValueError("ADAPTIVE_SIGMA_STRIDE must be an integer >= PACE")
        if self.sigma_min:
            if len(self.sigma_min) != n or any(
                not isfinite(s) or s < 0 for s in self.sigma_min
            ):
                raise ValueError("SIGMA_MIN needs one nonnegative value per CV")
            if not self.adaptive and any(
                lo > s for lo, s in zip(self.sigma_min, self.sigma)
            ):
                raise ValueError("SIGMA_MIN should be smaller than SIGMA")
        if not _positive(self.eps):
            raise ValueError("EPSILON must be positive; is the BARRIER too high?")
        if not _positive(self.cutoff):
            raise ValueError("KERNEL_CUTOFF must be positive")
        t = self.compression_threshold
        if not isfinite(t) or t < 0 or (t != 0 and t * t >= self.cutoff**2):
            raise ValueError("COMPRESSION_THRESHOLD must be in [0, KERNEL_CUTOFF)")

    # Derived constants, matching the PLUMED constructor.
    @property
    def ncv(self) -> int:
        """Number of collective variables."""
        return self.dimension if self.adaptive else len(self.sigma)

    @property
    def adaptive(self) -> bool:
        """True for SIGMA=ADAPTIVE."""
        return isinstance(self.sigma, str)

    @property
    def gamma(self) -> float:
        """Bias factor; barrier/kT when not supplied."""
        return self.barrier / self.kT if self.bias_factor is None else float(self.bias_factor)

    @property
    def bias_prefactor(self) -> float:
        """1-1/gamma (one for an infinite bias factor)."""
        return 1.0 if np.isinf(self.gamma) else 1.0 - 1.0 / self.gamma

    @property
    def eps(self) -> float:
        """Probability regularization epsilon."""
        if self.epsilon is not None:
            return float(self.epsilon)
        return exp(-self.barrier / self.bias_prefactor / self.kT)

    @property
    def cutoff(self) -> float:
        """Kernel truncation distance in units of sigma."""
        if self.kernel_cutoff is not None:
            return float(self.kernel_cutoff)
        return sqrt(2.0 * self.barrier / self.bias_prefactor / self.kT)

    @property
    def stride(self) -> int:
        """Adaptive-sigma measuring stride in steps."""
        return self.adaptive_sigma_stride or 10 * self.pace

    def _periods(self, dtype):
        return jnp.asarray(self.periods or (0.0,) * self.ncv, dtype=dtype)

    def init(self, dtype: DTypeLike = jnp.float64) -> OPESState:
        """Return the initial state (PLUMED's state before the first deposition).

        Kernel arrays have fixed shapes (capacity, CVs) and (capacity,); the
        adaptive fields have shape (walkers, CVs). Float64 requires
        jax_enable_x64=True; only float32/float64 are accepted (ValueError).
        """
        if jnp.dtype(dtype) == jnp.float64 and not jax.config.jax_enable_x64:
            raise ValueError("Enable jax_enable_x64 before requesting float64 state")
        if jnp.dtype(dtype) not in (jnp.float32, jnp.float64):
            raise ValueError("OPES state requires float32 or float64")
        n, w = self.ncv, self.walkers
        if self.no_zed:
            sum_weights = 1.0
        else:
            sum_weights = self.eps**self.bias_prefactor
        sigma0 = np.zeros((w, n)) if self.adaptive else np.tile(self.sigma, (w, 1))
        return OPESState(
            jnp.zeros((self.capacity, n), dtype),
            jnp.ones((self.capacity, n), dtype),
            jnp.zeros(self.capacity, dtype),
            jnp.int32(0),
            jnp.int32(0),
            jnp.array(False),
            jnp.int32(1),
            jnp.asarray(sum_weights, dtype),
            jnp.asarray(sum_weights * sum_weights, dtype),
            jnp.asarray(1.0, dtype),
            jnp.zeros(w, dtype),
            jnp.int32(0),
            jnp.asarray(sigma0, dtype),
            jnp.zeros((w, n), dtype),
            jnp.zeros((w, n), dtype),
            jnp.zeros(w, bool),
        )

    def _kernel_sum(self, x, centers, sigmas, heights, count):
        """Unnormalized truncated-kernel sum at x, visiting every slot."""
        delta = periodic_difference(x, centers, self._periods(centers.dtype))
        norm2 = jnp.sum((delta / sigmas) ** 2, axis=-1)
        cutoff2 = self.cutoff**2
        active = (jnp.arange(centers.shape[0]) < count) & (norm2 < cutoff2)
        tail = exp(-0.5 * cutoff2)
        terms = jnp.where(active, heights * (jnp.exp(-0.5 * norm2) - tail), 0.0)
        if self.exact_gradient:
            return jnp.sum(terms)
        # PLUMED's derivative is -(s-c)/sigma**2 times the *shifted* kernel
        # value. The surrogate below has exactly that gradient with respect
        # to x and evaluates to exactly zero, so energies are unchanged.
        weights = jax.lax.stop_gradient(terms)
        surrogate = jnp.sum(weights * jnp.where(active, -0.5 * norm2, 0.0))
        return jax.lax.stop_gradient(jnp.sum(terms)) + (
            surrogate - jax.lax.stop_gradient(surrogate)
        )

    def probability(self, values: ArrayLike, state: OPESState) -> Array:
        """Return the normalized kernel density estimate P(s) (not divided by Z)."""
        values = self._values(values, state)
        total = self._kernel_sum(
            values, state.centers, state.sigmas, state.heights, state.count
        )
        return total / state.sum_weights

    def _values(self, values, state):
        values = jnp.atleast_1d(jnp.asarray(values, dtype=state.centers.dtype))
        if values.shape != (self.ncv,):
            raise ValueError("CV shape does not match the OPES dimension")
        return values

    def energy(self, values: ArrayLike, state: OPESState) -> Array:
        """Return the scalar OPES bias at one CV vector, in energy units.

        values has shape (CVs,), or is scalar for one CV, and is converted to
        the state dtype. Forces follow by differentiation. Before the first
        deposition the bias is the constant kT(1-1/gamma)log(epsilon), i.e.
        -BARRIER with default epsilon. No kernel is deposited.
        """
        prob = self.probability(values, state)
        return self.kT * self.bias_prefactor * jnp.log(prob / state.zed + self.eps)

    def free_energy(self, values: ArrayLike, state: OPESState) -> Array:
        """Return -kT log(P(s)/Z + epsilon) = -V(s)/(1-1/gamma).

        This is the regularized on-the-fly estimate; it saturates at
        approximately BARRIER in unexplored regions. For converged free
        energies reweight samples with exp(V/kT) instead.
        """
        return -self.energy(values, state) / self.bias_prefactor

    def diagnostics(self, state: OPESState) -> dict[str, Array]:
        """Return PLUMED's output components rct, zed, neff, nker (and work)."""
        out = {
            "rct": self.kT * jnp.log(state.sum_weights / state.counter),
            "zed": state.zed,
            "neff": (1 + state.sum_weights) ** 2 / (1 + state.sum_weights2),
            "nker": state.count,
        }
        if self.calc_work:
            out["work"] = state.work[0] if self.walkers == 1 else state.work
        return out

    # ------------------------------------------------------------------
    # Deposition
    def update(
        self,
        values: ArrayLike,
        state: OPESState,
        *,
        step: ArrayLike | None = None,
        bias_energy: ArrayLike | None = None,
        extra_bias: ArrayLike | None = None,
    ) -> OPESState:
        """Return the next state after a completed force evaluation (one walker).

        values has shape (CVs,), or is scalar for one CV. step is an
        increasing integer scalar; None advances state.step by one. Kernels
        are deposited at multiples of pace. bias_energy may reuse this bias's
        energy at these values and the old state (never total energy).
        extra_bias optionally adds other biases to the reweighting factor
        (PLUMED EXTRA_BIAS). With adaptive sigma, call update every step.
        Invalid order, nonfinite deposits or exhausted capacity set sticky
        overflow and deposit nothing. Requires walkers=1 (ValueError).
        """
        if self.walkers != 1:
            raise ValueError("Use update_walkers for a multiple-walker configuration")
        values = jnp.atleast_1d(jnp.asarray(values))
        as_batch = lambda x: None if x is None else jnp.reshape(jnp.asarray(x), (1,))
        if bias_energy is not None and jnp.shape(bias_energy) != ():
            raise ValueError("bias_energy must be a scalar")
        if extra_bias is not None and jnp.shape(extra_bias) != ():
            raise ValueError("extra_bias must be a scalar")
        return self.update_walkers(
            values[None, :],
            state,
            step=step,
            bias_energies=as_batch(bias_energy),
            extra_biases=as_batch(extra_bias),
        )

    def update_walkers(
        self,
        values: ArrayLike,
        state: OPESState,
        *,
        step: ArrayLike | None = None,
        bias_energies: ArrayLike | None = None,
        extra_biases: ArrayLike | None = None,
    ) -> OPESState:
        """Synchronously deposit one kernel per walker (PLUMED WALKERS_MPI).

        values has shape (walkers, CVs) with walkers fixed by the
        configuration. All reweighting factors use the same old state and
        each walker's own bias (optional bias_energies, shape (walkers,));
        the summed weights set n_eff and the shrunk sigma, then kernels are
        added and compressed in walker order, followed by one Z update,
        exactly as the MPI gather in PLUMED. extra_biases, shape (walkers,),
        follows update. step advances once and counter by walkers. Adaptive
        sigma statistics are kept per walker. Invalid batches deposit nothing
        and set sticky overflow. Coordinates/heights are detached.
        """
        dtype = state.centers.dtype
        values = jnp.asarray(values, dtype=dtype)
        w, n = self.walkers, self.ncv
        if values.shape != (w, n):
            raise ValueError("Walker CVs must have shape (walkers, number_of_CVs)")
        new_step = state.step + 1 if step is None else jnp.asarray(step, jnp.int32)
        if new_step.shape != ():
            raise ValueError("step must be a scalar")
        for name, x in (("bias_energies", bias_energies), ("extra_biases", extra_biases)):
            if x is not None and jnp.shape(x) != (w,):
                raise ValueError(f"{name} must have one scalar per walker")
        if bias_energies is None:
            bias_energies = jax.vmap(self.energy, in_axes=(0, None))(values, state)
        bias_energies = jnp.asarray(bias_energies, dtype)
        extra = jnp.zeros(w, dtype) if extra_biases is None else jnp.asarray(extra_biases, dtype)
        values = jax.lax.stop_gradient(values)
        log_weights = jax.lax.stop_gradient((bias_energies + extra) / self.kT)

        ordered = new_step > state.step
        periods = self._periods(dtype)
        measuring = jnp.array(False)
        if self.adaptive:
            # Welford's online variance with an exponentially decaying mean.
            counter = state.adaptive_counter + 1
            tau = jnp.minimum(counter, self.stride).astype(dtype)
            diff = periodic_difference(values, state.av_cv, periods)
            av_cv = state.av_cv + diff / tau
            av_m2 = state.av_m2 + diff * periodic_difference(values, av_cv, periods)
            adapted = state._replace(adaptive_counter=counter, av_cv=av_cv, av_m2=av_m2)
            state = jax.tree_util.tree_map(
                lambda new, old: jnp.where(ordered, new, old), adapted, state
            )
            measuring = (state.adaptive_counter < self.stride) & (state.counter == 1)
        due = ordered & (new_step % self.pace == 0) & ~measuring

        deposited, failed = jax.lax.cond(
            due & ~state.overflow,
            lambda s: self._deposit(values, log_weights, bias_energies, s),
            lambda s: (s, jnp.array(False)),
            state,
        )
        return deposited._replace(
            step=new_step, overflow=state.overflow | ~ordered | failed
        )

    def _deposit(self, values, log_weights, old_bias, state):
        dtype = state.centers.dtype
        w, n = self.walkers, self.ncv
        weights = jnp.exp(log_weights)
        counter = state.counter + w
        sum_weights = state.sum_weights + jnp.sum(weights)
        sum_weights2 = state.sum_weights2 + jnp.sum(weights * weights)
        neff = (1 + sum_weights) ** 2 / (1 + sum_weights2)
        sigma0, av_m2, floor = state.sigma0, state.av_m2, state.sigma_floor
        bad = jnp.array(False)
        user_min = (
            jnp.asarray(self.sigma_min, dtype) if self.sigma_min else None
        )
        if self.adaptive:
            factor = self.gamma
            count = state.adaptive_counter.astype(dtype)
            first = counter == 1 + w
            m2_first = av_m2 * self.gamma
            s0_first = jnp.sqrt(m2_first / count / factor)
            if user_min is None:
                bad = first & jnp.any(~(s0_first > _SIGMA_FLOOR))
            else:
                s0_first = jnp.maximum(s0_first, user_min)
            av_m2 = jnp.where(first, m2_first, av_m2)
            sigma0 = jnp.where(first, s0_first, sigma0)
            sigma = jnp.sqrt(av_m2 / count / factor)
            if user_min is None:
                floor = floor | jnp.any(sigma < _SIGMA_FLOOR, axis=1)
                sigma = jnp.maximum(sigma, _SIGMA_FLOOR)
            else:
                sigma = jnp.maximum(sigma, user_min)
        else:
            sigma = sigma0
        if not self.fixed_sigma:
            rescale = (neff * (n + 2.0) / 4.0) ** (-1.0 / (4 + n))
            sigma = sigma * rescale
            if user_min is not None:
                sigma = jnp.maximum(sigma, user_min)
            elif self.adaptive:
                sigma = jnp.where(floor[:, None], jnp.maximum(sigma, _SIGMA_FLOOR), sigma)
        heights = weights * jnp.prod(sigma0 / sigma, axis=1)

        kernels = (state.centers, state.sigmas, state.heights, state.count)

        def add(carry, hill):
            kernels, failed = carry
            kernels, full = self._add_kernel(kernels, *hill)
            return (kernels, failed | full), None

        (kernels, full), _ = jax.lax.scan(
            add, (kernels, jnp.array(False)), (values, sigma, heights)
        )
        centers, sigmas, kheights, count = kernels
        if self.no_zed:
            zed = state.zed
        else:
            zed = self._zed(centers, sigmas, kheights, count) / sum_weights / count.astype(dtype)
        new = state._replace(
            centers=centers,
            sigmas=sigmas,
            heights=kheights,
            count=count,
            counter=counter,
            sum_weights=sum_weights,
            sum_weights2=sum_weights2,
            zed=zed,
            sigma0=sigma0,
            av_m2=av_m2,
            sigma_floor=floor,
        )
        if self.calc_work:
            new_bias = jax.vmap(self.energy, in_axes=(0, None))(values, new)
            new = new._replace(work=state.work + new_bias - old_bias)
        good = (
            jnp.all(jnp.isfinite(values))
            & jnp.all(jnp.isfinite(heights))
            & jnp.all(jnp.isfinite(sigma))
            & jnp.all(jnp.isfinite(new.sigmas))
            & jnp.isfinite(zed)
            & ~bad
        )
        failed = full | ~good
        state = jax.tree_util.tree_map(lambda old, x: jnp.where(failed, old, x), state, new)
        return state, failed

    def _zed(self, centers, sigmas, heights, count):
        """Sum over active kernel centers of the unnormalized kernel sum."""
        capacity = centers.shape[0]
        block = min(capacity, 256)
        pad = -capacity % block
        index = jnp.arange(capacity + pad)
        points = jnp.concatenate([centers, jnp.zeros((pad, self.ncv), centers.dtype)])
        rows = jax.lax.map(
            jax.vmap(lambda x: self._kernel_sum(x, centers, sigmas, heights, count)),
            points.reshape(-1, block, self.ncv),
        ).reshape(-1)
        return jnp.sum(jnp.where(index < count, rows, 0.0))

    def _nearest(self, center, exclude, centers, sigmas, count):
        """PLUMED getMergeableKernel: first kernel with the smallest distance."""
        delta = periodic_difference(center, centers, self._periods(centers.dtype))
        norm2 = jnp.sum((delta / sigmas) ** 2, axis=-1)
        index = jnp.arange(centers.shape[0])
        ok = (index < count) & (index != exclude) & (norm2 < self.compression_threshold**2)
        return jnp.argmin(jnp.where(ok, norm2, jnp.inf)).astype(jnp.int32), jnp.any(ok)

    def _merge(self, c1, s1, h1, c2, s2, h2):
        """PLUMED mergeKernels: moment-preserving merge of kernel 2 into 1."""
        periods = self._periods(c1.dtype)
        periodic = periods > 0
        c1 = jnp.where(periodic, c2 + periodic_difference(c1, c2, periods), c1)
        h = h1 + h2
        c = (h1 * c1 + h2 * c2) / h
        ss = (h1 * (s1 * s1 + c1 * c1) + h2 * (s2 * s2 + c2 * c2)) / h - c * c
        low = jnp.asarray(
            self.periodic_min or tuple(-p / 2 for p in (self.periods or (0.0,) * self.ncv)),
            c1.dtype,
        )
        middle = low + periods / 2
        c = jnp.where(periodic, middle + periodic_difference(c, middle, periods), c)
        return c, jnp.sqrt(ss), h

    def _add_kernel(self, kernels, center, sigma, height):
        """PLUMED addKernel with optional recursive merging; returns (kernels, full)."""
        centers, sigmas, heights, count = kernels
        capacity = centers.shape[0]
        if self.compression_threshold == 0:
            found = jnp.array(False)
            taker = jnp.int32(0)
        else:
            taker, found = self._nearest(center, capacity, centers, sigmas, count)

        def append(k):
            centers, sigmas, heights, count = k
            slot = jnp.minimum(count, capacity - 1)
            return (
                centers.at[slot].set(center),
                sigmas.at[slot].set(sigma),
                heights.at[slot].set(height),
                count + 1,
            )

        def merge(k):
            centers, sigmas, heights, count = k
            c, s, h = self._merge(
                centers[taker], sigmas[taker], heights[taker], center, sigma, height
            )
            k = (centers.at[taker].set(c), sigmas.at[taker].set(s),
                 heights.at[taker].set(h), count)
            if not self.recursive_merge:
                return k

            def condition(carry):
                return carry[2]

            def body(carry):
                (centers, sigmas, heights, count), giver, _, taker = carry
                low, high = jnp.minimum(taker, giver), jnp.maximum(taker, giver)
                c, s, h = self._merge(
                    centers[low], sigmas[low], heights[low],
                    centers[high], sigmas[high], heights[high],
                )
                centers, sigmas, heights = (
                    centers.at[low].set(c), sigmas.at[low].set(s), heights.at[low].set(h)
                )
                # Erase the higher index, shifting later kernels down by one.
                index = jnp.arange(capacity)
                source = jnp.minimum(jnp.where(index >= high, index + 1, index), capacity - 1)
                keep = index < count - 1
                centers = jnp.where(keep[:, None], centers[source], 0.0)
                sigmas = jnp.where(keep[:, None], sigmas[source], 1.0)
                heights = jnp.where(keep, heights[source], 0.0)
                count = count - 1
                taker, found = self._nearest(centers[low], low, centers, sigmas, count)
                return (centers, sigmas, heights, count), low, found, taker

            first, again = self._nearest(k[0][taker], taker, k[0], k[1], count)
            k, *_ = jax.lax.while_loop(condition, body, (k, taker, again, first))
            return k

        full = ~found & (count >= capacity)
        updated = jax.lax.cond(found, merge, append, kernels)
        updated = jax.tree_util.tree_map(
            lambda old, new: jnp.where(full, old, new), kernels, updated
        )
        return updated, full

    def replay(
        self,
        values: ArrayLike,
        state: OPESState | None = None,
        *,
        reuse_bias: bool = True,
    ) -> tuple[OPESState, Array]:
        """Return (final_state, pre_deposition_energies) for a CV trajectory.

        values has shape (steps, CVs) for one walker, or (steps, walkers, CVs)
        when walkers > 1; energies has shape (steps,) or (steps, walkers).
        Row i is completed step state.step+i+1. state=None initializes using
        the trajectory dtype. Check final_state.overflow outside JIT.
        """
        values = jnp.asarray(values)
        state = self.init(values.dtype) if state is None else state
        batched = self.walkers > 1

        def advance(s, v):
            v = v if batched else v[None, :]
            energies = jax.vmap(self.energy, in_axes=(0, None))(v, s)
            s = self.update_walkers(v, s, bias_energies=energies if reuse_bias else None)
            return s, energies if batched else energies[0]

        return jax.lax.scan(advance, state, values)
