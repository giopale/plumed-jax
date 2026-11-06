"""OPES_METAD with three synchronous walkers sharing one bias, and a restart.

Coordinates Å, energy eV, unit masses, twenty reduced-time Verlet steps per
segment. This is a usage example, not a converged free-energy calculation.
"""
import tempfile
from pathlib import Path

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from plumed_jax import distance, OPESMetad, checkpoint

# The JAX counterpart of OPES_METAD ... WALKERS_MPI with three replicas.
opes = OPESMetad(sigma=(0.05,), barrier=0.2, pace=5, capacity=64, kT=0.025,
                 walkers=3)


def cv(r):
    return jnp.atleast_1d(distance(r, 0, 1))


def total_energy(r, state):
    d = distance(r, 0, 1)
    return 0.5 * (d - 1.)**2 + opes.energy(jnp.atleast_1d(d), state)


# Every walker reads the same shared state; positions/velocities are per walker.
energy_gradient = jax.jit(jax.vmap(jax.value_and_grad(total_energy), in_axes=(0, None)))
values_for = jax.vmap(cv)
biases_for = jax.vmap(opes.energy, in_axes=(0, None))


@jax.jit
def step(r, v, force, state, completed_step):
    dt = 0.005
    half_v = v + 0.5 * dt * force
    r = r + dt * half_v
    _, gradient = energy_gradient(r, state)
    v = half_v - 0.5 * dt * gradient  # complete the step with the OLD bias
    values = values_for(r)
    old_biases = biases_for(values, state)  # each walker's own OPES bias
    # One synchronous deposit: weights from the same old state, kernels added
    # and compressed in walker order, then a single Z_n update.
    state = opes.update_walkers(values, state, step=completed_step,
                                bias_energies=old_biases)
    energies, gradient = energy_gradient(r, state)  # refresh EVERY walker
    return r, v, -gradient, state, energies


def run(r, v, state, steps):
    _, gradient = energy_gradient(r, state)
    force = -gradient
    for completed_step in steps:
        r, v, force, state, energies = step(r, v, force, state, completed_step)
    return r, v, state, energies


def main():
    r = jnp.array([[[0., 0., 0.], [1.3, 0.1, 0.]],
                   [[0., 0., 0.], [0.9, 0.0, 0.1]],
                   [[0., 0., 0.], [1.1, -0.1, 0.]]])
    v = jnp.zeros_like(r)
    r, v, state, _ = run(r, v, opes.init(), range(1, 21))
    assert not bool(state.overflow)
    with tempfile.TemporaryDirectory() as tmp:
        # Native restart: save the shared bias with all engine states.
        path = Path(tmp) / 'opes.npz'
        checkpoint.save(path, state, opes)
        state = checkpoint.load(path, opes)
    r, v, state, energies = run(r, v, state, range(21, 41))
    assert not bool(state.overflow) and int(state.counter) == 1 + 3 * 8
    info = opes.diagnostics(state)
    print('OPES walkers: kernels', int(info['nker']), 'zed', float(info['zed']),
          'neff', float(info['neff']), 'energies', energies)


if __name__ == '__main__':
    main()
