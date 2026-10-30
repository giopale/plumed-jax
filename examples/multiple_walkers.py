"""Two synchronous walkers sharing one bias; twenty reduced-time Verlet steps.

Coordinates Å, energy eV, unit masses. No distributed communication is required.
"""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from plumed_jax import distance, Metadynamics

meta = Metadynamics(sigma=(0.15,), height=0.001, pace=5, capacity=16,
                    bias_factor=5., kT=0.025)  # height is PER WALKER


def cv(r):
    return jnp.atleast_1d(distance(r, 0, 1))


def total_energy(r, history):
    d = distance(r, 0, 1)
    return 0.5 * (d - 1.)**2 + meta.energy(jnp.atleast_1d(d), history)


# All walkers read the same history; each has separate positions/velocities.
energy_gradient = jax.jit(jax.vmap(jax.value_and_grad(total_energy), in_axes=(0, None)))
values_for = jax.vmap(cv)
biases_for = jax.vmap(meta.energy, in_axes=(0, None))


@jax.jit
def step(r, v, force, history, completed_step):
    dt = 0.005
    half_v = v + 0.5 * dt * force
    r = r + dt * half_v
    _, gradient = energy_gradient(r, history)
    v = half_v - 0.5 * dt * gradient  # complete with common OLD bias
    values = values_for(r)
    old_biases = biases_for(values, history)
    # One synchronized update: both heights use the same pre-deposition state.
    history = meta.update_walkers(values, history, step=completed_step,
                                  bias_energies=old_biases)
    energies, gradient = energy_gradient(r, history)  # refresh EVERY walker
    return r, v, -gradient, history, energies


def main():
    r = jnp.array([[[0., 0., 0.], [1.3, 0.1, 0.]],
                   [[0., 0., 0.], [1.1, 0.2, 0.]]])
    v = jnp.zeros_like(r)
    history = meta.init()
    _, gradient = energy_gradient(r, history)
    force = -gradient
    for completed_step in range(1, 21):
        r, v, force, history, energies = step(r, v, force, history, completed_step)
    assert not bool(history.overflow) and int(history.count) == 8
    assert bool(jnp.all(jnp.isfinite(force)))
    print('shared walkers: hills', int(history.count), 'energies', energies)


if __name__ == '__main__':
    main()
