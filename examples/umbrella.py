"""Twenty velocity-Verlet steps with a distance umbrella; Å, eV, unit masses.

Time is in reduced units consistent with unit mass, Å, and eV.
"""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from plumed_jax import distance, harmonic_restraint


def energy(r):
    d = distance(r, 0, 1)
    physical = 0.5 * (d - 1.)**2
    return physical + harmonic_restraint(d, centers=1.2, kappa=2.)


energy_gradient = jax.jit(jax.value_and_grad(energy))


@jax.jit
def step(r, v, force):
    dt = 0.005
    half_v = v + 0.5 * dt * force  # mass = 1
    r = r + dt * half_v
    value, gradient = energy_gradient(r)
    force = -gradient
    return r, half_v + 0.5 * dt * force, force, value


def main():
    r = jnp.array([[0., 0., 0.], [1.3, 0.1, 0.]])
    v = jnp.zeros_like(r)
    _, gradient = energy_gradient(r)
    force = -gradient
    for _ in range(20):
        r, v, force, value = step(r, v, force)
    assert bool(jnp.all(jnp.isfinite(force)))
    print('umbrella: energy', float(value), 'distance', float(distance(r, 0, 1)))


if __name__ == '__main__':
    main()
