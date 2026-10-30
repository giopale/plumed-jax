"""Single-walker well-tempered metadynamics, twenty reduced-time Verlet steps.

Coordinates Å, energy eV, unit masses; this is a usage example, not an FES run.
"""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from plumed_jax import distance, Metadynamics

meta = Metadynamics(sigma=(0.15,), height=0.002, pace=5, capacity=8,
                    bias_factor=5., kT=0.025)


def cv(r):
    return jnp.atleast_1d(distance(r, 0, 1))


def total_energy(r, history):
    d = distance(r, 0, 1)
    return 0.5 * (d - 1.)**2 + meta.energy(jnp.atleast_1d(d), history)


energy_gradient = jax.jit(jax.value_and_grad(total_energy))


@jax.jit
def step(r, v, force, history, completed_step):
    dt = 0.005
    half_v = v + 0.5 * dt * force
    r = r + dt * half_v
    # Finish the step using the OLD bias, including the second velocity kick.
    _, gradient = energy_gradient(r, history)
    v = half_v - 0.5 * dt * gradient
    values = cv(r)
    old_bias = meta.energy(values, history)  # individual bias, not total energy
    history = meta.update(values, history, step=completed_step, bias_energy=old_bias)
    # Refresh the cached force before the NEXT integration step.
    value, gradient = energy_gradient(r, history)
    return r, v, -gradient, history, value


def main():
    r = jnp.array([[0., 0., 0.], [1.3, 0.1, 0.]])
    v = jnp.zeros_like(r)
    history = meta.init()  # no step-zero hill
    _, gradient = energy_gradient(r, history)
    force = -gradient
    for completed_step in range(1, 21):
        r, v, force, history, value = step(r, v, force, history, completed_step)
    assert not bool(history.overflow) and int(history.count) == 4
    assert bool(jnp.all(jnp.isfinite(force)))
    print('single walker: hills', int(history.count), 'energy', float(value))


if __name__ == '__main__':
    main()
