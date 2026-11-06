"""OPES_METAD against PLUMED driver output (optional fixture).

The fixture is produced by tests/fixtures/generate_opes_reference.py with a
local PLUMED build; these tests are skipped when it is absent. They compare,
at every MD step, the pre-deposition bias, the atomic bias forces and the
rct/zed/neff/nker/work components, and the final STATE kernels and scalars.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import opes_cases
from plumed_jax.parser import parse

FIXTURE = Path(__file__).parent / "fixtures" / "opes-plumed.npz"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="run tests/fixtures/generate_opes_reference.py")


def fixture_cases():
    if not FIXTURE.exists():
        return []
    manifest = json.loads(FIXTURE.with_suffix(".json").read_text())
    return sorted(manifest["cases"])


def load(name):
    with np.load(FIXTURE, allow_pickle=False) as data:
        prefix = name + "/"
        return {k[len(prefix):]: data[k] for k in data.files if k.startswith(prefix)}


@pytest.mark.parametrize("name", fixture_cases())
def test_opes_matches_plumed_driver(name):
    ref = load(name)
    walkers = opes_cases.CASES[name]["walkers"]
    program = parse(opes_cases.parser_input(name), walkers=walkers, hill_capacity=1024)
    config = program.metad["opes"]
    positions = jnp.asarray(ref["positions"])
    state = program.initial_state()

    if walkers == 1:
        def total(r, s, step):
            energy, new = program.bias_and_update(r, s, step=step)
            return energy, (energy[None], new)
        bias_only = lambda r, s: program.bias(r, state=s)[None]
        frame = lambda t: positions[0, t]
    else:
        def total(r, s, step):
            energies, new = program.bias_and_update_walkers(r, s, step=step)
            return jnp.sum(energies), (energies, new)
        bias_only = jax.vmap(lambda r, s: program.bias(r, state=s), in_axes=(0, None))
        frame = lambda t: positions[:, t]
    advance = jax.jit(jax.value_and_grad(total, has_aux=True))
    initial = jax.jit(jax.vmap(jax.grad(lambda r, s: program.bias(r, state=s)), in_axes=(0, None)))

    def check(t, energies, gradient, s):
        np.testing.assert_allclose(energies, ref["bias"][:, t], rtol=1e-10, atol=1e-9)
        np.testing.assert_allclose(-gradient, ref["forces"][:, t], rtol=1e-8, atol=1e-8)
        diagnostics = config.diagnostics(s["opes"])
        for key in ("rct", "zed", "neff", "nker", "work"):
            if key in ref and key in diagnostics:
                expected = ref[key][:, t] if key == "work" else ref[key][0, t]
                got = np.atleast_1d(diagnostics[key]) if key == "work" else diagnostics[key]
                np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-9,
                                           err_msg=f"{key} at step {t}")

    x0 = frame(0)
    check(0, bias_only(x0, state),
          initial(x0 if walkers > 1 else x0[None], state).reshape(walkers, 4, 3), state)
    for t in range(1, positions.shape[1]):
        (_, (energies, state_next)), gradient = advance(frame(t), state, jnp.int32(t))
        check(t, energies, gradient.reshape(walkers, 4, 3), state_next)
        state = state_next

    final = state["opes"]
    assert not bool(final.overflow)
    k = int(final.count)
    assert k == len(ref["state_heights"])
    np.testing.assert_allclose(final.centers[:k], ref["state_centers"], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(final.sigmas[:k], ref["state_sigmas"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(final.heights[:k], ref["state_heights"], rtol=1e-10)
    for key in ("zed", "sum_weights", "sum_weights2", "counter", "adaptive_counter"):
        if "state_" + key in ref:
            np.testing.assert_allclose(getattr(final, key), ref["state_" + key], rtol=1e-10)
    for key, field in (("sigma0", "sigma0"), ("av_cv", "av_cv"), ("av_M2", "av_m2")):
        if "state_" + key in ref:
            np.testing.assert_allclose(getattr(final, field), ref["state_" + key],
                                       rtol=1e-9, atol=1e-12)
