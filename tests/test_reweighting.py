import jax
import jax.numpy as jnp
import numpy as np
import pytest
from plumed_jax.reweighting import metadynamics_offset


@pytest.mark.parametrize("factor", [None, 5.0])
def test_offset_constant_bias_and_arbitrary_energy_zero(factor):
    fn = jax.jit(lambda v: metadynamics_offset(v, kT=2.5, bias_factor=factor))
    np.testing.assert_allclose(fn(jnp.full(21, 7.3)), 7.3, atol=2e-14)
    values = jnp.linspace(-4.0, 13.0, 21)
    c = fn(values)
    shifted_c = fn(values + 1000.0)
    np.testing.assert_allclose(shifted_c, c + 1000.0, atol=3e-13)
    np.testing.assert_allclose((values-c)/2.5, (values+1000-shifted_c)/2.5, atol=2e-13)


def test_ordinary_limit_and_stable_large_bias():
    values = jnp.array([0.0, 1.0, 2.0])
    expected = 2.5 * np.log(np.mean(np.exp(np.asarray(values)/2.5)))
    np.testing.assert_allclose(metadynamics_offset(values,kT=2.5),expected,atol=2e-14)
    np.testing.assert_allclose(metadynamics_offset(values,kT=2.5,bias_factor=1e8),expected,atol=1e-8)
    assert np.isfinite(metadynamics_offset(jnp.array([1e5,1e5+2]),kT=2.5,bias_factor=5.))


def test_invalid_reweighting_parameters():
    with pytest.raises(ValueError):metadynamics_offset([1.,2.],kT=0.)
    with pytest.raises(ValueError):metadynamics_offset([1.,2.],kT=2.,bias_factor=1.)
    with pytest.raises(ValueError):metadynamics_offset([1.],kT=2.)
