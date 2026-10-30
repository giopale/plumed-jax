"""CUSTOM extrema regressions, including the frozen PLUMED GPU failure."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from plumed_jax.parser import _expression, parse


@pytest.mark.parametrize("name", ["min", "max"])
@pytest.mark.parametrize("operands", [(1., 2.), (2., 1.), (2., 2.)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_custom_extremum_branch_derivatives(name, operands, dtype):
    expression = _expression(f"{name}(x,y)")
    fn = lambda xy: expression(dict(x=xy[0], y=xy[1]))
    xy = jnp.asarray(operands, dtype=dtype)
    value, gradient = jax.jit(jax.value_and_grad(fn))(xy)
    # PLUMED's Lepton derivative uses x<y for min and x>=y for max,
    # choosing one branch rather than averaging derivatives at equality.
    first = operands[0] < operands[1] if name == "min" else operands[0] >= operands[1]
    np.testing.assert_array_equal(gradient, [float(first), float(not first)])
    assert float(value) == (min(operands) if name == "min" else max(operands))


@pytest.mark.parametrize("name", ["min", "max"])
@pytest.mark.parametrize("literal_first", [False, True])
def test_custom_extremum_integer_literal_and_broadcast(name, literal_first):
    expression = _expression(f"{name}(2,x)" if literal_first else f"{name}(x,2)")
    value, gradient = jax.jit(jax.value_and_grad(
        lambda x: jnp.sum(expression(dict(x=x)))))(jnp.array([1., 3.]))
    np.testing.assert_array_equal(gradient, [1., 0.] if name == "min" else [0., 1.])
    assert float(value) == (3. if name == "min" else 5.)


def test_custom_max_frozen_periodic_plumed_derivative():
    # First four positions of random-n64-seed1701-orthorhombic; the remaining
    # atoms are unselected but preserve the 64-atom GPU differentiation shape.
    positions = jnp.zeros((64, 3)).at[:4].set(jnp.array([
        [1.9455976384367872, 2.5219215065185763, 1.1773972471561631],
        [2.5789532540285567, 1.9869108743115942, 2.201157916248619],
        [11.525647656132096, 12.571899122626528, 2.0681047685402763],
        [0.9427452804208363, 0.11193609491544709, 13.473050200090217],
    ]))
    cell = jnp.diag(jnp.array([12., 13., 14.]))
    program = parse("""UNITS LENGTH=A
a: DISTANCE ATOMS=1,2
b: DISTANCE ATOMS=2,3
c: DISTANCE ATOMS=3,4
cv: CUSTOM ARG=a,b,c FUNC=max(x,y) PERIODIC=NO
""")
    fn = lambda x, h: program.values(x, h)["cv"]
    value, gradient = jax.jit(jax.value_and_grad(fn))(positions, cell)
    # Independent native PLUMED 2.10.1 reference, retained in the benchmark.
    expected = np.zeros((64, 3))
    expected[1] = [0.7838616391419776, 0.6199952836450858, 0.03415814601967792]
    expected[2] = -expected[1]
    np.testing.assert_allclose(value, 3.8952098756084266, rtol=0, atol=2e-11)
    np.testing.assert_allclose(gradient, expected, rtol=0, atol=5e-10)
