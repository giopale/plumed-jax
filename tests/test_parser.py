import jax
import jax.numpy as jnp
import numpy as np
import pytest
from plumed_jax.parser import parse
from plumed_jax.bias import harmonic_restraint
from plumed_jax.metadata import Units
from plumed_jax.precision import compare


def test_parser_units_composition_and_virtual_atoms():
    p=parse('''UNITS LENGTH=A ENERGY=eV
water: GROUP ATOMS=1-3
v: COM ATOMS=water NOPBC
a: DISTANCE ATOMS=v,4 NOPBC
b: DISTANCE ATOMS=2,4 NOPBC
d: COMBINE ARG=a,b COEFFICIENTS=1,-1 PERIODIC=NO
h: RESTRAINT ARG=d AT=0.2 KAPPA=3.0
PRINT ARG=d,h.bias FILE=COLVAR
''')
    x=jnp.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[3.,0.,1.]])
    masses=jnp.array([1.,2.,1.,12.])
    values=jax.jit(lambda r:p.values(r,masses=masses))(x)
    got=jax.jit(lambda r:p.bias(r,masses=masses))(x)
    np.testing.assert_allclose(got,harmonic_restraint(values['d'],.2,3.),atol=1e-14)
    assert np.all(np.isfinite(jax.jit(jax.grad(lambda r:p.bias(r,masses=masses)))(x)))
    assert np.isclose(Units('nm','kj/mol').convert_energy(96.48533212331002),1.)


def test_parser_metad_state_scan_and_multiline():
    p=parse('''UNITS LENGTH=A ENERGY=eV
d: DISTANCE ATOMS=1,2 NOPBC
m: METAD ...
 ARG=d SIGMA=0.3 HEIGHT=0.2 PACE=2
...
''',hill_capacity=8)
    x=jnp.array([[0.,0.,0.],[1.,0.,0.]])
    def step(state,shift):
        r=x.at[1,0].add(shift)
        return p.update(r,state),p.bias(r,state=state)
    state,energies=jax.jit(lambda s:jax.lax.scan(step,s,jnp.linspace(0.,.4,6)))(p.initial_state())
    assert int(state['m'].count)==3 and energies[2]>0
    with pytest.raises(ValueError):p.bias(x)


@pytest.mark.parametrize('text',[
    'x: UNKNOWN ATOMS=1,2',
    'd: DISTANCE ATOMS=1,2 SILENTLYIGNORE=1',
    'c: CUSTOM ARG=x FUNC=__import__("os") PERIODIC=NO',
    'c: CUSTOM ARG=x FUNC=x.__class__ PERIODIC=NO',
    'd: DISTANCE ATOMS=1,2\nd: DISTANCE ATOMS=1,2',
    'd: DISTANCE ...\n ATOMS=1,2',
    'd: DISTANCE',
    'd: DISTANCE ATOMS=1,2 NOPBC=false',
    'd: DISTANCE ATOMS=1,2\ne: DISTANCE ATOMS=2,3\nb: RESTRAINT ARG=d,e AT=1 KAPPA=2,2',
    'd: DISTANCE ATOMS=1,2\nb: RESTRAINT ARG=missing AT=1 KAPPA=2',
])
def test_parser_rejects_unknown_and_unsafe(text):
    with pytest.raises(ValueError):parse(text)


def test_precision_distinguishes_signed_zero_and_one_ulp():
    metric=compare(np.array([0.,1.]),np.array([-0.,np.nextafter(1.,2.)]))
    assert not metric['bitwise_equal'] and metric['max_ulp']==1

@pytest.mark.parametrize('selection',['0,2','1,9'])
def test_parser_bounds_fail_before_jax_index_clamping(selection):
    p=parse(f'd: DISTANCE ATOMS={selection}')
    with pytest.raises(ValueError):jax.jit(p.values)(jnp.zeros((3,3)))


def test_combined_bias_and_update_uses_each_metad_history_before_deposition():
    program=parse('''UNITS LENGTH=A
p: POSITION ATOM=1 NOPBC
m: METAD ARG=p.x SIGMA=.12 HEIGHT=.2 PACE=1 BIASFACTOR=5 TEMP=300 GRID_MIN=-3 GRID_MAX=3 GRID_BIN=128
n: METAD ARG=p.y SIGMA=.2 HEIGHT=.3 PACE=2
r: RESTRAINT ARG=p.x AT=.1 KAPPA=2
''',hill_capacity=8)
    state=program.initial_state()
    state=program.update(jnp.array([[.1,-.2,.3]]),state)
    positions=jnp.array([[.14,-.23,.3]])
    separate=jax.jit(jax.value_and_grad(lambda r,s:program.bias(r,state=s)))
    combined=jax.jit(jax.value_and_grad(program.bias_and_update,has_aux=True))
    expected,expected_grad=separate(positions,state)
    (energy,updated),gradient=combined(positions,state)
    np.testing.assert_allclose(energy,expected,atol=1e-13)
    np.testing.assert_allclose(gradient,expected_grad,atol=5e-13)
    manual=program.update(positions,state)
    for key in manual:
        for a,b in zip(updated[key],manual[key]):np.testing.assert_allclose(a,b,atol=3e-14)
    assert int(updated['m'].count)==2 and int(updated['n'].count)==1
