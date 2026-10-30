import jax
import jax.numpy as jnp
import numpy as np
import pytest

from plumed_jax.geometry import displacement, prepare_cell
from plumed_jax.parser import parse


@pytest.mark.parametrize('dtype',[jnp.float64,jnp.float32])
def test_prepared_cell_refresh_and_coordinate_cell_gradients(dtype):
    r=jnp.array([[.4,.21,.4],[7.2,-2.,3.1]],dtype=dtype)
    cells=[jnp.diag(jnp.array([4.,5.,6.],dtype=dtype)),
        jnp.array([[1.,0.,0.],[9.7,.3,0.],[.2,.1,2.]],dtype=dtype)]
    def energy(r,h,shared):
        v=displacement(jnp.zeros(3,dtype=r.dtype),r,prepare_cell(h) if shared else h)
        return jnp.sum(v*v)
    original=jax.jit(jax.value_and_grad(lambda r,h:energy(r,h,False),(0,1)))
    shared=jax.jit(jax.value_and_grad(lambda r,h:energy(r,h,True),(0,1)))
    # Reuse the same executable with a different cell, crossing the orthogonal
    # branch, to detect stale factors as well as missing cell derivatives.
    for h in cells:
        for a,b in zip(jax.tree_util.tree_leaves(original(r,h)),
                jax.tree_util.tree_leaves(shared(r,h))):
            assert np.asarray(a).tobytes()==np.asarray(b).tobytes()


def test_pair_cache_distinguishes_periodic_free_and_virtual_atoms():
    program=parse('''c: COM ATOMS=1,2 NOPBC
d: DISTANCE ATOMS=1,2
f: DISTANCE ATOMS=1,2 NOPBC
e: DISTANCE ATOMS=1,2
v: DISTANCE ATOMS=c,3
a: ANGLE ATOMS=1,2,3
t: TORSION ATOMS=1,2,3,4
u: RESTRAINT ARG=d,f,e,v,a,t AT=1,1,1,1,1,1 KAPPA=.2,.2,.2,.2,.2,.2
''')
    r=jnp.array([[.1,.2,.3],[4.8,.6,.4],[2.,1.,.6],[2.4,1.4,2.]])
    h=jnp.array([[5.,0.,0.],[.7,5.,0.],[.2,.4,5.]])
    masses=jnp.array([12.,1.,16.,1.])
    values=[jax.jit(lambda r,h:program.values(r,h,masses=masses,share_geometry=s))(r,h)
        for s in (False,True)]
    assert not np.isclose(values[1]['d'],values[1]['f'])
    assert values[1]['d']==values[1]['e']
    for key in values[0]:np.testing.assert_allclose(values[1][key],values[0][key],atol=1e-13)
    outputs=[jax.jit(jax.value_and_grad(
        lambda r,h:program.bias(r,h,masses=masses,share_geometry=s),(0,1)))(r,h)
        for s in (False,True)]
    for a,b in zip(jax.tree_util.tree_leaves(outputs[1]),jax.tree_util.tree_leaves(outputs[0])):
        np.testing.assert_allclose(a,b,rtol=1e-13,atol=1e-13)


@pytest.mark.parametrize('action',[
    'd: DISTANCE ATOMS=1,2,3',
    'a: ANGLE ATOMS=1,2',
    't: TORSION ATOMS=1,2,3',
    'd: DISTANCE ATOMS=missing,2',
    'd: DISTANCE ATOMS=c,2\nc: COM ATOMS=1,3 NOPBC',
])
def test_prepared_geometry_rejects_invalid_selections(action):
    program=parse(action)
    for shared in (False,True):
        with pytest.raises(ValueError):
            program.values(jnp.zeros((3,3)),masses=jnp.ones(3),share_geometry=shared)


def test_prepared_selection_follows_edited_options_and_position_count():
    program=parse('d: DISTANCE ATOMS=01,2 NOPBC')
    positions=jnp.array([[0.,0.,0.],[1.,0.,0.],[3.,0.,0.]])
    assert program.values(positions)['d']==1.
    program.actions[0].options['ATOMS']='1,3'
    assert program.values(positions)['d']==3.
    with pytest.raises(ValueError,match='outside position array'):
        program.values(positions[:2])
