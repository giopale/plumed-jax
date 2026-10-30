import jax
import jax.numpy as jnp
import numpy as np
import pytest
from plumed_jax.bias import harmonic_restraint, make_biased_energy
from plumed_jax.metadynamics import Metadynamics
from plumed_jax import checkpoint
from plumed_jax.reference import rmsd, path_msd
from plumed_jax.structure import steinhardt, tetrahedral_order, contact_matrix
from plumed_jax.geometry import displacement


def test_periodic_restraint_and_complete_energy_gradient():
    fn=lambda x:harmonic_restraint(x,3.1,2.,periods=2*np.pi)
    x=jnp.array(-3.1)
    np.testing.assert_allclose(jax.grad(fn)(x),2*(2*np.pi-6.2),atol=1e-14)
    combined=make_biased_energy(lambda r,k:k*jnp.sum(r*r),lambda r:r[0,0],
                              lambda v:harmonic_restraint(v,1.,3.))
    r=jnp.array([[.2,.3,.4]])
    e,g=jax.jit(jax.value_and_grad(combined))(r,2.)
    expected=4*r; expected=expected.at[0,0].add(3*(r[0,0]-1))
    np.testing.assert_allclose(g,expected,atol=1e-14)

@pytest.mark.parametrize('factor',[None,5.])
def test_metadynamics_replay_restart_bytes_and_overflow(tmp_path,factor):
    config=Metadynamics((.3,),.2,2,8,bias_factor=factor,kT=.03)
    values=jnp.array([[.1],[.2],[-.4],[.6],[.7],[-.1]])
    final,energies=jax.jit(config.replay)(values)
    first,_=config.replay(values[:3])
    path=tmp_path/'state.npz'; checkpoint.save(path,first,config)
    restored=checkpoint.load(path,config)
    for a,b in zip(first,restored): assert np.asarray(a).tobytes()==np.asarray(b).tobytes()
    resumed,_=config.replay(values[3:],restored)
    for a,b in zip(final,resumed): assert np.asarray(a).tobytes()==np.asarray(b).tobytes()
    assert int(final.count)==3 and int(final.step)==6
    assert energies[0]==0 and energies[1]==0 and energies[2]>0
    tiny=Metadynamics((.3,),.2,1,1)
    overflow,_=jax.jit(tiny.replay)(values)
    assert bool(overflow.overflow) and int(overflow.count)==1
    with pytest.raises(ValueError): checkpoint.load(path,tiny)


def test_rmsd_weighted_gradient_alignment_and_symmetry():
    ref=jnp.array([[1.,1.,1.],[-1.,-1.,1.],[-1.,1.,-1.],[1.,-1.,-1.]])
    theta=.7
    rotation=jnp.array([[jnp.cos(theta),-jnp.sin(theta),0.],[jnp.sin(theta),jnp.cos(theta),0.],[0.,0.,1.]])
    x=ref@rotation+4
    fn=lambda r:rmsd(r,ref,squared=True)
    np.testing.assert_allclose(jax.jit(fn)(x),0,atol=1e-28)
    assert np.all(np.isfinite(jax.jit(jax.grad(fn))(x)))
    x=x.at[0,0].add(.12)
    fn=lambda r:rmsd(r,ref,alignment_weights=jnp.array([1.,2.,3.,4.]),
                    displacement_weights=jnp.array([4.,3.,2.,1.]),squared=True)
    gradient=jax.jit(jax.grad(fn))(x)
    eps=1e-5
    finite=(fn(x.at[0,1].add(eps))-fn(x.at[0,1].add(-eps)))/(2*eps)
    np.testing.assert_allclose(gradient[0,1],finite,atol=1e-9)
    s,z=jax.jit(lambda r:path_msd(r,jnp.stack([ref,ref.at[0,0].add(.2)]),10.))(x)
    assert 1<=s<=2 and np.isfinite(z)


def test_order_tetrahedron_and_empty_neighbor_gradients():
    x=jnp.array([[0.,0.,0.],[1.,1.,1.],[-1.,-1.,1.],[-1.,1.,-1.],[1.,-1.,-1.]])
    np.testing.assert_allclose(tetrahedral_order(x,0,[1,2,3,4]),1.,atol=1e-14)
    q=jax.jit(lambda r:steinhardt(r,[0],[1,2,3,4],l=4,r0=3.))(x)
    np.testing.assert_allclose(q,np.sqrt(7/27),atol=1e-14)
    fn=lambda r:jnp.sum(steinhardt(r,[0],[0],l=6,r0=3.))
    assert fn(x)==0 and np.all(np.isfinite(jax.jit(jax.grad(fn))(x)))
    assert np.all(np.diag(contact_matrix(x,[0,1,2],r0=3.))==0)


def test_skew_cell_exact_lattice_search_and_gradient():
    cell=jnp.array([[1.,0.,0.],[9.7,.3,0.],[.2,.1,2.]])
    delta=jnp.array([.4,.21,.4])
    got=jax.jit(lambda r,h:displacement(jnp.zeros(3),r,h))(delta,cell)
    import itertools
    shifts=np.array(list(itertools.product(range(-20,21),range(-3,4),range(-2,3))))
    images=np.asarray(delta)-shifts@np.asarray(cell)
    closest=images[np.argmin(np.sum(images**2,axis=1))]
    np.testing.assert_allclose(got,closest,atol=1e-14)
    fn=lambda r,h:jnp.sum(displacement(jnp.zeros(3),r,h)**2)
    gr,gc=jax.jit(jax.grad(fn,(0,1)))(delta,cell)
    np.testing.assert_allclose(gr,2*got,atol=1e-14)
    eps=1e-6
    finite=(fn(delta,cell.at[1,1].add(eps))-fn(delta,cell.at[1,1].add(-eps)))/(2*eps)
    np.testing.assert_allclose(gc[1,1],finite,atol=1e-8)

def test_empty_static_neighbor_groups():
    x=jnp.array([[0.,0.,0.],[1.,0.,0.]])
    fn=lambda r:jnp.sum(steinhardt(r,[0],[],l=6,r0=3.))
    value,gradient=jax.jit(jax.value_and_grad(fn))(x)
    assert value==0 and np.all(gradient==0)
    assert contact_matrix(x,[0],[],r0=3.).shape==(1,0)


@pytest.mark.parametrize('degree',[0,4,6,12])
def test_harmonic_order_matches_pairwise_and_remains_smooth_at_poles(degree):
    rng=np.random.default_rng(118)
    x=jnp.asarray(rng.normal(size=(65,3))).at[0].set(0.)
    x=x.at[1].set(jnp.array([0.,0.,1.])).at[2].set(jnp.array([0.,0.,-1.]))
    fn=lambda method:lambda r:jnp.sum(steinhardt(r,[0,3],range(65),l=degree,r0=2.,method=method))
    a=jax.jit(jax.value_and_grad(fn('pairwise')))(x)
    b=jax.jit(jax.value_and_grad(fn('harmonics')))(x)
    assert np.isfinite(b[0]) and np.all(np.isfinite(b[1]))
    np.testing.assert_allclose(a[0],b[0],atol=2e-13)
    np.testing.assert_allclose(a[1],b[1],atol=2e-12)
    # The order norm is invariant to rotation, including a change of polar axis.
    rotation=jnp.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
    np.testing.assert_allclose(fn('harmonics')(x@rotation),b[0],atol=3e-13)


def test_harmonic_order_empty_neighborhood_gradient():
    x=jnp.array([[0.,0.,0.],[0.,0.,1.]])
    for neighbors in ([],[0]):
        fn=lambda r:jnp.sum(steinhardt(r,[0],neighbors,l=6,r0=2.,method='harmonics'))
        value,gradient=jax.jit(jax.value_and_grad(fn))(x)
        assert value==0 and np.all(gradient==0)
    with pytest.raises(ValueError):steinhardt(x,[0],[1],method='unknown')
