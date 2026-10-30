import jax
import jax.numpy as jnp
import numpy as np
import pytest

from plumed_jax import GridMetadynamics, Metadynamics, checkpoint
from plumed_jax.parser import parse


@pytest.mark.parametrize('dimension',[1,2,3])
def test_grid_nodes_and_conservative_continuous_forces(dimension):
    grid=GridMetadynamics((.3,)*dimension,.2,1,(-1.,)*dimension,(1.,)*dimension,(40,)*dimension)
    direct=Metadynamics((.3,)*dimension,.2,1,8)
    centers=jnp.array([[.12]*dimension,[-.23]*dimension,[.37]*dimension])
    state,_=jax.jit(grid.replay)(centers)
    history,_=jax.jit(direct.replay)(centers)
    x=jnp.full(dimension,.1)  # an exact grid vertex, well inside Gaussian support
    value,grad=jax.jit(jax.value_and_grad(grid.energy))(x,state)
    expected,expected_grad=jax.value_and_grad(direct.energy)(x,history)
    np.testing.assert_allclose(value,expected,atol=3e-14)
    np.testing.assert_allclose(grad,expected_grad,atol=3e-13)
    # Check a force at an off-grid point against independent energy differences.
    x=jnp.linspace(.117,.163,dimension)
    grad=jax.grad(grid.energy)(x,state)
    for axis in range(dimension):
        eps=1e-6
        numerical=(grid.energy(x.at[axis].add(eps),state)-grid.energy(x.at[axis].add(-eps),state))/(2*eps)
        np.testing.assert_allclose(grad[axis],numerical,atol=3e-9)
    # Adjacent cells must return the same limiting force at their shared face.
    x=x.at[0].set(.1)
    left=jax.grad(grid.energy)(x.at[0].add(-1e-10),state)
    right=jax.grad(grid.energy)(x.at[0].add(1e-10),state)
    np.testing.assert_allclose(left,right,atol=3e-8)


@pytest.mark.parametrize('factor',[None,5.])
def test_grid_checkpoint_resume_bytes_and_well_tempering(tmp_path,factor):
    meta=GridMetadynamics((.25,),.3,2,(-1.,),(1.,),(80,),bias_factor=factor,kT=2.5)
    values=jnp.linspace(-.3,.3,12)[:,None]
    whole,_=jax.jit(meta.replay)(values)
    part,_=meta.replay(values[:5])
    path=tmp_path/'grid.npz';checkpoint.save(path,part,meta)
    loaded=checkpoint.load(path,meta)
    for a,b in zip(part,loaded):assert np.asarray(a).tobytes()==np.asarray(b).tobytes()
    resumed,_=jax.jit(meta.replay)(values[5:],loaded)
    for a,b in zip(whole,resumed):assert np.asarray(a).tobytes()==np.asarray(b).tobytes()
    assert int(whole.count)==6 and not bool(whole.overflow)
    assert float(whole.last_height)<=meta.height
    if factor:assert float(whole.last_height)<meta.height
    np.testing.assert_allclose(meta.offset(meta.init()),0.,atol=1e-14)


def test_periodic_seam_upper_bound_and_invalid_coordinates():
    meta=GridMetadynamics((.3,),.2,1,(-np.pi,),(np.pi,),(128,),periods=(2*np.pi,))
    state=meta.update(jnp.array([3.09]),meta.init())
    fn=jax.jit(jax.value_and_grad(meta.energy))
    for x in (-np.pi,np.pi,3*np.pi):
        e,g=fn(jnp.array([x]),state)
        a,b=fn(jnp.array([-np.pi]),state)
        np.testing.assert_allclose(e,a,atol=2e-14)
        np.testing.assert_allclose(g,b,atol=2e-13)
    lo=fn(jnp.array([-np.pi+1e-10]),state)
    hi=fn(jnp.array([np.pi-1e-10]),state)
    np.testing.assert_allclose(lo[1],hi[1],atol=2e-8)
    bounded=GridMetadynamics((.3,),.2,2,(-1.,),(1.,),(40,))
    state=bounded.init()
    for x in (-1.,1.):assert np.isfinite(bounded.energy(jnp.array([x]),state))
    for x in (-1.01,np.nan,np.inf):
        assert np.isnan(bounded.energy(jnp.array([x]),state))
        bad=jax.jit(bounded.update)(jnp.array([x]),state)
        assert bool(bad.overflow) and int(bad.count)==0
    bad=bounded.update(jnp.array([.2]),state,step=0)
    assert bool(bad.overflow)


def test_near_zero_spline_slopes_and_cutoff_remain_finite():
    meta=GridMetadynamics((.2,),.2,1,(0.,),(1.,),(2,))
    state=meta.init()._replace(values=jnp.array([1e-9,1.,2.]),
        derivatives=jnp.array([[.01],[.01],[.01]]))
    np.testing.assert_allclose(jax.grad(meta.energy)(jnp.array([0.]),state),0.,atol=1e-14)
    state=state._replace(values=state.values.at[0].set(1e-7))
    np.testing.assert_allclose(jax.grad(meta.energy)(jnp.array([0.]),state),.01,atol=1e-14)
    two=GridMetadynamics((.2,.2),.2,1,(-1.,-1.),(1.,1.),(80,80))
    deposited=two.update(jnp.array([.1,.1]),two.init())
    # Close to the stretched Gaussian's cutoff, tiny grid values must not
    # generate singular mixed-derivative corrections.
    value,gradient=jax.value_and_grad(two.energy)(jnp.array([-.39,-.39]),deposited)
    assert np.isfinite(value) and np.all(np.isfinite(gradient)) and abs(value)<.01
    three=GridMetadynamics((.2,)*3,.2,1,(-1.,)*3,(1.,)*3,(2,)*3)
    tiny=three.init()._replace(values=jnp.full(three.grid_shape,1e-250))
    value,gradient=jax.value_and_grad(three.energy)(jnp.array([.17,.21,-.34]),tiny)
    assert np.isfinite(value) and np.all(np.isfinite(gradient))


def test_grid_parser_spacing_defaults_and_compiled_state():
    base='p: POSITION ATOM=1 NOPBC\nm: METAD ARG=p.x SIGMA=.2 HEIGHT=.1 PACE=2 '
    program=parse(base+'GRID_MIN=-1 GRID_MAX=1 GRID_BIN=40 GRID_SPACING=.02')
    assert isinstance(program.metad['m'],GridMetadynamics)
    assert program.metad['m'].grid_bin==(100,)
    default=parse(base+'GRID_MIN=-1 GRID_MAX=1')
    assert default.metad['m'].grid_bin==(50,)
    r=jnp.array([[.1,0.,0.]])
    state=jax.jit(lambda s:program.update(r,s,step=2))(program.initial_state())
    assert int(state['m'].count)==1
    assert np.isfinite(jax.jit(lambda s:program.bias(r,state=s))(state))
    for extra in ('GRID_BIN=40','GRID_MIN=-1 GRID_MAX=1 GRID_BIN=1',
                  'GRID_MIN=-1 GRID_MAX=1 GRID_SPACING=-.1',
                  'GRID_MIN=-1 GRID_MAX=1 GRID_BIN=4.5'):
        with pytest.raises(ValueError):parse(base+extra)


def test_invalid_grid_configuration_and_checkpoint(tmp_path):
    with pytest.raises(ValueError):GridMetadynamics((.2,),.2,1,(-1.,),(1.,),(40,),periods=(3.,))
    with pytest.raises(ValueError):GridMetadynamics((.2,),.2,1,(-1.,),(1.,),(40.5,))
    with pytest.raises(ValueError):GridMetadynamics((.2,)*4,.2,1,(-1.,)*4,(1.,)*4,(10,)*4)
    grid=GridMetadynamics((.2,),.2,1,(-1.,),(1.,),(40,))
    direct=Metadynamics((.2,),.2,1,40)
    with pytest.raises(ValueError):checkpoint.save(tmp_path/'bad.npz',direct.init(),grid)


def test_float32_grid_preserves_precision_under_jit(tmp_path):
    meta=GridMetadynamics((.2,),.2,1,(-np.pi,),(np.pi,),(80,),
                         periods=(2*np.pi,),bias_factor=5.,kT=2.5)
    state=jax.jit(meta.update)(jnp.array([.17],dtype=jnp.float32),meta.init(jnp.float32))
    value,gradient=jax.jit(jax.value_and_grad(meta.energy))(jnp.array([.22],dtype=jnp.float32),state)
    assert value.dtype==gradient.dtype==state.values.dtype==state.derivatives.dtype==jnp.float32
    assert np.isfinite(value) and np.all(np.isfinite(gradient)) and not bool(state.overflow)
    path=tmp_path/'float32-grid.npz';checkpoint.save(path,state,meta)
    restored=checkpoint.load(path,meta)
    for a,b in zip(state,restored):assert np.asarray(a).tobytes()==np.asarray(b).tobytes()


@pytest.mark.parametrize('dimension',[1,2,3])
@pytest.mark.parametrize('periodic',[False,True])
def test_local_deposition_covers_full_gaussian_at_bounds_and_seams(dimension,periodic):
    periods=(0.,)*(dimension-1)+(6. if periodic else 0.,)
    meta=GridMetadynamics((.12,)*dimension,.2,1,(-3.,)*dimension,(3.,)*dimension,(48,)*dimension,
                         periods=periods,bias_factor=5.,kT=2.5)
    centers=jnp.array([[-2.97]*dimension,[2.97]*dimension,[.137]*dimension,[-.243]*dimension])
    def run(local):
        def advance(state,x):
            cached=meta.energy(x,state) if local else None
            return meta.update(x,state,dense=not local,bias_energy=cached),None
        return jax.jit(lambda x:jax.lax.scan(advance,meta.init(),x)[0])(centers)
    local,dense=run(True),run(False)
    for a,b in zip(local,dense):np.testing.assert_allclose(a,b,rtol=3e-14,atol=3e-14)
    assert not bool(local.overflow) and int(local.count)==len(centers)
    # No contribution may be missed at the boundary or wrapped seam.
    probe=jnp.array([2.91]*dimension)
    a=jax.jit(jax.value_and_grad(meta.energy))(probe,local)
    b=jax.jit(jax.value_and_grad(meta.energy))(probe,dense)
    np.testing.assert_allclose(a[0],b[0],atol=5e-14)
    np.testing.assert_allclose(a[1],b[1],atol=5e-13)


def test_float32_local_deposition_and_cached_height():
    meta=GridMetadynamics((.1,.15),.2,2,(-3.,-3.),(3.,3.),(80,80),bias_factor=5.,kT=2.5)
    state=meta.init(jnp.float32)
    local=jax.jit(lambda x,s,e:meta.update(x,s,step=2,bias_energy=e))
    dense=jax.jit(lambda x,s:meta.update(x,s,step=2,dense=True))
    x=jnp.array([2.93,-2.96],dtype=jnp.float32)
    a=local(x,state,meta.energy(x,state));b=dense(x,state)
    for aa,bb in zip(a,b):
        assert aa.dtype==bb.dtype
        np.testing.assert_allclose(aa,bb,rtol=5e-6,atol=5e-6)
    assert a.values.dtype==a.derivatives.dtype==jnp.float32


def test_sparse_grid_checkpoint_continuation_bytes(tmp_path):
    meta=GridMetadynamics((.12,.15),.2,2,(-3.,-3.),(3.,3.),(80,80),bias_factor=5.,kT=2.5)
    centers=jnp.stack((jnp.linspace(-.3,.3,18),jnp.linspace(.2,-.2,18)),axis=1)
    whole,_=jax.jit(meta.replay)(centers)
    first,_=jax.jit(meta.replay)(centers[:7])
    path=tmp_path/'sparse.npz';checkpoint.save(path,first,meta)
    resumed,_=jax.jit(meta.replay)(centers[7:],checkpoint.load(path,meta))
    for a,b in zip(whole,resumed):assert np.asarray(a).tobytes()==np.asarray(b).tobytes()


def test_extreme_cutoff_falls_back_without_stencil_overflow():
    meta=GridMetadynamics((.2,),.2,1,(-1.,),(1.,),(20,),cutoff=1e308)
    x=jnp.array([.12]);state=meta.init()
    auto=jax.jit(lambda x,s:meta.update(x,s,bias_energy=0.))(x,state)
    dense=jax.jit(lambda x,s:meta.update(x,s,dense=True))(x,state)
    for a,b in zip(auto,dense):assert np.asarray(a).tobytes()==np.asarray(b).tobytes()
