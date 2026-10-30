import jax
import jax.numpy as jnp
import numpy as np
import pytest

from plumed_jax import GridMetadynamics, Metadynamics, checkpoint


def config(grid, factor=5., capacity=24):
    options=dict(sigma=(.17,.21),height=.3,pace=2,
                 periods=(0.,2*np.pi),bias_factor=factor,kT=2.5)
    return (GridMetadynamics(**options,grid_min=(-2.,-np.pi),
            grid_max=(2.,np.pi),grid_bin=(80,96)) if grid else
            Metadynamics(**options,capacity=capacity))


def same_bytes(a,b):
    for x,y in zip(a,b):
        assert np.asarray(x).dtype==np.asarray(y).dtype
        assert np.asarray(x).tobytes()==np.asarray(y).tobytes()


@pytest.mark.parametrize('grid',[False,True])
@pytest.mark.parametrize('factor',[None,5.])
def test_one_walker_preserves_existing_bytes(grid,factor):
    m=config(grid,factor)
    x=jnp.array([.12,3.13]); state=m.init()
    energy=m.energy(x,state)
    single=jax.jit(m.update)(x,state,step=2,bias_energy=energy)
    batch=jax.jit(m.update_walkers)(x[None,:],state,step=2,bias_energies=energy[None])
    same_bytes(single,batch)


@pytest.mark.parametrize('grid',[False,True])
@pytest.mark.parametrize('dtype',[jnp.float32,jnp.float64])
def test_shared_predeposition_heights_and_conservative_forces(grid,dtype):
    m=config(grid)
    state=m.update(jnp.array([.1,3.12],dtype=dtype),m.init(dtype),step=2)
    x=jnp.array([[.13,3.10],[.14,-3.12],[.16,3.09]],dtype=dtype)
    energies=jax.vmap(m.energy,in_axes=(0,None))(x,state)
    expected=m.height*jnp.exp(-energies/(m.kT*(m.bias_factor-1)))
    update=jax.jit(m.update_walkers)
    a=update(x,state,step=4,bias_energies=energies)
    b=update(x,state,step=4)
    tolerance=3e-6 if dtype==jnp.float32 else 3e-14
    for aa,bb in zip(a,b):np.testing.assert_allclose(aa,bb,atol=tolerance,rtol=tolerance)
    assert int(a.count)==4 and int(a.step)==4 and not bool(a.overflow)
    assert (a.values.dtype if grid else a.heights.dtype)==dtype
    if grid:
        # Add ordinary hills in the same order using independently fixed heights.
        expected_state=state
        for v,h in zip(x,expected):
            ordinary=GridMetadynamics(m.sigma,float(h),1,m.grid_min,m.grid_max,
                m.grid_bin,periods=m.periods)
            expected_state=ordinary.update(v,expected_state)
        np.testing.assert_allclose(a.values,expected_state.values,atol=tolerance,rtol=tolerance)
        np.testing.assert_allclose(a.derivatives,expected_state.derivatives,atol=tolerance,rtol=tolerance)
        np.testing.assert_allclose(a.last_height,expected.sum(),atol=tolerance)
    else:
        np.testing.assert_allclose(a.heights[1:4],expected,atol=tolerance,rtol=tolerance)
    probe=jnp.array([.111,3.08],dtype=dtype)
    gradient=jax.jit(jax.grad(m.energy))(probe,a)
    if dtype==jnp.float64:
        for axis in range(2):
            h=1e-6
            numerical=(m.energy(probe.at[axis].add(h),a)-m.energy(probe.at[axis].add(-h),a))/(2*h)
            np.testing.assert_allclose(gradient[axis],numerical,atol=2e-8)
    assert np.all(np.isfinite(gradient))


@pytest.mark.parametrize('grid',[False,True])
def test_atomic_invalid_batch_and_step_accounting(grid):
    m=config(grid,capacity=2)
    state=m.init(); x=jnp.array([[.1,3.1],[.2,-3.1],[.3,3.12]])
    skipped=jax.jit(m.update_walkers)(x,state,step=1)
    assert int(skipped.count)==0 and not bool(skipped.overflow)
    bad=m.update_walkers(x.at[1,0].set(jnp.nan),state,step=2)
    assert bool(bad.overflow) and int(bad.count)==0
    if not grid:
        full=m.update_walkers(x,state,step=2)
        assert bool(full.overflow) and int(full.count)==0
    else:
        bad=m.update_walkers(x.at[1,0].set(3.),state,step=2)
        assert bool(bad.overflow) and int(bad.count)==0
        full=m.update_walkers(x,state._replace(count=jnp.int32(2**31-2)),step=2)
        assert bool(full.overflow) and int(full.count)==2**31-2
    repeated=m.update_walkers(x,state,step=0)
    assert bool(repeated.overflow) and int(repeated.count)==0
    for invalid in (jnp.ones(2),jnp.ones((0,2)),jnp.ones((3,1))):
        with pytest.raises(ValueError):m.update_walkers(invalid,state)
    with pytest.raises(ValueError):m.update_walkers(x,state,bias_energies=jnp.ones((3,1)))
    with pytest.raises(ValueError):m.update_walkers(x,state,step=jnp.array([2]))


@pytest.mark.parametrize('grid',[False,True])
def test_batch_checkpoint_resume_and_detached_deposition(grid,tmp_path):
    m=config(grid)
    trajectory=jnp.array([[[.1,3.1],[.2,-3.1]],[[.13,3.05],[.23,-3.05]],
                          [[.15,3.0],[.25,-3.0]],[[.17,2.95],[.27,-2.95]]])
    def run(x,s):
        return jax.lax.scan(lambda s,v:(m.update_walkers(v,s),None),s,x)[0]
    compiled=jax.jit(run)
    whole=compiled(trajectory,m.init())
    part=compiled(trajectory[:2],m.init())
    path=tmp_path/'walkers.npz'; checkpoint.save(path,part,m)
    same_bytes(part,checkpoint.load(path,m))
    resumed=compiled(trajectory[2:],checkpoint.load(path,m))
    same_bytes(whole,resumed)
    field='values' if grid else 'heights'
    fn=lambda x:jnp.sum(getattr(m.update_walkers(x,m.init(),step=2),field))
    np.testing.assert_array_equal(jax.jit(jax.grad(fn))(trajectory[0]),np.zeros((2,2)))
