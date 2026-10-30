import jax
import jax.numpy as jnp
import numpy as np

import plumed_jax as cv


def test_geometry_values_and_force_balance():
    xyz = jnp.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [1., 1., 1.]])
    assert np.isclose(cv.distance(xyz, 0, 1), 1.)
    assert np.isclose(cv.angle(xyz, 0, 1, 2), np.pi / 2)
    assert np.isclose(abs(cv.torsion(xyz, 0, 1, 2, 3)), np.pi / 2)
    for fn in (lambda x: cv.distance(x, 0, 1),
               lambda x: cv.angle(x, 0, 1, 2),
               lambda x: cv.torsion(x, 0, 1, 2, 3)):
        forces = -jax.jit(jax.grad(fn))(xyz)
        np.testing.assert_allclose(np.sum(forces, axis=0), 0., atol=1e-12)


def test_periodic_images_and_cell_gradient():
    xyz = jnp.array([[0.1, 0., 0.], [9.9, 0., 0.]])
    cell = jnp.diag(jnp.array([10., 10., 10.]))
    fn = jax.jit(lambda x, h: cv.distance(x, 0, 1, h))
    assert np.isclose(fn(xyz, cell), 0.2)
    assert np.isclose(cv.distance(xyz, 0, 1), 9.8)
    dx, dh = jax.grad(fn, (0, 1))(xyz, cell)
    np.testing.assert_allclose(dx[:, 0], [1., -1.], atol=1e-12)
    assert np.isclose(dh[0, 0], 1.)
    np.testing.assert_allclose(cv.position(xyz, 1, cell), [-0.1, 0., 0.], atol=1e-12)


def test_vmap_of_multiple_distances():
    xyz = jnp.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.]])
    pairs = jnp.array([[0, 1], [0, 2], [1, 2]])
    values = jax.jit(jax.vmap(lambda pair: cv.distance(xyz, pair[0], pair[1])))(pairs)
    np.testing.assert_allclose(values, [1., 2., np.sqrt(5.)], atol=1e-12)


def test_coordination_excludes_self_and_counts_unordered_pairs():
    xyz = jnp.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    cross = cv.coordination_number(xyz, [0, 1], [1, 2], r0=1.)
    single = cv.coordination_number(xyz, [0, 1, 2], r0=1.)
    raw = 2 / (1 + 1**6) + 1 / (1 + 2**6)
    at_max = 1.0 / (1.0 + 1e5)
    expected = (raw - 3 * at_max) / (1.0 - at_max)
    assert np.isclose(single, expected)
    assert np.isclose(cross, expected)
    grad = jax.jit(jax.grad(lambda x: cv.coordination_number(x, [0, 1, 2], r0=1.)))(xyz)
    np.testing.assert_allclose(np.sum(grad, axis=0), 0., atol=1e-12)
    overlap = jax.jit(jax.grad(
        lambda x: cv.coordination_number(x, [0, 1], [1, 2], r0=1.)))(xyz)
    assert np.isfinite(np.asarray(overlap)).all()


def test_general_rational_switch_is_finite_at_r0():
    xyz = jnp.array([[0., 0., 0.], [1.5, 0., 0.]])
    fn = lambda x: cv.coordination_number(x, [0], [1], r0=1.5, n=4, m=7,
                                           stretch=False)
    assert np.isclose(fn(xyz), 4 / 7)
    assert np.isfinite(np.asarray(jax.jit(jax.grad(fn))(xyz))).all()


def test_center_of_mass_rebuilds_an_ordered_molecule():
    xyz = jnp.array([[9.8, 0., 0.], [0.2, 0., 0.], [0.4, 0., 0.]])
    cell = jnp.eye(3) * 10.
    masses = jnp.array([1., 2., 1.])
    center = cv.center_of_mass(xyz, [0, 1, 2], masses, cell)
    assert np.isclose(center[0], 10.15)
    assert np.isclose(cv.radius_of_gyration(xyz, [0, 1, 2], cell=cell),
                      np.sqrt(np.mean((np.array([9.8, 10.2, 10.4]) - 10.133333333333333)**2)))


def test_proton_transfer_changes_sign_under_donor_swap():
    xyz = jnp.array([[0., 0., 0.], [1., 0., 0.], [3., 0., 0.]])
    f = jax.jit(lambda x: cv.proton_transfer(x, 0, 1, 2))
    assert np.isclose(f(xyz), -1.)
    assert np.isclose(f(xyz), -cv.proton_transfer(xyz, 2, 1, 0))
