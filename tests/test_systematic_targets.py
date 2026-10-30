"""Check that targeted validation inputs exercise their stated conditions."""
import itertools
import jax
import jax.numpy as jnp
import numpy as np
import reference_cases as module


def test_targets_are_reproducible_and_nonsingular():
    first, second = module.targets(), module.targets()
    assert len(first) == 19
    assert len({fixture["name"] for fixture in first}) == len(first)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left["positions"], right["positions"])
        assert left["positions"].shape == (5, 3)
        np.testing.assert_array_equal(left["masses"], right["masses"])
        x = jnp.asarray(left["positions"])
        h = None if left["cell"] is None else jnp.asarray(left["cell"])
        for case in left["cases"]:
            value, gradient = jax.value_and_grad(case.fn)(x, h)
            assert np.isfinite(value), (left["name"], case.name)
            assert np.all(np.isfinite(gradient)), (left["name"], case.name)
            if case.name in left["expected_values"]:
                np.testing.assert_allclose(value, left["expected_values"][case.name], atol=2e-12, rtol=1e-10)


def test_skew_case_requires_more_than_fractional_rounding():
    case = next(item for item in module.targets() if item["name"] == "skew_nearest_image")
    cell = case["cell"]
    displacement = case["positions"][1] - case["positions"][0]
    rounded = displacement - np.round(displacement @ np.linalg.inv(cell)) @ cell
    shifts = np.array(list(itertools.product(range(-3, 4), repeat=3))) @ cell
    nearest = np.min(np.linalg.norm(displacement - shifts, axis=1))
    assert np.linalg.norm(rounded) > 2 * nearest
    np.testing.assert_allclose(case["expected_values"]["distance"], nearest, atol=1e-14)
