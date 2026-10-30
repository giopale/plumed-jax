"""The systematic manifest must actually exercise the requested sizes and groups."""
import numpy as np
import reference_cases as module


def test_random_clouds_are_reproducible_and_separated():
    for n in module.SIZES:
        for seed in module.SEEDS:
            a = module.cloud(n, seed)
            np.testing.assert_array_equal(a, module.cloud(n, seed))
            d = np.linalg.norm(a[:, None] - a[None, :], axis=-1)
            np.fill_diagonal(d, np.inf)
            assert a.shape == (n, 3) and d.min() > .45


def test_manifest_includes_every_size_seed_and_boundary_and_global_groups():
    fixtures = list(module.fixtures())
    baseline = [f for f in fixtures if f["kind"] == "random"]
    assert len(baseline) == 36
    assert len(fixtures) == 55
    assert sum(len(f["cases"]) for f in fixtures) == 1190
    for fixture in baseline:
        n = len(fixture["positions"])
        cases = {c.name: c for c in fixture["cases"]}
        for name in ("gyration", "coordination", "coordination_single", "com_distance", "rmsd_OPTIMAL", "path_0", "q6"):
            assert cases[name].atoms == tuple(range(n))
        if fixture["cell"] is not None:
            fractional = fixture["positions"] @ np.linalg.inv(fixture["cell"])
            assert np.all(fractional >= -1e-14) and np.all(fractional < 1 + 1e-14)
