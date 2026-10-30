"""Representative, offline PLUMED 2.10.1 value and derivative regressions."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

FIXTURES = Path(__file__).parent / 'fixtures'
MANIFEST = json.loads((FIXTURES / 'plumed-2.10.1.json').read_text())
ROWS = MANIFEST['rows']
import reference_cases as core


@lru_cache(None)
def case_map():
    return {(source, f['name'], c.name): c
            for source, fixtures in [('core', core.fixtures()), ('options', core.option_fixtures())]
            for f in fixtures for c in f['cases']}


@lru_cache(None)
def evaluator(source, kind, n, name, fixture):
    case = case_map()[source, fixture, name]
    def deformed(eps, x, h):
        transform = jnp.eye(3) + eps
        return case.fn(x @ transform, None if h is None else h @ transform)
    return jax.jit(jax.value_and_grad(deformed, argnums=(0, 1)))


@lru_cache(None)
def reference_arrays():
    with np.load(FIXTURES / 'plumed-2.10.1.npz', allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def test_reference_integrity_and_all_95_rows():
    assert hashlib.sha256((FIXTURES / 'plumed-2.10.1.npz').read_bytes()).hexdigest() == MANIFEST['sha256']
    table = MANIFEST['table_a1']
    assert [r['row'] for r in table] == list(range(1, 96))
    assert len({r['case'] for r in table}) == 95
    assert {r['case'] for r in table} <= {r['case'] for r in ROWS}
    assert {r['n_atoms'] for r in ROWS} == {5, 16, 64}
    for row in ROWS:
        assert (row['source'], row['fixture'], row['case']) in case_map()


@pytest.mark.parametrize('index,row', list(enumerate(ROWS)),
                         ids=[f"{r['case']}--{r['fixture']}" for r in ROWS])
def test_plumed_value_coordinate_and_affine_derivatives(index, row):
    data = reference_arrays()
    x = jnp.asarray(data[f'{index}_positions'])
    box = data[f'{index}_cell']
    h = None if box.size == 0 else jnp.asarray(box)
    # Reuse compiled baseline functions across cells/seeds of a given shape.
    fixture = (f"random-n{row['n_atoms']}-seed1701-nonperiodic"
               if row['kind'] == 'random' else row['fixture'])
    evaluate = evaluator(row['source'], row['kind'], row['n_atoms'], row['case'], fixture)
    value, (affine, gradient) = evaluate(jnp.zeros((3, 3)), x, h)
    expected = data[f'{index}_value']
    if row['periodic_value']:
        value = jnp.arctan2(jnp.sin(value - expected), jnp.cos(value - expected))
        expected = 0.
    np.testing.assert_allclose(value, expected, rtol=0, atol=2e-11)
    np.testing.assert_allclose(gradient, data[f'{index}_gradient'], rtol=0, atol=5e-10)
    np.testing.assert_allclose(affine, data[f'{index}_virial'], rtol=0, atol=5e-10)
