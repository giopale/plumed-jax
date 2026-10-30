# PLUMED-JAX

plumed-jax is an AI-assisted adaptation of selected [PLUMED](https://www.plumed.org/) functionality for JAX-based simulations. It builds on decades of scientific insight, careful implementation, and sustained community effort by PLUMED’s developers and contributors, whose work I deeply appreciate.

This is not an improvement over PLUMED. It makes a small subset of its utilities available to people using JAX for atomistic simulations who are limited by the short timescales accessible to molecular dynamics. My hope is that it will be helpful to them.

The accompanying article exists to demonstrate how the port was validated, with implementation details and a few examples. I do not feel it warrants a citation, as the intellectual credit belongs to the people whose work made this port possible. If you use or build upon plumed-jax, I would be grateful if you acknowledged PLUMED’s developers and contributors and cited the original PLUMED papers and the methods you use, rather than this article. See [PLUMED’s citation guidance](https://www.plumed.org/cite).

## Installation

From a checkout (Python 3.10 or newer):

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

For development use `python -m pip install -e '.[test]'`.
CPU JAX is installed by default. Use a native Python build for your architecture.
For NVIDIA GPUs on Linux, first install the appropriate JAX accelerator package
in the same environment, then install this library. For supported CUDA 13 systems:

```sh
python -m pip install --upgrade 'jax[cuda13]'
python -m pip install .
python -c 'import jax; print(jax.devices())'
```

Check [JAX's installation guide](https://docs.jax.dev/en/latest/installation.html)
for driver/platform requirements and CUDA 12 alternatives. Installing this
library alone does not enable GPU execution. No JAX-MD, ASE, or molecular
potential is required; engines supply their own physical energy and integrator.

## Using plumed-jax

The functions below add a distance umbrella to a small JAX potential.
Lengths are in Å and energies in eV:

```python
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from plumed_jax import distance, harmonic_restraint

positions = jnp.array([[0., 0., 0.], [1.3, 0.1, 0.]])

def total_energy(r):
    d = distance(r, 0, 1)  # no periodic cell
    physical = 0.5 * (d - 1.0)**2
    umbrella = harmonic_restraint(d, centers=1.2, kappa=2.0)
    return physical + umbrella

energy, gradient = jax.jit(jax.value_and_grad(total_energy))(positions)
forces = -gradient
print('energy (eV):', energy)
print('forces (eV/Å):', forces)
```

Small runnable examples show umbrella sampling, well-tempered metadynamics,
and multiple walkers sharing a bias:

```sh
python examples/umbrella.py
python examples/well_tempered.py
python examples/multiple_walkers.py
```

These use a toy potential and 20 integration steps. They show pre-deposition
forces, hill ordering, and force refresh after every bias update. They are
usage demonstrations, not converged sampling calculations.

## Conventions and limits

- Positions are `(N, 3)`; Python atom indices start at zero, parsed input at one.
  Cell vectors are rows of `(3, 3)`; `cell=None` disables PBC.
- Use consistent units throughout the engine and bias. The functional API does
  no conversion. Angles are radians. Parsed defaults are nm, kJ/mol, ps;
  declaring `UNITS` does not convert coordinates or cells. PDB references use Å.
- Harmonic energy is `0.5*kappa*(CV-center)**2`, plus an optional linear slope.
  Forces are minus coordinate gradients. Joint affine derivatives deform both
  coordinates and cell; they are not bare cell gradients.
- Enable `jax_enable_x64` before creating arrays for reference-level precision.
  Float32 is supported but has different error bounds. Cross-backend bitwise
  equality is not promised.
- Choose atom selections, reference shapes, capacities, grid bounds, and
  configuration flags before JIT. Pass positions, cells, and explicit bias
  state as runtime arrays. Keep integration/deposition in `jax.lax.scan` for
  long compiled loops; check sticky `state.overflow` on the host.
- A completed step reads the old bias, then deposits, then refreshes cached
  forces. Cached well-tempered update energies must be from that individual
  bias at the current CV and old state. Shared walkers deposit once per step
  using the same old bias; hill height is per walker.
- Minimum-image derivatives follow the selected image. Image ties, coincident
  atoms, collinear geometry, zero unsquared RMSD, and nonunique alignment can
  be nonsmooth. Very ill-conditioned triclinic cells can be expensive.

## Available functionality

The following subset is available through JAX functions. PLUMED 2.10.1 is used
as a numerical reference and is not needed at runtime.

| Family | Scope |
|---|---|
| Geometry | Distance, Cartesian position, angle, torsion, ordered COM, gyration, proton-transfer distance difference |
| Contacts | Rational, EXP, GAUSSIAN, TANH, SMAP, COSINUS, CUBIC switches; coordination and static contact matrices |
| Reference/order | Weighted SIMPLE/OPTIMAL RMSD, PATHMSD, Steinhardt Q_l, explicit four-neighbor angular water order |
| Bias/sampling | Harmonic restraints/slopes, upper/lower walls, ordinary/well-tempered metadynamics, 1–3D grids, synchronous shared walkers, NPZ checkpoints |
| Composition | Strict optional text parser, CONSTANT, COMBINE, restricted CUSTOM expressions |

Support is limited to the documented options. Unsupported functionality includes
general PLUMED input compatibility, dynamic atom selection, neighbor-list construction,
OPES, adaptive hills, HILLS text restart, distributed walker synchronization, or
PLUMED file-output scheduling. Water angular order is not PLUMED's TETRAHEDRAL
action. Q_l normalization and rational cutoff defaults must be chosen explicitly
when comparing different PLUMED actions.

The [API guide](docs/api.md) covers function contracts, state handling, and the
supported input options. Public functions also document their arguments in
Python docstrings.

## Testing and development

```sh
python -m pip install -e '.[test]'
JAX_ENABLE_X64=true python -m pytest -q
```

Tests are self-contained and require no PLUMED installation. The frozen PLUMED
2.10.1 fixtures compare values, coordinate derivatives, and joint coordinate/cell
deformation derivatives for all 95 scalar outputs and option combinations.
The exact mapping, inputs, and provenance are in
`tests/fixtures/plumed-2.10.1.json`. These 494 representative comparisons are
separate from the full article benchmark.

The installed-wheel suite passed 595 tests on ARM macOS with Python 3.11.6,
JAX/jaxlib 0.10.2, NumPy 2.4.6, and pytest 9.2.0. One JAX future warning concerns
float64-to-float32 scatter casting in the float32 walker test with x64 enabled.
This release check did not rerun GPU verification; earlier A6000 runs covered
1,190 baseline and 2,232 additional comparisons, with one extrema failure
followed by a successful 72-case corrected min/max rerun. A CPU pass does not
establish GPU correctness. CI checks wheel installation and tests on Python
3.10/3.12; other dependency combinations are not all verified.

Implementation lives in `src/plumed_jax/`: geometry/CV/reference/order functions,
bias and metadynamics state, and the optional parser. Contributions should stay
focused and include independent value and coordinate/affine derivative tests,
explicit units, supported options, and rejected inputs. Keep fixtures small and
preserve numerical conventions and hill ordering.

Licensed under [Apache 2.0](LICENSE). See [NOTICE](NOTICE) for PLUMED attribution.
