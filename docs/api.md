# Python API reference

## Common conventions

- `Array` means a JAX array; `ArrayLike` includes numeric scalars and NumPy/JAX
  arrays. Shapes below describe one configuration unless a method explicitly
  accepts walkers. Use `jax.vmap` to evaluate batches.
- Atomic positions have shape `(N, 3)`, where `N` is the number of atoms.
  Python indices start at zero; parsed PLUMED indices start at one. Choose
  selections before compilation. Direct CV functions generally **assume valid
  indices**; the parser checks bounds when evaluating a program.
- Cells have shape `(3, 3)`, with lattice vectors as rows. `None` disables
  periodic wrapping. CVs also accept `geometry.PreparedCell` returned by
  `prepare_cell`. Rebuild it when the cell changes.
- Choose consistent length and energy units. Angles are radians. The functional
  API does not convert units. Parsed defaults are nm, kJ/mol, ps; the Python
  `Units` record defaults to Å, eV, ps. Declaring units does not convert arrays.
- Configuration flags and selections described as **static** must be fixed
  before `jax.jit`, usually by capturing them in a function. Coordinates, cells,
  and bias states remain runtime inputs. Forces are minus energy gradients.
- Function docstrings and type hints give the complete signatures, preconditions,
  errors, and JIT requirements. Type hints do not add runtime validation.

## Collective variables

All functions below are exported from `plumed_jax`.

| Function | Selection/configuration | Output shape and units |
|---|---|---|
| `distance` | Atom indices `i, j` | Scalar length |
| `position` | Atom index `i` | `(3,)` closest-image position relative to origin; length |
| `angle` | Atom indices `i, j, k` | Scalar angle in `[0, π]`; radians |
| `torsion` | Atom indices `i, j, k, l` | Scalar signed angle in `[-π, π]`; radians |
| `proton_transfer` | `donor_a, proton, donor_b` | Scalar distance difference; length; negative nearer donor A |
| `center_of_mass` | Ordered `indices`, per-atom `masses` | `(3,)` mass-weighted position; length |
| `radius_of_gyration` | Ordered `indices`, optional per-atom `masses` | Scalar length |
| `contact_values` | Explicit `(P, 2)` atom pairs; rational switch parameters | `(P,)` dimensionless contact weights; `P` is pair count |
| `coordination_number` | `group_a`, optional `group_b`; rational parameters | Scalar dimensionless contact sum |
| `contact_matrix` | `group_a`, optional `group_b`; `switching` options | `(A, B)` dimensionless weights; `A/B` are group sizes |
| `switching` | Distances and static switch configuration | Same shape as distances; dimensionless |
| `rmsd` | Selected atoms, reference, alignment/displacement weights | Scalar length; length squared with `squared=True` |
| `path_msd` | Reference frames, positive `lam` in inverse length squared | `(s, z)`: scalar dimensionless frame label and scalar squared-length distance |
| `steinhardt` | Centers, neighbors, static degree `l`, normalization/method | `(C,)` dimensionless order values; `C` is center count |
| `tetrahedral_order` | One center and exactly four neighbors | Scalar dimensionless water order |

COM, gyration, and reference alignment rebuild periodic molecules in the supplied
atom order; that order must connect the intended molecule. Selections are not
inferred from connectivity or nearest-neighbor sorting. Undefined derivatives
at coincident atoms, collinear bonds, or nonunique alignments are not regularized.

## Bias functions and energy composition

| Function | Contract |
|---|---|
| `periodic_difference` | Broadcast signed differences; positive periods wrap, zero means nonperiodic |
| `harmonic_restraint` | Scalar summed energy; centers/periods use CV units, `kappa` energy/CV-unit², `slopes` energy/CV-unit |
| `wall` | Scalar upper/lower power-law energy; centers/offset/scale use CV units, `kappa` energy units; no periodic wrapping |
| `make_biased_energy` | Returns a callable adding base and bias energies; forwards arguments only to the base potential; reads optional `bias_state` without updating it |

## Metadynamics configuration and explicit state

`Metadynamics` stores a direct hill history; `GridMetadynamics` stores accumulated
bias values and derivatives on a fixed one-to-three-dimensional grid. Both are
frozen configurations. Each CV's `sigma`, bounds, and period use that CV's units;
`height` and `kT` use energy units. `pace` counts completed integration steps.
`capacity` is allocated hills for the direct history; `grid_bin` counts grid
intervals for the grid. `bias_factor=None` selects ordinary metadynamics.

| Method | Inputs and return |
|---|---|
| `init(dtype=...)` | Returns fresh `HillState` or `GridState`; float64 requires JAX x64 enabled |
| `energy(values, state)` | One CV vector of shape `(D,)` → scalar bias energy; does not update state; `D` is the CV count |
| `update(values, state, ...)` | Returns the next state; explicit `step` must increase, otherwise advances by one |
| `update_walkers(values, state, ...)` | `(W, D)` CVs → one shared next state; `W` walkers deposit synchronously at one step |
| `replay(values, state=None, ...)` | A `(T, D)` CV trajectory → `(final_state, energies)`; `energies` has shape `(T,)`; `T` is step count |
| `free_energy(values, state)` | Scalar asymptotic negative-bias estimate; finite sampling need not be converged |
| `GridMetadynamics.offset(state)` | Scalar time-dependent `c(t)` normalization in energy units; requires `kT` |

**Evaluate energy and forces before depositing the current hill.**
An optional `bias_energy` passed to `update` must be that individual bias's
energy at the same CV values and pre-deposition state. Fix `dense` and
`reuse_bias` algorithm flags before JIT. Array shapes remain fixed.

State fields are documented on `HillState` and `GridState`. Both have a sticky
`overflow` flag: check it on the host after a compiled block. Grid energies
outside nonperiodic bounds return NaN; periodic coordinates wrap. Native
checkpoints preserve bytes, but neither interpolation nor floating-point
arithmetic promises universal byte equality across devices.

### Shared bias across walkers

`update_walkers` accepts a fixed walker count and optional `bias_energies` of
shape `(W,)`. Heights for every walker are calculated from the **same old bias**,
then added in walker order. A due update advances `count` by `W`, and advances
`step` once. Grid `last_height` is the sum of deposited heights. One walker
delegates to the existing single-walker update. `height` remains per walker.
Invalid due batches are rejected as a whole; no partial set of hills is written.
Grid coordinates must also be valid on steps without deposition, following the
single-walker grid contract. Check sticky `overflow` on the host.

```python
# values: (walkers, CVs); shared is one GridState or HillState.
energies = jax.vmap(meta.energy, in_axes=(0, None))(values, shared)
shared = meta.update_walkers(values, shared, step=step, bias_energies=energies)
```

Batch the independent engine states with `vmap`, while broadcasting the shared
bias to every force evaluation. Refresh all cached engine forces after depositing.
Walkers must use the same CV definitions and units; the usual multiple-walker
setup also uses the same physical potential and temperature with independent
positions, momenta and random keys.
The grid's `dense` option is also accepted by `update_walkers` and remains static.
There are no configuration exchanges or distributed communications in this API.
Checkpoint the shared bias and all engine states together for a restart.

## OPES_METAD

`OPESMetad` follows PLUMED 2.10's `OPES_METAD`. The bias is
`V(s) = (1-1/γ) kT log(P(s)/Z + ε)`, where `P` is a weighted density estimate
built from truncated Gaussian kernels that are compressed on the fly.
`OPESState` holds the kernels and running sums. Kernel slots are fixed by
`capacity`.

| PLUMED keyword | `OPESMetad` field |
|---|---|
| `ARG` periodicity | `periods` (and optional `periodic_min` for the domain lower bound) |
| `SIGMA` | `sigma` tuple, or `"adaptive"` with `dimension` |
| `BARRIER`, `PACE`, `TEMP` | `barrier`, `pace`, `kT` (energy units) |
| `BIASFACTOR` | `bias_factor`; `None` uses `barrier/kT`, `float("inf")` is a uniform target |
| `EPSILON`, `KERNEL_CUTOFF` | `epsilon`, `kernel_cutoff`; `None` uses the barrier-derived defaults |
| `COMPRESSION_THRESHOLD`, `SIGMA_MIN`, `ADAPTIVE_SIGMA_STRIDE` | same names in lowercase |
| `FIXED_SIGMA`, `NO_ZED`, `CALC_WORK`, `RECURSIVE_MERGE_OFF` | `fixed_sigma`, `no_zed`, `calc_work`, `recursive_merge=False` |
| `WALKERS_MPI` with W replicas | `walkers=W` and `update_walkers` |
| `EXTRA_BIAS` | `extra_bias` / `extra_biases` argument of `update` / `update_walkers` |

| Method | Inputs and return |
|---|---|
| `init(dtype=...)` | Fresh `OPESState`; the bias starts at `kT(1-1/γ) log ε` (−BARRIER by default) |
| `energy(values, state)` | `(D,)` CVs → scalar bias; differentiate for forces |
| `update(values, state, step=, bias_energy=, extra_bias=)` | One-walker deposit/compression; requires `walkers=1` |
| `update_walkers(values, state, ...)` | `(W, D)` CVs → next shared state, synchronous like WALKERS_MPI |
| `replay(values, state=None)` | `(T, D)` or `(T, W, D)` CV trajectory → `(final_state, pre-deposition energies)` |
| `probability`, `free_energy` | Normalized density estimate; `-V/(1-1/γ)` regularized free-energy estimate |
| `diagnostics(state)` | PLUMED components `rct`, `zed`, `neff`, `nker` (and `work`) |

The step convention is the same as for metadynamics. `init()` is PLUMED's state
after its skipped step zero, and step `n` deposits when `n % pace == 0`. Reweight
samples with `exp(V/kT)`; OPES needs no `c(t)` correction. With adaptive sigma,
call `update` at every step, because the CV variance is measured on every step.

**Walkers.** As in WALKERS_MPI, every walker's weight uses its own bias from the
same old state. The summed weights set `n_eff` and the shrunk sigma, kernels are
added and compressed in walker order, and `Z` is updated once. `counter`
advances by `W`. Adaptive-sigma statistics are kept per walker. Walkers run in
one process. Batch their engines with `vmap`, as in `examples/opes.py`.

**Capacity and failures.** A deposit that would need a new kernel slot when
none is free, or that produces nonfinite values, writes nothing, not even the
weight sums, and sets the sticky `overflow` flag. A PLUMED run with adaptive
sigma aborts when the first measured sigma is below 1e-6 and no `SIGMA_MIN` is
set; here the same case sets `overflow` instead.

**Restart.** `checkpoint.save/load` store every state array, including the
per-walker adaptive statistics. Resuming is byte-identical to an uninterrupted
run. PLUMED STATE and KERNELS text files are not read or written.

Not implemented: `OPES_METAD_EXPLORE`, `OPES_EXPANDED`, `NLIST`,
`EXCLUDED_REGION`, `STORE_STATES` and file I/O. `Z` is recomputed in full,
O(K²), at each deposit. PLUMED switches to an incremental update when there are
many kernels; the two differ only in rounding.

## Checkpoints and reweighting

`plumed_jax.checkpoint.save(path, state, config)` and `load(path, config)` perform
host file operations outside JIT. Loading checks configuration, shapes, and
precision for hill, grid, and OPES states. `metadynamics_offset(bias_grid, kT=..., bias_factor=...)` is the
standalone reweighting normalization for a uniform one-dimensional array of
grid-point bias values, including flattened multidimensional grids.

## Parsed programs

`from plumed_jax.parser import parse` is the text entry point.
`parse(text, base_dir='.', hill_capacity=1024, walkers=1)` constructs a `Program`
outside JIT. `base_dir` resolves reference PDB paths; `hill_capacity` applies to
direct METAD actions and OPES_METAD kernels. `walkers` is the replica count for
OPES_METAD actions that declare `WALKERS_MPI`. Unsupported actions and options raise errors.

| Method | Return and state behavior |
|---|---|
| `initial_state()` | Dictionary from each METAD/OPES_METAD label to an empty bias state |
| `values(positions, cell=None, ...)` | Dictionary of CV arrays keyed by action/component label; requires masses for COM |
| `bias(positions, cell=None, state=..., ...)` | Scalar total bias energy; requires explicit history for every METAD action |
| `update(positions, state, cell=None, ...)` | Next METAD state dictionary; no mutation of input state |
| `bias_and_update(positions, state, cell=None, ...)` | `(pre_deposition_energy, next_state)`; shares CV and individual bias evaluations |
| `update_walkers(positions, state, cell=None, ...)` | `(W, N, 3)` positions → next shared state for every METAD/OPES_METAD action |
| `bias_and_update_walkers(positions, state, cell=None, ...)` | `((W,) pre-deposition energies, next shared state)`; differentiate the summed energies for per-walker forces |

Compile `jax.value_and_grad(program.bias_and_update, has_aux=True)` to receive
`((energy, next_state), gradient)`. State is auxiliary output; forces use the old
history. `share_geometry=True` is an optional static choice with different
derivative rounding. Editing program definitions requires rebuilding compiled
callables. PRINT/FILE/FMT options describe output; the caller writes files.

`CVMetadata` describes names, component units, and periods without calculating
CVs. `Units.convert_length` and `convert_energy` explicitly scale scalar/array
values to a supported target unit.

## Supported input options

- `UNITS`: LENGTH=A/angstrom/nm/bohr; ENERGY=eV/kj/mol/kcal/mol; TIME=ps/fs.
- `GROUP`: static atom lists and increasing ranges.
- `DISTANCE`, `ANGLE`, `TORSION`: explicit atoms or COM virtual atoms; NOPBC.
- `POSITION`: Cartesian `.x/.y/.z`; `COM`: engine masses and ordered atom selection.
- `GYRATION`: TYPE=RADIUS with equal weights.
- `COORDINATION`: static GROUPA/GROUPB, simplified rational parameters or the
  documented switch set. Non-rational switches require explicit D_MAX.
- `RMSD`: one PDB reference, SIMPLE/OPTIMAL, SQUARED, NOPBC.
- `PATHMSD`: multi-frame PDB reference and positive LAMBDA; `.sss/.zzz`.
- `COMBINE`: coefficients, parameters, powers and NORMALIZE.
- `CUSTOM`: up to three scalar arguments (x,y,z), pi, arithmetic operators,
  sin/cos/tan/exp/log/sqrt/abs/tanh and two-argument min/max.
- `CONSTANT`: VALUE or VALUES.
- `RESTRAINT`, `UPPER_WALLS`, `LOWER_WALLS`: the parameters in the direct API.
- `METAD`: ARG, SIGMA, HEIGHT, PACE, optional BIASFACTOR/TEMP; FILE/FMT are
  descriptive metadata. Direct-history capacity is a parser construction argument.
  Optional GRID_MIN/GRID_MAX and GRID_BIN/GRID_SPACING select a 1–3D bias grid.
  Defaults use sigma/5 spacing; both bin and spacing options select the finer grid.
  Native grid c(t) is available through the API; CALC_RCT/RCT_USTRIDE input options,
  GRID_WFILE/GRID_RFILE and sparse/no-spline grid variants remain unsupported.
- `OPES_METAD`: ARG, PACE, BARRIER, TEMP (required: the parser has no engine
  temperature), SIGMA (numbers or ADAPTIVE, the default), SIGMA_MIN,
  ADAPTIVE_SIGMA_STRIDE, BIASFACTOR (number or inf), EPSILON, KERNEL_CUTOFF,
  COMPRESSION_THRESHOLD, FIXED_SIGMA, RECURSIVE_MERGE_OFF, NO_ZED, CALC_WORK,
  WALKERS_MPI, and SERIAL (ignored). FILE/FMT are descriptive metadata.
  STATE_RFILE/STATE_WFILE, NLIST, EXCLUDED_REGION and EXTRA_BIAS raise errors;
  use NPZ checkpoints and the direct API's `extra_bias` instead.
- `PRINT`: ARG/FILE/STRIDE/FMT are descriptive metadata. The caller implements
  output scheduling/writing; parsing does not produce files.

`COMBINE` and `CUSTOM` require PERIODIC=NO or a lower,upper pair. Bare flags
must not be written as `NOPBC=false`. Supplied parameter lists require exactly
one entry per ARG. Atoms start at one and must lie within the engine array;
JAX's out-of-range index clamping is not accepted as input semantics.
Unknown actions/options, unsafe expressions and forward scalar dependencies
raise errors. This is a strict useful subset, not a drop-in PLUMED input reader.
`Program.bias_and_update` combines evaluation and deposition with shared CVs and
separate cached energies for each METAD action. The direct APIs expose
`bias_energy` reuse; dense/uncached modes remain available for byte comparisons.

## Scope limits

Selections and array sizes are static. OPES is limited to OPES_METAD (no
EXPLORE/EXPANDED variants). There is no neighbor-list builder, adaptive hill covariance, distributed walker communication, or PLUMED HILLS/grid
file restart. RMSD supports SIMPLE/OPTIMAL alignment and optional squaring;
PATHMSD uses optimal squared RMSD. Q_l has conventional and PLUMED normalization;
angular water order requires four explicit neighbors and is not PLUMED's
TETRAHEDRAL action. Direct functions and parser actions have different scopes:
contact matrices and order functions are available through Python, while the
parser accepts only the actions listed above. Function docstrings specify
individual arguments, defaults, and exclusions.
