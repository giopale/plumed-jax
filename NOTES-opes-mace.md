# OPES in plumed-jax + MACE on Daint — where things stand

Branch: `claude/sleepy-ptolemy-0wu9j4` on `giopale/plumed-jax` (latest: `22a09b7`).

## Done and verified

- **OPES_METAD in plumed-jax** (`src/plumed_jax/opes.py`), following PLUMED 2.10's
  `OPESmetad.cpp`. It includes kernel compression and recursive merging, sigma
  shrinking, `SIGMA=ADAPTIVE`, `SIGMA_MIN`, `NO_ZED`, `FIXED_SIGMA`,
  `BIASFACTOR=inf` and `CALC_WORK`. Walkers sharing one bias (`update_walkers`)
  behave like `WALKERS_MPI`. Restarts use native NPZ checkpoints. The parser
  accepts `OPES_METAD`, and `parse(..., walkers=W)` sets the walker count.
- **Forces match PLUMED by default.** PLUMED's OPES force is not the exact
  gradient of its own bias near the kernel cutoff, and plumed-jax reproduces that
  force. Use `exact_gradient=True` for conservative forces.
- **Validated against PLUMED 2.10.0** on Daint. The fixture is
  `tests/fixtures/opes-plumed.npz`, generated with
  `tests/fixtures/generate_opes_reference.py`. All 8 cases pass: bias, atomic
  forces, rct/zed/neff/nker/work at every step, and the final STATE, including
  the 3-walker MPI cases. The full suite has 640+ tests, all passing.
- **MACE driver** `examples/mace_opes_walkers.py` (JAX-MD fork `working-branch` +
  MACE-JAX):
  - It converts the PyTorch `.model` on the fly and reads one LAMMPS data file per
    walker. It runs Nosé–Hoover NVT with all walkers sharing one OPES bias, and
    adapts the LAMMPS/PLUMED input automatically.
  - `--check` agrees with PyTorch MACE (float64): maximum force difference
    1.7e-5 eV/Å, energy difference 0.004 meV/atom.
  - The **multi-image neighbor list is required**: the cell heights are 10.4 and
    11.4 Å, less than 2·(6 + skin) Å. The minimum-image list gave 0.1 eV/Å force
    errors.
  - Speed with `--cueq fused` (the default): about 6.7× faster than without.

  | Setup | ms/step | ns/day per walker | ns/day aggregate |
  |---|---|---|---|
  | 2 walkers, 1 GPU, cueq off | 90.5 | 0.48 | 0.95 |
  | 2 walkers, 1 GPU, cueq fused | 13.6 | 3.2 | 6.4 |
  | 4 walkers, 1 GPU, cueq fused | 24.5 | 1.8 | 7.1 |

  One GPU is nearly saturated at about 2 walkers of 408 atoms, so you need all
  4 GPUs for 16 walkers.

## Open problem: `--shard` (4 GPUs)

- The first version, which placed the arrays with `device_put`, did not split the
  work. With 4 walkers it was slower than one GPU (37 ms/step), and with too many
  walkers it ran out of memory on GPU_0.
- It is now rewritten with `pmap`: each GPU integrates its own walkers, and the
  positions are all-gathered for the shared OPES update (`a4c2294`). The OPES
  part was checked on 4 simulated CPU devices: the state is identical on all
  devices and equal to the 1-device result.
- **On Daint, the `pmap` version died silently** for both 4 and 16 walkers: no
  output after the warnings, no traceback. This was likely a hard crash or a
  kill. `22a09b7` adds line-buffered output and `faulthandler`, so the next run
  shows where it stops.

### Next time, start here

1. Update the branch and confirm the version:
   ```bash
   cd ~/scripts/plumed-jax && LD_LIBRARY_PATH=${LD_LIBRARY_PATH#$CONDA_PREFIX/lib:} git pull --rebase && git log --oneline -1
   ```
   The last line should be `22a09b7` or newer.
2. Rerun the 4-walker sharded test, unfiltered, and record the exit code:
   ```bash
   S=/ritom/scratch/cscs/pgiorgio/LAMMPS/000-structures-start/selected; BASE="--model /ritom/scratch/cscs/pgiorgio/MLIPS_MACE/atio2-h2o-neutral-mace-pbe0-0.410-opes/iters-000/train-mace/tasks/000/mace_model.model --type-map H,O,Ti --plumed /ritom/scratch/cscs/pgiorgio/LAMMPS/MLIPS_DP/atio2-h2o-neutral-mace-pbe0-0.410-opes/A/000/plumed.plumed --steps 500"
   ```
   ```bash
   python examples/mace_opes_walkers.py $BASE --shard --data $S/cn1/00000[0-3].lammps.data --outdir /ritom/scratch/cscs/pgiorgio/test-mace-opes-w4-shard > /ritom/scratch/cscs/pgiorgio/test-mace-opes-w4-shard.log 2>&1; echo "exit code $?"
   ```
   ```bash
   tail -40 /ritom/scratch/cscs/pgiorgio/test-mace-opes-w4-shard.log
   ```
   - Exit code 137 or 9 means the process was killed, usually out of memory: check
     `sacct -j <JOBID> --format=JobID,State,MaxRSS`.
   - Exit code 139 means a segfault; the faulthandler trace shows where.
3. If sharding still fails, isolate the cause:
   - Run `--shard --cueq off`. If this works, the problem is cuEq inside `pmap`.
   - Run 2 walkers `--shard` with `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
   - Check that `python -c "import jax; print(jax.local_devices())"` lists 4 GPUs
     in the job step.
4. Fallback that works today: 4 walkers per GPU, either as 4 independent jobs
   (separate biases), or 4 walkers on one GPU with a shared bias (~7 ns/day
   aggregate).

## Production setup (once sharding works)

- **16 walkers:** use `--data $S/cn1/00000[0-7].lammps.data $S/cn2/00000[0-7].lammps.data`.
  Note that `0000*` matches 72 files.
- `--max-hours` should be a bit under the wall time. Continue with `--restart`
  and the same `--outdir`.
- The bias starts at **step 10,000** (5 ps). Before that, `ADAPTIVE_SIGMA_STRIDE=10000`
  only measures the CV fluctuations: `nker 0`, bias = −BARRIER, no OPES force,
  only the walls act. After that, kernels are added every `PACE=100` steps.
- Write outputs to scratch with `--outdir /ritom/scratch/cscs/pgiorgio/...`.

## Environment gotchas on Daint (container)

- The container sets `LD_PRELOAD` to the system OpenSSL. The env's activation
  script unsets it. If conda's `lib` is in `LD_LIBRARY_PATH`, git and ssh break:
  run git with `LD_LIBRARY_PATH=${LD_LIBRARY_PATH#$CONDA_PREFIX/lib:} git ...`.
- Host `srun` doesn't work inside the container (glibc 2.38). Run PLUMED MPI
  jobs with `mpiexec -launcher fork -n N`. Run `srun --jobid=<JOBID> --overlap --pty bash`
  from the login node.
- mace-jax must be at commit `fe19806`: newer versions renamed
  `CuEquivarianceConfig`, which the JAX-MD fork imports.
  `pip install --no-deps --force-reinstall 'mace-jax @ git+https://github.com/ACEsuit/mace-jax@fe19806'`.
- `git pull` complained about multiple upstream branches. Fix with
  `git config --unset-all branch.claude/sleepy-ptolemy-0wu9j4.merge; git branch -u origin/claude/sleepy-ptolemy-0wu9j4`.

## Not done / later

- Using the DeePMD models through DeePMD-kit 3.2's JAX backend: convert the
  **uncompressed** `.pb` with `dp convert-backend model.pb model.jax`, then use
  `deepmd.jax.jax_md.as_jax_md`.
- dplr-q integration: compatible in principle, by adding the OPES state to its MD
  scan. Electron-centre CVs need implicit differentiation through the
  `while_loop` ground-state solver.
- PLUMED STATE file import/export; OPES_METAD_EXPLORE, NLIST, EXCLUDED_REGION.
