"""OPES_METAD multiple walkers with a MACE potential in JAX-MD.

Requirements (in addition to plumed-jax):
  * the JAX-MD fork used in the plumed-jax paper (venkatkapil24/jax-md,
    branch working-branch), which provides ``energy.mace_neighbor_list`` and
    ``jax_md._nn.mace.mace_jax_from_torch``;
  * mace-jax, plus torch and mace-torch to convert the PyTorch model on the fly
    and (with --check) to compare energies/forces against PyTorch MACE.

Units: Å, eV, amu internally; the PLUMED input keeps PLUMED's defaults for
energy (kJ/mol unless UNITS ENERGY=... is given) and is converted to eV.

Typical use (16 walkers, one LAMMPS data file per walker):

  python examples/mace_opes_walkers.py --model model.model --type-map H,O,Ti \
      --plumed plumed.plumed --data cn1/0000*.lammps.data cn2/0000*.lammps.data \
      --temperature 350 --dt-fs 0.5 --tau-fs 100 --steps 100000 --check

The PLUMED input may be a LAMMPS/PLUMED file: ``adapt_plumed_input`` rewrites
``LABEL=`` lines, single-atom GROUP labels used as DISTANCE atoms, and drops
STATE_* keywords (restart uses NPZ checkpoints written by this script).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

KJ_MOL_PER_EV = 96.48533212331002
KB_EV = 8.617333262e-5  # eV/K
TIME_FS = 10.180505710774116  # internal time unit sqrt(amu Å^2/eV) in fs
ELEMENT_Z = {"H": 1, "He": 2, "Li": 3, "C": 6, "N": 7, "O": 8, "F": 9, "Na": 11,
             "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "K": 19,
             "Ca": 20, "Ti": 22, "Fe": 26, "Cu": 29, "Zn": 30}
ELEMENT_MASS = {"H": 1.00784, "O": 15.999, "Ti": 47.867, "C": 12.011,
                "N": 14.007, "Li": 6.94, "Na": 22.990, "Si": 28.085}


# ---------------------------------------------------------------------------
# Plain-Python helpers (no JAX-MD/MACE needed)

def adapt_plumed_input(text: str) -> str:
    """Rewrite a LAMMPS/PLUMED input into the plumed-jax parser subset.

    * ``ACTION ...`` blocks with a ``LABEL=x`` line become ``x: ACTION ...``;
    * single-atom GROUP labels used in DISTANCE/ANGLE/TORSION ATOMS are
      replaced by the atom number;
    * STATE_RFILE/STATE_WFILE/STATE_WSTRIDE/STORE_STATES are removed.
    Everything else is passed through unchanged for the strict parser.
    """
    lines = [line.split("#", 1)[0].rstrip() for line in text.splitlines()]
    out, pending, block_label = [], None, None
    for line in lines:
        stripped = line.strip()
        if pending is None and re.match(r"^(\w+:\s*)?\w+\s+\.\.\.$", stripped):
            pending, block_label = [stripped], None
            continue
        if pending is not None:
            if stripped.startswith("..."):
                head = pending[0]
                if block_label and ":" not in head.split()[0]:
                    head = f"{block_label}: {head}"
                out.append(head)
                out.extend(pending[1:])
                out.append("...")
                pending = None
                continue
            match = re.fullmatch(r"LABEL=(\S+)", stripped)
            if match:
                block_label = match.group(1)
            elif stripped:
                pending.append(stripped)
            continue
        out.append(stripped)
    text = "\n".join(out)
    text = re.sub(r"(?m)^\s*(STATE_RFILE|STATE_WFILE|STATE_WSTRIDE)=\S+\s*$\n?", "", text)
    text = re.sub(r"\s(STATE_RFILE|STATE_WFILE|STATE_WSTRIDE)=\S+", "", text)
    text = re.sub(r"(?m)^\s*STORE_STATES\s*$\n?", "", text)
    single = dict(re.findall(r"(?m)^(\w+):\s*GROUP\s+ATOMS=(\d+)\s*$", text))

    def replace_atoms(match):
        atoms = ",".join(single.get(a, a) for a in match.group(2).split(","))
        return f"{match.group(1)}ATOMS={atoms}"

    return re.sub(r"(?m)^(\w+:\s*(?:DISTANCE|ANGLE|TORSION)\s+.*?)ATOMS=(\S+)",
                  replace_atoms, text)


def read_lammps_data(path: str | Path):
    """Read an atom_style atomic LAMMPS data file.

    Returns (positions (N,3) Å sorted by atom id, types (N,) 1-based,
    cell (3,3) with lattice vectors as rows, masses {type: mass}).
    """
    lines = Path(path).read_text().splitlines()
    natoms = None
    lo = np.zeros(3)
    hi = np.zeros(3)
    tilt = np.zeros(3)
    masses, atoms = {}, []
    i = 0
    while i < len(lines):
        line = lines[i].split("#", 1)[0].strip()
        words = line.split()
        if line.endswith(" atoms") and len(words) == 2:
            natoms = int(words[0])
        for axis, key in enumerate(("xlo xhi", "ylo yhi", "zlo zhi")):
            if line.endswith(key):
                lo[axis], hi[axis] = float(words[0]), float(words[1])
        if line.endswith("xy xz yz"):
            tilt[:] = [float(w) for w in words[:3]]
        section = line.split()[0] if words else ""
        if section in ("Masses", "Atoms", "Velocities"):
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            while i < len(lines) and lines[i].strip():
                fields = lines[i].split("#", 1)[0].split()
                if section == "Masses":
                    masses[int(fields[0])] = float(fields[1])
                elif section == "Atoms":
                    atoms.append((int(fields[0]), int(fields[1]),
                                  *map(float, fields[2:5])))
                i += 1
            continue
        i += 1
    if natoms is None or len(atoms) != natoms:
        raise ValueError(f"{path}: expected {natoms} atoms, read {len(atoms)}")
    atoms.sort()
    types = np.array([a[1] for a in atoms])
    positions = np.array([a[2:] for a in atoms]) - lo
    lx, ly, lz = hi - lo
    xy, xz, yz = tilt
    cell = np.array([[lx, 0.0, 0.0], [xy, ly, 0.0], [xz, yz, lz]])
    return positions, types, cell, masses


# ---------------------------------------------------------------------------
# MACE + JAX-MD driver

def load_mace(model_path: str, cueq: str = "off"):
    """Convert a PyTorch MACE model to a callable MACE-JAX model.

    cueq selects cuEquivariance GPU kernels: "off" (plain e3nn-jax), "on"
    (all optimizations) or "fused" (all optimizations plus conv fusion).
    Always validate a new setting with --check.
    """
    import torch
    from flax import nnx
    from mace.tools.scripts_utils import extract_config_mace_model
    from jax_md._nn.mace.mace_jax_from_torch import convert_model

    torch_model = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(torch_model, dict) and "model" in torch_model:
        torch_model = torch_model["model"]
    torch_model = torch_model.float().eval()
    config = extract_config_mace_model(torch_model)
    if "error" in config:
        raise RuntimeError(config["error"])
    config["torch_model_class"] = torch_model.__class__.__name__
    cueq_config = None
    if cueq != "off":
        from mace_jax.modules.wrapper_ops import CuEquivarianceConfig
        cueq_config = CuEquivarianceConfig(enabled=True, layout="mul_ir", optimize_all=True,
                                           conv_fusion=cueq == "fused")
    graphdef, state, config = convert_model(torch_model, config, cueq_config=cueq_config)
    return nnx.merge(graphdef, state), config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="PyTorch MACE .model file")
    parser.add_argument("--type-map", required=True, help="element per LAMMPS type, e.g. H,O,Ti")
    parser.add_argument("--plumed", required=True)
    parser.add_argument("--data", nargs="+", required=True, help="one LAMMPS data file per walker")
    parser.add_argument("--temperature", type=float, default=350.0)
    parser.add_argument("--dt-fs", type=float, default=0.5)
    parser.add_argument("--tau-fs", type=float, default=100.0)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--print-every", type=int, default=100, help="COLVAR stride (one compiled block)")
    parser.add_argument("--dump-every", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=10000)
    parser.add_argument("--kernel-capacity", type=int, default=20000)
    parser.add_argument("--skin", type=float, default=0.5,
                        help="neighbor-list skin in Å (edges up to r_max+skin are evaluated)")
    parser.add_argument("--cueq", choices=("off", "on", "fused"), default="off",
                        help="cuEquivariance kernels for MACE-JAX (validate with --check)")
    parser.add_argument("--capacity-multiplier", type=float, default=1.5)
    parser.add_argument("--neighbor", choices=("auto", "mic", "multi-image"), default="auto",
                        help="minimum-image list, or explicit periodic images for small cells "
                             "(auto: multi-image if a cell height < 2*(r_max+skin))")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument("--check", action="store_true",
                        help="compare JAX vs PyTorch MACE energy/forces on walker 0, then continue")
    args = parser.parse_args(argv)

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax_md import energy, quantity, simulate, space
    from plumed_jax import checkpoint
    from plumed_jax.parser import parse

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    elements = args.type_map.split(",")
    walkers = len(args.data)

    # Structures: same composition and cell for every walker (NVT).
    frames = [read_lammps_data(p) for p in args.data]
    types, cell_rows = frames[0][1], frames[0][2]
    for p, (_, t, c, _) in zip(args.data, frames):
        if not np.array_equal(t, types) or not np.allclose(c, cell_rows):
            raise ValueError(f"{p}: walkers need identical atom types and cell")
    symbols = [elements[t - 1] for t in types]
    z_atomic = jnp.asarray([ELEMENT_Z[s] for s in symbols], jnp.int32)
    masses = frames[0][3]
    mass = jnp.asarray([masses.get(t, ELEMENT_MASS.get(s)) for t, s in zip(types, symbols)])
    positions0 = jnp.asarray(np.stack([f[0] for f in frames]))
    cell = jnp.asarray(cell_rows)            # rows = lattice vectors (plumed-jax)
    box = cell.T                             # columns = lattice vectors (JAX-MD)

    # PLUMED program and shared OPES state.
    text = adapt_plumed_input(Path(args.plumed).read_text())
    (out / "plumed.adapted.dat").write_text(text)
    program = parse(text, walkers=walkers, hill_capacity=args.kernel_capacity)
    to_ev = 1.0 / {"kj/mol": KJ_MOL_PER_EV, "ev": 1.0, "kcal/mol": KJ_MOL_PER_EV / 4.184}[
        program.units.energy.lower()]
    if program.units.length.lower() not in ("a", "angstrom"):
        raise ValueError("Use UNITS LENGTH=A in the PLUMED input (coordinates are in Å)")
    opes_labels = [a.label for a in program.actions if a.name == "OPES_METAD"]
    print("PLUMED actions:", [f"{a.label}:{a.name}" for a in program.actions])

    # MACE energy with a JAX-MD neighbor list.
    model, config = load_mace(args.model, args.cueq)
    r_max = float(config["r_max"])
    heights = cell_heights(cell_rows)
    mode = args.neighbor
    if mode == "auto":
        mode = "multi-image" if heights.min() < 2 * (r_max + args.skin) else "mic"
    print(f"cell heights {np.round(heights, 3)} Å, r_max {r_max} Å, skin {args.skin} Å "
          f"-> {mode} neighbor list")
    if mode == "mic" and heights.min() < 2 * (r_max + args.skin):
        print("WARNING: minimum-image list misses neighbors in this cell; use --neighbor multi-image")
    displacement, shift = space.periodic_general(box, fractional_coordinates=False)
    if mode == "multi-image":
        from jax_md import partition
        from jax_md.custom_partition import neighbor_list_multi_image
        from jax_md._nn.mace.featurizer import mace_multi_image_featurizer
        neighbor_options = dict(neighbor_list_fn=neighbor_list_multi_image,
                                featurizer_fn=mace_multi_image_featurizer,
                                format=partition.Dense)
    else:
        neighbor_options = dict(disable_cell_list=True)
    neighbor_fn, mace_energy = energy.mace_neighbor_list(
        displacement, box, model=model, config=config, z_atomic=z_atomic,
        r_cutoff=r_max, dr_threshold=args.skin, fractional_coordinates=False,
        capacity_multiplier=args.capacity_multiplier, **neighbor_options)

    def total_energy(R, neighbor, bias_state, **unused):
        bias = program.bias(R, cell, state=bias_state)
        return mace_energy(R, neighbor=neighbor) + bias * to_ev

    if args.check:
        check_against_torch(args.model, symbols, np.asarray(positions0[0]), cell_rows,
                            mace_energy, neighbor_fn, r_max)

    # Neighbor lists with one common capacity for all walkers.
    capacity = max(neighbor_fn.allocate(R).idx.shape[1] for R in positions0)

    def allocate_all(R_all, capacity):
        lists = []
        for R in R_all:
            nbr = neighbor_fn.allocate(R)
            lists.append(neighbor_fn.allocate(R, extra_capacity=capacity - nbr.idx.shape[1]))
        return jax.tree_util.tree_map(lambda *x: jnp.stack(x), *lists)

    kT = KB_EV * args.temperature
    init_fn, apply_fn = simulate.nvt_nose_hoover(
        total_energy, shift, args.dt_fs / TIME_FS, kT, tau=args.tau_fs / TIME_FS)
    force_fn = quantity.force(total_energy)
    pace = {label: program.metad[label].pace for label in opes_labels}

    def build_block(length):
        def one_step(carry, _):
            state, nbrs, bias_state, step = carry
            state = jax.vmap(lambda s, n: apply_fn(s, neighbor=n, bias_state=bias_state))(state, nbrs)
            nbrs = jax.vmap(lambda R, n: n.update(R))(state.position, nbrs)
            step = step + 1
            # Bias at the new positions with the OLD state, then one synchronous deposit.
            _, new_bias_state = program.bias_and_update_walkers(
                state.position, bias_state, cell, step=step)
            due = jnp.any(jnp.array([step % p == 0 for p in pace.values()]))
            # Refresh cached forces when the bias changed.
            state = jax.lax.cond(
                due,
                lambda s: s.set(force=jax.vmap(lambda R, n: force_fn(R, neighbor=n, bias_state=new_bias_state))(s.position, nbrs)),
                lambda s: s, state)
            return (state, nbrs, new_bias_state, step), None

        @jax.jit
        def block(carry):
            carry, _ = jax.lax.scan(one_step, carry, None, length=length)
            state, nbrs, bias_state, step = carry
            values = jax.vmap(lambda R: program.values(R, cell))(state.position)
            biases = jax.vmap(lambda R: program.bias(R, cell, state=bias_state))(state.position)
            pe = jax.vmap(lambda R, n: mace_energy(R, neighbor=n))(state.position, nbrs)
            temps = jax.vmap(lambda p, m: quantity.temperature(momentum=p, mass=m))(
                state.momentum, state.mass) / KB_EV
            return carry, values, biases, pe, temps, jnp.any(nbrs.did_buffer_overflow)
        return block

    # Initial or restarted state.
    md_path = out / "md_state.npz"
    if args.restart:
        bias_state = {label: checkpoint.load(out / f"opes_state.{label}.npz", program.metad[label])
                      for label in opes_labels}
    else:
        bias_state = program.initial_state()
    nbrs = allocate_all(positions0, capacity)
    keys = jax.random.split(jax.random.PRNGKey(args.seed), walkers)
    state = jax.vmap(lambda k, R, n: init_fn(k, R, mass=mass, neighbor=n, bias_state=bias_state))(
        keys, positions0, nbrs)
    step = jnp.int32(0)
    if args.restart:
        with np.load(md_path) as data:
            leaves = [jnp.asarray(data[f"leaf{i}"]) for i in range(len(data.files) - 1)]
            step = jnp.int32(int(data["step"]))
        state = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(state), leaves)
        nbrs = allocate_all(state.position, capacity)

    block = build_block(args.print_every)
    colvar = [open(out / f"COLVAR.{w}", "a") for w in range(walkers)]
    cv_names = None
    started = time.time()
    first_block = True
    while int(step) < args.steps:
        previous = (state, nbrs, bias_state, step)
        tick = time.time()
        (state, nbrs, bias_state, step), values, biases, pe, temps, overflow = block(previous)
        overflow = bool(overflow)  # waits for the device
        elapsed = time.time() - tick
        if overflow:
            capacity = int(capacity * 1.3) + 1
            print(f"neighbor list overflow: reallocating with capacity {capacity}")
            state, _, bias_state, step = previous
            nbrs = allocate_all(state.position, capacity)
            block = build_block(args.print_every)
            continue
        bad = [label for label in opes_labels if bool(bias_state[label].overflow)]
        if bad:
            raise RuntimeError(f"OPES overflow in {bad}: raise --kernel-capacity or check the CVs")
        n = int(step)
        if cv_names is None:
            cv_names = sorted(values)
            for f in colvar:
                f.write("#! FIELDS step time " + " ".join(cv_names) + " bias "
                        + " ".join(f"{l}.{k}" for l in opes_labels for k in ("rct", "zed", "neff", "nker"))
                        + " temperature mace_energy\n")
        diag = {l: program.metad[l].diagnostics(bias_state[l]) for l in opes_labels}
        for w, f in enumerate(colvar):
            fields = [n, n * args.dt_fs * 1e-3] + [float(np.ravel(values[k][w])[0]) for k in cv_names]
            fields += [float(biases[w])] + [float(diag[l][k]) for l in opes_labels
                                            for k in ("rct", "zed", "neff", "nker")]
            fields += [float(temps[w]), float(pe[w])]
            f.write(" ".join(f"{x:.10g}" for x in fields) + "\n")
            f.flush()
        ms_per_step = 1000 * elapsed / args.print_every
        ns_day = args.dt_fs * 1e-6 * 86400 / (elapsed / args.print_every)
        timing = ("(includes compilation)" if first_block else
                  f"{ms_per_step:.1f} ms/step, {ns_day:.2f} ns/day per walker, "
                  f"{ns_day * walkers:.2f} ns/day aggregate")
        first_block = False
        print(f"step {n}  T {np.mean(temps):7.1f} K  E_mace {np.mean(pe):.4f} eV  {timing}  "
              + "  ".join(f"{l}: nker {int(diag[l]['nker'])} neff {float(diag[l]['neff']):.1f}" for l in opes_labels),
              flush=True)
        if n % args.dump_every == 0:
            for w in range(walkers):
                with open(out / f"traj.{w}.xyz", "a") as f:
                    f.write(f"{len(symbols)}\nLattice=\"{' '.join(map(str, cell_rows.ravel()))}\" step={n}\n")
                    for s, x in zip(symbols, np.asarray(state.position[w])):
                        f.write(f"{s} {x[0]:.6f} {x[1]:.6f} {x[2]:.6f}\n")
        timeout = args.max_hours is not None and time.time() - started > 3600 * args.max_hours
        if n % args.checkpoint_every == 0 or n >= args.steps or timeout:
            save_state(md_path, state, step)
            for label in opes_labels:
                checkpoint.save(out / f"opes_state.{label}.npz", bias_state[label], program.metad[label])
        if timeout:
            print("time limit reached; checkpoint written")
            break
    for f in colvar:
        f.close()


def save_state(path, state, step):
    import jax
    leaves = jax.tree_util.tree_leaves(state)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as stream:
        np.savez(stream, step=np.asarray(step), **{f"leaf{i}": np.asarray(x) for i, x in enumerate(leaves)})
    tmp.replace(path)


def cell_heights(cell_rows):
    """Perpendicular widths of the cell (rows = lattice vectors)."""
    a, b, c = np.asarray(cell_rows, float)
    volume = abs(np.dot(a, np.cross(b, c)))
    return np.array([volume / np.linalg.norm(np.cross(b, c)),
                     volume / np.linalg.norm(np.cross(c, a)),
                     volume / np.linalg.norm(np.cross(a, b))])


def count_pairs(positions, cell_rows, r_cut):
    """Brute-force number of directed pairs (i, j, image) with 0 < d < r_cut."""
    cell_rows = np.asarray(cell_rows, float)
    n = np.ceil(r_cut / cell_heights(cell_rows)).astype(int)
    shifts = np.array(np.meshgrid(*[np.arange(-k, k + 1) for k in n], indexing="ij")).reshape(3, -1).T
    total = 0
    for shift in shifts @ cell_rows:
        d = np.linalg.norm(positions[None, :, :] + shift - positions[:, None, :], axis=-1)
        total += int(np.sum((d > 1e-8) & (d < r_cut)))
    return total


def count_edges(neighbor, positions):
    """Valid neighbor slots (an upper bound that includes the skin)."""
    idx = np.asarray(neighbor.idx)
    n = len(positions)
    if idx.ndim == 2 and idx.shape[0] == n:
        return int(np.sum(idx < n))
    return int(np.sum(idx[0] < n))


def check_against_torch(model_path, symbols, positions, cell_rows, mace_energy, neighbor_fn, r_max):
    """Compare MACE-JAX energy/forces with PyTorch MACE (float64) on one structure."""
    import jax
    import jax.numpy as jnp
    from ase import Atoms
    from mace.calculators import MACECalculator

    atoms = Atoms(symbols, positions=positions, cell=cell_rows, pbc=True)
    atoms.calc = MACECalculator(model_paths=model_path, device="cpu", default_dtype="float64")
    e_torch, f_torch = atoms.get_potential_energy(), atoms.get_forces()
    R = jnp.asarray(positions)
    nbr = neighbor_fn.allocate(R)
    e_jax, g = jax.value_and_grad(lambda r: mace_energy(r, neighbor=nbr))(R)
    f_jax = -np.asarray(g)
    report = {
        "energy_torch_float64_eV": float(e_torch), "energy_jax_float32_eV": float(e_jax),
        "energy_diff_meV_per_atom": 1000 * abs(float(e_jax) - e_torch) / len(symbols),
        "force_max_abs_diff_eV_A": float(np.max(np.abs(f_jax - f_torch))),
        "force_rms_diff_eV_A": float(np.sqrt(np.mean((f_jax - f_torch) ** 2))),
        "force_rms_eV_A": float(np.sqrt(np.mean(f_torch**2))),
        "pairs_within_r_max_bruteforce": count_pairs(positions, cell_rows, r_max),
        "neighbor_slots_within_r_max_plus_skin": count_edges(nbr, positions),
        "worst_atom": int(np.argmax(np.abs(f_jax - f_torch).max(axis=1))) + 1,
    }
    print("MACE check:", json.dumps(report, indent=1))
    # float32 rounding of a ~1e5 eV total can reach ~1 meV/atom; forces are the real test.
    if report["energy_diff_meV_per_atom"] > 5.0 or report["force_max_abs_diff_eV_A"] > 2e-2:
        raise RuntimeError("MACE-JAX disagrees with PyTorch MACE; see report above")


if __name__ == "__main__":
    main()
