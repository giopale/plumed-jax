"""Generate PLUMED OPES_METAD reference data for tests/test_opes_plumed.py.

Run from the repository root with a PLUMED build that has the opes module
(check with `plumed config module opes`). Multiple-walker cases need an MPI
build and an MPI launcher; they are skipped without --mpirun:

    python tests/fixtures/generate_opes_reference.py --plumed plumed
    python tests/fixtures/generate_opes_reference.py --plumed plumed --mpirun "mpirun -np"

The script writes tests/fixtures/opes-plumed.npz and opes-plumed.json. It runs
`plumed driver` serially (OMP_NUM_THREADS=1, SERIAL) on deterministic
trajectories from tests/opes_cases.py and stores per-step bias, components and
atomic forces plus the final STATE file. Nothing is computed with plumed_jax.
"""
import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
import opes_cases  # noqa: E402


def read_plumed_table(path):
    """Read a PLUMED text file: (#! SET constants, {field: column})."""
    constants, fields, rows = {}, None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#! FIELDS"):
            fields = line.split()[2:]
        elif line.startswith("#! SET"):
            _, _, key, value = line.split(None, 3)
            constants[key] = value
        elif line.strip() and not line.startswith("#"):
            rows.append(line.split())
    columns = {f: np.array([float(r[i]) for r in rows]) for i, f in enumerate(fields)}
    return constants, columns


def read_forces(path, natoms):
    lines = Path(path).read_text().splitlines()
    frames = []
    for start in range(0, len(lines), natoms + 2):
        block = lines[start + 2:start + 2 + natoms]
        frames.append([[float(x) for x in line.split()[1:4]] for line in block])
    return np.array(frames)


def write_xyz(path, frames):
    with open(path, "w") as stream:
        for frame in frames:
            stream.write(f"{len(frame)}\n0 0 0\n")
            for x in frame:
                stream.write("X " + " ".join(f"{v:.17g}" for v in x) + "\n")


def run_case(name, plumed, mpirun):
    case = opes_cases.CASES[name]
    positions = opes_cases.trajectory(name)
    walkers = case["walkers"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "plumed.dat").write_text(opes_cases.plumed_input(name))
        command = [plumed, "driver", "--plumed", "plumed.dat", "--timestep", "1",
                   "--dump-forces", "forces", "--dump-forces-fmt", "%.17g"]
        if walkers == 1:
            write_xyz(tmp / "traj.xyz", positions[0])
            command += ["--ixyz", "traj.xyz"]
        else:
            for w in range(walkers):
                write_xyz(tmp / f"traj.{w}.xyz", positions[w])
            command = shlex.split(mpirun) + [str(walkers)] + command
            command += ["--ixyz", "traj.xyz", "--multi", str(walkers)]
        env = dict(os.environ, OMP_NUM_THREADS="1", PLUMED_NUM_THREADS="1")
        subprocess.run(command, cwd=tmp, env=env, check=True,
                       stdout=subprocess.DEVNULL)
        data = {"positions": positions}
        suffix = [""] if walkers == 1 else [f".{w}" for w in range(walkers)]
        colvars = [read_plumed_table(tmp / f"COLVAR{s}")[1] for s in suffix]
        for key in colvars[0]:
            if key.startswith("opes."):
                data[key[5:]] = np.stack([c[key] for c in colvars])
        data["forces"] = np.stack([read_forces(tmp / f"forces{s}", 4) for s in suffix])
        constants, kernels = read_plumed_table(tmp / "STATE")
        for key in ("zed", "sum_weights", "sum_weights2", "counter", "adaptive_counter"):
            if key in constants:
                data["state_" + key] = np.array(float(constants[key]))
        args = ["d"] if case["cvs"] == "1d" else ["d", "t"]
        data["state_centers"] = np.stack([kernels[a] for a in args], axis=-1)
        data["state_sigmas"] = np.stack([kernels["sigma_" + a] for a in args], axis=-1)
        data["state_heights"] = kernels["height"]
        for prefix in ("sigma0", "av_cv", "av_M2"):
            keys = ([f"{prefix}_{a}" for a in args] if walkers == 1 else
                    [[f"{prefix}_{a}_{w}" for a in args] for w in range(walkers)])
            flat = np.ravel(keys)
            if all(k in constants for k in flat):
                data["state_" + prefix] = np.array(
                    [float(constants[k]) for k in flat]).reshape(walkers, len(args))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plumed", default="plumed")
    parser.add_argument("--mpirun", default=None,
                        help='launcher prefix taking the process count, e.g. "mpirun -np"')
    options = parser.parse_args()
    version = subprocess.run([options.plumed, "info", "--version"], check=True,
                             capture_output=True, text=True).stdout.strip()
    has_opes = subprocess.run([options.plumed, "config", "module", "opes"]).returncode == 0
    if not has_opes:
        sys.exit("This PLUMED build does not include the opes module")
    arrays, manifest = {}, {"plumed_version": version, "cases": {}}
    for name, case in opes_cases.CASES.items():
        if case["walkers"] > 1 and options.mpirun is None:
            print(f"skipping {name}: needs --mpirun and an MPI build")
            continue
        data = run_case(name, options.plumed, options.mpirun)
        arrays.update({f"{name}/{k}": v for k, v in data.items()})
        manifest["cases"][name] = {
            "plumed_input": opes_cases.plumed_input(name),
            "walkers": case["walkers"],
            "frames": case["frames"],
        }
        print(f"{name}: {len(data['state_heights'])} kernels")
    output = ROOT / "tests" / "fixtures" / "opes-plumed.npz"
    np.savez(output, **arrays)
    manifest["sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest["provenance"] = (
        "plumed driver outputs printed with %.17g (COLVAR, STATE, --dump-forces); "
        "generated by tests/fixtures/generate_opes_reference.py")
    (output.with_suffix(".json")).write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
