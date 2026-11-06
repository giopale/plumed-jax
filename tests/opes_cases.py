"""OPES_METAD cases shared by the PLUMED fixture generator and its test.

Each case is a PLUMED input fragment for the parser subset plus the extra
output-only lines PLUMED needs. Trajectories are deterministic four-atom
random walks in nm; frame t is MD step t (PLUMED skips updating at step 0).
"""
import numpy as np

CVS = {
    "1d": "d: DISTANCE ATOMS=1,2 NOPBC\n",
    "2d": "d: DISTANCE ATOMS=1,2 NOPBC\nt: TORSION ATOMS=1,2,3,4 NOPBC\n",
}

CASES = {
    "default_1d": dict(cvs="1d", walkers=1, frames=161,
        opes="ARG=d PACE=5 BARRIER=20 TEMP=300 SIGMA=0.01"),
    "periodic_2d": dict(cvs="2d", walkers=1, frames=151,
        opes="ARG=d,t PACE=3 BARRIER=15 TEMP=300 SIGMA=0.01,0.15 SIGMA_MIN=0.002,0.03"),
    "adaptive_2d": dict(cvs="2d", walkers=1, frames=201,
        opes="ARG=d,t PACE=2 BARRIER=25 TEMP=300 ADAPTIVE_SIGMA_STRIDE=30"),
    "uniform_work": dict(cvs="1d", walkers=1, frames=121,
        opes="ARG=d PACE=4 BARRIER=10 TEMP=300 SIGMA=0.015 BIASFACTOR=inf "
             "RECURSIVE_MERGE_OFF COMPRESSION_THRESHOLD=0.6 CALC_WORK"),
    "fixed_nozed": dict(cvs="1d", walkers=1, frames=101,
        opes="ARG=d PACE=2 BARRIER=12 TEMP=300 SIGMA=0.01 FIXED_SIGMA NO_ZED "
             "BIASFACTOR=6 EPSILON=1e-4 KERNEL_CUTOFF=4"),
    "no_compression": dict(cvs="2d", walkers=1, frames=81,
        opes="ARG=d,t PACE=2 BARRIER=20 TEMP=300 SIGMA=0.01,0.15 COMPRESSION_THRESHOLD=0"),
    "walkers_mpi": dict(cvs="2d", walkers=3, frames=101,
        opes="ARG=d,t PACE=4 BARRIER=20 TEMP=300 SIGMA=0.01,0.15 WALKERS_MPI CALC_WORK"),
    "walkers_adaptive": dict(cvs="1d", walkers=2, frames=121,
        opes="ARG=d PACE=2 BARRIER=20 TEMP=300 ADAPTIVE_SIGMA_STRIDE=20 WALKERS_MPI"),
}


def trajectory(name):
    """Return positions of shape (walkers, frames, 4, 3) in nm."""
    case = CASES[name]
    seed = sum(map(ord, name))
    rng = np.random.default_rng(seed)
    start = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.5, 0.35, 0.0], [0.9, 0.4, 0.3]])
    steps = rng.normal(0.0, 0.006, (case["walkers"], case["frames"], 4, 3))
    steps[:, 0] = 0.0
    walk = start + np.cumsum(steps, axis=1)
    return walk + rng.normal(0.0, 0.02, (case["walkers"], 1, 4, 3))


def parser_input(name):
    """Input accepted by plumed_jax.parser (no file-output keywords)."""
    case = CASES[name]
    return CVS[case["cvs"]] + f"opes: OPES_METAD {case['opes']}\n"


def plumed_input(name):
    """Input for PLUMED: adds KERNELS/STATE files and per-step output."""
    case = CASES[name]
    pace = int(case["opes"].split("PACE=")[1].split()[0])
    args = "d" if case["cvs"] == "1d" else "d,t"
    components = "opes.bias,opes.rct,opes.zed,opes.neff,opes.nker"
    if "CALC_WORK" in case["opes"]:
        components += ",opes.work"
    return (
        CVS[case["cvs"]]
        + f"opes: OPES_METAD {case['opes']} SERIAL FILE=KERNELS FMT=%.17g "
        + f"STATE_WFILE=STATE STATE_WSTRIDE={pace}\n"
        + f"PRINT ARG={args},{components} FILE=COLVAR STRIDE=1 FMT=%.17g\n"
    )
