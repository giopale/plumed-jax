"""Strict parser for the documented CV/bias subset of PLUMED input.

Positions/cells and returned energies use the declared UNITS (PLUMED defaults:
nm, kJ/mol, ps). Unsupported actions/options raise errors. No text is executed.
"""

from __future__ import annotations

from collections.abc import Callable
from jax import Array, custom_jvp
from jax.typing import ArrayLike
from .metadynamics import HillState
from .grid_metadynamics import GridState
from .opes import OPESMetad, OPESState
from dataclasses import dataclass
from functools import lru_cache
import ast
import re
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from . import cvs
from .bias import harmonic_restraint, wall
from .metadata import Units, LENGTH_TO_A
from .metadynamics import Metadynamics
from .grid_metadynamics import GridMetadynamics
from .reference import rmsd, path_msd
from .switches import switching
from .geometry import displacement, prepare_cell, PreparedCell

FUNCTIONS = {
    name: getattr(jnp, name)
    for name in ("sin", "cos", "tan", "exp", "log", "sqrt", "abs", "tanh")
}

@custom_jvp
def _minimum(x, y):
    return jnp.minimum(x, y)


@_minimum.defjvp
def _minimum_jvp(primals, tangents):
    x, y = primals
    dx, dy = tangents
    # Lepton selects the second derivative at a tie for min.
    return _minimum(x, y), jnp.where(x < y, dx, dy)


@custom_jvp
def _maximum(x, y):
    return jnp.maximum(x, y)


@_maximum.defjvp
def _maximum_jvp(primals, tangents):
    x, y = primals
    dx, dy = tangents
    # Compare operands, not equality with the result: GPU fusion can round
    # those evaluations differently. Lepton selects the first branch at ties.
    return _maximum(x, y), jnp.where(x >= y, dx, dy)


def _extremum_inputs(x, y):
    # Promote literal integers before AD so their tangents are floating zeros.
    dtype = jnp.result_type(x, y)
    return jnp.asarray(x, dtype=dtype), jnp.asarray(y, dtype=dtype)


FUNCTIONS.update(
    min=lambda x, y: _minimum(*_extremum_inputs(x, y)),
    max=lambda x, y: _maximum(*_extremum_inputs(x, y)),
)


def _expression(text):
    tree = ast.parse(text.replace("^", "**"), mode="eval")

    def evaluate(node, variables):
        if isinstance(node, ast.Expression):
            return evaluate(node.body, variables)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.Name):
            if node.id == "pi":
                return jnp.pi
            if node.id not in variables:
                raise ValueError(f"Unknown expression variable {node.id}")
            return variables[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = evaluate(node.operand, variables)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            a, b = evaluate(node.left, variables), evaluate(node.right, variables)
            operators = {
                ast.Add: lambda: a + b,
                ast.Sub: lambda: a - b,
                ast.Mult: lambda: a * b,
                ast.Div: lambda: a / b,
                ast.Pow: lambda: a**b,
            }
            if type(node.op) in operators:
                return operators[type(node.op)]()
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in FUNCTIONS
            and not node.keywords
        ):
            return FUNCTIONS[node.func.id](*(evaluate(n, variables) for n in node.args))
        raise ValueError("Unsupported or unsafe expression")

    # Validate the complete AST before returning a JIT-able expression.
    allowed = (
        ast.Expression,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.UnaryOp,
        ast.USub,
        ast.UAdd,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Call,
    )
    if any(not isinstance(node, allowed) for node in ast.walk(tree)):
        raise ValueError("Unsupported or unsafe expression")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in {
            "x",
            "y",
            "z",
            "pi",
            *FUNCTIONS,
        }:
            raise ValueError("Unknown expression name")
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name)
            or node.func.id not in FUNCTIONS
            or node.keywords
        ):
            raise ValueError("Unsupported function call")
    return lambda variables: evaluate(tree, variables)


def _pdb(path, unit):
    frames = []
    coordinates = []
    indices = []
    aw = []
    dw = []
    all_indices = None
    for line in Path(path).read_text().splitlines() + ["END"]:
        if line.startswith(("ATOM  ", "HETATM")):
            indices.append(int(line[6:11]) - 1)
            coordinates.append(
                [
                    float(line[a:b]) / LENGTH_TO_A[unit]
                    for a, b in ((30, 38), (38, 46), (46, 54))
                ]
            )
            aw.append(float(line[54:60]))
            dw.append(float(line[60:66]))
        elif line.startswith("END") and coordinates:
            if all_indices is None:
                all_indices = tuple(indices)
                weights = (np.array(aw), np.array(dw))
            if tuple(indices) != all_indices:
                raise ValueError("Reference frames select different atoms")
            if not np.array_equal(weights[0], aw) or not np.array_equal(weights[1], dw):
                raise ValueError("Reference frames change weights")
            frames.append(np.array(coordinates))
            coordinates = []
            indices = []
            aw = []
            dw = []
    if not frames or any(np.any(w < 0) or np.sum(w) <= 0 for w in weights):
        raise ValueError("Invalid PDB reference weights/frames")
    return jnp.array(frames), all_indices, weights


@dataclass(frozen=True)
class Action:
    """One parsed action: label, uppercase name, and string/bare-flag options."""

    label: str
    name: str
    options: dict


_ALLOWED = {
    "GROUP": {"ATOMS"},
    "DISTANCE": {"ATOMS", "NOPBC"},
    "ANGLE": {"ATOMS", "NOPBC"},
    "TORSION": {"ATOMS", "NOPBC"},
    "POSITION": {"ATOM", "NOPBC"},
    "COM": {"ATOMS", "NOPBC"},
    "GYRATION": {"ATOMS", "TYPE", "NOPBC"},
    "COORDINATION": {"GROUPA", "GROUPB", "R_0", "D_0", "NN", "MM", "SWITCH", "NOPBC"},
    "RMSD": {"REFERENCE", "TYPE", "SQUARED", "NOPBC"},
    "PATHMSD": {"REFERENCE", "LAMBDA", "NOPBC"},
    "COMBINE": {"ARG", "COEFFICIENTS", "PARAMETERS", "POWERS", "PERIODIC", "NORMALIZE"},
    "CUSTOM": {"ARG", "FUNC", "PERIODIC"},
    "CONSTANT": {"VALUE", "VALUES"},
    "RESTRAINT": {"ARG", "AT", "KAPPA", "SLOPE"},
    "UPPER_WALLS": {"ARG", "AT", "KAPPA", "EPS", "EXP", "OFFSET"},
    "LOWER_WALLS": {"ARG", "AT", "KAPPA", "EPS", "EXP", "OFFSET"},
    "METAD": {
        "ARG",
        "SIGMA",
        "HEIGHT",
        "PACE",
        "BIASFACTOR",
        "TEMP",
        "FILE",
        "FMT",
        "GRID_MIN",
        "GRID_MAX",
        "GRID_BIN",
        "GRID_SPACING",
    },
    "OPES_METAD": {
        "ARG",
        "PACE",
        "BARRIER",
        "TEMP",
        "SIGMA",
        "SIGMA_MIN",
        "ADAPTIVE_SIGMA_STRIDE",
        "BIASFACTOR",
        "EPSILON",
        "KERNEL_CUTOFF",
        "COMPRESSION_THRESHOLD",
        "FIXED_SIGMA",
        "RECURSIVE_MERGE_OFF",
        "NO_ZED",
        "CALC_WORK",
        "WALKERS_MPI",
        "SERIAL",
        "FILE",
        "FMT",
    },
    "PRINT": {"ARG", "FILE", "STRIDE", "FMT"},
}
_REQUIRED = {
    "GROUP": {"ATOMS"},
    "DISTANCE": {"ATOMS"},
    "ANGLE": {"ATOMS"},
    "TORSION": {"ATOMS"},
    "POSITION": {"ATOM"},
    "COM": {"ATOMS"},
    "GYRATION": {"ATOMS"},
    "COORDINATION": {"GROUPA"},
    "RMSD": {"REFERENCE"},
    "PATHMSD": {"REFERENCE", "LAMBDA"},
    "COMBINE": {"ARG", "PERIODIC"},
    "CUSTOM": {"ARG", "FUNC", "PERIODIC"},
    "RESTRAINT": {"ARG", "AT"},
    "UPPER_WALLS": {"ARG", "AT", "KAPPA"},
    "LOWER_WALLS": {"ARG", "AT", "KAPPA"},
    "METAD": {"ARG", "SIGMA", "HEIGHT", "PACE"},
    "OPES_METAD": {"ARG", "PACE", "BARRIER", "TEMP"},
    "PRINT": {"ARG"},
}
_FLAGS = {
    "NOPBC",
    "SQUARED",
    "NORMALIZE",
    "FIXED_SIGMA",
    "RECURSIVE_MERGE_OFF",
    "NO_ZED",
    "CALC_WORK",
    "WALKERS_MPI",
    "SERIAL",
}
_STATEFUL_BIASES = ("METAD", "OPES_METAD")
_GEOMETRY_ATOM_COUNTS = {"DISTANCE": 2, "ANGLE": 3, "TORSION": 4}
_CELL_GEOMETRY_ACTIONS = (
    *_GEOMETRY_ATOM_COUNTS,
    "POSITION",
    "COM",
    "GYRATION",
    "COORDINATION",
    "RMSD",
    "PATHMSD",
)


@lru_cache(maxsize=128)
def _geometry_selection(spec):
    """Prepare static tokens only; never cache coordinates or cell factors."""
    atoms = tuple(spec.split(","))
    indices = tuple(int(t) - 1 if re.fullmatch(r"\d+", t) else t for t in atoms)
    return atoms, indices


def _numbers(text):
    return tuple(
        float(x.replace("-pi", str(-np.pi)).replace("pi", str(np.pi)))
        for x in str(text).split(",")
    )


def _tokens(text):
    return re.findall(r"[^\s=]+\s*=\s*(?:\{[^}]*\}|[^\s]+)|[^\s]+", text)


def _action_lines(text):
    """Join PLUMED continuation lines while stripping comments."""
    pending = ""
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.endswith("...") and not line.startswith("..."):
            if pending:
                raise ValueError("Nested multiline action")
            pending = line[:-3].strip() + " "
            continue
        if line.startswith("..."):
            if not pending:
                raise ValueError("Unexpected multiline terminator")
            line = pending
            pending = ""
        elif pending:
            pending += line + " "
            continue
        yield line
    if pending:
        raise ValueError("Unclosed multiline action")


def _read_action(line):
    """Read a label, action name, and options without interpreting values."""
    tokens = _tokens(line)
    label = None
    if tokens[0].endswith(":"):
        label = tokens.pop(0)[:-1]
    name = tokens.pop(0).upper()
    opts = {}
    for token in tokens:
        pair = token.split("=", 1)
        key = pair[0].strip().upper()
        if key in opts:
            raise ValueError(f"Duplicate option {key}")
        opts[key] = True if len(pair) == 1 else pair[1].strip()
    return label, name, opts


def _validate_action(name, opts, actions):
    """Validate supported options and dependencies on preceding actions."""
    if name not in _ALLOWED or set(opts) - _ALLOWED[name]:
        raise ValueError(
            f"Unsupported {name} options: {set(opts) - _ALLOWED.get(name, set())}"
        )
    missing = _REQUIRED.get(name, set()) - set(opts)
    if missing:
        raise ValueError(f"{name} requires {sorted(missing)}")
    if any(opts[k] is not True for k in _FLAGS & set(opts)):
        raise ValueError("Flags are bare keywords, not keyword=value options")
    if any(opts[k] is True for k in set(opts) - _FLAGS):
        raise ValueError("Option requires a value")
    if name == "CONSTANT" and (("VALUE" in opts) + ("VALUES" in opts) != 1):
        raise ValueError("CONSTANT requires VALUE or VALUES")
    if name == "RESTRAINT" and not ({"KAPPA", "SLOPE"} & set(opts)):
        raise ValueError("RESTRAINT requires KAPPA or SLOPE")
    if name == "COORDINATION" and not ({"R_0", "SWITCH"} & set(opts)):
        raise ValueError("COORDINATION requires R_0 or SWITCH")
    if name == "METAD" and "BIASFACTOR" in opts and "TEMP" not in opts:
        raise ValueError("BIASFACTOR requires TEMP")
    if "ARG" in opts and name != "PRINT":
        for arg in opts["ARG"].split(","):
            if arg.split(".")[0] not in {a.label for a in actions}:
                raise ValueError(f"Unknown preceding CV {arg}")
        n_args = len(opts["ARG"].split(","))
        for key in (
            "AT",
            "KAPPA",
            "SLOPE",
            "EPS",
            "EXP",
            "OFFSET",
            "SIGMA",
            "SIGMA_MIN",
            "COEFFICIENTS",
            "PARAMETERS",
            "POWERS",
        ):
            if key == "SIGMA" and str(opts.get(key)).upper() == "ADAPTIVE":
                continue
            if key in opts and len(_numbers(opts[key])) != n_args:
                raise ValueError(f"{key} needs one value per ARG")


def _group_selection(spec, groups):
    """Resolve one-based static GROUP lists/ranges to zero-based indices."""
    if spec in groups:
        return groups[spec]
    result = []
    for item in str(spec).split(","):
        if item in groups:
            result.extend(groups[item])
            continue
        if re.fullmatch(r"\d+-\d+", item):
            lo, hi = map(int, item.split("-"))
            if hi < lo:
                raise ValueError("Reversed atom range")
            result.extend(range(lo - 1, hi))
        elif re.fullmatch(r"\d+", item):
            result.append(int(item) - 1)
        else:
            raise ValueError(f"Invalid static atom selection {item}")
    if any(i < 0 for i in result):
        raise ValueError("PLUMED atom indices start at 1")
    return tuple(result)


def _metadynamics_config(opts, units, periods, hill_capacity):
    """Construct a direct-history or grid bias from validated METAD options."""
    args = opts["ARG"].split(",")
    factor = float(opts["BIASFACTOR"]) if "BIASFACTOR" in opts else None
    # PLUMED 2.10.1's published-unit convention; validated against replay.
    from .metadata import ENERGY_TO_EV

    kT = (
        0.0083144621
        * float(opts["TEMP"])
        * ENERGY_TO_EV["kj/mol"]
        / ENERGY_TO_EV[units.energy]
        if factor
        else None
    )
    sigma = _numbers(opts["SIGMA"])
    cv_periods = tuple(periods.get(arg, 0.0) for arg in args)
    if any(key.startswith("GRID_") for key in opts):
        if not {"GRID_MIN", "GRID_MAX"} <= set(opts):
            raise ValueError("Grid requires GRID_MIN and GRID_MAX")
        lo, hi = _numbers(opts["GRID_MIN"]), _numbers(opts["GRID_MAX"])
        if len(lo) != len(args) or len(hi) != len(args):
            raise ValueError("Grid bounds need one value per ARG")
        if any(
            not np.isfinite(a) or not np.isfinite(b) or b <= a for a, b in zip(lo, hi)
        ):
            raise ValueError("Grid bounds must be finite and increasing")
        bins = None
        if "GRID_BIN" in opts:
            tokens = opts["GRID_BIN"].split(",")
            if len(tokens) != len(args) or any(
                not re.fullmatch(r"\d+", t) for t in tokens
            ):
                raise ValueError("GRID_BIN needs one integer per ARG")
            bins = tuple(int(t) for t in tokens)
            if any(b < 2 for b in bins):
                raise ValueError("GRID_BIN must be >=2")
        if "GRID_SPACING" in opts or bins is None:
            dx = (
                _numbers(opts["GRID_SPACING"])
                if "GRID_SPACING" in opts
                else tuple(s / 5 for s in sigma)
            )
            if len(dx) != len(args) or any(not np.isfinite(d) or d <= 0 for d in dx):
                raise ValueError("GRID_SPACING needs one positive value per ARG")
            computed = tuple(
                max(2, int(np.ceil((b - a) / d))) for a, b, d in zip(lo, hi, dx)
            )
            bins = (
                computed
                if bins is None
                else tuple(max(a, b) for a, b in zip(bins, computed))
            )
        return GridMetadynamics(
            sigma,
            float(opts["HEIGHT"]),
            int(opts["PACE"]),
            lo,
            hi,
            bins,
            cv_periods,
            factor,
            kT,
        )
    return Metadynamics(
        sigma,
        float(opts["HEIGHT"]),
        int(opts["PACE"]),
        hill_capacity,
        cv_periods,
        factor,
        kT,
    )


def _opes_config(opts, units, periods, lower, capacity, walkers):
    """Construct OPESMetad from validated OPES_METAD options."""
    from .metadata import ENERGY_TO_EV

    args = opts["ARG"].split(",")
    kT = (
        0.0083144621
        * float(opts["TEMP"])
        * ENERGY_TO_EV["kj/mol"]
        / ENERGY_TO_EV[units.energy]
    )
    sigma = opts.get("SIGMA", "ADAPTIVE")
    adaptive = str(sigma).upper() == "ADAPTIVE"
    factor = opts.get("BIASFACTOR")
    if factor is not None:
        factor = float("inf") if factor.upper() == "INF" else float(factor)
    stride = opts.get("ADAPTIVE_SIGMA_STRIDE")
    if stride is not None and not re.fullmatch(r"\d+", stride):
        raise ValueError("ADAPTIVE_SIGMA_STRIDE must be an integer")
    if not re.fullmatch(r"\d+", opts["PACE"]):
        raise ValueError("PACE must be an integer")
    cv_periods = tuple(periods.get(arg, 0.0) for arg in args)
    return OPESMetad(
        sigma="adaptive" if adaptive else _numbers(sigma),
        barrier=float(opts["BARRIER"]),
        pace=int(opts["PACE"]),
        capacity=capacity,
        kT=kT,
        periods=cv_periods if any(cv_periods) else (),
        periodic_min=tuple(lower.get(arg, -p / 2) for arg, p in zip(args, cv_periods))
        if any(cv_periods)
        else (),
        bias_factor=factor,
        epsilon=float(opts["EPSILON"]) if "EPSILON" in opts else None,
        kernel_cutoff=float(opts["KERNEL_CUTOFF"]) if "KERNEL_CUTOFF" in opts else None,
        compression_threshold=float(opts.get("COMPRESSION_THRESHOLD", 1.0)),
        sigma_min=_numbers(opts["SIGMA_MIN"]) if "SIGMA_MIN" in opts else (),
        adaptive_sigma_stride=None if stride is None else int(stride),
        fixed_sigma="FIXED_SIGMA" in opts,
        recursive_merge="RECURSIVE_MERGE_OFF" not in opts,
        no_zed="NO_ZED" in opts,
        calc_work="CALC_WORK" in opts,
        walkers=walkers if "WALKERS_MPI" in opts else 1,
        dimension=len(args) if adaptive else None,
    )


def _coordination(positions, group_a, group_b, options, cell):
    """Evaluate simplified rational or explicit SWITCH coordination."""
    if "SWITCH" not in options:
        n = int(options.get("NN", 6))
        m = int(options.get("MM", 0)) or 2 * n
        return cvs.coordination_number(
            positions,
            group_a,
            group_b,
            r0=float(options["R_0"]),
            d0=float(options.get("D_0", 0)),
            n=n,
            m=m,
            cell=cell,
        )
    pairs = (
        [
            (group_a[i], group_a[j])
            for i in range(len(group_a))
            for j in range(i + 1, len(group_a))
        ]
        if group_b is None
        else [(i, j) for i in group_a for j in group_b if i != j]
    )
    params = _tokens(options["SWITCH"].strip("{}"))
    kind = params.pop(0)
    parameters = dict(t.split("=", 1) for t in params)
    mapping = {
        "R_0": "r0",
        "D_0": "d0",
        "D_MAX": "dmax",
        "NN": "n",
        "MM": "m",
        "A": "a",
        "B": "b",
    }
    if set(parameters) - set(mapping):
        raise ValueError("Unsupported switch options")
    kwargs = {
        mapping[k]: (int(v) if k in ("NN", "MM") else float(v))
        for k, v in parameters.items()
    }
    if kind == "RATIONAL" and "dmax" not in kwargs:
        nn = kwargs.get("n", 6)
        mm = kwargs.get("m", 2 * nn)
        kwargs["m"] = mm
        kwargs["dmax"] = kwargs.get("d0", 0.0) + kwargs.get("r0", 1.0) * 1e-5 ** (
            1 / (nn - mm)
        )
    if kind != "RATIONAL" and "dmax" not in kwargs:
        raise ValueError("Parser requires explicit D_MAX for non-rational switches")
    if not pairs:
        return jnp.array(0.0, dtype=positions.dtype)
    pairs_array = jnp.array(pairs)
    distances = jnp.linalg.norm(
        displacement(positions[pairs_array[:, 0]], positions[pairs_array[:, 1]], cell),
        axis=-1,
    )
    return jnp.sum(switching(distances, kind=kind, stretch="dmax" in kwargs, **kwargs))


def parse(
    text: str,
    *,
    base_dir: str | Path = ".",
    hill_capacity: int = 1024,
    walkers: int = 1,
) -> Program:
    """Construct a Program from the documented PLUMED input subset, outside JIT.

    text is the input string; base_dir resolves PDB REFERENCE paths.
    hill_capacity fixes direct-history METAD storage and the compressed
    kernel capacity of OPES_METAD (default 1024); grid METAD uses its
    specified bounds/bins instead. walkers is the number of synchronous
    replicas for OPES_METAD ... WALKERS_MPI (default 1); with walkers > 1
    every OPES_METAD action must declare WALKERS_MPI. Defaults are nm, kJ/mol,
    ps unless UNITS overrides them. Coordinates must already use these units.
    Unsupported actions/options/dependencies raise ValueError; numeric and
    file errors propagate. PRINT/FILE/FMT do not write output. Atom bounds
    and required engine masses are checked when the program is evaluated.
    """
    actions = []
    units = Units("nm", "kj/mol", "ps")
    groups = {}
    references = {}
    expressions = {}
    periods = {}
    lower = {}
    metad = {}
    if type(walkers) is not int or walkers < 1:
        raise ValueError("walkers must be a positive integer")
    for line in _action_lines(text):
        label, name, opts = _read_action(line)
        if name == "UNITS":
            if actions:
                raise ValueError("UNITS must precede actions")
            if set(opts) - {"LENGTH", "ENERGY", "TIME"}:
                raise ValueError("Unsupported UNITS option")
            units = Units(
                opts.get("LENGTH", "nm"),
                opts.get("ENERGY", "kj/mol"),
                opts.get("TIME", "ps"),
            )
            continue
        _validate_action(name, opts, actions)
        label = label or f"_action_{len(actions)}"
        if label in {a.label for a in actions}:
            raise ValueError("Duplicate action label")
        action = Action(label, name, opts)
        actions.append(action)
        if name == "GROUP":
            groups[label] = _group_selection(opts["ATOMS"], groups)
        elif name in ("RMSD", "PATHMSD"):
            references[label] = _pdb(Path(base_dir) / opts["REFERENCE"], units.length)
        elif name == "CUSTOM":
            expressions[label] = _expression(opts["FUNC"])
        if name == "TORSION":
            periods[label] = 2 * np.pi
            lower[label] = -np.pi
        elif name in ("COMBINE", "CUSTOM"):
            declared = opts.get("PERIODIC")
            if declared is None:
                raise ValueError(f"{name} requires explicit PERIODIC")
            bounds = None if declared == "NO" else _numbers(declared)
            if bounds is not None and (len(bounds) != 2 or bounds[1] <= bounds[0]):
                raise ValueError("Invalid periodic interval")
            periods[label] = 0.0 if bounds is None else bounds[1] - bounds[0]
            if bounds is not None:
                lower[label] = bounds[0]
        else:
            periods.setdefault(label, 0.0)
        if name == "METAD":
            metad[label] = _metadynamics_config(opts, units, periods, hill_capacity)
        if name == "OPES_METAD":
            if walkers > 1 and "WALKERS_MPI" not in opts:
                raise ValueError("OPES_METAD with walkers>1 requires WALKERS_MPI")
            metad[label] = _opes_config(
                opts, units, periods, lower, hill_capacity, walkers
            )
    return Program(
        tuple(actions), units, groups, references, expressions, periods, metad
    )


@dataclass
class Program:
    """Parsed static CV/bias definitions with separate caller-owned histories.

    Construct using parse; evaluate with values/bias and advance histories
    with update/bias_and_update. Store initial_state() separately. Atom
    selections are static, while positions/cell/state are runtime arrays.
    This object is mutable for compatibility; editing captured definitions
    requires rebuilding compiled callables. No file output is scheduled.
    """

    actions: tuple[Action, ...]
    units: Units
    groups: dict[str, tuple[int, ...]]
    references: dict
    expressions: dict[str, Callable]
    periods: dict[str, float]
    metad: dict[str, Metadynamics | GridMetadynamics | OPESMetad]

    def __post_init__(self):
        for action in self.actions:
            if action.name in _GEOMETRY_ATOM_COUNTS and "ATOMS" in action.options:
                _geometry_selection(action.options["ATOMS"])

    def initial_state(self) -> dict[str, HillState | GridState | OPESState]:
        """Return a dictionary mapping METAD labels to fresh hill/grid states.

        The dictionary is empty without METAD actions. Floating arrays default
        to float64 and require jax_enable_x64=True. State is explicit: keep the
        returned dictionary and pass it into bias/update calls.
        """
        return {name: config.init() for (name, config) in self.metad.items()}

    def _selection(self, text):
        result = []
        for item in text.split(","):
            if item in self.groups:
                result.extend(self.groups[item])
            elif "-" in item:
                lo, hi = map(int, item.split("-"))
                if hi < lo:
                    raise ValueError("Reversed atom range")
                result.extend(range(lo - 1, hi))
            else:
                result.append(int(item) - 1)
        if any(i < 0 for i in result):
            raise ValueError("Atom indices start at 1")
        return tuple(result)

    def _validate_selections(self, positions):
        # Selections are static, so these checks run while tracing, not per step.
        virtual_atoms = {
            action.label for action in self.actions if action.name == "COM"
        }
        for action in self.actions:
            for key in ("ATOM", "ATOMS", "GROUPA", "GROUPB"):
                if key not in action.options:
                    continue
                spec = action.options[key]
                if action.name in _GEOMETRY_ATOM_COUNTS:
                    _, indices = _geometry_selection(spec)
                    if any(
                        isinstance(i, str) and i not in virtual_atoms for i in indices
                    ):
                        raise ValueError(
                            "This geometry action accepts atoms and COM virtual atoms only"
                        )
                    ids = [i for i in indices if isinstance(i, int)]
                else:
                    ids = self._selection(spec)
                if any(i < 0 or i >= positions.shape[0] for i in ids):
                    raise ValueError("Atom selection outside position array")
            if action.label in self.references:
                if any(
                    i < 0 or i >= positions.shape[0]
                    for i in self.references[action.label][1]
                ):
                    raise ValueError("Reference atom outside position array")

    def values(
        self,
        positions: Array,
        cell: ArrayLike | PreparedCell | None = None,
        *,
        masses: ArrayLike | None = None,
        share_geometry: bool = False,
    ) -> dict[str, Array]:
        """Return a dictionary of CV arrays keyed by action/component labels.

        positions has shape (N, 3); cell is None, a (3, 3) row-vector cell, or
        PreparedCell. Coordinates/cell already use this Program's length unit.
        masses has shape (N,) and is required for COM. Outputs have the shapes
        and units of each action (including labels such as position.x).
        Invalid selections, missing masses, or unsupported action variants raise
        ValueError. Fix share_geometry before JIT; True shares cell/pair work
        within this evaluation and can change derivative rounding. No bias
        history is read or updated. Rebuild JIT callables after editing actions.
        """
        if (
            share_geometry
            and cell is not None
            and not isinstance(cell, PreparedCell)
            and any(
                action.name in _CELL_GEOMETRY_ACTIONS
                and not action.options.get("NOPBC")
                for action in self.actions
            )
        ):
            cell = prepare_cell(cell)
        pair_vectors = {}

        def cached_displacement(atoms, points, i, j, pair_cell):
            key = (atoms[i], atoms[j], pair_cell is not None)
            if key not in pair_vectors:
                pair_vectors[key] = displacement(points[i], points[j], pair_cell)
            return pair_vectors[key]

        self._validate_selections(positions)
        result = {}
        for action in self.actions:
            name, options, label = action.name, action.options, action.label
            action_cell = None if options.get("NOPBC") else cell
            if name in (
                "GROUP",
                "PRINT",
                "METAD",
                "OPES_METAD",
                "RESTRAINT",
                "UPPER_WALLS",
                "LOWER_WALLS",
            ):
                continue
            if name in _GEOMETRY_ATOM_COUNTS:
                atoms, indices = _geometry_selection(options["ATOMS"])
                if any(
                    isinstance(i, str) and t not in result
                    for t, i in zip(atoms, indices)
                ):
                    raise ValueError("COM virtual atoms must precede geometry actions")
                points = [
                    result[t] if t in result else positions[i]
                    for t, i in zip(atoms, indices)
                ]
                atom_count = _GEOMETRY_ATOM_COUNTS[name]
                if len(points) != atom_count:
                    raise ValueError(f"{name} requires {atom_count} atoms")
                if not share_geometry:
                    xyz = jnp.stack(points)
                    result[label] = getattr(cvs, name.lower())(
                        xyz, *range(atom_count), cell=action_cell
                    )
                elif name == "DISTANCE":
                    result[label] = jnp.linalg.norm(
                        cached_displacement(atoms, points, 0, 1, action_cell)
                    )
                elif name == "ANGLE":
                    result[label] = cvs._angle_from_vectors(
                        cached_displacement(atoms, points, 1, 0, action_cell),
                        cached_displacement(atoms, points, 1, 2, action_cell),
                    )
                else:
                    result[label] = cvs._torsion_from_vectors(
                        cached_displacement(atoms, points, 0, 1, action_cell),
                        cached_displacement(atoms, points, 1, 2, action_cell),
                        cached_displacement(atoms, points, 2, 3, action_cell),
                    )
            elif name == "POSITION":
                result[label] = cvs.position(
                    positions, int(options["ATOM"]) - 1, action_cell
                )
                for i, component in enumerate("xyz"):
                    result[label + "." + component] = result[label][i]
            elif name == "COM":
                if masses is None:
                    raise ValueError("COM requires engine masses")
                result[label] = cvs.center_of_mass(
                    positions, self._selection(options["ATOMS"]), masses, action_cell
                )
            elif name == "GYRATION":
                if options.get("TYPE", "RADIUS") != "RADIUS":
                    raise ValueError("Only GYRATION TYPE=RADIUS is supported")
                result[label] = cvs.radius_of_gyration(
                    positions, self._selection(options["ATOMS"]), cell=action_cell
                )
            elif name == "COORDINATION":
                group_a = self._selection(options["GROUPA"])
                group_b = (
                    self._selection(options["GROUPB"]) if "GROUPB" in options else None
                )
                result[label] = _coordination(
                    positions, group_a, group_b, options, action_cell
                )
            elif name in ("RMSD", "PATHMSD"):
                refs, indices, (aw, dw) = self.references[label]
                kwargs = dict(
                    alignment_weights=aw, displacement_weights=dw, cell=action_cell
                )
                if name == "RMSD":
                    typ = options.get("TYPE", "SIMPLE")
                    if typ not in ("SIMPLE", "OPTIMAL") or len(refs) != 1:
                        raise ValueError("Single-reference SIMPLE/OPTIMAL RMSD only")
                    result[label] = rmsd(
                        positions,
                        refs[0],
                        indices,
                        optimal=typ == "OPTIMAL",
                        squared=bool(options.get("SQUARED")),
                        **kwargs,
                    )
                else:
                    s, z = path_msd(
                        positions, refs, float(options["LAMBDA"]), indices, **kwargs
                    )
                    result[label + ".sss"] = s
                    result[label + ".zzz"] = z
            elif name in ("COMBINE", "CUSTOM"):
                values = jnp.array([result[arg] for arg in options["ARG"].split(",")])
                if name == "COMBINE":
                    coefficients = jnp.array(
                        _numbers(
                            options.get("COEFFICIENTS", ",".join(["1"] * len(values)))
                        )
                    )
                    if options.get("NORMALIZE"):
                        coefficients = coefficients / jnp.sum(coefficients)
                    parameters = jnp.array(
                        _numbers(
                            options.get("PARAMETERS", ",".join(["0"] * len(values)))
                        )
                    )
                    powers = jnp.array(
                        _numbers(options.get("POWERS", ",".join(["1"] * len(values))))
                    )
                    if any(
                        v.shape != values.shape
                        for v in (coefficients, parameters, powers)
                    ):
                        raise ValueError("COMBINE component lengths differ")
                    result[label] = jnp.sum(
                        coefficients * (values - parameters) ** powers
                    )
                else:
                    if len(values) > 3:
                        raise ValueError("CUSTOM supports at most x,y,z")
                    result[label] = self.expressions[label](
                        dict(zip(("x", "y", "z"), values))
                    )
            elif name == "CONSTANT":
                result[label] = jnp.asarray(
                    _numbers(options.get("VALUES", options.get("VALUE")))
                ).squeeze()
            else:
                raise ValueError(f"Unsupported action {name}")
        return result

    def bias(
        self,
        positions: Array,
        cell: ArrayLike | PreparedCell | None = None,
        *,
        masses: ArrayLike | None = None,
        state: dict[str, HillState | GridState | OPESState] | None = None,
        share_geometry: bool = False,
    ) -> Array:
        """Return the scalar sum of restraints, walls, and METAD bias energies.

        positions/cell/masses follow values. Output uses this Program's energy
        unit. state maps METAD labels to their current histories; missing
        METAD state raises ValueError. Evaluation does not deposit hills.
        Fix share_geometry before JIT; use -jax.grad for atomic bias forces.
        """
        values = self.values(
            positions, cell, masses=masses, share_geometry=share_geometry
        )
        return self._bias_from_values(values, positions.dtype, state)[0]

    def _bias_from_values(self, values, dtype, state):
        energy = jnp.array(0.0, dtype=dtype)
        metad_energies = {}
        for action in self.actions:
            options = action.options
            if action.name not in (
                "RESTRAINT", *_STATEFUL_BIASES, "UPPER_WALLS", "LOWER_WALLS"
            ):
                continue
            args = options["ARG"].split(",")
            v = jnp.array([values[t] for t in args])
            if action.name in _STATEFUL_BIASES:
                if state is None or action.label not in state:
                    raise ValueError("Explicit METAD state is required")
                term = self.metad[action.label].energy(v, state[action.label])
                metad_energies[action.label] = term
                energy += term
            elif action.name == "RESTRAINT":
                energy += harmonic_restraint(
                    v,
                    _numbers(options["AT"]),
                    _numbers(options.get("KAPPA", "0")),
                    slopes=_numbers(options.get("SLOPE", "0")),
                    periods=jnp.array([self.periods.get(t, 0.0) for t in args]),
                )
            else:
                energy += wall(
                    v,
                    _numbers(options["AT"]),
                    _numbers(options["KAPPA"]),
                    upper=action.name == "UPPER_WALLS",
                    scale=jnp.array(_numbers(options.get("EPS", "1"))),
                    exponent=jnp.array(_numbers(options.get("EXP", "2"))),
                    offset=jnp.array(_numbers(options.get("OFFSET", "0"))),
                )
        return energy, metad_energies

    def update(
        self,
        positions: Array,
        state: dict[str, HillState | GridState | OPESState],
        cell: ArrayLike | PreparedCell | None = None,
        *,
        masses: ArrayLike | None = None,
        step: ArrayLike | None = None,
        bias_energies: dict[str, ArrayLike] | None = None,
        share_geometry: bool = False,
    ) -> dict[str, HillState | GridState | OPESState]:
        """Return a dictionary of next METAD histories after force evaluation.

        positions/cell/masses follow values; state maps METAD labels to histories.
        step is an increasing integer scalar, or None advances each history.
        bias_energies optionally maps each METAD label to its own scalar bias
        at these coordinates and the pre-deposition history, not total energy.
        Fix share_geometry before JIT. Check each returned state's overflow
        on the host; the input dictionary is not mutated.
        """
        values = self.values(
            positions, cell, masses=masses, share_geometry=share_geometry
        )
        return self._update_from_values(values, state, step, bias_energies)

    def _update_from_values(self, values, state, step, bias_energies):
        energies = {} if bias_energies is None else bias_energies
        return {
            action.label: self.metad[action.label].update(
                jnp.array([values[t] for t in action.options["ARG"].split(",")]),
                state[action.label],
                step=step,
                bias_energy=energies.get(action.label),
            )
            for action in self.actions
            if action.name in _STATEFUL_BIASES
        }

    def bias_and_update(
        self,
        positions: Array,
        state: dict[str, HillState | GridState | OPESState],
        cell: ArrayLike | PreparedCell | None = None,
        *,
        masses: ArrayLike | None = None,
        step: ArrayLike | None = None,
        share_geometry: bool = False,
    ) -> tuple[Array, dict[str, HillState | GridState | OPESState]]:
        """Return (scalar pre-deposition bias energy, next METAD state dictionary).

        positions/cell/masses follow values; state/step follow update. CVs and
        each individual METAD bias are evaluated once and reused for deposition.
        Fix share_geometry before JIT. With jax.value_and_grad(..., has_aux=True),
        the next state is auxiliary output and atomic gradients use the old
        history. Check overflow on the host; the input state is not mutated.
        """
        values = self.values(
            positions, cell, masses=masses, share_geometry=share_geometry
        )
        energy, metad_energies = self._bias_from_values(values, positions.dtype, state)
        return energy, self._update_from_values(values, state, step, metad_energies)

    def _walker_values(self, positions, cell, masses, share_geometry):
        if positions.ndim != 3:
            raise ValueError("Walker positions must have shape (walkers, atoms, 3)")
        return jax.vmap(
            lambda r: self.values(r, cell, masses=masses, share_geometry=share_geometry)
        )(positions)

    def _update_walkers_from_values(self, values, state, step, bias_energies):
        energies = {} if bias_energies is None else bias_energies
        return {
            action.label: self.metad[action.label].update_walkers(
                jnp.stack(
                    [values[t] for t in action.options["ARG"].split(",")], axis=-1
                ),
                state[action.label],
                step=step,
                bias_energies=energies.get(action.label),
            )
            for action in self.actions
            if action.name in _STATEFUL_BIASES
        }

    def update_walkers(
        self,
        positions: Array,
        state: dict[str, HillState | GridState | OPESState],
        cell: ArrayLike | PreparedCell | None = None,
        *,
        masses: ArrayLike | None = None,
        step: ArrayLike | None = None,
        bias_energies: dict[str, Array] | None = None,
        share_geometry: bool = False,
    ) -> dict[str, HillState | GridState | OPESState]:
        """Synchronously update every shared bias from a batch of walkers.

        positions has shape (walkers, atoms, 3); cell and masses are shared by
        all walkers. Each METAD/OPES_METAD history receives one deposit per
        walker through its update_walkers method (OPES_METAD requires
        WALKERS_MPI and parse(..., walkers=W)). bias_energies optionally maps
        each label to its (walkers,) energies from the old state. Check every
        returned state's overflow on the host; the input is not mutated.
        """
        values = self._walker_values(positions, cell, masses, share_geometry)
        return self._update_walkers_from_values(values, state, step, bias_energies)

    def bias_and_update_walkers(
        self,
        positions: Array,
        state: dict[str, HillState | GridState | OPESState],
        cell: ArrayLike | PreparedCell | None = None,
        *,
        masses: ArrayLike | None = None,
        step: ArrayLike | None = None,
        share_geometry: bool = False,
    ) -> tuple[Array, dict[str, HillState | GridState | OPESState]]:
        """Return ((walkers,) pre-deposition bias energies, next shared state).

        positions has shape (walkers, atoms, 3). All walkers read the same old
        state, then update_walkers deposits synchronously; CVs and individual
        bias energies are evaluated once. For forces differentiate the sum of
        the returned energies (walkers are independent) with has_aux=True.
        """
        values = self._walker_values(positions, cell, masses, share_geometry)
        energies, metad_energies = jax.vmap(
            lambda v: self._bias_from_values(v, positions.dtype, state)
        )(values)
        return energies, self._update_walkers_from_values(
            values, state, step, metad_energies
        )
