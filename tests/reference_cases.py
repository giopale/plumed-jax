"""Pure case definitions used by the offline PLUMED reference tests."""
from dataclasses import dataclass, replace
import itertools
import jax.numpy as jnp
import numpy as np
import plumed_jax as cv
from plumed_jax import cvs
from plumed_jax.bias import harmonic_restraint, wall
from plumed_jax.geometry import displacement
from plumed_jax.parser import parse
from plumed_jax.reference import rmsd, path_msd
from plumed_jax.structure import steinhardt, tetrahedral_order, contact_matrix
from plumed_jax.switches import switching


BASE = np.array([
    [0.2, 0.3, 0.4],
    [1.4, 0.5, 0.2],
    [1.7, 1.8, 0.7],
    [2.2, 2.1, 1.9],
    [3.2, 0.8, 1.1],
], dtype=float)

MASSES = np.array([12.0, 1.0, 16.0, 1.0, 16.0])

@dataclass(frozen=True)
class Case:
    name: str
    lines: tuple[str, ...]
    output: str
    fn: object
    atoms: tuple[int, ...]
    gradient: bool = True
    periodic_value: bool = False
    direct_bias: bool = False
    files: tuple = ()

CASES = (
    Case("distance", ("cv: DISTANCE ATOMS=1,2",), "cv",
         lambda x, h: cv.distance(x, 0, 1, h), (0, 1)),
    Case("angle", ("cv: ANGLE ATOMS=1,2,3",), "cv",
         lambda x, h: cv.angle(x, 0, 1, 2, h), (0, 1, 2)),
    Case("torsion", ("cv: TORSION ATOMS=1,2,3,4",), "cv",
         lambda x, h: cv.torsion(x, 0, 1, 2, 3, h), (0, 1, 2, 3),
         periodic_value=True),
    Case("coordination", ("cv: COORDINATION GROUPA=1,2 GROUPB=3,4 R_0=1.5",),
         "cv", lambda x, h: cv.coordination_number(x, [0, 1], [2, 3], r0=1.5, cell=h),
         (0, 1, 2, 3)),
    Case("coordination_single", ("cv: COORDINATION GROUPA=1,2,3,4 R_0=1.5",),
         "cv", lambda x, h: cv.coordination_number(x, [0, 1, 2, 3], r0=1.5, cell=h),
         (0, 1, 2, 3)),
    Case("coordination_overlap", ("cv: COORDINATION GROUPA=1,2 GROUPB=2,3 R_0=1.5",),
         "cv", lambda x, h: cv.coordination_number(x, [0, 1], [1, 2], r0=1.5, cell=h),
         (0, 1, 2)),
    Case("coordination_shifted",
         ("cv: COORDINATION GROUPA=1,2 GROUPB=3,4 R_0=1.5 D_0=0.3 NN=4 MM=8",),
         "cv", lambda x, h: cv.coordination_number(
             x, [0, 1], [2, 3], r0=1.5, d0=0.3, n=4, m=8, cell=h),
         (0, 1, 2, 3)),
    Case("coordination_general",
         ("cv: COORDINATION GROUPA=1,2 GROUPB=3,4 R_0=1.5 D_0=0.2 NN=4 MM=7",),
         "cv", lambda x, h: cv.coordination_number(
             x, [0, 1], [2, 3], r0=1.5, d0=0.2, n=4, m=7, cell=h),
         (0, 1, 2, 3)),
    Case("gyration", ("cv: GYRATION ATOMS=1,2,3,4 TYPE=RADIUS",), "cv",
         lambda x, h: cv.radius_of_gyration(x, [0, 1, 2, 3], cell=h), (0, 1, 2, 3)),
    Case("position_x", ("cv: POSITION ATOM=1",), "cv.x",
         lambda x, h: cv.position(x, 0, h)[0], (0,)),
    Case("position_y", ("cv: POSITION ATOM=1",), "cv.y",
         lambda x, h: cv.position(x, 0, h)[1], (0,)),
    Case("position_z", ("cv: POSITION ATOM=1",), "cv.z",
         lambda x, h: cv.position(x, 0, h)[2], (0,)),
    Case("com_distance", ("v: COM ATOMS=1,2,3,4", "cv: DISTANCE ATOMS=v,5"),
         "cv", lambda x, h: jnp.linalg.norm(displacement(
             cv.center_of_mass(x, [0, 1, 2, 3], jnp.asarray(MASSES), h), x[4], h)),
         (0, 1, 2, 3, 4)),
    Case("proton_transfer", ("a: DISTANCE ATOMS=1,2", "b: DISTANCE ATOMS=3,2",
                              "cv: COMBINE ARG=a,b COEFFICIENTS=1,-1 PERIODIC=NO"),
         "cv", lambda x, h: cv.proton_transfer(x, 0, 1, 2, h), (0, 1, 2)),
)

def pdb(frame,alignment=None,measurement=None):
    alignment=np.ones(len(frame)) if alignment is None else alignment
    measurement=np.ones(len(frame)) if measurement is None else measurement
    return ''.join(f'ATOM  {i+1:5d}  C   MOL A   1    {x:8.3f}{y:8.3f}{z:8.3f}{a:6.2f}{d:6.2f}           C\n'
                   for i,((x,y,z),a,d) in enumerate(zip(frame,alignment,measurement)))+'END\n'

AW=np.array([1.,2.,3.,4.,5.])

DW=AW[::-1].copy()

REFS=np.stack([BASE,BASE+np.array([[.1,.2,0.],[0.,-.1,.2],[.2,0.,.1],[0.,.1,-.1],[-.1,0.,.1]]),BASE+np.array([[.3,.2,0.],[0.,-.2,.3],[.2,0.,.4],[0.,.1,-.2],[-.1,0.,.1]])])

def extended_cases():
    yield Case('harmonic_distance',('cv: DISTANCE ATOMS=1,2','b: RESTRAINT ARG=cv AT=1.7 KAPPA=2.3 SLOPE=0.4'),
               'b.bias',lambda x,h:harmonic_restraint(cvs.distance(x,0,1,h),1.7,2.3,slopes=.4),tuple(range(5)),direct_bias=True)
    yield Case('harmonic_torsion',('cv: TORSION ATOMS=1,2,3,4','b: RESTRAINT ARG=cv AT=-3.1 KAPPA=2.3'),
               'b.bias',lambda x,h:harmonic_restraint(cvs.torsion(x,0,1,2,3,h),-3.1,2.3,periods=2*np.pi),tuple(range(5)),direct_bias=True)
    for upper in (True,False):
        action='UPPER_WALLS' if upper else 'LOWER_WALLS'
        yield Case(action.lower(),('cv: DISTANCE ATOMS=1,2',f'b: {action} ARG=cv AT=1.5 KAPPA=2.3 EPS=0.7 EXP=2 OFFSET=0.1'),
                   'b.bias',lambda x,h,u=upper:wall(cvs.distance(x,0,1,h),1.5,2.3,upper=u,scale=.7,offset=.1),tuple(range(5)),direct_bias=True)
    for optimal in (False,True):
        typ='OPTIMAL' if optimal else 'SIMPLE'
        yield Case('rmsd_'+typ, (f'cv: RMSD REFERENCE=ref.pdb TYPE={typ} SQUARED',),'cv',
                   lambda x,h,o=optimal:rmsd(x,jnp.array(BASE),alignment_weights=jnp.array(AW),displacement_weights=jnp.array(DW),optimal=o,squared=True,cell=h),tuple(range(5)),files=(('ref.pdb',pdb(BASE,AW,DW)),))
        yield Case('rmsd_unsquared_'+typ,(f'cv: RMSD REFERENCE=ref.pdb TYPE={typ}',),'cv',
                   lambda x,h,o=optimal:rmsd(x,jnp.array(BASE),alignment_weights=jnp.array(AW),displacement_weights=jnp.array(DW),optimal=o,squared=False,cell=h),tuple(range(5)),files=(('ref.pdb',pdb(BASE,AW,DW)),))
    for component in (0,1):
        yield Case('path_'+str(component),('cv: PATHMSD REFERENCE=path.pdb LAMBDA=7.0',),('cv.sss','cv.zzz')[component],
                   lambda x,h,c=component:path_msd(x,jnp.array(REFS),7.,cell=h)[c],tuple(range(5)),files=(('path.pdb',''.join(pdb(r) for r in REFS)),))
    for kind in ('EXP','GAUSSIAN','TANH','SMAP','COSINUS','CUBIC'):
        extra=' A=4 B=8' if kind=='SMAP' else ''
        switch=f'{kind} R_0=1.4 D_0=0.2 D_MAX=3.0'+extra
        if kind=='CUBIC': switch='CUBIC D_0=0.2 D_MAX=3.0'
        yield Case('switch_'+kind,(f'cv: COORDINATION GROUPA=1 GROUPB=2 SWITCH={{{switch}}}',),'cv',
                   lambda x,h,k=kind:switching(cvs.distance(x,0,1,h),kind=k,r0=1.4,d0=.2,dmax=3.,a=4.,b=8.,stretch=True),tuple(range(5)))
    lines=[];terms=[]
    import itertools
    for k,(i,j) in enumerate(itertools.combinations(range(2,6),2)):
        lines.extend([f'a{k}: ANGLE ATOMS={i},1,{j}',f't{k}: CUSTOM ARG=a{k} FUNC=(cos(x)+1/3)^2 PERIODIC=NO'])
        terms.append(f't{k}')
    lines.extend(['s: COMBINE ARG='+','.join(terms)+' PERIODIC=NO','cv: CUSTOM ARG=s FUNC=1-3*x/8 PERIODIC=NO'])
    yield Case('water_tetrahedral_composite',tuple(lines),'cv',lambda x,h:tetrahedral_order(x,0,[1,2,3,4],cell=h),tuple(range(5)))

SEEDS = (1701, 2903, 4309)

SIZES = (5, 16, 64)

BOUNDARIES = ("nonperiodic", "cubic", "orthorhombic", "triclinic")

def cloud(n, seed):
    """Uniform points in a 4-Angstrom cube, minimum separation 0.45 Angstrom."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, n]))
    points = []
    attempts = 0
    while len(points) < n:
        attempts += 1
        if attempts > 100000:
            raise RuntimeError("Point-cloud rejection sampler did not terminate")
        point = rng.uniform(.5, 4.5, 3)
        if all(np.linalg.norm(point - other) > .45 for other in points):
            points.append(point)
    result = np.array(points)
    # Explicitly exclude singular selections used by angle and torsion CVs.
    bonds = np.diff(result[:4], axis=0)
    if min(np.linalg.norm(np.cross(bonds[i], bonds[i + 1])) for i in (0, 1)) < .05:
        raise RuntimeError("Seed produced a nearly singular selected torsion")
    return result

def baseline_cases(n):
    """All original scalar families, with group/reference observables using N atoms."""
    all_atoms = tuple(range(n))
    group_a, group_b = tuple(range(n // 2)), tuple(range(n // 2, n))
    overlap_a = tuple(range(n // 2 + 1))
    masses = 1. + np.arange(n) % 5
    result = []
    for case in CASES:
        if case.name.startswith("coordination"):
            single = case.name == "coordination_single"
            a = overlap_a if case.name == "coordination_overlap" else group_a
            b = None if single else group_b
            if single:
                a = all_atoms
            params = {"r0": 1.5}
            if case.name == "coordination_shifted":
                params.update(d0=.3, n=4, m=8)
            elif case.name == "coordination_general":
                params.update(d0=.2, n=4, m=7)
            selection = "GROUPA=" + ",".join(str(i + 1) for i in a)
            if b is not None:
                selection += " GROUPB=" + ",".join(str(i + 1) for i in b)
            settings = " R_0=1.5"
            if "d0" in params:
                settings += f" D_0={params['d0']} NN={params['n']} MM={params['m']}"
            result.append(replace(case, lines=(f"cv: COORDINATION {selection}{settings}",),
                fn=lambda x, h, a=a, b=b, p=params: cvs.coordination_number(x, a, b, cell=h, **p),
                atoms=all_atoms))
        elif case.name == "gyration":
            result.append(replace(case, lines=(f"cv: GYRATION ATOMS=1-{n} TYPE=RADIUS",),
                fn=lambda x, h: cvs.radius_of_gyration(x, all_atoms, cell=h), atoms=all_atoms))
        elif case.name == "com_distance":
            selection = tuple(range(n - 1))
            result.append(replace(case, lines=(f"v: COM ATOMS=1-{n-1}", f"cv: DISTANCE ATOMS=v,{n}"),
                fn=lambda x, h: jnp.linalg.norm(displacement(
                    cvs.center_of_mass(x, selection, jnp.asarray(masses), h), x[n - 1], h)),
                atoms=all_atoms))
        else:
            result.append(case)

    # PDB coordinates and weights have finite printed precision; share exactly
    # those rounded numbers with JAX instead of comparing different references.
    reference = np.round(cloud(n, 99991), 3)
    rng = np.random.default_rng(83000 + n)
    references = np.stack([reference, np.round(reference + rng.normal(0, .12, reference.shape), 3),
                           np.round(reference + rng.normal(0, .24, reference.shape), 3)])
    alignment = 1. + np.arange(n) % 5
    measurement = alignment[::-1].copy()
    for case in extended_cases():
        if case.name.startswith("rmsd"):
            optimal = case.name.endswith("OPTIMAL")
            squared = "unsquared" not in case.name
            result.append(replace(case,
                fn=lambda x, h, o=optimal, s=squared: rmsd(x, jnp.asarray(reference),
                    alignment_weights=jnp.asarray(alignment), displacement_weights=jnp.asarray(measurement),
                    optimal=o, squared=s, cell=h), atoms=all_atoms,
                files=(("ref.pdb", pdb(reference, alignment, measurement)),)))
        elif case.name.startswith("path_"):
            component = int(case.name[-1])
            result.append(replace(case, fn=lambda x, h, c=component:
                path_msd(x, jnp.asarray(references), 7., cell=h)[c], atoms=all_atoms,
                files=(("path.pdb", "".join(pdb(frame) for frame in references)),)))
        else:
            result.append(case)
    result.append(Case("q6", (f"q: Q6 SPECIESA=1 SPECIESB=2-{n} SWITCH={{RATIONAL R_0=1.5}} MEAN",),
        "q.mean", lambda x, h: steinhardt(x, [0], tuple(range(1, n)), l=6, r0=1.5,
            n=6, m=12, normalization="plumed", cell=h)[0], all_atoms))
    assert len(result) == 32
    return result, masses

def fixtures():
    for n in SIZES:
        cases, masses = baseline_cases(n)
        for seed in SEEDS:
            points = cloud(n, seed)
            for boundary in BOUNDARIES:
                cells = {"nonperiodic": None, "cubic": np.eye(3) * 12.,
                         "orthorhombic": np.diag([12., 13., 14.]),
                         "triclinic": np.array([[12., 0., 0.], [4.3, 11.7, 0.], [-1.1, 2.4, 13.2]])}
                cell = cells[boundary]
                positions = points.copy()
                if cell is not None:
                    translated = positions + np.array([10.1, 11.2, 12.3])
                    positions = ((translated @ np.linalg.inv(cell)) % 1.) @ cell
                yield dict(name=f"random-n{n}-seed{seed}-{boundary}", positions=positions, cell=cell,
                    masses=masses, cases=cases, kind="random", seed=seed,
                    note="Uniform cube with minimum pair separation; translated/wrapped for PBC.")
    for fixture in targets():
        yield dict(fixture, kind="targeted", seed=None)

def _fixture(name, positions, cases, note, *, cell=None, expected=None):
    return dict(
        name=name,
        positions=np.asarray(positions, dtype=float).copy(),
        cell=None if cell is None else np.asarray(cell, dtype=float).copy(),
        cases=list(cases),
        masses=MASSES.copy(),
        note=note,
        expected_values={} if expected is None else dict(expected),
    )

def _pair(distance):
    positions = BASE.copy()
    positions[0] = (0.0, 0.0, 0.0)
    positions[1] = (distance, 0.0, 0.0)
    return positions

def targets():
    """Return 19 fixed geometries and their applicable scalar comparisons.

    Entries use the same ``Case`` interface as ``verify_against_plumed``.
    References are rounded to the PDB coordinate precision before constructing
    either implementation's input; this avoids comparing two different
    reference geometries. The optional ``expected_values`` are additional
    analytic/construction checks and do not replace the PLUMED oracle.
    """
    core = {case.name: case for case in CASES}
    extended = {case.name: case for case in extended_cases()}
    result = []
    rational = Case(
        "rational_removable_singularity",
        ("cv: COORDINATION GROUPA=1 GROUPB=2 R_0=1.5 D_0=0.2 NN=4 MM=7",),
        "cv", lambda x, h: cvs.coordination_number(
            x, [0], [1], r0=1.5, d0=.2, n=4, m=7, cell=h
        ), tuple(range(5)),
    )
    for suffix, offset in (("below", -1e-5), ("exact", 0.0), ("above", 1e-5)):
        result.append(_fixture(
            f"rational_r0_{suffix}", _pair(1.7 + offset), [rational],
            "Rational switch at reduced distance 1, a removable algebraic "
            "singularity with finite value and derivative.",
        ))

    switches = [extended[f"switch_{kind}"] for kind in
                ("EXP", "GAUSSIAN", "TANH", "SMAP", "COSINUS", "CUBIC")]
    for suffix, offset in (("below", -1e-5), ("exact", 0.0), ("above", 1e-5)):
        selected = [extended["switch_CUBIC"]] if offset == 0 else switches
        result.append(_fixture(
            f"explicit_cutoff_{suffix}", _pair(3.0 + offset), selected,
            "One-sided tests at the explicit 3 Angstrom switch cutoff; "
            "the exact cutoff tests only CUBIC, whose first derivative is "
            "continuous and zero there.",
            expected={c.name: 0.0 for c in selected} if offset >= 0 else None,
        ))

    for suffix, theta in (("negative", -np.pi + 1e-3), ("positive", np.pi - 1e-3)):
        positions = np.array([
            [1., 0., 0.], [0., 0., 0.], [0., 0., 1.],
            [np.cos(theta), np.sin(theta), 1.], [2., 2., 2.],
        ])
        result.append(_fixture(
            f"torsion_wrap_{suffix}", positions,
            [core["torsion"], extended["harmonic_torsion"]],
            "A well-defined torsion 0.001 radian from the signed pi wrap, "
            "also tested through a periodic harmonic bias.",
            expected={"torsion": theta},
        ))

    for suffix, theta in (("near_zero", 1e-3), ("near_pi", np.pi - 1e-3)):
        positions = np.array([
            [1., 0., 0.], [0., 0., 0.],
            [np.cos(theta), np.sin(theta), 0.], [0., 0., 1.], [1., 2., 3.],
        ])
        result.append(_fixture(
            f"angle_{suffix}", positions, [core["angle"]],
            "Nearly collinear angle; the exact singular configuration is avoided.",
            expected={"angle": theta},
        ))

    wall_cases = []
    for upper in (True, False):
        action = "UPPER_WALLS" if upper else "LOWER_WALLS"
        wall_cases.append(Case(
            f"wall_activation_{action.lower()}",
            ("cv: DISTANCE ATOMS=1,2",
             f"b: {action} ARG=cv AT=1.5 KAPPA=2.3 EPS=0.7 EXP=2 OFFSET=0"),
            "b.bias",
            lambda x, h, u=upper: wall(cvs.distance(x, 0, 1, h), 1.5, 2.3,
                                      upper=u, scale=.7),
            tuple(range(5)), direct_bias=True,
        ))
    for suffix, offset in (("below", -1e-4), ("exact", 0.0), ("above", 1e-4)):
        expected = {
            wall_cases[0].name: 2.3 * (max(offset, 0) / .7) ** 2,
            wall_cases[1].name: 2.3 * (max(-offset, 0) / .7) ** 2,
        }
        result.append(_fixture(
            f"wall_activation_{suffix}", _pair(1.5 + offset), wall_cases,
            "Quadratic walls immediately below, at, and above activation; "
            "both the energy and its first derivative are continuous.",
            expected=expected,
        ))

    skew = np.array([[5., 0., 0.], [4.6, 1.4, 0.], [.5, .7, 4.8]])
    positions = _pair(1.0)
    positions[1] = np.array([.49, .49, .13]) @ skew
    shifts = np.array(list(itertools.product(range(-3, 4), repeat=3))) @ skew
    expected_distance = np.linalg.norm(positions[1] - shifts, axis=1).min()
    result.append(_fixture(
        "skew_nearest_image", positions, [core["distance"]],
        "Tilted-cell displacement for which rounding each fractional "
        "coordinate does not select the nearest image; the expected distance "
        "comes from independent finite lattice enumeration.",
        cell=skew, expected={"distance": expected_distance},
    ))
    positions = BASE.copy()
    positions[0] = (.10, .15, .20)
    positions[1] = (4.90, 4.85, 4.80)
    result.append(_fixture(
        "cube_corner_crossing", positions, [core["distance"]],
        "The nearest pair displacement crosses all three cubic-cell faces.",
        cell=np.eye(3) * 5., expected={"distance": np.linalg.norm([.2, .3, .4])},
    ))

    reference = np.round(BASE, 3)
    rmsd_cases = []
    for optimal in (False, True):
        kind = "OPTIMAL" if optimal else "SIMPLE"
        rmsd_cases.append(Case(
            f"target_rmsd_squared_{kind}",
            (f"cv: RMSD REFERENCE=ref.pdb TYPE={kind} SQUARED",), "cv",
            lambda x, h, o=optimal: rmsd(x, jnp.asarray(reference), optimal=o,
                                          squared=True, cell=h),
            tuple(range(5)), files=(("ref.pdb", pdb(reference)),),
        ))
    result.append(_fixture(
        "squared_rmsd_zero", reference, rmsd_cases,
        "Exact reference geometry: squared RMSD is differentiable at zero; "
        "unsquared RMSD is deliberately not tested here.",
        expected={case.name: 0.0 for case in rmsd_cases},
    ))
    theta = .7
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0.],
                         [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
    distorted = reference.copy()
    distorted[2] += [.12, -.08, .06]
    distorted = distorted @ rotation + [1.4, -.6, .8]
    result.append(_fixture(
        "rmsd_rotated_distorted", distorted, rmsd_cases,
        "A local distortion followed by a known rigid rotation and translation; "
        "translation-only and rotational alignment must give different results.",
    ))

    references = np.stack([reference, reference.copy(), reference.copy()])
    references[1, 2] += [.2, -.1, .1]
    references[2, 2] += [.5, -.2, .3]
    references = np.round(references, 3)
    path_cases = []
    for component, name in enumerate(("progress", "distance")):
        path_cases.append(Case(
            f"target_path_{name}",
            ("cv: PATHMSD REFERENCE=path.pdb LAMBDA=7.0",),
            ("cv.sss", "cv.zzz")[component],
            lambda x, h, c=component: path_msd(x, jnp.asarray(references), 7., cell=h)[c],
            tuple(range(5)), files=(("path.pdb", "".join(pdb(r) for r in references)),),
        ))
    for index in (0, 2):
        result.append(_fixture(
            f"path_endpoint_{index + 1}", references[index], path_cases,
            "A configuration equal to one path reference; smooth path progress "
            "is not assumed to equal that reference's integer index because "
            "all reference weights contribute.",
        ))
    assert len(result) == 19
    return result

def cases_for(n):
    atoms = tuple(range(n))
    masses = jnp.asarray(1. + np.arange(n) % 5)
    cases = []

    def add(name, lines, fn, output="cv", files=()):
        cases.append(Case(name, tuple(lines), output, fn, atoms, files=files))

    for component, axis in enumerate("xyz"):
        add(f"com_{axis}", (f"c: COM ATOMS=1-{n}",
            "cv: POSITION ATOM=c NOPBC"),
            lambda x, h, c=component: cvs.center_of_mass(x, atoms, masses, h)[c],
            output=f"cv.{axis}")
    add("gyration_mass_weighted", (f"cv: GYRATION ATOMS=1-{n} TYPE=RADIUS MASS_WEIGHTED",),
        lambda x, h: cvs.radius_of_gyration(x, atoms, masses=masses, cell=h))

    for stretched in (False, True):
        for i, j in ((0, 1), (1, 2), (0, 3)):
            line = (f"cv: COORDINATION GROUPA={i+1} GROUPB={j+1} R_0=1.5"
                if stretched else
                f"cv: COORDINATION GROUPA={i+1} GROUPB={j+1} SWITCH={{RATIONAL R_0=1.5 NOSTRETCH}}")
            add(f"contact_pair_{i}_{j}_stretch{int(stretched)}", (line,),
                lambda x, h, i=i, j=j, s=stretched:
                    cvs.contact_values(x, jnp.array([[i, j]]), 1.5, stretch=s, cell=h)[0])

    switches = {
        "RATIONAL": ("RATIONAL R_0=1.5 NOSTRETCH", dict(kind="RATIONAL", r0=1.5)),
        "EXP": ("EXP R_0=1.4 D_0=.2 D_MAX=3", dict(kind="EXP", r0=1.4, d0=.2, dmax=3., stretch=True)),
        "GAUSSIAN": ("GAUSSIAN R_0=1.4 D_0=.2 D_MAX=3", dict(kind="GAUSSIAN", r0=1.4, d0=.2, dmax=3., stretch=True)),
        "TANH": ("TANH R_0=1.4 D_0=.2 D_MAX=3", dict(kind="TANH", r0=1.4, d0=.2, dmax=3., stretch=True)),
        "SMAP": ("SMAP R_0=1.4 D_0=.2 D_MAX=3 A=4 B=8", dict(kind="SMAP", r0=1.4, d0=.2, dmax=3., a=4., b=8., stretch=True)),
        "COSINUS": ("COSINUS R_0=1.4 D_0=.2 D_MAX=3", dict(kind="COSINUS", r0=1.4, d0=.2, dmax=3., stretch=True)),
        "CUBIC": ("CUBIC D_0=.2 D_MAX=3", dict(kind="CUBIC", d0=.2, dmax=3., stretch=True)),
    }
    for mode in ("one", "two"):
        a, b = (0, 1), None if mode == "one" else (1, 2)
        selection = "ATOMS=1,2" if b is None else "GROUPA=1,2 GROUPB=2,3"
        for kind, (switch_text, settings) in switches.items():
            # Full 2x2 rational matrices; a nonself element for each additional
            # switch and group mode. Generic switch functions have own checks.
            entries = ((0, 0), (0, 1), (1, 0), (1, 1)) if kind == "RATIONAL" else ((0, 1),)
            for row, col in entries:
                add(f"matrix_{mode}_{kind}_{row}_{col}",
                    (f"m: CONTACT_MATRIX {selection} SWITCH={{{switch_text}}}",
                     f"s: SELECT_COMPONENTS ARG=m COMPONENTS={row+1}.{col+1}",
                     "cv: SUM ARG=s PERIODIC=NO"),
                    lambda x, h, a=a, b=b, r=row, c=col, settings=settings:
                        contact_matrix(x, a, b, cell=h, **settings)[r, c])

    reference = np.round(cloud(n, 99991), 3)
    for optimal in (False, True):
        name = "OPTIMAL" if optimal else "SIMPLE"
        add(f"rmsd_equal_unsquared_{name}",
            (f"cv: RMSD REFERENCE=ref.pdb TYPE={name}",),
            lambda x, h, o=optimal: rmsd(x, jnp.asarray(reference), optimal=o, cell=h),
            files=(("ref.pdb", pdb(reference)),))

    for degree in (1, 3, 4, 6):
        for normalization in ("plumed", "conventional"):
            if degree == 6 and normalization == "plumed":
                continue
            factor = np.sqrt(4*np.pi/(2*degree+1)) if normalization == "conventional" else 1.
            add(f"q{degree}_{normalization}",
                (f"q: Q{degree} SPECIESA=1 SPECIESB=2-{n} SWITCH={{RATIONAL R_0=1.5}} MEAN",
                 f"cv: CUSTOM ARG=q.mean FUNC={factor:.17g}*x PERIODIC=NO"),
                lambda x, h, l=degree, norm=normalization:
                    steinhardt(x, [0], tuple(range(1, n)), l=l, r0=1.5,
                        normalization=norm, cell=h)[0])

    for name, text, options in (
        ("raw", "RATIONAL R_0=1.5 NOSTRETCH", dict(r0=1.5)),
        ("cutoff_raw", "RATIONAL R_0=1.5 D_0=.2 D_MAX=3 NOSTRETCH", dict(r0=1.5, d0=.2, dmax=3.)),
        ("cutoff_stretched", "RATIONAL R_0=1.5 D_0=.2 D_MAX=3 NN=4 MM=7", dict(r0=1.5, d0=.2, dmax=3., n=4, m=7, stretch=True)),
    ):
        add(f"generic_rational_{name}",
            (f"cv: COORDINATION GROUPA=1 GROUPB=2 SWITCH={{{text}}}",),
            lambda x, h, opts=options: switching(cvs.distance(x, 0, 1, h), **opts))

    def parsed(name, lines, result="cv", component=None):
        program = parse("UNITS LENGTH=A\n" + "\n".join(lines))
        def evaluate(x, h):
            value = program.values(x, h)[result]
            return value if component is None else value[component]
        return evaluate

    lines = ("cv: CONSTANT VALUE=1.7",)
    add("constant_scalar", ("c: CONSTANT VALUE=1.7", "d: DISTANCE ATOMS=1,2",
        "cv: CUSTOM ARG=c,d FUNC=x+0*y PERIODIC=NO"), parsed("constant_scalar", lines))
    for component in range(3):
        definitions = ("c: CONSTANT VALUES=.3,1.7,-.4",)
        add(f"constant_vector_{component}", definitions +
            (f"s: SELECT_COMPONENTS ARG=c COMPONENTS={component+1}",
             "v: SUM ARG=s PERIODIC=NO", "d: DISTANCE ATOMS=1,2",
             "cv: CUSTOM ARG=v,d FUNC=x+0*y PERIODIC=NO"),
            parsed("constant_vector", definitions, result="c", component=component))

    base = ("a: DISTANCE ATOMS=1,2", "b: DISTANCE ATOMS=2,3", "c: DISTANCE ATOMS=3,4")
    for name, options in (
        ("linear", "COEFFICIENTS=1.2,-.4,.8"),
        ("powers", "COEFFICIENTS=1.2,-.4,.8 POWERS=2,3,1"),
        ("shifted", "COEFFICIENTS=1.2,-.4,.8 PARAMETERS=.2,.1,-.3 POWERS=2,3,1"),
        ("normalized", "COEFFICIENTS=1.2,.4,.8 PARAMETERS=.2,.1,-.3 POWERS=2,3,1 NORMALIZE"),
    ):
        lines = base + (f"cv: COMBINE ARG=a,b,c {options} PERIODIC=NO",)
        add(f"combine_{name}", lines, parsed(name, lines))

    expressions = {
        "arithmetic": "x+2*y-z/3+x^2", "pi": "pi*x/3",
        "sin": "sin(x)", "cos": "cos(x)", "tan": "tan(x/10)",
        "exp": "exp(x/10)", "log": "log(x+1)", "sqrt": "sqrt(x+1)",
        "abs": "abs(x-y)", "tanh": "tanh(x)",
        "min": "min(x,y)", "max": "max(x,y)",
    }
    for name, expression in expressions.items():
        lines = base + (f"cv: CUSTOM ARG=a,b,c FUNC={expression} PERIODIC=NO",)
        add(f"custom_{name}", lines, parsed(name, lines))
    return cases

def option_fixtures():
    for fixture in fixtures():
        if fixture["kind"] != "random":
            continue
        cases = cases_for(len(fixture["positions"]))
        if fixture["cell"] is None:
            # These actions otherwise try to unwrap a box that does not exist.
            cases = [replace(c, lines=tuple(
                line + " NOPBC" if any(f": {a} " in line for a in
                    ("COM", "CONTACT_MATRIX")) and "NOPBC" not in line else line
                for line in c.lines)) for c in cases]
        yield dict(fixture, cases=cases, note="Missing Table A1 options; same seeded clouds as original benchmark.")
