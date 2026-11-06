"""OPES_METAD: transcription agreement, forces, restarts, walkers and parsing."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opes_reference import Reference
from plumed_jax import OPESMetad, checkpoint
from plumed_jax.parser import parse

BASE = dict(sigma=(0.1,), barrier=5.0, pace=5, kT=1.0)
CASES = {
    "default": (BASE, 1, 1),
    "fixed_sigma": ({**BASE, "fixed_sigma": True}, 1, 1),
    "merge_off": ({**BASE, "recursive_merge": False}, 1, 1),
    "no_compression_no_zed": ({**BASE, "no_zed": True, "compression_threshold": 0.0}, 1, 1),
    "uniform_target": ({**BASE, "bias_factor": float("inf"), "calc_work": True}, 1, 1),
    "explicit_constants": ({**BASE, "bias_factor": 4.0, "epsilon": 1e-3,
                            "kernel_cutoff": 3.0, "compression_threshold": 0.7}, 1, 1),
    "periodic_2d": (dict(sigma=(0.1, 0.2), barrier=5.0, pace=3, kT=1.0,
                         periods=(0.0, 2 * np.pi), sigma_min=(0.05, 0.05)), 1, 2),
    "shifted_domain": (dict(sigma=(0.3,), barrier=6.0, pace=2, kT=1.0,
                            periods=(2 * np.pi,), periodic_min=(0.0,)), 1, 1),
    "walkers": ({**BASE, "calc_work": True}, 3, 1),
    "adaptive_walkers": (dict(sigma="adaptive", dimension=2, barrier=8.0, pace=4, kT=1.0,
                              periods=(0.0, 2 * np.pi)), 2, 2),
    "adaptive_floor": (dict(sigma="adaptive", dimension=1, barrier=8.0, pace=4, kT=1.0,
                            sigma_min=(0.02,), adaptive_sigma_stride=12), 1, 1),
}


def trajectory(name, steps, walkers, dimension, periodic):
    rng = np.random.default_rng(sum(map(ord, name)))
    x = np.cumsum(rng.normal(0, 0.05, (steps, walkers, dimension)), axis=0)
    if periodic:
        x[..., -1] = (x[..., -1] * 6 + np.pi) % (2 * np.pi) - np.pi
    return x


def reference_for(options, walkers):
    renamed = {("adaptive_stride" if k == "adaptive_sigma_stride" else k): v
               for k, v in options.items()}
    return Reference(**renamed, walkers=walkers)


@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_line_by_line_transcription_of_plumed(name):
    options, walkers, dimension = CASES[name]
    config = OPESMetad(**options, walkers=walkers, capacity=256)
    steps = 241
    x = trajectory(name, steps, walkers, dimension, bool(options.get("periods")))
    ref = reference_for(options, walkers)
    ref_energies = np.array([ref.update(t, [list(v) for v in x[t]]) for t in range(steps)])
    rows = jnp.asarray(x[1:] if walkers > 1 else x[1:, 0])
    state, energies = jax.jit(config.replay)(rows)
    energies = np.asarray(energies).reshape(steps - 1, walkers)
    # Frame zero only evaluates the initial bias.
    np.testing.assert_allclose(ref_energies[0], options["kT"] * config.bias_prefactor
                               * np.log(config.eps))
    np.testing.assert_allclose(energies, ref_energies[1:], rtol=1e-11, atol=1e-11)
    k = int(state.count)
    assert not bool(state.overflow) and k == len(ref.kernels) and k > 1
    np.testing.assert_allclose(state.heights[:k], [h for h, _, _ in ref.kernels], rtol=1e-11)
    np.testing.assert_allclose(state.centers[:k], [c for _, c, _ in ref.kernels], atol=1e-12)
    np.testing.assert_allclose(state.sigmas[:k], [s for _, _, s in ref.kernels], rtol=1e-11)
    np.testing.assert_allclose(state.zed, ref.zed, rtol=1e-11)
    np.testing.assert_allclose(state.sum_weights, ref.sum_weights, rtol=1e-12)
    np.testing.assert_allclose(state.sum_weights2, ref.sum_weights2, rtol=1e-12)
    assert int(state.counter) == ref.counter
    np.testing.assert_allclose(state.sigma0, ref.sigma0, rtol=1e-12)
    if config.calc_work:
        np.testing.assert_allclose(state.work, ref.work, rtol=1e-10, atol=1e-11)
    if config.adaptive:
        assert int(state.adaptive_counter) == ref.adaptive_counter
        np.testing.assert_allclose(state.av_m2, ref.av_M2, rtol=1e-12)
    # Unoccupied slots stay inert and finite.
    assert np.all(np.asarray(state.sigmas[k:]) == 1) and np.all(np.asarray(state.heights[k:]) == 0)


def test_recursive_merging_is_exercised_and_changes_the_kernel_list():
    options = {**BASE, "pace": 1}
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(0, 0.05, 400))[:, None]
    counts = []
    for recursive in (True, False):
        ref = reference_for({**options, "recursive_merge": recursive}, 1)
        for t in range(len(x)):
            ref.update(t, [list(x[t])])
        config = OPESMetad(**options, recursive_merge=recursive, capacity=128)
        state, energies = jax.jit(config.replay)(jnp.asarray(x[1:]))
        assert int(state.count) == len(ref.kernels)
        k = int(state.count)
        np.testing.assert_allclose(state.centers[:k], [c for _, c, _ in ref.kernels], atol=1e-12)
        np.testing.assert_allclose(state.heights[:k], [h for h, _, _ in ref.kernels], rtol=1e-11)
        np.testing.assert_allclose(state.zed, ref.zed, rtol=1e-11)
        counts.append((len(ref.kernels), ref.recursive_merges))
    assert counts[0][1] > 0 and counts[1][1] == 0
    assert counts[0][0] < counts[1][0]


def test_initial_bias_forces_and_free_energy():
    config = OPESMetad(**BASE, capacity=16)
    state = config.init()
    value, gradient = jax.value_and_grad(config.energy)(jnp.array([0.3]), state)
    np.testing.assert_allclose(value, -BASE["barrier"], rtol=1e-14)
    assert gradient[0] == 0
    np.testing.assert_allclose(config.free_energy(0.3, state), BASE["barrier"] / config.bias_prefactor)
    diagnostics = config.diagnostics(state)
    assert int(diagnostics["nker"]) == 0 and float(diagnostics["zed"]) == 1


@pytest.mark.parametrize("periodic", [False, True])
def test_forces_are_conservative(periodic):
    options = dict(sigma=(0.2, 0.3), barrier=5.0, pace=1, kT=1.0,
                   periods=(0.0, 2 * np.pi) if periodic else ())
    config = OPESMetad(**options, capacity=64)
    x = trajectory("forces", 30, 1, 2, periodic)[:, 0]
    state, _ = config.replay(jnp.asarray(x))
    point = jnp.asarray(x[-1]) + jnp.array([0.05, -0.04])
    if periodic:
        point = point.at[1].set(3.12)  # near the seam
    gradient = jax.jit(jax.grad(config.energy))(point, state)
    for axis in range(2):
        eps = 1e-6
        plus = config.energy(point.at[axis].add(eps), state)
        minus = config.energy(point.at[axis].add(-eps), state)
        np.testing.assert_allclose(gradient[axis], (plus - minus) / (2 * eps), rtol=1e-6, atol=1e-8)
    assert np.all(np.isfinite(gradient))


@pytest.mark.parametrize("name", ["default", "adaptive_walkers", "walkers"])
def test_checkpoint_restart_is_bytewise_identical(tmp_path, name):
    options, walkers, dimension = CASES[name]
    config = OPESMetad(**options, walkers=walkers, capacity=128)
    x = trajectory(name, 120, walkers, dimension, bool(options.get("periods")))
    rows = jnp.asarray(x if walkers > 1 else x[:, 0])
    replay = jax.jit(config.replay)
    whole, energies = replay(rows)
    part, first = replay(rows[:53])
    path = tmp_path / "opes.npz"
    checkpoint.save(path, part, config)
    loaded = checkpoint.load(path, config)
    for a, b in zip(part, loaded):
        assert np.asarray(a).dtype == np.asarray(b).dtype
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    resumed, rest = replay(rows[53:], loaded)
    for a, b in zip(whole, resumed):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    assert np.asarray(jnp.concatenate([first, rest])).tobytes() == np.asarray(energies).tobytes()
    other = OPESMetad(**{**options, "barrier": options["barrier"] + 1}, walkers=walkers, capacity=128)
    with pytest.raises(ValueError):
        checkpoint.load(path, other)
    corrupt = part._replace(count=jnp.int32(500))
    checkpoint.save(path, corrupt, config)
    with pytest.raises(ValueError):
        checkpoint.load(path, config)


def test_capacity_overflow_and_step_order_are_sticky_without_partial_writes():
    config = OPESMetad(**{**BASE, "compression_threshold": 0.0, "pace": 1}, capacity=3, walkers=2)
    state = config.init()
    batch = jnp.array([[0.0], [1.0]])
    state = config.update_walkers(batch, state)
    assert int(state.count) == 2 and not bool(state.overflow)
    full = config.update_walkers(batch + 2, state)
    assert bool(full.overflow) and int(full.step) == 2
    for name in ("centers", "sigmas", "heights", "count", "counter", "sum_weights", "zed"):
        assert np.asarray(getattr(full, name)).tobytes() == np.asarray(getattr(state, name)).tobytes()
    later = config.update_walkers(batch + 4, full)
    assert bool(later.overflow) and int(later.count) == 2
    single = OPESMetad(**BASE, capacity=4)
    s = single.update(0.1, single.init(), step=5)
    repeated = single.update(0.2, s, step=5)
    assert bool(repeated.overflow) and int(repeated.count) == 1
    nan = single.update(jnp.nan, single.init(), step=5)
    assert bool(nan.overflow) and int(nan.count) == 0


def test_walker_order_and_shared_old_bias():
    config = OPESMetad(**{**BASE, "pace": 1}, capacity=32, walkers=2)
    state = config.update_walkers(jnp.array([[0.0], [0.5]]), config.init())
    batch = jnp.array([[0.05], [0.9]])
    energies = jax.vmap(config.energy, in_axes=(0, None))(batch, state)
    cached = config.update_walkers(batch, state, bias_energies=energies)
    computed = jax.jit(config.update_walkers)(batch, state)
    for a, b in zip(cached, computed):  # eager vs compiled rounding only
        np.testing.assert_allclose(a, b, rtol=1e-14, atol=1e-16)
    far = jnp.array([[2.0], [3.0]])  # two new kernels: slots follow walker order
    ordered = config.update_walkers(far, state)
    swapped = config.update_walkers(far[::-1], state)
    np.testing.assert_array_equal(ordered.centers[2:4, 0], [2.0, 3.0])
    np.testing.assert_array_equal(swapped.centers[2:4, 0], [3.0, 2.0])
    with pytest.raises(ValueError):
        config.update(0.1, state)
    with pytest.raises(ValueError):
        config.update_walkers(jnp.zeros((3, 1)), state)
    with pytest.raises(ValueError):
        config.update_walkers(batch, state, bias_energies=jnp.zeros(3))


def test_extra_bias_enters_the_weights():
    config = OPESMetad(**{**BASE, "pace": 1}, capacity=8)
    state = config.init()
    plain = config.update(0.1, state)
    extra = config.update(0.1, state, extra_bias=2.0)
    np.testing.assert_allclose(extra.sum_weights - state.sum_weights,
                               (plain.sum_weights - state.sum_weights) * np.exp(2.0), rtol=1e-13)


def test_float32_scan_and_jit():
    config = OPESMetad(sigma=(0.1, 0.2), barrier=5.0, pace=2, kT=1.0,
                       periods=(0.0, 2 * np.pi), capacity=64)
    x = jnp.asarray(trajectory("f32", 80, 1, 2, True)[:, 0], jnp.float32)
    state, energies = jax.jit(config.replay)(x)
    assert state.centers.dtype == jnp.float32 and energies.dtype == jnp.float32
    assert not bool(state.overflow) and np.all(np.isfinite(energies))
    reference, ref_energies = config.replay(jnp.asarray(x, jnp.float64))
    np.testing.assert_allclose(energies, ref_energies, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("bad", [
    dict(sigma=(0.0,)), dict(sigma="fixed"), dict(sigma="adaptive"),
    dict(barrier=-1.0), dict(pace=0), dict(capacity=0), dict(kT=0.0),
    dict(bias_factor=1.0), dict(barrier=0.5),  # default gamma = barrier/kT <= 1
    dict(compression_threshold=10.0), dict(epsilon=0.0), dict(kernel_cutoff=-1.0),
    dict(sigma_min=(0.5,)), dict(adaptive_sigma_stride=10), dict(periods=(1.0, 2.0)),
    dict(walkers=0), dict(sigma="adaptive", dimension=1, bias_factor=float("inf")),
    dict(sigma="adaptive", dimension=1, adaptive_sigma_stride=2),
])
def test_invalid_configuration(bad):
    with pytest.raises(ValueError):
        OPESMetad(**{**BASE, "capacity": 4, **bad})


def test_default_constants_follow_plumed():
    config = OPESMetad(**BASE, capacity=4)
    gamma = BASE["barrier"] / BASE["kT"]
    assert config.gamma == gamma
    np.testing.assert_allclose(config.eps, np.exp(-BASE["barrier"] / (1 - 1 / gamma)))
    np.testing.assert_allclose(config.cutoff, np.sqrt(2 * BASE["barrier"] / (1 - 1 / gamma)))
    uniform = OPESMetad(**BASE, bias_factor=float("inf"), capacity=4)
    assert uniform.bias_prefactor == 1 and np.isclose(uniform.eps, np.exp(-5.0))


def test_parser_builds_opes_and_shares_walkers():
    text = """
    d: DISTANCE ATOMS=1,2
    t: TORSION ATOMS=1,2,3,4
    opes: OPES_METAD ...
        ARG=d,t PACE=2 BARRIER=20 TEMP=300 SIGMA=0.05,0.2 BIASFACTOR=inf
        FIXED_SIGMA NO_ZED RECURSIVE_MERGE_OFF CALC_WORK WALKERS_MPI SERIAL
    ...
    """
    program = parse(text, walkers=2, hill_capacity=40)
    config = program.metad["opes"]
    np.testing.assert_allclose(config.kT, 0.0083144621 * 300)
    assert np.isinf(config.gamma) and config.fixed_sigma and config.no_zed
    assert not config.recursive_merge and config.calc_work and config.walkers == 2
    assert config.periods == (0.0, 2 * np.pi) and config.periodic_min[1] == -np.pi
    r = jnp.asarray(np.random.default_rng(3).normal(size=(2, 4, 3)))
    state = program.initial_state()

    def total(r, s):
        energies, new = program.bias_and_update_walkers(r, s, step=2)
        return jnp.sum(energies), (energies, new)

    (_, (energies, new)), gradient = jax.jit(jax.value_and_grad(total, has_aux=True))(r, state)
    np.testing.assert_allclose(energies, -20.0)
    assert int(new["opes"].count) >= 1 and gradient.shape == r.shape
    direct = config.update_walkers(
        jnp.stack([jax.vmap(lambda x: program.values(x)[k])(r) for k in "dt"], -1), state["opes"],
        step=2)
    for a, b in zip(direct, new["opes"]):
        np.testing.assert_allclose(a, b, rtol=1e-14)
    later = program.update_walkers(r + 0.01, new, step=4)
    assert int(later["opes"].step) == 4


def test_parser_defaults_units_and_errors():
    program = parse("UNITS ENERGY=eV\nd: DISTANCE ATOMS=1,2\n"
                    "o: OPES_METAD ARG=d PACE=10 BARRIER=0.5 TEMP=300\n")
    config = program.metad["o"]
    assert config.adaptive and config.dimension == 1 and config.stride == 100
    np.testing.assert_allclose(config.kT, 0.0083144621 * 300 / 96.48533212331002)
    r = jnp.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
    np.testing.assert_allclose(program.bias(r, state=program.initial_state()), -0.5)
    head = "d: DISTANCE ATOMS=1,2\n"
    for line in [
        "o: OPES_METAD ARG=d PACE=10 BARRIER=20",  # TEMP is required
        "o: OPES_METAD ARG=d PACE=10 BARRIER=20 TEMP=300 NLIST",
        "o: OPES_METAD ARG=d PACE=10 BARRIER=20 TEMP=300 STATE_WFILE=STATE",
        "o: OPES_METAD ARG=d PACE=10 BARRIER=20 TEMP=300 SIGMA=0.1,0.2",
        "o: OPES_METAD ARG=d PACE=10 BARRIER=20 TEMP=300 NO_ZED=yes",
        "o: OPES_METAD ARG=d PACE=1.5 BARRIER=20 TEMP=300 SIGMA=0.1",
    ]:
        with pytest.raises(ValueError):
            parse(head + line)
    with pytest.raises(ValueError):
        parse(head + "o: OPES_METAD ARG=d PACE=10 BARRIER=20 TEMP=300 SIGMA=0.1", walkers=2)


def test_fixture_generator_reads_plumed_files(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "generate_opes_reference",
        Path(__file__).parent / "fixtures" / "generate_opes_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = tmp_path / "STATE"
    state.write_text(
        "#! FIELDS time d t sigma_d sigma_t height\n"
        "#! SET action OPES_METAD_state\n#! SET zed 0.25\n#! SET counter 7\n"
        "#! SET min_t -pi\n#! SET max_t pi\n"
        " 4 0.4 -3.1 0.01 0.2 1.5\n 8 0.5 3.0 0.02 0.1 2.5\n")
    constants, columns = module.read_plumed_table(state)
    assert constants["zed"] == "0.25" and constants["counter"] == "7"
    np.testing.assert_array_equal(columns["t"], [-3.1, 3.0])
    np.testing.assert_array_equal(columns["height"], [1.5, 2.5])
    forces = tmp_path / "forces"
    forces.write_text("2\n0 0 0\nX 1 2 3\nX 4 5 6\n2\n0 0 0\nX -1 -2 -3\nX -4 -5 -6\n")
    np.testing.assert_array_equal(module.read_forces(forces, 2)[1, 1], [-4, -5, -6])
    module.write_xyz(tmp_path / "t.xyz", np.ones((1, 2, 3)) / 3)
    assert (tmp_path / "t.xyz").read_text().splitlines()[2] == "X " + " ".join(["0.33333333333333331"] * 3)
