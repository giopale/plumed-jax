"""Lossless hill/grid-state checkpoints with configuration validation."""

from __future__ import annotations

from .metadynamics import Metadynamics
from dataclasses import asdict
import json
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from .metadynamics import HillState
from .grid_metadynamics import GridMetadynamics, GridState


def save(
    path: str | Path,
    state: HillState | GridState,
    config: Metadynamics | GridMetadynamics,
) -> None:
    """Atomically write a hill/grid state and its configuration to an NPZ file.

    path is a string or Path; parent directories are created. config is
    Metadynamics or GridMetadynamics; state must have the corresponding
    HillState/GridState type (ValueError otherwise). The host reads all
    state arrays, preserving dtypes and bytes. File-system errors propagate.
    This file operation is outside JIT and returns None.
    """
    state_class = GridState if isinstance(config, GridMetadynamics) else HillState
    if not isinstance(state, state_class):
        raise ValueError("Checkpoint state does not match configuration type")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        np.savez(
            stream,
            config=json.dumps(asdict(config), sort_keys=True),
            state_type="grid" if state_class is GridState else "hills",
            **{name: np.asarray(x) for name, x in zip(state_class._fields, state)},
        )
    tmp.replace(path)


def load(
    path: str | Path, config: Metadynamics | GridMetadynamics
) -> HillState | GridState:
    """Load a checkpoint matching config, preserving every state array's dtype.

    path is a string or Path; config is Metadynamics or GridMetadynamics.
    Returns the corresponding HillState/GridState on the default JAX device.
    Configuration/type mismatches, malformed state, or precision truncation
    raise ValueError; file/archive errors propagate. Enable x64 before
    loading float64. This file operation is outside JIT.
    """
    expected = json.loads(json.dumps(asdict(config)))
    is_grid = isinstance(config, GridMetadynamics)
    state_class = GridState if is_grid else HillState
    with np.load(path, allow_pickle=False) as data:
        if json.loads(str(data["config"])) != expected:
            raise ValueError("Metadynamics configuration differs from checkpoint")
        kind = str(data["state_type"]) if "state_type" in data else "hills"
        if kind != ("grid" if is_grid else "hills"):
            raise ValueError("Checkpoint state type differs from configuration")
        try:
            arrays = [data[name] for name in state_class._fields]
        except KeyError as error:
            raise ValueError("Incomplete checkpoint state") from error
    shape0 = config.grid_shape if is_grid else (config.capacity, len(config.sigma))
    shape1 = config.grid_shape + (len(config.sigma),) if is_grid else (config.capacity,)
    if (
        arrays[0].shape != shape0
        or arrays[1].shape != shape1
        or any(a.shape != () for a in arrays[2:])
        or arrays[0].dtype != arrays[1].dtype
        or arrays[0].dtype not in (np.dtype("float32"), np.dtype("float64"))
        or arrays[2].dtype != np.int32
        or arrays[3].dtype != np.int32
        or arrays[4].dtype != np.bool_
        or not 0
        <= int(arrays[2])
        <= (np.iinfo(np.int32).max if is_grid else config.capacity)
        or int(arrays[3]) < 0
        or not all(np.all(np.isfinite(a)) for a in arrays[:2])
        or (
            is_grid
            and (
                arrays[5].dtype != arrays[0].dtype
                or not np.isfinite(arrays[5])
                or arrays[5] < 0
            )
        )
    ):
        raise ValueError("Invalid checkpoint state")
    state = state_class(*(jnp.asarray(a) for a in arrays))
    if any(np.asarray(a).dtype != b.dtype for a, b in zip(state, arrays)):
        raise ValueError("Checkpoint precision changed; enable float64 before loading")
    return state
