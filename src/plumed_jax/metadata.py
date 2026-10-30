"""Static conventions and explicit conversions for MD integration."""

from __future__ import annotations

from jax.typing import ArrayLike
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CVMetadata:
    """Static name, units, periods, and optional component names for a CV.

    units/periods must have equal lengths; components is empty or has the
    same length. Periods are finite nonnegative values in each CV's unit;
    zero means nonperiodic. Invalid metadata raises ValueError. This record
    describes conventions and does not calculate CVs or convert units.
    """

    name: str
    units: tuple[str, ...]
    periods: tuple[float, ...]
    components: tuple[str, ...] = ()

    def __post_init__(self):
        if len(self.units) != len(self.periods):
            raise ValueError("One unit and period per component is required")
        if self.components and len(self.components) != len(self.units):
            raise ValueError("Component names do not match the output shape")
        if any(not math.isfinite(p) or p < 0 for p in self.periods):
            raise ValueError("Periods must be finite and nonnegative")


# Conversion factors to Angstrom, eV and ps. Angles always use radians.
LENGTH_TO_A = {"A": 1.0, "angstrom": 1.0, "nm": 10.0, "bohr": 0.529177210903}
ENERGY_TO_EV = {
    "eV": 1.0,
    "kj/mol": 1 / 96.48533212331002,
    "kcal/mol": 4.184 / 96.48533212331002,
}
TIME_TO_PS = {"ps": 1.0, "fs": 0.001}


@dataclass(frozen=True)
class Units:
    """Static coordinate/energy/time conventions and explicit conversions.

    length supports A/angstrom/nm/bohr, energy eV/kj/mol/kcal/mol, and time
    ps/fs. Defaults are A/eV/ps for this Python class; parse defaults to
    nm/kj/mol/ps. Unsupported names raise ValueError. Values are not
    converted automatically; angles always use radians.
    """

    length: str = "A"
    energy: str = "eV"
    time: str = "ps"

    def __post_init__(self):
        if (
            self.length not in LENGTH_TO_A
            or self.energy not in ENERGY_TO_EV
            or self.time not in TIME_TO_PS
        ):
            raise ValueError("Unsupported unit convention")

    def convert_energy(self, value: ArrayLike, target: str = "eV") -> ArrayLike:
        """Convert scalar/array value from self.energy to target's energy unit.

        Returns the scaled value; target must be a key of ENERGY_TO_EV.
        An unknown target raises KeyError. No coordinate conversion is implied.
        """
        return value * ENERGY_TO_EV[self.energy] / ENERGY_TO_EV[target]

    def convert_length(self, value: ArrayLike, target: str = "A") -> ArrayLike:
        """Convert scalar/array value from self.length to target's length unit.

        Returns the scaled value; target must be a key of LENGTH_TO_A.
        An unknown target raises KeyError. Cell arrays can be converted the same way.
        """
        return value * LENGTH_TO_A[self.length] / LENGTH_TO_A[target]
