"""Local boiling point of the regenerative-cooling coolant.

The thermal export has no coolant-identity or boiling-point column, so the coolant is identified by
density: of the propellants in the performance file, the one whose density at the file's coolant
temperature and pressure (``Tc``, ``pc``) matches the file's ``rho`` column is taken as the coolant.
Its saturation temperature at the local coolant pressure is the boiling point (CoolProp, reference
equation of state).  Everything degrades gracefully: if CoolProp is missing or nothing matches,
``boiling_curve`` returns None and the plot simply omits the curve.
"""
from __future__ import annotations

import numpy as np

try:
    from CoolProp.CoolProp import PropsSI
except Exception:                                  # noqa: BLE001 - optional dependency
    PropsSI = None

# RPA propellant name (before any comma / phase tag) -> CoolProp fluid name
FLUIDS = {
    "C2H5OH": "Ethanol", "CH3OH": "Methanol", "H2O": "Water", "CH4": "Methane",
    "H2": "Hydrogen", "O2": "Oxygen", "NH3": "Ammonia", "N2O": "NitrousOxide",
    "C3H8": "Propane", "C2H6": "Ethane", "N2": "Nitrogen",
}
MAX_DENSITY_MISMATCH = 0.08                        # relative; beyond this, don't guess a coolant


def _candidates(propellants) -> list[str]:
    out = []
    for name, _ in propellants:
        key = name.split(",")[0].split("(")[0].strip()
        if key in FLUIDS and FLUIDS[key] not in out:
            out.append(FLUIDS[key])
    return out


def _density_mismatch(fluid: str, T, p_mpa, rho) -> float:
    try:
        calc = np.array([PropsSI("D", "T", float(t), "P", float(p) * 1e6, fluid)
                         for t, p in zip(T, p_mpa)])
    except Exception:                              # noqa: BLE001 - e.g. state outside the EOS range
        return np.inf
    return float(np.max(np.abs(calc - rho) / rho))


def boiling_curve(propellants, cols: dict[str, np.ndarray]):
    """Return (fluid_name, T_sat_in_K_per_station) or None."""
    if PropsSI is None or not propellants:
        return None
    T, p, rho = cols["Tc"], cols["pc"], cols["rho_c"]
    scored = sorted((_density_mismatch(f, T, p, rho), f) for f in _candidates(propellants))
    if not scored or scored[0][0] > MAX_DENSITY_MISMATCH:
        return None
    fluid = scored[0][1]
    p_crit = PropsSI("Pcrit", fluid) / 1e6
    tsat = np.array([PropsSI("T", "P", float(pp) * 1e6, "Q", 0, fluid) if pp < 0.999 * p_crit
                     else np.nan for pp in p])
    return fluid, tsat
