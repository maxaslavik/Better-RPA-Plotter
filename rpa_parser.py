"""Parsers for Rocket Propulsion Analysis (RPA) text exports.

Two files are understood:

* the *performance* export  (propellants, Table 1 thermodynamic properties at
  four stations, Table 2 species fractions at the same stations)
* the *thermal* export      (wall/coolant quantities along the engine contour)

``build_geometry`` combines them so every plot shares one x-axis: the injector
face at x = 0 mm and the hot-gas flow toward +x.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

STATION_NAMES = ("Injector", "Nozzle inlet", "Throat", "Nozzle exit")   # always present


def station_label(name: str) -> str:
    """Display text for a station key ('A/At=1.200 (conv.)' -> 'A/A_t = 1.2')."""
    m = re.match(r"A/At=([0-9.]+)", name)
    return f"$A/A_t$ = {float(m.group(1)):g}" if m else name
_STATION_PREFIX = (
    ("inj", "Injector"),
    ("nozzle inl", "Nozzle inlet"),
    ("nozzle thr", "Throat"),
    ("nozzle exi", "Nozzle exit"),
)

THERMAL_COLUMNS = (
    "x", "r", "h_conv", "q_conv", "q_rad", "q_tot",
    "Twg", "Twi", "Twc", "Tc", "pc", "wc", "rho_c",
)


def canonical_station(raw: str) -> str:
    key = raw.strip().lower()
    for prefix, name in _STATION_PREFIX:
        if key.startswith(prefix):
            return name
    return raw.strip()


def read_text(path: str | Path) -> str:
    """RPA writes single-byte (cp1252) text: units contain '·' and '²'."""
    data = Path(path).read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def _to_float(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# performance file
# --------------------------------------------------------------------------- #
@dataclass
class ThermoRow:
    name: str
    unit: str
    values: np.ndarray           # one value per station
    raw: list[str]               # the printed strings (to judge precision)


@dataclass
class PerformanceData:
    engine_name: str | None = None
    propellants: list[tuple[str, float]] = field(default_factory=list)  # (name, mass fraction)
    of_ratio: float | None = None
    stations: list[str] = field(default_factory=list)
    thermo: list[ThermoRow] = field(default_factory=list)
    species: list[str] = field(default_factory=list)
    mass_frac: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    mole_frac: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    notes: list[str] = field(default_factory=list)       # things the user should know about

    def row(self, name: str) -> ThermoRow | None:
        for r in self.thermo:
            if r.name == name:
                return r
        return None


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.lstrip("#").rstrip("\n").split("\t")]


def parse_performance(path: str | Path) -> PerformanceData:
    perf = PerformanceData()
    section = None                     # "prop", "t1", "t2", or None
    header: list[list[str]] = []
    species_cols: list[tuple[str, str]] = []
    mass_rows: list[list[float]] = []
    mole_rows: list[list[float]] = []

    for line in read_text(path).splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        m = re.match(r"#\s*Engine name:\s*(.+)", line)
        if m:
            perf.engine_name = m.group(1).strip()
            continue
        if re.match(r"#\s*Propellant Specification", line):
            section = "prop"
            continue
        m = re.match(r"#\s*Table\s+(\d+)\.", line)
        if m:
            section = {"1": "t1", "2": "t2"}.get(m.group(1))
            header = []
            continue
        if re.fullmatch(r"#-+", stripped) or stripped == "#":
            continue

        if section == "prop" and stripped.startswith("#"):
            m = re.search(r"O/F:\s*([0-9.]+)", line)
            if m and "O/F 0" not in line:
                perf.of_ratio = float(m.group(1))
                continue
            cells = _cells(line)
            if len(cells) >= 3 and cells[0] and not cells[0].startswith("Total"):
                mass = _to_float(cells[1])
                if mass is not None and _to_float(cells[2]) is not None:
                    perf.propellants.append((cells[0].split()[0], mass))
            continue

        if section in ("t1", "t2"):
            if stripped.startswith("#"):
                header.append(_cells(line))
                continue
            cells = [c.strip() for c in line.split("\t")]
            name = cells[0]
            if section == "t1":
                stations = [canonical_station(c) for c in header[0][1:]
                            if c and c.lower() != "unit"]
                n = len(stations)
                raw = cells[1:1 + n]
                vals = [_to_float(c) for c in raw]
                if any(v is None for v in vals):
                    continue
                perf.stations = stations
                unit = cells[1 + n] if len(cells) > 1 + n else ""
                perf.thermo.append(ThermoRow(name, unit, np.array(vals), raw))
            else:
                if not species_cols:
                    stations = header[0][1:]
                    kinds = header[1][1:] if len(header) > 1 else []
                    species_cols = [(canonical_station(s), "mass" if "mass" in k.lower() else "mole")
                                    for s, k in zip(stations, kinds) if s]
                vals = [_to_float(c) for c in cells[1:1 + len(species_cols)]]
                if len(vals) < len(species_cols) or any(v is None for v in vals):
                    continue
                perf.species.append(name)
                mass_rows.append([v for v, (_, k) in zip(vals, species_cols) if k == "mass"])
                mole_rows.append([v for v, (_, k) in zip(vals, species_cols) if k == "mole"])

    if mass_rows:
        perf.mass_frac = np.array(mass_rows)
        perf.mole_frac = np.array(mole_rows)
    if not perf.thermo:
        raise ValueError(f"No 'Thermodynamic properties' table found in {Path(path).name}")
    _finalize_stations(perf)
    return perf


def _finalize_stations(perf: PerformanceData) -> None:
    """RPA can export extra stations at fixed area ratios ('A/At=1.200').  Drop exact duplicate
    columns and tag each area-ratio station as convergent or divergent, so every column has a
    unique name that build_geometry can locate on the contour."""
    names = list(perf.stations)
    n = len(names)
    has_species = perf.mass_frac.ndim == 2 and perf.mass_frac.shape[1] == n

    def column(i):
        parts = [row.values[i] for row in perf.thermo]
        if has_species:
            parts += list(perf.mass_frac[:, i]) + list(perf.mole_frac[:, i])
        return np.array(parts)

    keep: list[int] = []
    for i in range(n):
        dup = next((j for j in keep if names[j] == names[i] and np.array_equal(column(j), column(i))), None)
        if dup is None:
            keep.append(i)
        else:
            perf.notes.append(f"Station column {i + 1} ('{names[i]}') is an exact duplicate of "
                              f"column {dup + 1} and was ignored.")
    if len(keep) != n:
        for row in perf.thermo:
            row.values = row.values[keep]
            row.raw = [row.raw[i] for i in keep]
        if has_species:
            perf.mass_frac, perf.mole_frac = perf.mass_frac[:, keep], perf.mole_frac[:, keep]
        names = [names[i] for i in keep]

    if "Throat" in names:
        i_throat = names.index("Throat")
        names = [f"{nm} ({'conv.' if i < i_throat else 'div.'})" if nm.startswith("A/At") else nm
                 for i, nm in enumerate(names)]
    perf.stations = names


# --------------------------------------------------------------------------- #
# thermal file
# --------------------------------------------------------------------------- #
@dataclass
class ThermalData:
    engine_name: str | None
    cols: dict[str, np.ndarray]
    comments: list[str]

    @property
    def x(self) -> np.ndarray:
        return self.cols["x"]

    @property
    def r(self) -> np.ndarray:
        return self.cols["r"]

    @property
    def coolant_direction(self) -> int:
        """-1: coolant flows toward the injector (counterflow), +1: with the hot gas, 0: unknown."""
        text = " ".join(self.comments).lower()
        if "opposite" in text or "counter" in text:
            return -1
        if "parallel" in text or "co-flow" in text or "same direction" in text:
            return 1
        return 0

    def reversed(self) -> "ThermalData":
        return ThermalData(self.engine_name, {k: v[::-1].copy() for k, v in self.cols.items()},
                           self.comments[::-1])


def parse_thermal(path: str | Path) -> ThermalData:
    engine = None
    rows: list[list[float]] = []
    comments: list[str] = []
    for line in read_text(path).splitlines():
        s = line.strip()
        if not s:
            continue
        if not s.startswith("#") and not re.match(r"[-+]?\d", s):
            engine = engine or s
            continue
        if s.startswith("#"):
            continue
        tokens = s.split()
        nums = [_to_float(t) for t in tokens[:len(THERMAL_COLUMNS)]]
        if len(nums) < len(THERMAL_COLUMNS) or any(v is None for v in nums):
            continue
        rows.append(nums)
        comments.append(" ".join(tokens[len(THERMAL_COLUMNS):]))
    if not rows:
        raise ValueError(f"No thermal-analysis rows found in {Path(path).name}")
    arr = np.array(rows)
    return ThermalData(engine, {k: arr[:, i] for i, k in enumerate(THERMAL_COLUMNS)}, comments)


# --------------------------------------------------------------------------- #
# geometry shared by all plots
# --------------------------------------------------------------------------- #
@dataclass
class Geometry:
    x: np.ndarray                      # mm, injector face at 0, increasing downstream
    r: np.ndarray                      # mm, inner wall radius
    stations: dict[str, float]         # name -> x [mm]

    @property
    def length(self) -> float:
        return float(self.x[-1] - self.x[0])


def build_geometry(thermal: ThermalData, perf: PerformanceData | None = None
                   ) -> tuple[Geometry, ThermalData]:
    """Return the engine contour (injector at x = 0, flow toward +x) and the
    thermal data re-ordered to match.

    If the thermal file is written exit-first, the performance file's injector /
    exit area ratios identify the injector end and the data are mirrored.
    """
    th = thermal
    x, r = th.x, th.r
    r_min = r.min()

    if perf is not None:
        area = perf.row("Area ratio")
        if area is not None and len(area.values) >= 4:
            ar_inj, ar_exit = area.values[0], area.values[-1]
            ar_first, ar_last = (r[0] / r_min) ** 2, (r[-1] / r_min) ** 2
            if abs(ar_inj - ar_exit) > 0.05 * ar_inj:   # otherwise the test is ambiguous
                err_keep = abs(ar_first - ar_inj) + abs(ar_last - ar_exit)
                err_flip = abs(ar_last - ar_inj) + abs(ar_first - ar_exit)
                if err_flip < err_keep:
                    th = th.reversed()
    if th.x[0] > th.x[-1]:                              # still descending: mirror the axis
        th = th.reversed()

    x = th.x - th.x[0]
    r = th.r
    th = ThermalData(th.engine_name, {**th.cols, "x": x}, th.comments)

    i_throat = int(np.argmin(r))
    # end of the cylindrical chamber = last point of the initial run at (almost) maximum radius
    chamber = np.where(r >= r[0] * (1 - 1e-3))[0]
    run_end = chamber[0]
    for idx in chamber[1:]:
        if idx != run_end + 1:
            break
        run_end = idx
    stations = {
        "Injector": 0.0,
        "Nozzle inlet": float(x[run_end]),
        "Throat": float(x[i_throat]),
        "Nozzle exit": float(x[-1]),
    }

    # extra stations at a given area ratio: find them on the convergent / divergent contour
    area = perf.row("Area ratio") if perf is not None else None
    if area is not None:
        ar = (r / r_min) ** 2
        for i, name in enumerate(perf.stations):
            if name in stations or i >= len(area.values):
                continue
            target = float(area.values[i])
            if name.endswith("(conv.)"):
                seg = slice(run_end, i_throat + 1)
                stations[name] = float(np.interp(target, ar[seg][::-1], x[seg][::-1]))
            else:
                seg = slice(i_throat, None)
                stations[name] = float(np.interp(target, ar[seg], x[seg]))
        stations = dict(sorted(stations.items(), key=lambda kv: kv[1]))
    return Geometry(x, r, stations), th
