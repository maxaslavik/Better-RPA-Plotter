"""Report-quality plots of RPA engine data.

Public entry point: :func:`generate`.  Every plot shares

* one x-axis - axial position in mm, injector face at x = 0, hot gas flowing toward +x,
* an engine-contour strip under the data with the four RPA stations marked,
* the colour scheme / font chosen in :class:`PlotOptions`,
* a dedicated legend row (legends never sit on top of data or tick labels).
"""
from __future__ import annotations

import math
import re
from functools import lru_cache
from itertools import product
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.textpath import text_to_path
from scipy.interpolate import PchipInterpolator
from matplotlib.ticker import AutoMinorLocator, LogLocator, MaxNLocator, NullFormatter

import coolant
from rpa_parser import (STATION_NAMES, Geometry, PerformanceData, ThermalData, ThermoRow,
                        build_geometry, station_label)
from themes import DEFAULT_THEME, THEMES, Theme

FONTS = {
    "Serif (Times)": dict(family=["Times New Roman", "STIXGeneral", "DejaVu Serif"], math="stix"),
    "Sans (Arial)": dict(family=["Arial", "DejaVu Sans"], math="stixsans"),
}
PLOT_KINDS = ("thermo", "species", "cooling")
PLOT_TITLES = {
    "thermo": "Thermodynamic properties",
    "species": "Species transport",
    "cooling": "Regenerative cooling",
}

FIG_W = 7.4                      # in - fits a two-column page width
FS_LEGEND = 6.8


@dataclass
class PlotOptions:
    engine_name: str = ""
    scheme: str = DEFAULT_THEME
    font: str = "Serif (Times)"
    n_species: int | None = 8            # None = ALL
    species_scale: str = "log"           # "log" | "linear"
    species_basis: str = "both"          # "both" | "mass" | "mole"
    const_tol_pct: float = 5.0           # spread below this => "effectively constant", not plotted
    thermo_panels: int = 2
    thermo_yscale: str = "linear"        # "linear" | "symlog"
    annotate: bool = True                # label peak values in the cooling plot
    subtitle: bool = True


# --------------------------------------------------------------------------- #
# style helpers
# --------------------------------------------------------------------------- #
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "p")
MARKER_SCALE = {"o": 1.0, "s": 0.9, "^": 1.15, "D": 0.85, "v": 1.15,
                "P": 1.3, "X": 1.25, "<": 1.15, ">": 1.15, "p": 1.1}
LINESTYLES = ("-", (0, (5, 1.8)), (0, (5, 1.5, 1.2, 1.5)))
MS = 4.2
LW = 1.1


def series_style(theme: Theme, i: int) -> dict:
    """Colour, marker and dash for the i-th series.  Colour order is fixed; past the
    end of the cycle the marker shifts and the dash changes, so identities stay unique."""
    n = len(theme.cycle)
    tier = i // n
    marker = MARKERS[(i + tier) % len(MARKERS)]
    return dict(color=theme.cycle[i % n], marker=marker,
                ls=LINESTYLES[tier % len(LINESTYLES)], ms=MS * MARKER_SCALE[marker])


def make_rc(theme: Theme, font: str) -> dict:
    f = FONTS.get(font, FONTS["Serif (Times)"])
    return {
        "font.family": f["family"], "mathtext.fontset": f["math"],
        "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": FS_LEGEND,
        "axes.unicode_minus": True, "axes.formatter.useoffset": False,
        "axes.formatter.limits": (-5, 6),
        "figure.facecolor": theme.fig_bg, "savefig.facecolor": theme.fig_bg,
        "axes.facecolor": theme.axes_bg, "axes.edgecolor": theme.spine,
        "text.color": theme.text, "axes.labelcolor": theme.text,
        "xtick.color": theme.text, "ytick.color": theme.text,
        "grid.color": theme.grid, "grid.linewidth": theme.grid_lw,
        "legend.edgecolor": theme.grid, "legend.facecolor": theme.fig_bg,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
        "lines.solid_capstyle": "round", "lines.dash_capstyle": "butt",
    }


def _style_axes(ax, theme: Theme, grid: str = "both") -> None:
    ax.set_facecolor(theme.axes_bg)
    for side, sp in ax.spines.items():
        if theme.spines == "none":
            sp.set_visible(False)
        elif theme.spines == "open" and side in ("top", "right"):
            sp.set_visible(False)
        else:
            sp.set_linewidth(0.7)
            sp.set_color(theme.spine)
    ax.set_axisbelow(True)
    ax.grid(True, which="major", axis=grid, color=theme.grid, lw=theme.grid_lw)
    length = 3.4 if theme.ticks else 0
    ax.tick_params(which="major", direction="in", length=length, width=0.7,
                   colors=theme.text, top=False, right=False, pad=3)
    ax.tick_params(which="minor", direction="in", length=length * 0.55, width=0.5,
                   colors=theme.text, top=False, right=False)


def _text_w_pt(s: str, fs: float) -> float:
    """Rendered width of s in points (cheap: no outline extents)."""
    w, _, _ = text_to_path.get_text_width_height_descent(s, FontProperties(size=fs), "$" in s)
    return float(w)


def chem(name: str) -> str:
    """'N2O(L),298.15K' -> 'N$_{2}$O(L)'."""
    name = name.split(",")[0].strip()
    return re.sub(r"(?<=[A-Za-z\)])(\d+)", r"$_{\1}$", name)


def _wrap(text: str, width_in: float, fs: float) -> list[str]:
    lines, cur = [], ""
    for word in text.split(" "):
        trial = f"{cur} {word}".strip()
        if cur and _text_w_pt(trial, fs) > width_in * 72:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def _wrap_parts(parts: list[str], width_in: float, fs: float, sep: str = "   ·   ") -> list[str]:
    """Greedy-pack subtitle fragments into lines no wider than width_in."""
    lines, cur = [], ""
    for part in parts:
        trial = f"{cur}{sep}{part}" if cur else part
        if cur and _text_w_pt(trial, fs) > width_in * 72:
            lines.append(cur)
            cur = part
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


@dataclass
class LegendPlan:
    ncol: int
    rows: int

    @property
    def height_in(self) -> float:
        return self.rows * 1.75 * FS_LEGEND / 72 + 0.24


def _plan_legend(labels: list[str], avail_in: float) -> LegendPlan:
    widths = [_text_w_pt(lab, FS_LEGEND) for lab in labels]
    handle, pad, gap = 2.6 * FS_LEGEND, 0.6 * FS_LEGEND, 1.6 * FS_LEGEND
    n = len(labels)
    for c in range(min(n, 6), 0, -1):
        rows = math.ceil(n / c)
        c = math.ceil(n / rows)                              # balanced: no near-empty columns
        colw = [max(w, default=0) for w in (list(a) for a in np.array_split(widths, c))]
        if sum(handle + pad + w for w in colw) + gap * (c - 1) <= avail_in * 72:
            return LegendPlan(c, rows)
    return LegendPlan(1, n)


def _legend(ax, handles, labels, theme: Theme, plan: LegendPlan, title: str | None = None,
            loc: str = "upper left"):
    ax.axis("off")
    leg = ax.legend(handles, labels, loc=loc, ncol=plan.ncol, frameon=False,
                    handlelength=2.6, handletextpad=0.6, columnspacing=1.6, labelspacing=0.5,
                    borderaxespad=0.1, title=title, title_fontsize=FS_LEGEND,
                    alignment="left" if loc == "upper left" else "center")
    leg.set_in_layout(False)
    for t in leg.get_texts():
        t.set_color(theme.text)
    if title:
        leg.get_title().set_color(theme.muted)
    return leg


def _proxy(theme: Theme, style: dict, **over) -> Line2D:
    kw = dict(color=style["color"], marker=style["marker"], ls=style["ls"], lw=LW,
              ms=style["ms"], mec=theme.axes_bg, mew=0.6)
    kw.update(over)
    return Line2D([], [], **kw)


# --------------------------------------------------------------------------- #
# figure scaffolding
# --------------------------------------------------------------------------- #
def _new_figure(height_rows: list[float], ncols: int, decor_in: float, dpi: int = 100):
    fig = Figure(figsize=(FIG_W, sum(height_rows) + decor_in), dpi=dpi, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.07, h_pad=0.06, hspace=0.015, wspace=0.03)
    gs = fig.add_gridspec(len(height_rows), ncols, height_ratios=height_rows)
    return fig, gs


HEADER_FS = 7.8


def _header_height(sub_lines: list[str]) -> float:
    return 0.40 + 0.155 * len(sub_lines)


def _header(fig, cell, title: str, sub_lines: list[str], theme: Theme) -> None:
    ax = fig.add_subplot(cell)
    ax.axis("off")
    texts = [ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(0, 0),
                         textcoords="offset points", va="top", ha="left", fontsize=12.5,
                         fontweight="bold", color=theme.text)]
    if sub_lines:
        texts.append(ax.annotate("\n".join(sub_lines), xy=(0, 1), xycoords="axes fraction",
                                 xytext=(0, -19), textcoords="offset points", va="top",
                                 ha="left", fontsize=HEADER_FS, color=theme.muted,
                                 linespacing=1.5))
    for t in texts:                       # text must never drive the axes layout
        t.set_in_layout(False)
    ax.plot([0, 1], [0, 0], transform=ax.transAxes, color=theme.grid if not theme.dark else theme.spine,
            lw=0.8, clip_on=False, solid_capstyle="butt")
    ax.plot([0, 0.07], [0, 0], transform=ax.transAxes, color=theme.accent, lw=2.4,
            clip_on=False, solid_capstyle="butt")


def _footer(fig, cell, lines: list[str], theme: Theme) -> None:
    ax = fig.add_subplot(cell)
    ax.axis("off")
    ax.text(0, 1, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
            fontsize=6.8, color=theme.muted, linespacing=1.45)


def _footer_height(lines: list[str]) -> float:
    return len(lines) * 6.8 * 1.45 / 72 + 0.12


def _station_xs(geom: Geometry) -> list[float]:
    return list(geom.stations.values())


XLABEL = "Axial position from injector face, $x$ (mm)"
SCHEM_W = 3.4                    # in - width of the to-scale engine drawing


def _setup_x(axes, geom: Geometry, theme: Theme, extras: bool = True) -> None:
    """x-limits, tick locators and station lines shared by every plot."""
    pad = 0.035 * geom.length
    for ax in axes:
        ax.set_xlim(-pad, geom.length + pad)
        for name, xs in geom.stations.items():
            if name == "Injector":
                ax.axvline(xs, color=theme.accent, lw=1.3, zorder=2.5, solid_capstyle="butt")
            elif name in STATION_NAMES:
                ax.axvline(xs, color=theme.muted, lw=0.6, ls=(0, (3, 2.2)), alpha=0.75, zorder=1.5)
            elif extras:
                ax.axvline(xs, color=theme.muted, lw=0.5, ls=(0, (1, 2.2)), alpha=0.6, zorder=1.4)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))
        ax.xaxis.set_minor_locator(AutoMinorLocator())


def _smooth(xs, ys, log: bool = False, n: int = 160):
    """Shape-preserving (PCHIP) curve through the station values - never overshoots the data."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    xf = np.linspace(xs[0], xs[-1], n)
    if log:
        return xf, 10 ** PchipInterpolator(xs, np.log10(ys))(xf)
    return xf, PchipInterpolator(xs, ys)(xf)


ROW_PT = 9.0                                   # vertical pitch of station-label rows


@lru_cache(maxsize=64)
def _stagger_cached(items: tuple, width_in: float, length: float, max_rows: int = 3) -> tuple:
    """Assign each label a row (0 = lowest).  A layout is valid when no two labels on the same
    row overlap and no label's leader line passes through a lower-row label.  Of the valid
    layouts take the one with the fewest rows; if none is valid, the one with fewest clashes."""
    span = 1.07 * length
    cen, half = [], []
    for name, x in items:
        fs = 7.2 if name in STATION_NAMES else 6.4
        cen.append((x + 0.035 * length) / span * width_in)
        half.append(_text_w_pt(station_label(name), fs) / 72 / 2 + 0.04)
    n = len(items)

    def clashes(rows):
        bad = 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if rows[i] == rows[j] and i < j and abs(cen[i] - cen[j]) < half[i] + half[j]:
                    bad += 1
                if rows[i] < rows[j] and abs(cen[j] - cen[i]) < half[i] + 0.01:
                    bad += 1                               # j's leader crosses i's text
        return bad

    best, best_key = None, None
    for rows in product(range(max_rows), repeat=n):
        key = (clashes(rows), max(rows), sum(rows))
        if best_key is None or key < best_key:
            best, best_key = rows, key
            if key[0] == 0 and key[1] == 0:
                break
    return best


def _stagger_rows(items: list[tuple[str, float]], length: float, width_in: float) -> list[int]:
    return list(_stagger_cached(tuple(items), round(width_in, 2), round(length, 3)))


def _station_labels(ax, geom: Geometry, theme: Theme, width_in: float) -> int:
    """Names of the four main stations above the axes, each joined to the edge by a short leader,
    staggered into rows when they would collide.  Extra (area-ratio) stations are not labelled:
    they sit within ~0.2 in of the throat, so any label would collide; they are explained in the
    footnote instead.  Returns the number of rows used."""
    items = [(n, x) for n, x in geom.stations.items() if n in STATION_NAMES]
    rows = _stagger_rows(items, geom.length, width_in)
    for (name, x), r in zip(items, rows):
        main = name in STATION_NAMES
        inj = name == "Injector"
        ax.annotate(station_label(name), xy=(x, 1), xycoords=("data", "axes fraction"),
                    xytext=(0, 3.5 + ROW_PT * r), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.2 if main else 6.4,
                    fontweight="bold" if inj else "normal",
                    color=theme.accent if inj else (theme.text if main else theme.muted),
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=theme.muted, shrinkA=0,
                                    shrinkB=0, alpha=1.0 if main else 0.7))
    return 1 + max(rows)


def _schematic_height(geom: Geometry) -> float:
    return SCHEM_W * (2 * 1.12 * float(geom.r.max())) / (1.07 * geom.length)


def _schematic(fig, cell, geom: Geometry, theme: Theme, coolant_dir: int = 0, extras: bool = True):
    """Engine contour drawn once per sheet at true scale (equal x and y), stations marked."""
    side = max((FIG_W - SCHEM_W) / 2, 0.1)
    sub = cell.subgridspec(1, 3, width_ratios=[side, SCHEM_W, side])
    ax = fig.add_subplot(sub[0, 1])
    _style_axes(ax, theme, grid="x")
    ax.fill_between(geom.x, -geom.r, geom.r, facecolor=theme.engine_fill,
                    edgecolor=theme.engine_edge, lw=0.7, zorder=2)
    ax.axhline(0, color=theme.muted, lw=0.5, ls=(0, (6, 2, 1, 2)), zorder=3)
    half = 1.12 * float(geom.r.max())
    ax.set_ylim(-half, half)
    ax.set_yticks([])
    _setup_x([ax], geom, theme, extras)
    ax.set_box_aspect(2 * half / (1.07 * geom.length))      # equal scaling in x and y
    ax.set_xlabel("Engine contour, drawn to scale — $x$ (mm)")
    _station_labels(ax, geom, theme, SCHEM_W)
    _gas_arrow(ax, geom, theme)
    if coolant_dir:
        x_t, x_e = geom.stations["Throat"], geom.stations["Nozzle exit"]
        a, b = x_t + 0.18 * (x_e - x_t), x_t + 0.92 * (x_e - x_t)
        if coolant_dir < 0:
            a, b = b, a
        _flow_arrow(ax, a, b, "coolant flow", theme.accent, theme)
    return ax


def _flow_arrow(ax, x0: float, x1: float, label: str, color: str, theme: Theme) -> None:
    ax.annotate("", xy=(x1, 0), xytext=(x0, 0), zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0, mutation_scale=7,
                                shrinkA=0, shrinkB=0))
    ax.annotate(label, xy=((x0 + x1) / 2, 0), xytext=(0, 2.5), textcoords="offset points",
                ha="center", va="bottom", fontsize=6.2, zorder=6,
                color=theme.text)


def _gas_arrow(ax, geom: Geometry, theme: Theme) -> None:
    x_in = geom.stations["Nozzle inlet"]
    _flow_arrow(ax, 0.12 * x_in, 0.88 * x_in, "hot-gas flow", theme.text, theme)


# --------------------------------------------------------------------------- #
# labels for RPA quantities
# --------------------------------------------------------------------------- #
LABELS = {
    "Pressure": "Pressure, $p$",
    "Temperature": "Temperature, $T$",
    "Enthalpy": "Enthalpy, $h$",
    "Entropy": "Entropy, $s$",
    "Internal energy": "Internal energy, $u$",
    "Specific heat (p=const)": "Specific heat, $c_p$",
    "Specific heat (V=const)": "Specific heat, $c_v$",
    "Gamma": r"Heat-capacity ratio, $\gamma$",
    "Isentropic exponent": r"Isentropic exponent, $n_s$",
    "Gas constant": "Gas constant, $R$",
    "Molecular weight (M)": "Molar mass, $M$",
    "Molecular weight (MW)": r"Molar mass, $M_w$",
    "Density": r"Density, $\rho$",
    "Sonic velocity": "Sonic velocity, $a$",
    "Velocity": "Velocity, $v$",
    "Mach number": "Mach number, $\\mathrm{Ma}$",
    "Area ratio": "Area ratio, $A/A_t$",
    "Mass flux": "Mass flux, $G$",
    "Mass flux (relative)": "Specific mass flux",
    "Viscosity": r"Viscosity, $\mu$",
    "Conductivity, frozen": r"Conductivity, $k_{\mathrm{fr}}$",
    "Specific heat (p=const), frozen": r"Specific heat, $c_{p,\mathrm{fr}}$",
    "Prandtl number, frozen": r"Prandtl no., $\mathrm{Pr}_{\mathrm{fr}}$",
    "Conductivity, effective": r"Conductivity, $k_{\mathrm{eff}}$",
    "Specific heat (p=const), effective": r"Specific heat, $c_{p,\mathrm{eff}}$",
    "Prandtl number, effective": r"Prandtl no., $\mathrm{Pr}_{\mathrm{eff}}$",
}


def _unit(u: str) -> str:
    u = u.strip()
    return f" [{u}]" if u else " [–]"


def _sig_digits(s: str) -> int:
    digits = re.sub(r"[^0-9]", "", s.replace("e", "E").split("E")[0]).lstrip("0")
    return len(digits)


@dataclass
class Quantity:
    row: ThermoRow
    label: str
    max_abs: float

    @property
    def log_mag(self) -> float:
        return math.log10(self.max_abs) if self.max_abs > 0 else -99.0


def _select_quantities(perf: PerformanceData, tol_pct: float):
    plotted, constant, low_precision = [], [], []
    for row in perf.thermo:
        label = LABELS.get(row.name, row.name) + _unit(row.unit)
        v = row.values
        q = Quantity(row, label, float(np.max(np.abs(v))))
        biggest = row.raw[int(np.argmax(np.abs(v)))]
        if _sig_digits(biggest) < 3:
            low_precision.append(q)
        elif (v.max() - v.min()) / max(float(np.mean(np.abs(v))), 1e-300) < tol_pct / 100.0:
            constant.append(q)
        else:
            plotted.append(q)
    return plotted, constant, low_precision


def _group_by_magnitude(items: list[Quantity], n_panels: int) -> list[list[Quantity]]:
    """Split into n_panels groups at the largest gaps in log10(max |value|); each group keeps
    file order.  Panels run from largest to smallest magnitude."""
    n_panels = max(1, min(n_panels, len(items)))
    order = sorted(range(len(items)), key=lambda i: -items[i].log_mag)
    gaps = sorted(((items[order[k]].log_mag - items[order[k + 1]].log_mag, k)
                   for k in range(len(order) - 1)), reverse=True)
    cuts = sorted(k for _, k in gaps[:n_panels - 1])
    groups, start = [], 0
    for c in cuts + [len(order) - 1]:
        members = sorted(order[start:c + 1])
        groups.append([items[i] for i in members])
        start = c + 1
    return groups


def _panel_title(tag: str, group: list[Quantity]) -> str:
    lo = math.floor(min(q.log_mag for q in group))
    hi = math.floor(max(q.log_mag for q in group))
    order = f"$10^{{{lo}}}$" if lo == hi else f"$10^{{{lo}}}$–$10^{{{hi}}}$"
    return f"({tag}) Quantities of order {order}"


def _fmt_range(q: Quantity) -> str:
    lo, hi = q.row.values.min(), q.row.values.max()
    name = q.label.rsplit(" [", 1)[0]
    unit = q.row.unit.strip()
    val = f"{lo:.4g}" if f"{lo:.3g}" == f"{hi:.3g}" else f"{lo:.4g}–{hi:.4g}"
    return f"{name} = {val}" + (f" {unit}" if unit else "")


# --------------------------------------------------------------------------- #
# subtitles
# --------------------------------------------------------------------------- #
def _subtitle(perf: PerformanceData, extra: list[str] | None = None) -> list[str]:
    parts = []
    if perf.propellants:
        parts.append(" / ".join(chem(n) for n, _ in perf.propellants))
    if perf.of_ratio:
        parts.append(f"O/F = {perf.of_ratio:.2f}")
    p = perf.row("Pressure")
    if p is not None:
        parts.append(f"$p_{{\\mathrm{{inj}}}}$ = {p.values[0]:.3f} {p.unit.strip()}")
    parts += extra or []
    return _wrap_parts(parts, FIG_W - 0.3, HEADER_FS)


def _extra_station_note(perf: PerformanceData) -> str:
    extra = [n for n in perf.stations if n not in STATION_NAMES]
    if not extra:
        return ""
    parts = []
    for n in extra:
        side = "convergent" if "(conv.)" in n else "divergent"
        parts.append(f"{station_label(n)} ({side})")
    return "Dotted lines mark the additional RPA stations at " + ", ".join(parts) + "."


def _engine_name(opts: PlotOptions, perf: PerformanceData, th: ThermalData) -> str:
    return opts.engine_name.strip() or perf.engine_name or th.engine_name or "Engine"


# --------------------------------------------------------------------------- #
# Plot 1 - thermodynamic properties
# --------------------------------------------------------------------------- #
def _plot_thermo(perf, geom, th, theme, opts):
    plotted, constant, low_prec = _select_quantities(perf, opts.const_tol_pct)
    if not plotted:
        raise ValueError("Every thermodynamic quantity is effectively constant - lower the "
                         "'constant if spread <' tolerance.")
    groups = _group_by_magnitude(plotted, opts.thermo_panels)
    n = len(groups)
    panel_w = (FIG_W - 0.5) / n - 0.6
    plans = [_plan_legend([q.label for q in g], panel_w) for g in groups]
    xs = [geom.stations[s] for s in perf.stations]

    notes = ["Hot gas flows toward +x from the injector face at x = 0. Markers are the "
             f"{len(perf.stations)} RPA stations; curves are shape-preserving (PCHIP) "
             "interpolations between them."]
    if _extra_station_note(perf):
        notes.append(_extra_station_note(perf))
    if constant:
        notes.append(f"Effectively constant (spread < {opts.const_tol_pct:g} %), not plotted: "
                     + "; ".join(_fmt_range(q) for q in constant) + ".")
    if low_prec:
        notes.append("Not plotted (printed with fewer than 3 significant digits in the RPA file): "
                     + ", ".join(q.label.rsplit(" [", 1)[0] for q in low_prec) + ".")
    foot = [ln for note in notes for ln in _wrap(note, FIG_W - 0.9, 6.8)]

    leg_h = max(p.height_in for p in plans)
    sub = _subtitle(perf) if opts.subtitle else []
    fig, gs = _new_figure([_header_height(sub), 3.5, leg_h, _schematic_height(geom),
                           _footer_height(foot)], n, decor_in=1.75)
    name = _engine_name(opts, perf, th)
    _header(fig, gs[0, :], f"{name} — {PLOT_TITLES['thermo']}", sub, theme)

    for c, group in enumerate(groups):
        ax = fig.add_subplot(gs[1, c])
        _style_axes(ax, theme)
        _setup_x([ax], geom, theme)
        ax.set_xlabel(XLABEL)
        label_rows = _station_labels(ax, geom, theme, panel_w)

        handles, labels = [], []
        for k, q in enumerate(group):
            st = series_style(theme, k)
            xf, yf = _smooth(xs, q.row.values)
            ax.plot(xf, yf, color=st["color"], ls=st["ls"], lw=LW, zorder=3)
            ax.plot(xs, q.row.values, ls="none", color=st["color"], marker=st["marker"],
                    ms=st["ms"], mec=theme.axes_bg, mew=0.6, clip_on=False, zorder=4)
            handles.append(_proxy(theme, st))
            labels.append(q.label)

        nz = np.abs(np.concatenate([q.row.values for q in group]))
        nz = nz[nz > 0]
        if opts.thermo_yscale == "symlog" and nz.max() / nz.min() >= 30:   # else linear is clearer
            ax.set_yscale("symlog", linthresh=10 ** math.floor(math.log10(nz.min())), linscale=0.6)
        else:
            ax.yaxis.set_major_locator(MaxNLocator(7, steps=[1, 2, 2.5, 5, 10]))
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.margins(y=0.07)
        ax.set_ylabel("Value (units as listed in legend)")
        ax.set_title(_panel_title("abcdefgh"[c], group), loc="left",
                     pad=19 + 9 * (label_rows - 1), fontsize=9, fontweight="bold",
                     color=theme.text)
        _legend(fig.add_subplot(gs[2, c]), handles, labels, theme, plans[c])

    _schematic(fig, gs[3, :], geom, theme)
    _footer(fig, gs[4, :], foot, theme)
    return fig


# --------------------------------------------------------------------------- #
# Plot 2 - species transport
# --------------------------------------------------------------------------- #
def species_ranking(perf: PerformanceData) -> list[int]:
    """Indices of species ordered by mean mass fraction over the stations, highest first."""
    return list(np.argsort(-perf.mass_frac.mean(axis=1), kind="stable"))


def _plot_species(perf, geom, th, theme, opts):
    if perf.mass_frac.size == 0:
        raise ValueError("No species table found in the performance file.")
    order = species_ranking(perf)
    n = len(order) if opts.n_species is None else max(1, min(opts.n_species, len(order)))
    sel = order[:n]
    xs = [geom.stations[s] for s in perf.stations]
    panel_w = (FIG_W - 1.0) / (2 if opts.species_basis == "both" else 1) - 0.7

    bases = []
    if opts.species_basis in ("both", "mass"):
        bases.append(("mass", "Mass fraction, $Y_i$", perf.mass_frac))
    if opts.species_basis in ("both", "mole"):
        bases.append(("mole", "Mole fraction, $X_i$", perf.mole_frac))
    nb = len(bases)

    log = opts.species_scale == "log"
    allv = np.concatenate([b[2][sel].ravel() for b in bases])
    pos = allv[allv > 0]
    has_zero = bool((allv <= 0).any())
    if log:
        lo_dec = math.floor(math.log10(pos.min()))
        floor = 10.0 ** (lo_dec - 1) if has_zero else 10.0 ** lo_dec
        ylim = (floor, 10.0 ** math.ceil(math.log10(pos.max() * 1.0001)))
    else:
        floor = 0.0
        ylim = (0.0, float(allv.max()) * 1.06)

    labels = [chem(perf.species[i]) for i in sel]
    styles = [series_style(theme, k) for k in range(n)]
    plan = _plan_legend(labels, FIG_W - 0.6)
    notes = ["Hot gas flows toward +x from the injector face at x = 0. Markers are the "
             f"{len(perf.stations)} RPA stations; curves are shape-preserving (PCHIP) "
             "interpolations between them (log-space on the log scale). "
             f"Species are ranked by mean mass fraction over the {len(perf.stations)} stations; "
             + (f"all {n} are shown" if n == len(order) else f"the top {n} of {len(order)} are shown")
             + " (legend order: highest first, reading down each column)."]
    if _extra_station_note(perf):
        notes.append(_extra_station_note(perf))
    if log and has_zero:
        notes.append("Hollow markers on the lower axis limit: reported as zero in the RPA output "
                     "(below its print resolution); dotted segments connect to them.")
    foot = [ln for note in notes for ln in _wrap(note, FIG_W - 0.9, 6.8)]

    sub = _subtitle(perf) if opts.subtitle else []
    fig, gs = _new_figure([_header_height(sub), 3.5, plan.height_in + 0.12,
                           _schematic_height(geom), _footer_height(foot)], nb, decor_in=1.75)
    name = _engine_name(opts, perf, th)
    _header(fig, gs[0, :], f"{name} — {PLOT_TITLES['species']}", sub, theme)

    for c, (kind, ylabel, data) in enumerate(bases):
        ax = fig.add_subplot(gs[1, c])
        _style_axes(ax, theme)
        _setup_x([ax], geom, theme)
        ax.set_xlabel(XLABEL)
        label_rows = _station_labels(ax, geom, theme, panel_w)

        for k, i in enumerate(sel):
            st = series_style(theme, k)
            y = data[i].astype(float)
            zero = y <= 0
            if log:
                y = np.where(zero, floor, y)
            xf, yf = _smooth(xs, y, log=log)
            common = dict(color=st["color"], lw=LW, clip_on=False, zorder=3)
            for a in range(len(xs) - 1):
                seg = (xf >= xs[a]) & (xf <= xs[a + 1])
                dotted = log and (zero[a] or zero[a + 1])
                ax.plot(xf[seg], yf[seg], ls=":" if dotted else st["ls"], **common)
            ax.plot(np.array(xs)[~zero], y[~zero], ls="none", marker=st["marker"], ms=st["ms"],
                    mfc=st["color"], mec=theme.axes_bg, mew=0.6, color=st["color"],
                    clip_on=False, zorder=4)
            if log and zero.any():
                ax.plot(np.array(xs)[zero], y[zero], ls="none", marker=st["marker"], ms=st["ms"],
                        mfc=theme.axes_bg, mec=st["color"], mew=1.0, clip_on=False, zorder=4)

        ax.set_ylim(*ylim)
        if log:
            ax.set_yscale("log")
            ax.set_ylim(*ylim)
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=12))
            ax.yaxis.set_minor_formatter(NullFormatter())
        else:
            ax.yaxis.set_major_locator(MaxNLocator(6, steps=[1, 2, 2.5, 5, 10]))
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_ylabel(ylabel)
        ax.set_title(f"({'ab'[c]}) {ylabel.split(',')[0]}s along the engine", loc="left",
                     pad=19 + 9 * (label_rows - 1), fontsize=9, fontweight="bold",
                     color=theme.text)

    handles = [_proxy(theme, st) for st in styles]
    _legend(fig.add_subplot(gs[2, :]), handles, labels, theme, plan,
            title="Species (ranked by mean mass fraction)", loc="upper center")
    _schematic(fig, gs[3, :], geom, theme)
    _footer(fig, gs[4, :], foot, theme)
    return fig


# --------------------------------------------------------------------------- #
# Plot 3 - regenerative cooling
# --------------------------------------------------------------------------- #
def _annotate_peak(ax, x, y, text, theme, dx=-12, dy=18, va="bottom"):
    ax.plot([x], [y], marker="o", ms=7.5, mfc="none", mec=theme.text, mew=0.9, zorder=5)
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points", ha="right",
                va=va, fontsize=7, color=theme.text, zorder=6,
                arrowprops=dict(arrowstyle="-", color=theme.muted, lw=0.7, shrinkA=0, shrinkB=4),
                bbox=dict(boxstyle="round,pad=0.25", fc=theme.axes_bg, ec=theme.grid, lw=0.5))


def _plot_cooling(perf, geom, th, theme, opts):
    c = dict(th.cols)
    x = th.x
    me = max(1, len(x) // 18)

    merged = bool(np.allclose(c["Twg"], c["Twi"], atol=0.05))
    temps = [("Twg", r"Hot-gas-side wall / inner wall, $T_{wg}=T_{wi}$" if merged
              else r"Hot-gas-side wall surface, $T_{wg}$")]
    if not merged:
        temps.append(("Twi", r"Inner wall surface, $T_{wi}$"))
    temps += [("Twc", r"Coolant-side wall, $T_{wc}$"), ("Tc", r"Coolant bulk, $T_c$")]
    boil = coolant.boiling_curve(perf.propellants, c)
    if boil:
        fluid, c["Tsat"] = boil
        temps.append(("Tsat", f"Local boiling point of {fluid.lower()}, "
                              r"$T_{sat}(p_c)$"))
    labels = [f"{lab} [K]" for _, lab in temps]
    plan = _plan_legend(labels, FIG_W - 0.6)

    direction = th.coolant_direction
    flow_txt = {-1: "counterflow (coolant enters at the nozzle exit)",
                1: "parallel flow (coolant enters at the injector)", 0: ""}[direction]
    extra = ["regenerative cooling" + (f", {flow_txt}" if flow_txt else "")]
    if direction:
        i_in = int(np.argmax(x)) if direction < 0 else 0
        i_out = int(np.argmin(x)) if direction < 0 else len(x) - 1
        extra += [f"$T_{{c,\\mathrm{{in}}}}$ = {c['Tc'][i_in]:.0f} K",
                  f"$\\Delta p_c$ = {abs(c['pc'][i_in] - c['pc'][i_out]):.3f} MPa"]

    notes = ["Hot gas flows toward +x from the injector face at x = 0. Markers show "
             + ("every other " if me == 2 else "the ") + "analysis station of the thermal "
             "calculation."]
    if boil:
        notes.append(f"Boiling point: saturation temperature of {boil[0].lower()} (CoolProp reference "
                     "equation of state) at the local coolant pressure; the coolant was identified "
                     "from the density column of the thermal file.")
    foot = [ln for note in notes for ln in _wrap(note, FIG_W - 0.9, 6.8)]

    sub = _subtitle(perf, extra) if opts.subtitle else []
    fig, gs = _new_figure([_header_height(sub), 1.75, 1.75, plan.height_in,
                           _schematic_height(geom), _footer_height(foot)], 2, decor_in=1.95)
    name = _engine_name(opts, perf, th)
    _header(fig, gs[0, :], f"{name} — {PLOT_TITLES['cooling']}", sub, theme)

    ax_t = fig.add_subplot(gs[1:3, 0])
    ax_p = fig.add_subplot(gs[1, 1], sharex=ax_t)
    ax_v = fig.add_subplot(gs[2, 1], sharex=ax_t)
    for ax in (ax_t, ax_p, ax_v):
        _style_axes(ax, theme)
    _setup_x([ax_t, ax_p, ax_v], geom, theme, extras=False)
    ax_t.set_xlabel(XLABEL)
    ax_v.set_xlabel(XLABEL)
    ax_p.tick_params(labelbottom=False)
    panel_w = (FIG_W - 1.0) / 2 - 0.7
    rows_t = _station_labels(ax_t, geom, theme, panel_w)
    rows_p = _station_labels(ax_p, geom, theme, panel_w)

    # (a) temperatures
    handles = []
    for k, (key, _) in enumerate(temps):
        st = series_style(theme, k)
        if key == "Tsat":                         # a limit, not a result: dashed
            st = {**st, "ls": (0, (5, 2.2))}
        ax_t.plot(x, c[key], color=st["color"], marker=st["marker"], ms=st["ms"] * 0.9, ls=st["ls"],
                  lw=LW, markevery=me, mec=theme.axes_bg, mew=0.6, zorder=3 + (len(temps) - k) * 0.01)
        handles.append(_proxy(theme, st))
    allT = np.concatenate([c[k][np.isfinite(c[k])] for k, _ in temps])
    rng = allT.max() - allT.min()
    ax_t.set_ylim(allT.min() - 0.06 * rng, allT.max() + (0.24 if opts.annotate else 0.08) * rng)
    ax_t.yaxis.set_major_locator(MaxNLocator(7, steps=[1, 2, 2.5, 5, 10]))
    ax_t.yaxis.set_minor_locator(AutoMinorLocator())
    ax_t.set_ylabel("Temperature (K)")
    ax_t.set_title("(a) Wall and coolant temperatures", loc="left", pad=19 + 9 * (rows_t - 1), fontsize=9,
                   fontweight="bold", color=theme.text)
    if opts.annotate:
        i = int(np.argmax(c["Twg"]))
        _annotate_peak(ax_t, x[i], c["Twg"][i],
                       f"Peak $T_{{wg}}$ = {c['Twg'][i]:.0f} K\n($x$ = {x[i]:.0f} mm)", theme)

    # (b), (c) coolant pressure and velocity
    st = series_style(theme, 0)
    for ax, key, ylabel, title in ((ax_p, "pc", "Pressure (MPa)", "(b) Coolant static pressure"),
                                   (ax_v, "wc", "Velocity (m/s)", "(c) Coolant velocity")):
        ax.plot(x, c[key], color=st["color"], marker=st["marker"], ms=st["ms"] * 0.9, ls=st["ls"],
                lw=LW, markevery=me, mec=theme.axes_bg, mew=0.6, zorder=3)
        ax.yaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_ylabel(ylabel)
        rg = c[key].max() - c[key].min()
        top_pad = 0.12
        ax.set_ylim(c[key].min() - 0.10 * rg, c[key].max() + top_pad * rg)
    ax_p.set_title("(b) Coolant static pressure", loc="left", pad=19 + 9 * (rows_p - 1), fontsize=9,
                   fontweight="bold", color=theme.text)
    ax_v.set_title("(c) Coolant velocity", loc="left", pad=6, fontsize=9, fontweight="bold",
                   color=theme.text)
    if opts.annotate:
        j = int(np.argmax(c["wc"]))
        where = " (throat)" if abs(x[j] - geom.stations["Throat"]) < 0.03 * geom.length else ""
        _annotate_peak(ax_v, x[j], c["wc"][j],
                       f"Peak $w_c$ = {c['wc'][j]:.1f} m/s{where}", theme, dx=-14, dy=0, va="center")

    _legend(fig.add_subplot(gs[3, :]), handles, labels, theme, plan, loc="upper center")
    _schematic(fig, gs[4, :], geom, theme, coolant_dir=direction, extras=False)
    _footer(fig, gs[5, :], foot, theme)
    return fig


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
_BUILDERS = {"thermo": _plot_thermo, "species": _plot_species, "cooling": _plot_cooling}
FILE_STEM = {"thermo": "1_thermodynamic_properties", "species": "2_species_transport",
             "cooling": "3_regenerative_cooling"}


def generate(kind: str, perf: PerformanceData, thermal: ThermalData, opts: PlotOptions,
             out_dir: str | Path, formats=("png", "pdf"), dpi: int = 600) -> list[Path]:
    """Build one plot and save it in each requested format.  Returns the written paths."""
    theme = THEMES[opts.scheme]
    geom, th = build_geometry(thermal, perf)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9]+", "_", _engine_name(opts, perf, th)).strip("_")
    written = []
    with mpl.rc_context(make_rc(theme, opts.font)):
        fig = _BUILDERS[kind](perf, geom, th, theme, opts)
        for fmt in formats:
            path = out_dir / f"{safe}_{FILE_STEM[kind]}.{fmt}"
            fig.savefig(path, dpi=dpi, facecolor=theme.fig_bg)
            written.append(path)
    return written
