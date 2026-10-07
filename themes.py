"""Colour schemes. One theme styles every plot identically.

Each theme carries a 10-colour categorical cycle.  Series beyond ten re-use the
cycle with a different marker *and* line style (see plots.series_style), so no
two series ever share the same colour + marker + dash combination.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    name: str
    fig_bg: str
    axes_bg: str
    text: str
    muted: str
    grid: str
    spine: str
    accent: str            # injector line, highlights
    engine_fill: str       # contour strip
    engine_edge: str
    cycle: tuple[str, ...]
    spines: str = "box"    # "box" | "open" | "none"
    ticks: bool = True
    grid_lw: float = 0.5
    dark: bool = False


THEMES: dict[str, Theme] = {t.name: t for t in (
    Theme(
        name="RXPI Light",
        fig_bg="#ffffff", axes_bg="#faf9f7", text="#16130f", muted="#6b645a",
        grid="#dcd8d2", spine="#16130f", accent="#d6001c",
        engine_fill="#f1eee8", engine_edge="#16130f",
        cycle=("#d6001c", "#8a8378", "#16130f", "#0e8fa3", "#7b4fa0",
               "#1f8a70", "#d98e00", "#d0509a", "#a8541a", "#2f6fa3"),
    ),
    Theme(
        name="RXPI Dark",
        fig_bg="#0f0d0b", axes_bg="#16130f", text="#f3efe8", muted="#9a9288",
        grid="#2e2924", spine="#6b645a", accent="#ff3b52",
        engine_fill="#26211c", engine_edge="#c7c1b7",
        cycle=("#ff3b52", "#5aa9e6", "#f06fb0", "#f3efe8", "#3fc79a",
               "#b48ae0", "#e8854a", "#35c6d6", "#f2b134", "#9a9288"),
        dark=True,
    ),
    Theme(
        name="Seaborn",
        fig_bg="#ffffff", axes_bg="#eaeaf2", text="#262626", muted="#555555",
        grid="#ffffff", spine="#ffffff", accent="#c44e52",
        engine_fill="#d4d4e2", engine_edge="#555555",
        cycle=("#4c72b0", "#55a868", "#da8bc3", "#937860", "#3aa4bf",
               "#dd8452", "#8172b3", "#c44e52", "#b7a03c", "#7f7f7f"),
        spines="none", ticks=False, grid_lw=1.0,
    ),
    Theme(
        name="ggplot",
        fig_bg="#ffffff", axes_bg="#e5e5e5", text="#333333", muted="#555555",
        grid="#ffffff", spine="#e5e5e5", accent="#e24a33",
        engine_fill="#cfcfcf", engine_edge="#555555",
        cycle=("#e24a33", "#1f9a8a", "#c98a00", "#348abd", "#8a5a2b",
               "#988ed5", "#4d4d4d", "#d9708a", "#8eba42", "#6f6f6f"),
        spines="none", ticks=False, grid_lw=1.0,
    ),
    Theme(
        name="Lavender",
        fig_bg="#fbf9fc", axes_bg="#f2eef4", text="#2e4045", muted="#5d6f73",
        grid="#ddd4e0", spine="#2e4045", accent="#5e3c58",
        engine_fill="#c7bbc9", engine_edge="#2e4045",
        cycle=("#5e3c58", "#3c8c9a", "#c27a3a", "#4a6fa5", "#b8a000",
               "#a35276", "#7b8f4e", "#9368a8", "#2e4045", "#8a7d79"),
    ),
    Theme(
        name="Pastel",
        fig_bg="#fffafb", axes_bg="#f6f1f8", text="#2b3252", muted="#626a8a",
        grid="#e0d8e8", spine="#2b3252", accent="#d9667a",
        engine_fill="#dec2cb", engine_edge="#2b3252",
        cycle=("#d9667a", "#5a74c4", "#c4577f", "#d98f3a", "#8a8fa8",
               "#2b3252", "#2f9a9c", "#8a68b8", "#5fa05a", "#b8609a"),
    ),
    Theme(
        name="Atari",
        fig_bg="#fdf8ea", axes_bg="#f7eed6", text="#4f372d", muted="#7a6258",
        grid="#e4d6b4", spine="#4f372d", accent="#cc2a36",
        engine_fill="#edc951", engine_edge="#4f372d",
        cycle=("#cc2a36", "#00a0b0", "#eb6841", "#2f5f9a", "#4f8f3a",
               "#7a4f9a", "#c99a00", "#d0509a", "#4f372d", "#a8715a"),
    ),
)}

DEFAULT_THEME = "RXPI Light"
