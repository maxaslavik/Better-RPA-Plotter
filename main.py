"""RPA Plotter - GUI by default, or headless with --perf / --thermal.

    python main.py
    python main.py --perf P.txt --thermal T.txt --scheme "RXPI Dark" --species ALL --out figs
"""
from __future__ import annotations

import argparse
import sys

from plots import FONTS, PLOT_KINDS, PlotOptions, generate
from rpa_parser import parse_performance, parse_thermal
from themes import DEFAULT_THEME, THEMES


def cli(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perf", required=True, help="RPA performance .txt")
    ap.add_argument("--thermal", required=True, help="RPA thermal-analysis .txt")
    ap.add_argument("--engine", default="", help="engine name (default: from the files)")
    ap.add_argument("--scheme", default=DEFAULT_THEME, choices=list(THEMES))
    ap.add_argument("--font", default="Serif (Times)", choices=list(FONTS))
    ap.add_argument("--plots", default=",".join(PLOT_KINDS), help="comma list of: " + ", ".join(PLOT_KINDS))
    ap.add_argument("--species", default="8", help="number of species for plot 2, or ALL")
    ap.add_argument("--species-scale", default="log", choices=["log", "linear"])
    ap.add_argument("--species-basis", default="both", choices=["both", "mass", "mole"])
    ap.add_argument("--panels", type=int, default=2)
    ap.add_argument("--tol", type=float, default=5.0, help="%% spread below which a quantity counts as constant")
    ap.add_argument("--yscale", default="linear", choices=["linear", "symlog"])
    ap.add_argument("--formats", default="png,pdf")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--out", default="output")
    a = ap.parse_args(argv)

    opts = PlotOptions(engine_name=a.engine, scheme=a.scheme, font=a.font,
                       n_species=None if a.species.upper() == "ALL" else int(a.species),
                       species_scale=a.species_scale, species_basis=a.species_basis,
                       const_tol_pct=a.tol, thermo_panels=a.panels, thermo_yscale=a.yscale)
    perf, thermal = parse_performance(a.perf), parse_thermal(a.thermal)
    for note in perf.notes:
        print("Note:", note)
    for kind in a.plots.split(","):
        for path in generate(kind.strip(), perf, thermal, opts, a.out, a.formats.split(","), a.dpi):
            print(path)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(cli(sys.argv[1:]))
    import gui
    gui.run()
