# Better RPA Plotter

Report-quality plots from Rocket Propulsion Analysis (RPA) exports.

```
pip install -r requirements.txt   # matplotlib, numpy, scipy
python main.py                      # small GUI
python main.py --perf P.txt --thermal T.txt --scheme "RXPI Dark" --species ALL --out figs   # headless
```

Inputs: the RPA *performance* export and the *thermal analysis* export. Output: PNG (default 600 dpi) / PDF / SVG.

| Plot | Content |
|---|---|
| 1 | Thermodynamic properties at the four stations, grouped by order of magnitude. Near-constant quantities (spread below a tolerance) and quantities printed with <3 significant digits are listed in the footnote instead of plotted. |
| 2 | Species mass / mole fractions. The N species with the highest mean mass fraction are shown (or ALL). |
| 3 | Wall and coolant temperatures; coolant pressure and velocity. |

All plots use x = axial position in mm, with the injector face at x = 0 and the hot gas flowing toward +x
(the thermal file is mirrored automatically if it is written exit-first). Station positions come from the engine contour in the thermal file.
Schemes: RXPI Light, RXPI Dark, Seaborn, ggplot, Lavender, Pastel, Atari (`themes.py`).
