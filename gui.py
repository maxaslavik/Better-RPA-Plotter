"""Small Tk front-end for the RPA plotter."""
from __future__ import annotations

import json
import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from plots import FONTS, PLOT_KINDS, PLOT_TITLES, PlotOptions, generate
from rpa_parser import parse_performance, parse_thermal
from themes import DEFAULT_THEME, THEMES

SETTINGS_FILE = Path.home() / ".rpa_plotter_settings.json"
SPECIES_CHOICES = ["ALL", "3", "5", "8", "10", "15"]


class App(ttk.Frame):
    def __init__(self, root: tk.Tk):
        super().__init__(root, padding=8)
        self.root = root
        self.pack(fill="both", expand=True)
        self.msgs: queue.Queue = queue.Queue()
        self.n_species_in_file: int | None = None

        v = self.vars = {
            "perf": tk.StringVar(), "thermal": tk.StringVar(), "engine": tk.StringVar(),
            "scheme": tk.StringVar(value=DEFAULT_THEME), "font": tk.StringVar(value="Serif (Times)"),
            "dpi": tk.StringVar(value="600"), "out": tk.StringVar(value=str(Path.cwd() / "output")),
            "png": tk.BooleanVar(value=True), "pdf": tk.BooleanVar(value=True),
            "svg": tk.BooleanVar(value=False), "open": tk.BooleanVar(value=True),
            "do_thermo": tk.BooleanVar(value=True), "do_species": tk.BooleanVar(value=True),
            "do_cooling": tk.BooleanVar(value=True),
            "n_species": tk.StringVar(value="8"), "sp_scale": tk.StringVar(value="log"),
            "sp_basis": tk.StringVar(value="both"),
            "tol": tk.StringVar(value="5"), "panels": tk.StringVar(value="2"),
            "yscale": tk.StringVar(value="linear"),
            "annotate": tk.BooleanVar(value=True), "subtitle": tk.BooleanVar(value=True),
        }
        self._load_settings()
        self._build()
        self.root.after(100, self._poll)
        if v["perf"].get():
            self._inspect_performance()

    # ------------------------------------------------------------------ layout
    def _build(self):
        v = self.vars
        pad = dict(padx=3, pady=2)

        files = ttk.LabelFrame(self, text="Input files", padding=6)
        files.pack(fill="x")
        files.columnconfigure(1, weight=1)
        for r, (label, key) in enumerate((("Performance", "perf"), ("Thermals", "thermal"))):
            ttk.Label(files, text=label).grid(row=r, column=0, sticky="w", **pad)
            ttk.Entry(files, textvariable=v[key], width=34).grid(row=r, column=1, sticky="ew", **pad)
            ttk.Button(files, text="…", width=3,
                       command=lambda k=key: self._browse_file(k)).grid(row=r, column=2, **pad)
        ttk.Label(files, text="Engine name").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=v["engine"]).grid(row=2, column=1, columnspan=2, sticky="ew", **pad)

        look = ttk.LabelFrame(self, text="Appearance (all plots)", padding=6)
        look.pack(fill="x", pady=(6, 0))
        ttk.Label(look, text="Color scheme").grid(row=0, column=0, sticky="w", **pad)
        ttk.Combobox(look, textvariable=v["scheme"], values=list(THEMES), state="readonly",
                     width=14).grid(row=0, column=1, **pad)
        ttk.Label(look, text="Font").grid(row=0, column=2, sticky="e", **pad)
        ttk.Combobox(look, textvariable=v["font"], values=list(FONTS), state="readonly",
                     width=13).grid(row=0, column=3, **pad)
        ttk.Checkbutton(look, text="Subtitle (propellants, O/F …)",
                        variable=v["subtitle"]).grid(row=1, column=0, columnspan=4, sticky="w", **pad)

        plots = ttk.LabelFrame(self, text="Plots", padding=6)
        plots.pack(fill="x", pady=(6, 0))
        ttk.Checkbutton(plots, text="1  " + PLOT_TITLES["thermo"],
                        variable=v["do_thermo"]).grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(plots, text="Panels").grid(row=1, column=0, sticky="e", **pad)
        ttk.Spinbox(plots, from_=1, to=4, width=3, textvariable=v["panels"]).grid(row=1, column=1, sticky="w")
        ttk.Label(plots, text="Constant if spread < (%)").grid(row=1, column=2, sticky="e", **pad)
        ttk.Spinbox(plots, from_=0, to=50, increment=1, width=4,
                    textvariable=v["tol"]).grid(row=1, column=3, sticky="w")
        ttk.Label(plots, text="Y scale").grid(row=2, column=0, sticky="e", **pad)
        ttk.Combobox(plots, textvariable=v["yscale"], values=["linear", "symlog"], state="readonly",
                     width=7).grid(row=2, column=1, sticky="w")

        ttk.Separator(plots).grid(row=3, column=0, columnspan=4, sticky="ew", pady=4)
        ttk.Checkbutton(plots, text="2  " + PLOT_TITLES["species"],
                        variable=v["do_species"]).grid(row=4, column=0, columnspan=4, sticky="w")
        ttk.Label(plots, text="No. species").grid(row=5, column=0, sticky="e", **pad)
        ttk.Combobox(plots, textvariable=v["n_species"], values=SPECIES_CHOICES,
                     width=6).grid(row=5, column=1, sticky="w")
        self.species_hint = ttk.Label(plots, text="top-N by mean mass fraction", foreground="#6b645a")
        self.species_hint.grid(row=5, column=2, columnspan=2, sticky="w", **pad)
        ttk.Label(plots, text="Scale").grid(row=6, column=0, sticky="e", **pad)
        ttk.Combobox(plots, textvariable=v["sp_scale"], values=["log", "linear"], state="readonly",
                     width=7).grid(row=6, column=1, sticky="w")
        ttk.Label(plots, text="Show").grid(row=6, column=2, sticky="e", **pad)
        ttk.Combobox(plots, textvariable=v["sp_basis"], values=["both", "mass", "mole"],
                     state="readonly", width=7).grid(row=6, column=3, sticky="w")

        ttk.Separator(plots).grid(row=7, column=0, columnspan=4, sticky="ew", pady=4)
        ttk.Checkbutton(plots, text="3  " + PLOT_TITLES["cooling"],
                        variable=v["do_cooling"]).grid(row=8, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(plots, text="Label peak values",
                        variable=v["annotate"]).grid(row=8, column=2, columnspan=2, sticky="w")

        out = ttk.LabelFrame(self, text="Output", padding=6)
        out.pack(fill="x", pady=(6, 0))
        out.columnconfigure(1, weight=1)
        ttk.Label(out, text="Folder").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(out, textvariable=v["out"]).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(out, text="…", width=3, command=self._browse_dir).grid(row=0, column=2, **pad)
        fmt = ttk.Frame(out)
        fmt.grid(row=1, column=0, columnspan=3, sticky="w")
        for text, key in (("PNG", "png"), ("PDF", "pdf"), ("SVG", "svg")):
            ttk.Checkbutton(fmt, text=text, variable=v[key]).pack(side="left", padx=3)
        ttk.Label(fmt, text="  DPI").pack(side="left")
        ttk.Combobox(fmt, textvariable=v["dpi"], values=["300", "600", "1200"], width=5).pack(side="left", padx=3)
        ttk.Checkbutton(fmt, text="Open when done", variable=v["open"]).pack(side="left", padx=8)

        self.button = ttk.Button(self, text="Generate plots", command=self._generate)
        self.button.pack(fill="x", pady=(8, 0))
        self.status = tk.Text(self, height=5, width=52, state="disabled", wrap="word",
                              font=("Segoe UI", 8), relief="flat", background="#f1eee8")
        self.status.pack(fill="both", expand=True, pady=(6, 0))

    # ----------------------------------------------------------------- helpers
    def _log(self, text: str):
        self.status.configure(state="normal")
        self.status.insert("end", text + "\n")
        self.status.see("end")
        self.status.configure(state="disabled")

    def _browse_file(self, key: str):
        start = Path(self.vars[key].get()).parent if self.vars[key].get() else Path.home() / "Downloads"
        path = filedialog.askopenfilename(initialdir=start, filetypes=[("RPA text export", "*.txt"),
                                                                       ("All files", "*.*")])
        if not path:
            return
        self.vars[key].set(path)
        if key == "perf":
            self._inspect_performance()
        elif not self.vars["perf"].get() and Path(path).name.lower().endswith("thermals.txt"):
            guess = Path(path).with_name(Path(path).name[:-len("Thermals.txt")] + "Performance.txt")
            if guess.exists():
                self.vars["perf"].set(str(guess))
                self._inspect_performance()

    def _browse_dir(self):
        d = filedialog.askdirectory(initialdir=self.vars["out"].get() or str(Path.cwd()))
        if d:
            self.vars["out"].set(d)

    def _inspect_performance(self):
        """Fill in the engine name and species count as soon as a file is chosen."""
        try:
            perf = parse_performance(self.vars["perf"].get())
        except Exception as exc:                      # noqa: BLE001 - shown to the user
            self._log(f"Could not read performance file: {exc}")
            return
        if perf.engine_name and not self.vars["engine"].get():
            self.vars["engine"].set(perf.engine_name)
        for note in perf.notes:
            self._log("Note: " + note)
        self.n_species_in_file = len(perf.species)
        self.species_hint.configure(text=f"of {len(perf.species)} in file, ranked by mean mass fraction")

    def _read_options(self) -> tuple[PlotOptions, list[str], list[str], int]:
        v = self.vars
        raw = v["n_species"].get().strip()
        if raw.upper() == "ALL":
            n_sp = None
        else:
            try:
                n_sp = int(raw)
                if n_sp < 1:
                    raise ValueError
            except ValueError:
                raise ValueError("No. species must be ALL or a whole number ≥ 1") from None
        try:
            tol = float(v["tol"].get())
            panels = int(v["panels"].get())
            dpi = int(v["dpi"].get())
        except ValueError:
            raise ValueError("Panels, constant-tolerance and DPI must be numbers") from None
        kinds = [k for k in PLOT_KINDS if v[f"do_{k}"].get()]
        formats = [f for f in ("png", "pdf", "svg") if v[f].get()]
        if not kinds:
            raise ValueError("Select at least one plot")
        if not formats:
            raise ValueError("Select at least one output format")
        opts = PlotOptions(engine_name=v["engine"].get(), scheme=v["scheme"].get(), font=v["font"].get(),
                           n_species=n_sp, species_scale=v["sp_scale"].get(),
                           species_basis=v["sp_basis"].get(), const_tol_pct=tol, thermo_panels=panels,
                           thermo_yscale=v["yscale"].get(), annotate=v["annotate"].get(),
                           subtitle=v["subtitle"].get())
        return opts, kinds, formats, dpi

    # ---------------------------------------------------------------- generate
    def _generate(self):
        v = self.vars
        try:
            opts, kinds, formats, dpi = self._read_options()
            if not Path(v["perf"].get()).is_file() or not Path(v["thermal"].get()).is_file():
                raise ValueError("Choose both the performance and the thermals file")
        except ValueError as exc:
            messagebox.showerror("RPA Plotter", str(exc))
            return
        self._save_settings()
        self.button.configure(state="disabled")
        args = (v["perf"].get(), v["thermal"].get(), opts, kinds, formats, dpi, v["out"].get(),
                v["open"].get())
        threading.Thread(target=self._worker, args=args, daemon=True).start()

    def _worker(self, perf_path, thermal_path, opts, kinds, formats, dpi, out_dir, open_after):
        try:
            perf, thermal = parse_performance(perf_path), parse_thermal(thermal_path)
            for kind in kinds:
                self.msgs.put(f"Rendering {PLOT_TITLES[kind].lower()} …")
                paths = generate(kind, perf, thermal, opts, out_dir, formats, dpi)
                for p in paths:
                    self.msgs.put(f"  saved {p.name}")
                if open_after:
                    first = next((p for p in paths if p.suffix == ".png"), paths[0])
                    os.startfile(first)                                      # noqa: S606 - Windows
            self.msgs.put(f"Done - files in {out_dir}")
        except Exception as exc:                                              # noqa: BLE001
            self.msgs.put(f"ERROR: {exc}")
        finally:
            self.msgs.put(None)

    def _poll(self):
        try:
            while True:
                msg = self.msgs.get_nowait()
                if msg is None:
                    self.button.configure(state="normal")
                else:
                    self._log(msg)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    # ---------------------------------------------------------------- settings
    def _save_settings(self):
        try:
            data = {k: var.get() for k, var in self.vars.items()}
            SETTINGS_FILE.write_text(json.dumps(data, indent=1), encoding="utf-8")
        except OSError:
            pass

    def _load_settings(self):
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for k, val in data.items():
            if k in self.vars and k != "engine":
                try:
                    self.vars[k].set(val)
                except tk.TclError:
                    pass


def run():
    root = tk.Tk()
    root.title("RPA Plotter")
    root.resizable(False, False)
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()
