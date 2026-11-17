"""Shared pieces of the c1 (case A) pre/post-processing: constants, PASS/WARN/FAIL log, file search, repo paths."""
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

C0 = 299792458.0
ETA0 = 376.730313668
R_EFF_FACTOR = 0.75        # MTLN/Holland thin wires behave as if r = 0.75 r_w (as in Paul D.28)
CFLN = 0.8                 # semba-fdtd corrects the time step to CFLN * dx / (c sqrt(3))
trapz = getattr(np, "trapezoid", None) or np.trapz

# What the tutorial (claude/crosstalk_fdtd_tutorial_ABC_EN.md) asks for; deviations are reported, not fatal.
EXPECTED = dict(height=0.035, length=2.0, radius=0.5e-3, dx=5e-3, cells=[168, 88, 400])


def solver_time_plan(dt_json, n_json, dx, dy=None, dz=None):
    """How semba-fdtd turns timeStep / numberOfSteps of the JSON into the run it really does.
    Mirrors OpenSEMBA/fdtd (checked on commit cd069a6):
      semba_fdtd.F90 ~l.625-660: dt_CFL = 1 / (c sqrt(1/dx^2 + 1/dy^2 + 1/dz^2)); if dt_json == 0 or
                                 dt_json > HEURCFL * dt_CFL, dt becomes HEURCFL * dt_CFL (HEURCFL = 0.8)
      semba_fdtd.F90 ~l.385-411: the final step is then int(dt_json / dt * N_json) (truncated), so the simulated
                                 time N * dt is kept to within one step. Not applied when dt_json == 0.
    The switches -cfl and -n (additionalArguments, or flags of the FDTD object) change this; the monitor of
    c1_solver reads the real numbers from _Report.txt anyway."""
    dy, dz = dy or dx, dz or dx
    dt_cfl = 1.0 / (C0 * math.sqrt(1 / dx ** 2 + 1 / dy ** 2 + 1 / dz ** 2))
    dt_limit = CFLN * dt_cfl
    corrected = dt_json == 0 or dt_json > dt_limit
    dt = dt_limit if corrected else dt_json
    n_final = int(dt_json / dt * n_json) if (corrected and dt_json != 0) else n_json
    return SimpleNamespace(dt_json=dt_json, n_json=n_json, t_json=dt_json * n_json, dt_cfl=dt_cfl, dt_limit=dt_limit,
                           corrected=corrected, dt=dt, n_final=n_final, t_final=n_final * dt)


class Log:
    """Collects the PASS / WARN / FAIL / INFO lines and prints the final summary."""

    def __init__(self):
        self.rows = []                                 # (level, name, detail)

    def check(self, ok, name, detail="", level="FAIL"):
        self.rows.append(("PASS" if ok else level, name, detail))
        return ok

    def info(self, name, detail=""):
        self.rows.append(("INFO", name, detail))

    def summary(self, title="case A (c1)"):
        print(f"\n=========== {title} summary ===========")
        for lvl, name, detail in self.rows:
            print(f"[{lvl:4s}] {name}" + (f": {detail}" if detail else ""))
        n = {k: sum(1 for r in self.rows if r[0] == k) for k in ("PASS", "WARN", "FAIL")}
        if sum(n.values()) == 0:                       # an empty log must never read as a success
            print("\n0 checks ran -> NOTHING WAS VALIDATED. Call load_case with this same Log object "
                  "(log = Log() once, before load_case, and do not create it again).")
            return 1
        print(f"\n{n['PASS']} passed, {n['WARN']} warnings, {n['FAIL']} failed"
              + ("  -> case A calibrated" if n["FAIL"] == 0 else "  -> fix the failed items before moving to case B"))
        return n["FAIL"]


def find_file(root, name_or_glob):
    """First match in root, then recursively (Elemwave writes the results to ugrfdtd/)."""
    root = Path(root)
    hits = sorted(root.glob(name_or_glob)) or sorted(root.rglob(name_or_glob))
    return hits[0] if hits else None


def find_repo(start):
    """Walk up from the case folder until a folder holding src_pyWrapper is found (any depth)."""
    start = Path(start)
    for p in [start, *start.parents]:
        if (p / "src_pyWrapper").is_dir():
            return p
    return start.parents[2] if len(start.parents) > 2 else start


def repo_paths(here):
    """(REPO, SEMBA_EXE, SPINIT_FOLDER); REPO_ROOT and SEMBA_EXE environment variables override the defaults."""
    repo = Path(os.environ["REPO_ROOT"]) if "REPO_ROOT" in os.environ else find_repo(here)
    exe = os.environ.get("SEMBA_EXE", str(repo / "build" / "bin" / "semba-fdtd"))
    return repo, exe, repo / "testData" / "spinit"
