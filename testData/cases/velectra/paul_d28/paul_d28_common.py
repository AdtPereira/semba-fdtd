"""Shared pieces of the Paul D.28 (Example 9.2) pre/post-processing: constants, PASS/WARN/FAIL log, file search,
repository paths.

Paul D.28 = two bare #20 AWG wires (r = 0.4064 mm), h = s = 20 mm above a ground plane, L = 4.674 m, 50 ohm at the four
ends, 1 V trapezoid (12.5 / 7.5 / 12.5 ns, from the book's SPICE card), simulated to 200 ns. It is the textbook benchmark
of the cross-talk validation ladder; the cases A, B and C (c1, c2, c3) use the same checks on the Velectra geometry.
"""
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

C0 = 299792458.0
ETA0 = 376.730313668
MU0 = 4e-7 * math.pi
EPS0 = 1.0 / (MU0 * C0 ** 2)
CFLN = 0.8                 # HEURCFL of semba-fdtd (fdetypes.F90): dt is corrected to CFLN * dt_CFL
trapz = getattr(np, "trapezoid", None) or np.trapz

ROLES = ("agg_near", "agg_far", "vic_near", "vic_far")      # the four probed wire ends
# What the tutorial (claude/tutorial_paul_D28_crosstalk_EN.md, sections 2 and 5) asks for; deviations are reported.
EXPECTED = dict(length=4.674, height=0.020, sep=0.020, radius=0.4064e-3, R=50.0, window=200e-9,
                x_ends=(0.102, 4.776), y_wires=(0.190, 0.210), riser=0.020,
                rise=12.5e-9, flat=7.5e-9, fall=12.5e-9, amp=1.0, dt_exc=0.05e-9)
# Case A (c1): Zc effective 313.1 ohm of one wire (r = 0.5 mm) at h = 35 mm over the plane. The plane formula turns it into
# the effective radius of the thin-wire model: r_eff = 0.7556 r. Paul D.28 is a different radius and height, so the
# factor (not the radius) is what is transferred to it.
H_A, R_A, ZC_A_MEAS = 0.035, 0.5e-3, 313.1
R_EFF_FACTOR = H_A / math.cosh(2 * math.pi * ZC_A_MEAS / ETA0) / R_A          # 0.7556
# The book (SPICEMTL_D28.OUT, paul_emc/cap9/D28): per-unit-length matrices and the two modes of the subcircuit.
BOOK = dict(L11=9.17859e-7, L12=1.60944e-7, C11=12.5068e-12, C12=-2.19302e-12, Z0=(323.4164, 226.9173),
            TD=(15.59081e-9, 15.59079e-9),
            NE_plateau=23.33e-3, FE_plateau=-31.99e-3,         # exact lossless MTL, first plateaus (tutorial, section 3.2)
            NE_later=(9.7e-3, -8.7e-3), FE_later=8.7e-3, NE_over_S_10kHz_dB=-66.2)
BAND = (3e6, 100e6)        # usable FDTD band for the transfer function comparison (Paul Fig. D.30 stops at 100 MHz)
SRC_FLOOR_DB = -40.0       # use the transfer function only where the source spectrum is above this
NULL_HALF_WIDTH = 4e6      # bins this close to a line resonance n*c/(2L) are skipped in the dB comparison
LATE_START = 150e-9        # last stretch of the 200 ns window, used for the tail check against the MTL
# Tolerances (the same spirit as c2 / c3)
TOL_PEAK_PCT = 10.0        # FAIL: NE / FE extreme against the primary MTL
TOL_PLATEAU_PCT = 5.0      # WARN: first plateaus against the primary MTL
TOL_RMS_AGG = 2.0          # WARN: aggressor currents, RMS in % of the own peak
TOL_RMS_VIC = 3.0          # WARN: NE / FE waveforms, RMS in % of the own peak
TOL_LAG = 0.15e-9          # WARN: cross-correlation lag NE / FE against the MTL
TOL_DB = 2.0               # WARN: |H| against the MTL, dB, resonance nulls skipped


def solver_time_plan(dt_json, n_json, dx, dy=None, dz=None):
    """How semba-fdtd turns timeStep / numberOfSteps of the JSON into the run it really does.
    Mirrors OpenSEMBA/fdtd (checked on commit cd069a6; the 12/11 run of D.28 reproduces it: 1.5407e-10 s -> 8.1284e-12 s,
    1298 -> 24602 steps):
      dt_CFL = 1 / (c sqrt(1/dx^2 + 1/dy^2 + 1/dz^2)) with the SMALLEST cell of each axis;
      if dt_json == 0 or dt_json > HEURCFL * dt_CFL (HEURCFL = 0.8), dt becomes HEURCFL * dt_CFL and the final step is
      int(dt_json / dt * N_json) (truncated), so the simulated time N * dt is kept to within one step."""
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

    def count(self):
        return {k: sum(1 for r in self.rows if r[0] == k) for k in ("PASS", "WARN", "FAIL")}

    def summary(self, title="Paul D.28 (Example 9.2)"):
        print(f"\n=========== {title} summary ===========")
        if not self.rows:
            print("0 checks ran -> NOTHING WAS VALIDATED (was load_case run on this log?)")
            return 1
        for lvl, name, detail in self.rows:
            print(f"[{lvl:4s}] {name}" + (f": {detail}" if detail else ""))
        n = self.count()
        print(f"\n{n['PASS']} passed, {n['WARN']} warnings, {n['FAIL']} failed"
              + ("  -> Paul D.28 validated" if n["FAIL"] == 0 else "  -> fix the failed items (geometry, terminals, "
                 "source, sign) before trusting the result"))
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
