"""Offline tests of the Paul D.28 pipeline (no solver, no probe data needed). Run:  python paul_d28_selftest.py

1. The trapezoid writer reproduces the .exc of the case byte for byte (to the printed precision).
2. The reference against the book (L, C, modes, -66.2 dB at 10 kHz) and the frequency-domain solver against the modal bounce.
3. Synthetic 'FDTD' data: the primary MTL (plus 0.3 % noise) must give 0 FAIL and 0 WARN; a sign flip, a 5 % wrong plateau,
   a 0.5 ns shifted record and the book's nominal-radius curves as 'FDTD' must be caught.
4. Tampered inputs (a copy of the JSON): a 75 ohm terminal, a PEC upper face, a missing probe, a shorter window, a wire moved
   off its height, a 10 mm cell that breaks the grading ratio, a wrong flat top: each must produce the expected FAIL.
Every test prints OK or BAD; the exit code is the number of BAD."""
import copy
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("MPLBACKEND", "Agg")

from paul_d28_analytic import book_check, compare_transfer, compare_with_mtl, mtl_time
from paul_d28_calibration import calibrate
from paul_d28_common import Log, ROLES
from paul_d28_crosstalk import crosstalk, late_time, transfer_function
from paul_d28_input import load_case
from paul_d28_make_exc import trapezoid

BAD = 0


def verdict(ok, name):
    global BAD
    BAD += 0 if ok else 1
    print(f"[{'OK ' if ok else 'BAD'}] {name}")


def workdir(mut=None, exc=None):
    """A temporary folder with the JSON (optionally modified by mut) and the .exc."""
    d = Path(tempfile.mkdtemp(prefix="d28test_"))
    inp = json.loads((HERE / "paul_d28.fdtd.json").read_text())
    if mut:
        mut(inp)
    (d / "paul_d28.fdtd.json").write_text(json.dumps(inp))
    if exc is None:
        shutil.copy(HERE / "paul_d28.exc", d / "paul_d28.exc")
    else:
        np.savetxt(d / "paul_d28.exc", np.column_stack(exc), fmt="%.6e")
    return d


def fails(mut=None, exc=None):
    d = workdir(mut, exc)
    log = Log()
    try:
        load_case(d, "paul_d28.fdtd.json", log)
    finally:
        os.chdir(HERE)
        shutil.rmtree(d, ignore_errors=True)
    return [r[1] for r in log.rows if r[0] == "FAIL"], [r[1] for r in log.rows if r[0] == "WARN"]


# 1 ---------------------------------------------------------------------------------------------------------------
t_e, v_e = trapezoid(200e-9)
ref = np.loadtxt(HERE / "paul_d28.exc")
verdict(ref.shape == (len(t_e), 2) and np.allclose(ref[:, 0], t_e, atol=1e-15) and np.allclose(ref[:, 1], v_e, atol=1e-6),
        "make_exc reproduces paul_d28.exc (4001 samples, 0 to 200 ns, 0.05 ns)")

# the case as delivered ---------------------------------------------------------------------------------------------
log = Log()
case = load_case(HERE, "paul_d28.fdtd.json", log)
f0, w0 = [r[1] for r in log.rows if r[0] == "FAIL"], [r[1] for r in log.rows if r[0] == "WARN"]
verdict(not f0 and len(w0) == 1 and "timeStep" in w0[0], f"delivered JSON: 0 FAIL, only the dt-correction WARN ({len(log.rows)} lines)")

# 2 ---------------------------------------------------------------------------------------------------------------
t = np.arange(0.5, 24602.5) * 8.1283929e-12
lg = Log()
book_check(case, t, lg)
verdict(lg.count()["FAIL"] == 0 and lg.count()["PASS"] == 4, "reference: book L/C, modes, -66.2 dB at 10 kHz, modal bounce (4 checks)")

# 3 ---------------------------------------------------------------------------------------------------------------
prim = case.refs[case.primary]
cur = mtl_time(case, prim, t)
rng = np.random.default_rng(1)


def pipeline(I):
    log = Log()
    load = load_case(HERE, "paul_d28.fdtd.json", Log())
    cal = calibrate(load, t, I, log)
    xt = crosstalk(load, t, I, log)
    curves = compare_with_mtl(load, cal, xt, t, I, log)
    tf = transfer_function(load, t, xt)
    compare_transfer(load, tf, t, curves, log)
    late_time(load, t, I, xt, curves, log)
    return log


def noisy(c, rel=0.003):
    return {k: v + rel * np.abs(v).max() * rng.standard_normal(len(v)) for k, v in c.items()}


import contextlib
import io

with contextlib.redirect_stdout(io.StringIO()):
    g = pipeline(noisy(cur))
n = g.count()
verdict(n["FAIL"] == 0 and n["WARN"] == 0, f"synthetic data = primary MTL + 0.3 % noise: 0 FAIL, 0 WARN ({n['PASS']} PASS)")

with contextlib.redirect_stdout(io.StringIO()):
    flipped = noisy(cur)
    flipped["vic_near"] = -flipped["vic_near"]
    g = pipeline(flipped)
verdict(any("polarity" in r[1] for r in g.rows if r[0] == "FAIL"), "a flipped victim near-end current is caught (polarity FAIL)")

with contextlib.redirect_stdout(io.StringIO()):
    g = pipeline({k: (1.15 * v if k.startswith("vic") else v) for k, v in noisy(cur).items()})
verdict(any("plateaus" in r[1] or "maximum" in r[1] for r in g.rows if r[0] in ("FAIL", "WARN")),
        "victim currents 15 % too large are caught (peak / plateau)")

dt = t[1] - t[0]
sh = int(round(0.5e-9 / dt))
with contextlib.redirect_stdout(io.StringIO()):
    g = pipeline({k: np.roll(v, sh) for k, v in noisy(cur).items()})
verdict(any("lag" in r[1] or "edges within" in r[1] for r in g.rows if r[0] in ("FAIL", "WARN")), "a record shifted by 0.5 ns is caught (lag / edges)")

nomc = mtl_time(case, case.refs[case.refs_names[0]], t)
with contextlib.redirect_stdout(io.StringIO()):
    g = pipeline(noisy(nomc))
verdict(any(r[0] in ("FAIL", "WARN") for r in g.rows if "Zin" in r[1] or "deficit" in r[1] or "waveforms" in r[1]),
        "the book's nominal-radius curves posing as FDTD are caught (Zin 268 vs 285 ohm, waveforms)")

# 4 ---------------------------------------------------------------------------------------------------------------
def term75(inp):
    for m in inp["materials"]:
        if m["type"] == "terminal":
            m["terminations"][0]["resistance"] = 75.0


def zupper_pec(inp):
    inp["boundary"]["zUpper"] = {"type": "pec"}


def drop_probe(inp):
    inp["probes"] = inp["probes"][:3]


def short_window(inp):
    inp["general"]["numberOfSteps"] = 600


def move_wire(inp):                                   # wire 2 top corners one cell higher: h differs between the wires
    for k in inp["mesh"]["coordinates"]:
        if k["id"] in (6, 7):
            k["relativePosition"][2] = 5


def bad_grading(inp):
    s = inp["mesh"]["grid"]["steps"]["z"]
    s[8] = 0.010                                       # a 10 mm cell next to 5 mm ones: ratio 2


def lowers_mur(inp):
    inp["boundary"]["zLower"] = {"type": "mur"}


def bigger_radius(inp):
    for m in inp["materials"]:
        if m["type"] == "wire":
            m["radius"] = 0.5e-3


cases = [("a 75 ohm terminal", term75, None, "50 ohm"), ("PEC on the upper z face", zupper_pec, None, "absorb"),
         ("a missing probe", drop_probe, None, "probes"), ("a window of 600 steps", short_window, None, "200 ns"),
         ("wire 2 moved up one cell", move_wire, None, "same x extent and height"),
         ("a 10 mm cell in the z grading", bad_grading, None, "ratio"), ("a Mur lower z face", lowers_mur, None, "PEC"),
         ("a 0.5 mm radius instead of 16 mil", bigger_radius, None, "geometry")]
for name, mut, exc, key in cases:
    f, w = fails(mut, exc)
    verdict(any(key in x for x in f + w), f"{name}: caught ({key})")
tw, vw = trapezoid(200e-9)
vw2 = np.interp(tw, [0, 12.5e-9, 32.5e-9, 45e-9], [0, 1, 1, 0], right=0)         # flat top 20 ns
f, w = fails(None, (tw, vw2))
verdict(any("PULSE" in x for x in f), "a trapezoid with a 20 ns flat top (the 50 % width mistaken for the flat top): caught")
print(f"\n{BAD} test(s) failed" if BAD else "\nall tests passed")
sys.exit(BAD)
