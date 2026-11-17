# %% Velectra cross-talk experiment, case B (c2): TWO thin wires over the same PEC ground plane (NEXT / FEXT).
#
# Purpose: validate the cross-talk of the FDTD against the analytic multiconductor line (MTL) before case C
# (closed cylinder). Case A (c1) already calibrated source, .exc file, delay, effective Zc and probe sign.
# Geometry (dx = 5 mm): two 1 mm wires, h = 35 mm above a PEC plane, s = 70 mm apart, L = 2 m, 50 ohm series
# terminals on all four ends, 1 V gaussian generator (sigma = 2.5 ns, t0 = 15 ns) in series with the aggressor
# near-end terminal (= R_S). 300 ns window: the mismatch (Zc ~ 300 ohm against 50 ohm) rings for ~150 ns.
#
# This file only orchestrates; each step lives in its own module (same folder):
#   c2_common.py         constants, PASS/WARN/FAIL log, file search, repository paths
#   c2_input.py          static check of c2.fdtd.json + the .exc file; roles of wires and probes from positions
#   c2_solver.py         run plan (steps/time as semba-fdtd computes them), run with live progress bar and ETA
#                        (follows <case>_Report.txt), clean stop on interrupt, loading of the four probe currents
#   c2_calibration.py    aggressor on the first pass: peaks, delay, Zc effective, probe sign
#   c2_crosstalk.py      NE/FE voltages, peaks, polarity, transfer function, late-time ring-down
#   c2_analytic.py       MTL reference (inductance matrix, exact in frequency) and the comparison checks
#   c2_solver_report.py  Energy.dat / Report.txt checks
#   c2_plots.py          4-panel figure
#
# Usage (from anywhere, or cell by cell in VS Code):
#   python c2.py            check the input, run semba-fdtd, analyse the result
#   python c2.py --no-run   check the input and analyse the results of a run already made
# Environment: SEMBA_EXE (path of semba-fdtd), REPO_ROOT (default: nearest parent folder with src_pyWrapper).
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE))                         # the c2_*.py modules

from c2_analytic import compare_transfer, compare_with_mtl
from c2_calibration import calibrate
from c2_common import Log, repo_paths
from c2_crosstalk import crosstalk, late_time, transfer_function
from c2_input import load_case
from c2_plots import plot_case
from c2_solver import load_currents, run_solver
from c2_solver_report import energy_check, report_check

NO_RUN = "--no-run" in sys.argv or os.environ.get("C2_NO_RUN") == "1"
CASE = "c2.fdtd.json"
REPO, SEMBA_EXE, SPINIT_FOLDER = repo_paths(HERE)
log = Log()

# %% 1. Static check of the input file (no solver needed)
case = load_case(HERE, CASE, log)

# %% 2. Run semba-fdtd, unless --no-run. Prints the run plan, then a progress bar updated in place (about 6x the time
# of case A: 300 ns instead of 50 ns; 3x slower on battery). Interrupt = clean stop; the solver reports every ~60 s.
solver = None if NO_RUN else run_solver(case, REPO, SEMBA_EXE, SPINIT_FOLDER)

# %% 3. Probe currents (positive along the polyline, near end -> far end): aggressor and victim, near and far
t, I = load_currents(case, solver)

# %% 4. Aggressor on the first pass of the pulse: peaks, delay, Zc effective, probe sign
cal = calibrate(case, t, I, log)

# %% 5. Victim: NE and FE voltages, peaks and polarity
xt = crosstalk(case, t, I, log)

# %% 6. Transfer function in the band of the source, and the ring-down of the mismatched line
tf = transfer_function(case, t, xt)
late_time(case, t, I, xt, log)

# %% 7. Analytic reference: MTL of the wire pair, time and frequency domain
curves = compare_with_mtl(case, cal, xt, t, I, log)
compare_transfer(case, tf, log)

# %% 8. Optional: energy decay and solver messages
energy_check(case.HERE, log)
report_check(case.HERE, case.dt_solver, log)

# %% 9. Summary
log.summary()

# %% 10. Save the results for the B -> C comparison (the closed cylinder should give about -10 %)
np.savez(case.WORK / "c2_results.npz", t=t, V_NE=xt.V["NE"], V_FE=xt.V["FE"], **{f"I_{k}": v for k, v in I.items()})

# %% 11. Plots
plot_case(case, cal, xt, tf, curves, t, I, case.WORK / "c2.png")

# %%