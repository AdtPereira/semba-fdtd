# %% Paul, Introduction to EMC (3rd ed.), Example 9.2 / Appendix D.4.1, Fig. D.28: two wires over a ground plane (NEXT / FEXT).
#
# The textbook benchmark of the cross-talk validation ladder (issue #5, Velectra project). Two bare #20 AWG wires, h = s = 20 mm,
# L = 4.674 m, 50 ohm at the four ends, 1 V trapezoid 12.5 / 7.5 / 12.5 ns, 200 ns. Same checks as cases A, B and C (c1, c2, c3):
# static check of the input, calibration of the generator wire, NE / FE against an exact analytic MTL (time and frequency).
#
# This file only orchestrates; each step lives in its own module (same folder):
#   paul_d28_common.py       constants, tolerances, the book's numbers, PASS/WARN/FAIL log, file search, repository paths
#   paul_d28_input.py        static check of paul_d28.fdtd.json + the .exc file; roles of wires and probes from positions;
#                            the references (book, r_eff, r_eff + riser inductance)
#   paul_d28_solver.py       run plan (steps/time as semba-fdtd computes them), run with live progress bar and ETA,
#                            clean stop on interrupt, loading of the four probe currents
#   paul_d28_calibration.py  generator wire on the first pass: 50 % edges, delay, Zin, probe sign
#   paul_d28_crosstalk.py    NE/FE voltages, extremes, plateaus, polarity, transfer functions, late-time tail
#   paul_d28_analytic.py     exact MTL (frequency domain) + independent modal-bounce solution, the book check, the comparisons
#   paul_d28_solver_report.py  Energy.dat / Report.txt / Warnings.txt checks
#   paul_d28_plots.py        six-panel figure
#   paul_d28_make_exc.py     writes the trapezoid .exc (200 ns as in the book, or longer with --window)
#   paul_d28_selftest.py     offline tests: the reference against the book, synthetic data, tampered inputs
#
# Usage (from anywhere, or cell by cell in VS Code):
#   python paul_d28.py            check the input, run semba-fdtd (about 3 min), analyse the result
#   python paul_d28.py --no-run   check the input and analyse the results of a run already made
# Environment: SEMBA_EXE (path of semba-fdtd), REPO_ROOT (default: nearest parent folder with src_pyWrapper),
#              D28_NO_RUN=1 (same as --no-run), D28_MCELLS_PER_S (speed for the a-priori time estimate).
# Needs next to it: paul_d28.fdtd.json and paul_d28.exc. Replaces paul_d28_prepost.py.
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE))                         # the paul_d28_*.py modules

from paul_d28_analytic import book_check, compare_transfer, compare_with_mtl
from paul_d28_calibration import calibrate
from paul_d28_common import Log, repo_paths
from paul_d28_crosstalk import band_info, crosstalk, late_time, node_transfer, transfer_function
from paul_d28_input import load_case
from paul_d28_plots import plot_case
from paul_d28_solver import load_currents, run_solver
from paul_d28_solver_report import energy_check, report_check

NO_RUN = "--no-run" in sys.argv or os.environ.get("D28_NO_RUN") == "1"
solver = None        # stays None when cell 2 is skipped or --no-run: the probes are then read from the .dat files
CASE = "paul_d28.fdtd.json"
REPO, SEMBA_EXE, SPINIT_FOLDER = repo_paths(HERE)
log = Log()

# %% 1. Static check of the input file (no solver needed): grid, wires and risers, terminals, source, time plan, references
case = load_case(HERE, CASE, log)

# %% 2. Run semba-fdtd, unless --no-run. Prints the run plan, then a progress bar updated in place (about 3 min).
# Interrupt = clean stop. A new run overwrites the .dat files of this folder: copy the folder first to keep the old one.
# solver = None if NO_RUN else run_solver(case, REPO, SEMBA_EXE, SPINIT_FOLDER)

# %% 3. Probe currents (positive along the polyline, near end -> far end): generator wire and receptor wire, near and far
t, I = load_currents(case, solver)

# %% 4. The reference itself: L, C and modes against the book, frequency domain against the independent time-domain solution
book_check(case, t, log)

# %% 5. Generator wire on the first pass of the trapezoid: edges, delay, Zin (about 285 ohm), probe sign
cal = calibrate(case, t, I, log)

# %% 6. Receptor wire: NE and FE voltages, extremes, first plateaus, polarity
xt = crosstalk(case, t, I, log)

# %% 7. Analytic MTL: time domain (book, r_eff, r_eff + risers), thin-wire deficit, residuals
curves = compare_with_mtl(case, cal, xt, t, I, log)

# %% 8. Transfer functions (to the generator and to the line-input node), and the tail of the record
tf = transfer_function(case, t, xt)
band_info(tf, log)
compare_transfer(case, tf, t, curves, log)
tfn = node_transfer(case, t, I, xt, tf)
late_time(case, t, I, xt, curves, log)

# %% 9. Optional: energy decay and solver messages
energy_check(case.HERE, log)
report_check(case.HERE, case.dt_solver, log)

# %% 10. Summary
log.summary()

# %% 11. Save the results
np.savez(case.WORK / "paul_d28_results.npz", t=t, V_NE=xt.V["NE"], V_FE=xt.V["FE"], **{f"I_{k}": v for k, v in I.items()})

# %% 12. Plots
plot_case(case, cal, xt, tf, tfn, curves, t, I, case.WORK / "paul_d28.png")

# %% 
