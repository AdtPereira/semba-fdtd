# %% Velectra cross-talk experiment, case A (c1): ONE thin wire over a PEC ground plane.
#
# Purpose: calibrate everything that case B (two wires, cross-talk) and case C (closed cylinder) rely on:
# source and .exc file, delay, effective characteristic impedance, probe sign, boundary handling.
# Geometry (dx = 5 mm): 1 mm wire, h = 35 mm above a PEC plane, L = 2 m, a series terminal on each end,
# 1 V gaussian generator (sigma = 2.5 ns, t0 = 15 ns) in series with the near-end terminal (= R_S).
#
# This file only orchestrates; each step lives in its own module (same folder):
#   c1_common.py         constants, PASS/WARN/FAIL log, file search, repository paths
#   c1_input.py          static check of c1.fdtd.json + gauss.exc
#   c1_solver.py         run plan (steps/time as semba-fdtd computes them), run with live progress bar and ETA, probes
#   c1_calibration.py    peaks, delay, Zc effective, Gamma_L, charge
#   c1_echo.py           echo = monopole (resistive) + doublet (reactive), late-time residual
#   c1_analytic.py       bounce-diagram reference of the line
#   c1_solver_report.py  Energy.dat / Report.txt checks
#   c1_plots.py          3-panel figure
#
# Usage (from anywhere, or cell by cell in VS Code):
#   python c1.py            check the input, run semba-fdtd (live progress bar), analyse the result
#   python c1.py --no-run   check the input and analyse the results of a run already made
# In the interactive window (VS Code / Jupyter): set os.environ["C1_NO_RUN"] = "1" before cell 1, or simply do not run
# cell 2; `solver` stays None and the probes are read from the .dat files of the case folder.
# Environment: SEMBA_EXE (path of semba-fdtd), REPO_ROOT (default: nearest parent folder with src_pyWrapper),
# C1_NO_RUN=1 (same as --no-run), C1_MCELLS_PER_S (speed for the a-priori time estimate; else the last Report.txt).
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE))                         # the c1_*.py modules

from c1_analytic import compare_with_line
from c1_calibration import calibrate
from c1_common import Log, repo_paths
from c1_echo import echo_analysis
from c1_input import load_case
from c1_plots import plot_case
from c1_solver import load_currents, run_solver
from c1_solver_report import energy_check, report_check

NO_RUN = "--no-run" in sys.argv or os.environ.get("C1_NO_RUN") == "1"
solver = None        # stays None when cell 2 is skipped or --no-run: the probes are then read from the .dat files
CASE = "c1.fdtd.json"
REPO, SEMBA_EXE, SPINIT_FOLDER = repo_paths(HERE)
log = Log()

# %% 1. Static check of the input file (no solver needed)
case = load_case(HERE, CASE, log)

# %% 2. Run semba-fdtd, unless --no-run. Prints the run plan, then a progress bar updated in place (about 6 min on a
# fast machine; 3x slower on battery). Interrupt = clean stop; the solver reports every ~60 s.
solver = None if NO_RUN else run_solver(case, REPO, SEMBA_EXE, SPINIT_FOLDER)

# %% 3. Probe currents (positive along the polyline, near end -> far end)
t, I_n, I_f = load_currents(case, solver)

# %% 4. Calibration numbers from the first pass of the pulse
cal = calibrate(case, t, I_n, I_f, log)

# %% 5. First echo at the near end: resistive part (monopole) and reactive part (doublet)
echo = echo_analysis(case, cal, t, I_n, I_f, log)

# %% 6. Analytic reference: bounce diagram of the lossless line
curves = compare_with_line(case, cal, t, I_n, I_f, log)

# %% 7. Optional: energy decay and solver messages
energy_check(case.HERE, log)
report_check(case.HERE, case.dt_solver, log)

# %% 8. Summary
log.summary()

# %% 9. Plots
plot_case(case, cal, echo, t, I_n, I_f, curves, case.WORK / "c1_prepost.png")

# %%