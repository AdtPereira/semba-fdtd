# %% Velectra cross-talk experiment, case A (c1): ONE thin wire over a PEC ground plane.
#
# Purpose: calibrate everything that case B (two wires, cross-talk) and case C (closed cylinder) rely on:
# source and .exc file, delay, probe sign, boundary handling. Two curves are compared, nothing else: semba-fdtd with the
# terminals exactly as they are in the JSON (359.24 ohm, not tuned to the solver's wire), and the analytic lossless line with
# the PHYSICAL radius of the wire (Zc = 359.2 ohm, so it is matched).
# Geometry (dx = 5 mm): 1 mm wire, h = 100 mm above a PEC plane (originally 35 mm), L = 2 m, a series terminal on each end,
# 1 V gaussian generator (sigma = 2.5 ns, t0 = 15 ns) in series with the near-end terminal (= R_S).
#
# This file only orchestrates; each step lives in its own module (same folder):
#   c1_common.py         constants, PASS/WARN/FAIL log, file search, repository paths
#   c1_input.py          static check of c1.fdtd.json + gauss.exc
#   c1_solver.py         run plan (steps/time as semba-fdtd computes them), run with live progress bar and ETA, probes
#   c1_calibration.py    peaks, delay, impedance seen by the source, charge (semba-fdtd only)
#   c1_echo.py           echo of the near-end current (measured, not fitted), late-time residual
#   c1_analytic.py       analytic line with the physical radius (bounce diagram) and its difference to semba-fdtd
#   c1_solver_report.py  Energy.dat / Report.txt checks
#   c1_plots.py          two figures: currents vs analytic (2 panels), near-end shape and echo (3 panels)
#
# One copy of the code, one subfolder per case (each holds its c1.fdtd.json, the .exc file and the results of its runs):
#   h100/ (h = 100 mm, PML, 359.24 ohm; the standard case), h35/ (h = 35 mm, PML, 296.29 ohm: the height of cases B and C),
#   h100_dx10/ (h100 with dx = 10 mm, 4 MPI ranks). Terminals always equal the Zc of the physical radius; boundaries always PML.
# Choose it with C1_CASE (subfolder name or a path; default h100): C1_CASE=h35 python c1.py
# Usage (from anywhere, or cell by cell in VS Code):
#   python c1.py            check the input, run semba-fdtd (live progress bar), analyse the result
#   python c1.py --no-run   check the input and analyse the results of a run already made
# In the interactive window (VS Code / Jupyter): set os.environ["C1_NO_RUN"] = "1" before cell 1, or simply do not run
# cell 2; `solver` stays None and the probes are read from the .dat files of the case folder.
# Environment: SEMBA_EXE (path of semba-fdtd), REPO_ROOT (default: nearest parent folder with src_pyWrapper),
# C1_NO_RUN=1 (same as --no-run), C1_MCELLS_PER_S (speed for the a-priori time estimate; else the last Report.txt).
# MPI is the DEFAULT: C1_MPI_RANKS=8 (1 = serial run with the serial build), C1_MPI_DIR=x (never z with wires), MPIRUN
# (default /usr/local/bin/mpirun if it exists), C1_MPI_FLAGS; SEMBA_EXE defaults to <repo>/build-mpi/bin/semba-fdtd when
# ranks > 1. See mpi_settings in c1_solver.py. On a machine without the MPI build (e.g. the laptop) set C1_MPI_RANKS=1.
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
from c1_plots import plot_case, plot_near_end
from c1_solver import load_currents, run_solver
from c1_solver_report import border_check, energy_check, report_check, warnings_check

NO_RUN = "--no-run" in sys.argv or os.environ.get("C1_NO_RUN") == "1"
solver = None        # stays None when cell 2 is skipped or --no-run: the probes are then read from the .dat files
CASE = "c1.fdtd.json"
CASE_NAME = os.environ.get("C1_CASE", "h100")         # subfolder of the code folder, or a path; set it before cell 1
CASE_DIR = Path(CASE_NAME) if Path(CASE_NAME).is_absolute() else HERE / CASE_NAME
if not (CASE_DIR / CASE).exists():
    found = sorted(p.parent.name for p in HERE.glob("*/" + CASE))
    raise SystemExit(f"{CASE_DIR / CASE} not found. Case folders next to the code: {found}. Set C1_CASE to one of them.")
# MPI by default (ewing scalability tests, 2026-11-17: 8 ranks cutting in x was the fastest; -mpidir z fails with wires).
# Every value can be overridden from the environment before cell 1; C1_MPI_RANKS=1 gives the serial run and the serial build.
os.environ.setdefault("C1_MPI_RANKS", "8")
os.environ.setdefault("C1_MPI_DIR", "x")
if os.path.exists("/usr/local/bin/mpirun"):           # the mpirun of ewing, built with Slurm support
    os.environ.setdefault("MPIRUN", "/usr/local/bin/mpirun")
REPO, SEMBA_EXE, SPINIT_FOLDER = repo_paths(HERE)
if "SEMBA_EXE" not in os.environ and os.environ["C1_MPI_RANKS"] != "1":
    SEMBA_EXE = str(REPO / "build-mpi" / "bin" / "semba-fdtd")     # MPI build, next to build/ in the repository
log = Log()

# %% 1. Static check of the input file (no solver needed)
case = load_case(CASE_DIR, CASE, log)

# %% 2. Run semba-fdtd, unless --no-run. Prints the run plan, then a progress bar updated in place (about 6 min on a
# fast machine; 3x slower on battery). Interrupt = clean stop; the solver reports every ~60 s.
solver = None if NO_RUN else run_solver(case, REPO, SEMBA_EXE, SPINIT_FOLDER)

# %% 3. Probe currents (positive along the polyline, near end -> far end)
t, I_n, I_f = load_currents(case, solver)

# %% 4. Numbers from the first pass of the pulse (semba-fdtd)
cal = calibrate(case, t, I_n, I_f, log)

# %% 5. First echo at the near end and late-time residual (semba-fdtd)
echo = echo_analysis(case, cal, t, I_n, I_f, log)

# %% 6. Analytic reference (physical radius) and its difference to semba-fdtd
curves = compare_with_line(case, cal, t, I_n, I_f, log)

# %% 7. Optional: energy decay and solver messages
energy_check(case.HERE, log, min_db=30.0)      # only ~3 reports in a 50 ns run, the last one at ~43 ns
border_check(case.HERE, case.bnd_xy, log)
warnings_check(case.HERE, log)
report_check(case.HERE, case.dt_solver, log)

# %% 8. Summary
log.summary()

# %% 9. Plots
plot_case(case, cal, echo, t, I_n, I_f, curves, case.WORK / "c1_prepost.png")
plot_near_end(case, cal, echo, t, I_n, case.WORK / "c1_near_end_shape.png")

# %%