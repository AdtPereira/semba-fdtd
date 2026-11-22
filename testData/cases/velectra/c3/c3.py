# %% Velectra cross-talk experiment, case C (c3): TWO thin wires inside the closed PEC cylinder (NEXT / FEXT).
#
# Purpose: case C = case B (c2) with ONE change: the ground plane becomes the closed PEC cylinder (R = 400 mm, L = 2 m).
# Wires, terminals, generator, probes, .exc file, time step and window are those of case B. Case A (c1) showed that thin-wire effects
# (about +6 % of L' at 35 mm) belong to the solver's wire model and not to the boundary or to a radius change, and case B (c2)
# found them again with two wires. So three curves are compared, nothing else: semba-fdtd with the terminals exactly as they are
# in the JSON, the analytic MTL of the wire pair inside the cylinder with the PHYSICAL radius (no fitted r_eff), and the case B
# FDTD result. The main test is the change B -> C against the analytic change: the model offset cancels in it.
# Geometry (dx = 5 mm): the wires of case B (j = 11, 70 mm apart, L = 2 m) in a grid of 168 x 168 x 400 cells that holds the
# cylinder (wall at j = 4 under the wires, h = 35 mm), PML on x and y (physically irrelevant behind the closed shell).
#
# This file only orchestrates; each step lives in its own module (same folder):
#   c3_common.py         constants, PASS/WARN/FAIL log, file search, repository paths, B -> C limits
#   c3_input.py          static check of c3.fdtd.json + the .exc file; the cylinder wall (ring, j = 4, closed); roles of wires and
#                        probes from positions; analytic inductance matrices (cylinder, and the plane of case B)
#   c3_solver.py         run plan (MPI by default, steps/time as semba-fdtd computes them), run with live progress bar and
#                        ETA (follows <case>_Report.txt), clean stop on interrupt, loading of the four probe currents
#   c3_calibration.py    aggressor on the first pass: peaks, delay, Zc seen by the source, effective L' and C', charge
#   c3_crosstalk.py      NE/FE voltages, peaks, polarity, transfer function, late-time ring-down (50 ohm case)
#   c3_analytic.py       MTL reference (circular image method, exact in frequency) and its difference to semba-fdtd
#   c3_bc.py             B -> C comparison against the c2 results of the same case (peak change, Zc, transfer function)
#   c3_solver_report.py  Energy.dat / Report.txt / Warnings.txt checks, absorbing borders listed in the Report
#   c3_plots.py          two figures, two vertical panels each: c3_time.png (I, V) and c3_db.png (transfer function, ring-down)
#
# One copy of the code, one subfolder per case (each holds its c3.fdtd.json, the .exc file and the results of its runs):
#   50ohm/   the real experiment: 50 ohm on the four ends, 300 ns window; compared with c2/50ohm
#   matched/ 290.80 ohm (Zc of one wire inside the cylinder, physical radius) on the four ends, ~40 ns window: no ring-down;
#            compared with c2/matched
# Boundaries are always PML. Choose the case with C3_CASE (subfolder name or a path; default 50ohm): C3_CASE=matched python c3.py
# Usage (from anywhere, or cell by cell in VS Code):
#   python c3.py            check the input, run semba-fdtd (live progress bar), analyse the result
#   python c3.py --no-run   check the input and analyse the results of a run already made
# In the interactive window (VS Code / Jupyter): set os.environ["C3_NO_RUN"] = "1" before cell 1, or simply do not run
# cell 2; `solver` stays None and the probes are read from the .dat files of the case folder.
# Needs for the B -> C cell: the c2 results of the same case, in ../c2/<case>/ (c2_results.npz for 50ohm, c2_results_matched.npz
# for matched) or the file named by C2_RESULTS.
# Environment: SEMBA_EXE (path of semba-fdtd), REPO_ROOT (default: nearest parent folder with src_pyWrapper),
# C3_NO_RUN=1 (same as --no-run), C3_MCELLS_PER_S (speed for the a-priori time estimate; else the last Report.txt).
# MPI is the DEFAULT: C3_MPI_RANKS=8 (1 = serial run with the serial build), C3_MPI_DIR=x (never z with wires), MPIRUN
# (default /usr/local/bin/mpirun if it exists), C3_MPI_FLAGS; SEMBA_EXE defaults to <repo>/build-mpi/bin/semba-fdtd when
# ranks > 1. See mpi_settings in c3_solver.py. On a machine without the MPI build (e.g. the laptop) set C3_MPI_RANKS=1.
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE))                         # the c3_*.py modules

from c3_analytic import compare_transfer, compare_with_mtl
from c3_bc import compare_with_b, load_b
from c3_calibration import calibrate
from c3_common import Log, repo_paths
from c3_crosstalk import crosstalk, late_time, transfer_function
from c3_input import load_case
from c3_plots import plot_db, plot_time
from c3_solver import load_currents, run_solver
from c3_solver_report import border_check, energy_check, report_check, warnings_check

NO_RUN = "--no-run" in sys.argv or os.environ.get("C3_NO_RUN") == "1"
solver = None        # stays None when cell 2 is skipped or --no-run: the probes are then read from the .dat files
CASE = "c3.fdtd.json"
CASE_NAME = os.environ.get("C3_CASE", "50ohm")        # subfolder of the code folder, or a path; set it before cell 1
CASE_DIR = Path(CASE_NAME) if Path(CASE_NAME).is_absolute() else HERE / CASE_NAME
if not (CASE_DIR / CASE).exists():
    found = sorted(p.parent.name for p in HERE.glob("*/" + CASE))
    raise SystemExit(f"{CASE_DIR / CASE} not found. Case folders next to the code: {found}. Set C3_CASE to one of them.")
# MPI by default (ewing scalability tests, 2026-11-17: 8 ranks cutting in x was the fastest; -mpidir z fails with wires).
# Every value can be overridden from the environment before cell 1; C3_MPI_RANKS=1 gives the serial run and the serial build.
os.environ.setdefault("C3_MPI_RANKS", "8")
os.environ.setdefault("C3_MPI_DIR", "x")
if os.path.exists("/usr/local/bin/mpirun"):           # the mpirun of ewing, built with Slurm support
    os.environ.setdefault("MPIRUN", "/usr/local/bin/mpirun")
REPO, SEMBA_EXE, SPINIT_FOLDER = repo_paths(HERE)
if "SEMBA_EXE" not in os.environ and os.environ["C3_MPI_RANKS"] != "1":
    SEMBA_EXE = str(REPO / "build-mpi" / "bin" / "semba-fdtd")     # MPI build, next to build/ in the repository
log = Log()

# %% 1. Static check of the input file (no solver needed)
case = load_case(CASE_DIR, CASE, log)

# %% 2. Run semba-fdtd, unless --no-run. Prints the run plan, then a progress bar updated in place (50ohm: about 13 min
# with 8 MPI ranks, 13.5 Mcells with the PML; matched: about 3 min). Interrupt = clean stop; the solver reports every ~60 s.
solver = None if NO_RUN else run_solver(case, REPO, SEMBA_EXE, SPINIT_FOLDER)

# %% 3. Probe currents (positive along the polyline, near end -> far end): aggressor and victim, near and far
t, I = load_currents(case, solver)

# %% 4. Aggressor on the first pass of the pulse: peaks, delay, Zc seen by the source, effective L' and C', charge
cal = calibrate(case, t, I, log)

# %% 5. Victim: NE and FE voltages, peaks and polarity
xt = crosstalk(case, t, I, log)

# %% 6. Transfer function in the band of the source, and the ring-down of the mismatched line (50 ohm case)
tf = transfer_function(case, t, xt)
late_time(case, t, I, xt, log)

# %% 7. Analytic reference (physical radius): MTL of the wire pair in the cylinder, time and frequency domain
curves = compare_with_mtl(case, cal, xt, t, I, log)
compare_transfer(case, tf, log)

# %% 7b. B -> C: the change against case B (same case name, c2 results) compared with the analytic change
bc = compare_with_b(case, cal, xt, tf, t, load_b(HERE, case, log), log)

# %% 8. Optional: energy decay and solver messages
energy_check(case.HERE, log, min_db=50.0 if not case.matched else 30.0)   # matched: ~40 ns, only a few reports
border_check(case.HERE, case.bnd_xy, log)
warnings_check(case.HERE, log)
report_check(case.HERE, case.dt_solver, log)

# %% 9. Summary
log.summary(f"case C (c3), {case.name}")

# %% 10. Save the results of case C (the case folder: c3_results.npz for 50ohm, c3_results_matched.npz for matched)
out_npz = "c3_results_matched.npz" if case.matched else "c3_results.npz"
np.savez(case.WORK / out_npz, t=t, V_NE=xt.V["NE"], V_FE=xt.V["FE"], **{f"I_{k}": v for k, v in I.items()})

# %% 11. Plots
plot_time(case, cal, xt, curves, t, I, case.WORK / "c3_time.png", bc)   # currents and voltages
plot_db(case, tf, curves, t, I, case.WORK / "c3_db.png", bc)            # transfer function and ring-down, in dB

# %%
