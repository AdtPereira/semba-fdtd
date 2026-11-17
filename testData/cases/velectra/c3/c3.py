# %% Velectra cross-talk experiment, case C (c3): TWO thin wires inside the closed PEC cylinder (NEXT / FEXT).
#
# Case C = case B (c2) with ONE change: the ground plane becomes the closed PEC cylinder (R = 400 mm, L = 2 m).
# Wires, 50 ohm terminals, generator, probes, .exc file, boundaries, time step and the 300 ns window are identical.
# Grid 168 x 168 x 400 cells of 5 mm (about 11.3 Mcells, twice B). Expected (tutorial 5.1, r_eff): NE +7.10 mV,
# FE -10.22 mV, i.e. about -10 % against case B (+7.82 / -11.28 mV measured there); Zc effective about 308 ohm.
#
# This file only orchestrates; each step lives in its own module (same folder):
#   c3_common.py         constants, PASS/WARN/FAIL log, file search, repository paths
#   c3_input.py          static check of c3.fdtd.json + the .exc file; the cylinder wall (ring, j = 4, closed);
#                        roles of wires and probes from positions
#   c3_solver.py         run plan (steps/time as semba-fdtd computes them), run with live progress bar and ETA
#                        (follows <case>_Report.txt), clean stop on interrupt, loading of the four probe currents
#   c3_calibration.py    aggressor on the first pass: peaks, delay, Zc effective, probe sign
#   c3_crosstalk.py      NE/FE voltages, peaks, polarity, transfer function, late-time ring-down
#   c3_analytic.py       MTL reference (circular image method, exact in frequency) and the comparison checks
#   c3_bc.py             B -> C comparison against c2_results.npz (peak change, transfer-function change)
#   c3_solver_report.py  Energy.dat / Report.txt checks
#   c3_plots.py          4-panel figure
#
# Usage (from anywhere, or cell by cell in VS Code):
#   python c3.py            check the input, run semba-fdtd, analyse the result
#   python c3.py --no-run   check the input and analyse the results of a run already made
# Environment: SEMBA_EXE (path of semba-fdtd), REPO_ROOT (default: nearest parent folder with src_pyWrapper),
#              C3_NO_RUN=1 (same as --no-run), C3_MCELLS_PER_S (speed for the a-priori time estimate).
# Needs next to it: c3.fdtd.json, gauss_sigma2p5ns_t15ns.exc and (for the B -> C cell) c2_results.npz.
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
from c3_plots import plot_case
from c3_solver import load_currents, run_solver
from c3_solver_report import energy_check, report_check

NO_RUN = "--no-run" in sys.argv or os.environ.get("C3_NO_RUN") == "1"
solver = None        # stays None when cell 2 is skipped or --no-run: the probes are then read from the .dat files
CASE = "c3.fdtd.json"
REPO, SEMBA_EXE, SPINIT_FOLDER = repo_paths(HERE)
log = Log()

# %% 1. Static check of the input file (no solver needed): grid, wires, terminals, time plan, and the cylinder wall
case = load_case(HERE, CASE, log)

# %% 2. Run semba-fdtd, unless --no-run. Prints the run plan, then a progress bar updated in place (about 1.0 to
# 1.1 h at 120 Mcells/s: 11.3 Mcells x 38944 steps; keep the laptop plugged in, high-performance power plan).
# Interrupt = clean stop; the solver reports every ~60 s. A new run overwrites the .dat files of this folder.
# solver = None if NO_RUN else run_solver(case, REPO, SEMBA_EXE, SPINIT_FOLDER)

# %% 3. Probe currents (positive along the polyline, near end -> far end): aggressor and victim, near and far
t, I = load_currents(case, solver)

# %% 4. Aggressor on the first pass of the pulse: peaks, delay, Zc effective (about 308 ohm), probe sign
cal = calibrate(case, t, I, log)

# %% 5. Victim: NE and FE voltages, peaks and polarity
xt = crosstalk(case, t, I, log)

# %% 6. Transfer function in the band of the source, and the ring-down (case B ended at -41.7 dB: thin margin)
tf = transfer_function(case, t, xt)
late_time(case, t, I, xt, log)

# %% 7. Analytic reference: MTL of the wire pair in the cylinder, time and frequency domain
curves = compare_with_mtl(case, cal, xt, t, I, log)
compare_transfer(case, tf, log)

# %% 7b. B -> C: the closed cylinder should give about -10 % in NE and FE (needs c2_results.npz)
bc = compare_with_b(case, xt, tf, t, load_b(HERE, log), log)

# %% 8. Optional: energy decay and solver messages
energy_check(case.HERE, log)
report_check(case.HERE, case.dt_solver, log)

# %% 9. Summary
log.summary()

# %% 10. Save the results of case C
np.savez(case.WORK / "c3_results.npz", t=t, V_NE=xt.V["NE"], V_FE=xt.V["FE"], **{f"I_{k}": v for k, v in I.items()})

# %% 11. Plots
plot_case(case, cal, xt, tf, curves, t, I, case.WORK / "c3.png", bc)

# %%