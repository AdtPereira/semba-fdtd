# %% Paul, Introduction to EMC, Example 9.2 / Fig. D.28: crosstalk between two wires over a ground plane
#
# Place this file, paul_d28.fdtd.json and paul_d28.exc in <semba-fdtd>/testData/cases/paul/
# (next to Paul's other cases) and run it from anywhere:  python paul_d28_prepost.py
# or cell by cell in VS Code (# %%). Paths are resolved from this file's folder.
import glob
import os
import shutil
import sys
from pathlib import Path
from sys import platform

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
os.chdir(HERE)                                   # FDTD runs and writes its outputs here
REPO = HERE.parents[2]                           # testData/cases/paul -> repository root

sys.path.append(str(REPO / "src_pyWrapper"))
from pyWrapper import FDTD, Probe                # noqa: E402

SEMBA_EXE = os.environ.get("SEMBA_EXE", str(REPO / "build" / "bin" / "semba-fdtd"))
SPINIT_FOLDER = REPO / "testData" / "spinit"

R = 50.0                                         # all four terminations (ohm)
CASE = "paul_d28.fdtd.json"
# Elemwave probe names and where they are (current positive from near end to far end):
PROBES = {
    "S":  "Wire probe001",                       # generator wire, near end
    "NE": "Wire probe002",                       # receptor wire, near end
    "L":  "Wire probe003",                       # generator wire, far end
    "FE": "Wire probe004",                       # receptor wire, far end
}


def makeCopy(dest_dir, src_path):
    for file in glob.glob(str(src_path)):
        shutil.copy2(file, os.path.join(dest_dir, os.path.basename(file)))


def setNgspice(tmp_path):
    """MTLN generators are ngspice XSPICE sources: spinit + code models must be next to the run."""
    sys_name = "windows" if platform == "win32" else "linux"
    os.environ["SPICE_SCRIPTS"] = "./"
    for f in ["spinit", "analog.cm", "digital.cm", "spice2poly.cm", "table.cm", "xtradev.cm", "xtraevt.cm"]:
        makeCopy(tmp_path, SPINIT_FOLDER / sys_name / f)


setNgspice(HERE)

# %% Run (about 2 min)
solver = FDTD(input_filename=CASE, path_to_exe=SEMBA_EXE)
solver.cleanUp()
solver.run()
if solver.output.returncode != 0:
    raise RuntimeError(solver.output.stdout.decode(errors="ignore")[-2000:])

# %% Post-process: terminal voltages from the wire-probe currents
I = {}
for end, name in PROBES.items():
    p = Probe(solver.getSolvedProbeFolders(name)[0])
    t = p["time"].to_numpy()
    I[end] = p["current_0"].to_numpy()

V = {
    "NE": -R * I["NE"],                          # current flows into the line at the near end
    "FE": +R * I["FE"],
    "L":  +R * I["L"],
}
for k in V:
    assert np.all(np.isfinite(V[k])) and not np.all(V[k] == 0.0), f"V_{k}: NaN/Inf or all zero"

plateau = 1e3 * V["NE"][(t > 14e-9) & (t < 19e-9)].mean()
print(f"V_NE first plateau: {plateau:.2f} mV   (exact lossless MTL: 23.33 mV; "
      f"MTLN effective radius gives about 21.2 mV)")
print(f"V_FE minimum:       {1e3 * V['FE'].min():.2f} mV  (exact: -31.99 mV)")

# %% Plot, with the exact solution if a reference file is present
ref_file = HERE / "paul_d28_exact.csv"           # optional: velectra runs/paul_d28/analytic/time.csv
ref = np.genfromtxt(ref_file, delimiter=",", names=True) if ref_file.exists() else None

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
for ax, k in zip(axes, ["NE", "FE"]):
    ax.plot(1e9 * t, 1e3 * V[k], "b", label=f"semba-fdtd V_{k}")
    if ref is not None:
        ax.plot(1e9 * ref["t"], 1e3 * ref[f"V_{k}"], "k--", label=f"exact MTL V_{k}")
    ax.set_ylabel(f"V_{k} (mV)")
    ax.grid(True)
    ax.legend()
axes[-1].set_xlabel("t (ns)")
axes[0].set_title("Paul Fig. D.28: near- and far-end crosstalk")
plt.tight_layout()
plt.show()

# %%
