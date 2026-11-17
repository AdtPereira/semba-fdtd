"""c3_bench.py - scaling benchmark of semba-fdtd on the c3 case (serial/OpenMP build and MPI build).

Same layout as c3_smoke.py (cells `# %%`, same helper modules). For every configuration it:
  1. makes a clean run folder  bench_runs/<label>/
  2. writes a short copy of c3.fdtd.json (BENCH_STEPS steps) and copies the .exc file(s)
  3. initialises ngspice exactly as c3_solver.setNgspice does (spinit + .cm code models copied into the run folder,
     SPICE_SCRIPTS="./") and runs the solver with cwd = run folder (every MPI rank inherits that cwd)
  4. measures wall time and Mcells/s and checks the probe files against the first configuration.

Usage
    python c3_bench.py                # full sweep, prints a table, writes c3_bench_results.csv
    python c3_bench.py --dry-run      # only prints the commands that would be launched
    python c3_bench.py --only omp     # only the OpenMP sweep   (or: --only mpi)

Environment (all optional)
    SEMBA_EXE        serial/OpenMP executable   (default <repo>/build/bin/semba-fdtd)
    SEMBA_EXE_MPI    MPI executable             (default <repo>/build-mpi/bin/semba-fdtd)
    REPO_ROOT        repository root            (default: found from this folder)
    MPIRUN           mpirun to use              (default: first mpirun in PATH)
    MPI_FLAGS        extra mpirun flags         (default: empty; e.g. "--bind-to core")
    SEMBA_ARGS_MPI   extra solver flags for MPI runs (default: empty)
    BENCH_STEPS      number of time steps       (default 1000)
    BENCH_THREADS    OpenMP sweep               (default "1,2,4,8,16")
    BENCH_RANKS      MPI sweep                  (default "1,2,4,8,16")
    BENCH_MPI_OMP    OMP_NUM_THREADS of each MPI rank (default "1")
    BENCH_REPEAT     repetitions, best is kept  (default 1)
    BENCH_KEEP       1 = keep the run folders   (default: delete everything but report/log)
"""
import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# %% 0. paths and helpers of the c3 project (same as c3_smoke.py)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from c3_common import repo_paths                       # noqa: E402
from c3_solver import parse_report, setNgspice, hms    # noqa: E402

CASE = "c3.fdtd.json"
BENCH_DIR = HERE / "bench_runs"
RESULTS_CSV = HERE / "c3_bench_results.csv"

STEPS = int(os.environ.get("BENCH_STEPS", "1000"))
THREADS = [int(x) for x in os.environ.get("BENCH_THREADS", "1,2,4,8,16").split(",") if x.strip()]
RANKS = [int(x) for x in os.environ.get("BENCH_RANKS", "1,2,4,8,16").split(",") if x.strip()]
MPI_OMP = os.environ.get("BENCH_MPI_OMP", "1")
REPEAT = int(os.environ.get("BENCH_REPEAT", "1"))
KEEP = os.environ.get("BENCH_KEEP", "0") == "1"
MPI_FLAGS = os.environ.get("MPI_FLAGS", "").split()
SEMBA_ARGS_MPI = os.environ.get("SEMBA_ARGS_MPI", "").split()


def make_bench_json(dst_folder, steps):
    """Copy of c3.fdtd.json cut to `steps` steps, plus the excitation file(s) it needs, inside dst_folder."""
    d = json.loads((HERE / CASE).read_text())
    d["general"]["numberOfSteps"] = int(steps)
    dst_folder.mkdir(parents=True, exist_ok=True)
    (dst_folder / CASE).write_text(json.dumps(d, indent=2))
    for exc in HERE.glob("*.exc"):
        shutil.copy2(exc, dst_folder / exc.name)


def load_average():
    try:
        return os.getloadavg()[0]
    except (AttributeError, OSError):
        return float("nan")


def read_probe_files(folder):
    """{name: array} of the probe .dat files written by the solver (Energy.dat excluded: depends on reports)."""
    out = {}
    for f in sorted(folder.rglob("*.dat")):
        if "Energy" in f.name:
            continue
        try:
            out[f.name] = np.atleast_2d(np.loadtxt(f))
        except Exception:
            pass
    return out


def compare(ref, new):
    """Largest difference relative to the peak of the reference, over the files both runs wrote."""
    worst, n = 0.0, 0
    for name, a in ref.items():
        b = new.get(name)
        if b is None or a.shape != b.shape:
            return float("inf"), n
        peak = np.abs(a[:, 1:]).max() if a.shape[1] > 1 else np.abs(a).max()
        if peak > 0:
            worst = max(worst, float(np.abs(a - b)[:, 1:].max() / peak) if a.shape[1] > 1
                        else float(np.abs(a - b).max() / peak))
        n += 1
    return worst, n


def run_one(label, cmd, env_extra, steps, dry_run):
    """Run `cmd` in a fresh folder with ngspice initialised; returns a dict with the measurements."""
    folder = BENCH_DIR / label
    if dry_run:
        envs = " ".join(f"{k}={v}" for k, v in env_extra.items())
        print(f"  [dry-run] cd {folder} && SPICE_SCRIPTS=./ {envs} {' '.join(cmd)}")
        return None
    if folder.exists():
        shutil.rmtree(folder)
    make_bench_json(folder, steps)
    setNgspice(folder, SPINIT_FOLDER)                           # spinit + *.cm in the run folder, SPICE_SCRIPTS="./"
    missing = [f for f in ("spinit",) if not (folder / f).exists()]
    if missing:
        raise RuntimeError(f"ngspice files not copied from {SPINIT_FOLDER}: {missing}")

    env = dict(os.environ, SPICE_SCRIPTS="./", **env_extra)
    log = folder / "stdout.log"
    load0 = load_average()
    t0 = time.time()
    with open(log, "wb") as out:
        rc = subprocess.run(cmd, cwd=folder, env=env, stdout=out, stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    if rc != 0:
        raise RuntimeError(f"{label}: exit code {rc}\n" + log.read_bytes()[-1500:].decode(errors="ignore"))

    rep = next(iter(sorted(folder.rglob("*Report.txt"))), None)
    st = parse_report(rep.read_text(errors="ignore")) if rep else parse_report("")
    mcells = st.mcells or MCELLS_FALLBACK
    res = dict(label=label, wall=wall, steps=steps, mcells=mcells,
               mcells_s_wall=mcells * steps / wall, mcells_s_report=st.speed_avg, load0=load0,
               probes=read_probe_files(folder))
    if not KEEP:                                                 # keep only report and log, the .dat are big
        for f in folder.iterdir():
            if f.is_file() and not (f.name.endswith("Report.txt") or f.name == "stdout.log"):
                f.unlink()
    return res


def best_of(label, cmd, env_extra, steps, dry_run):
    runs = [run_one(label, cmd, env_extra, steps, dry_run) for _ in range(REPEAT)]
    if dry_run:
        return None
    return min(runs, key=lambda r: r["wall"])


# %% 1. executables and mpirun
REPO, SEMBA_EXE, SPINIT_FOLDER = repo_paths(HERE)
SEMBA_EXE_MPI = os.environ.get("SEMBA_EXE_MPI", str(Path(REPO) / "build-mpi" / "bin" / "semba-fdtd"))
MPIRUN = os.environ.get("MPIRUN") or shutil.which("mpirun")
MCELLS_FALLBACK = 11.29                                          # Mcells of the c3 grid (168 x 168 x 400)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    ap.add_argument("--only", choices=["omp", "mpi"], help="run only one of the two sweeps")
    args = ap.parse_args()

    ncpu = os.cpu_count() or 1
    print(f"c3_bench: {STEPS} steps per run, {ncpu} logical CPUs, load average {load_average():.1f}")
    if load_average() > 2 and not args.dry_run:
        print("  WARNING: the machine is not idle (see btop / uptime); the numbers will be pessimistic.")

    plan = []                                                    # (group, label, cmd, env)
    if args.only != "mpi":
        for t in THREADS:
            plan.append(("omp", f"omp{t:02d}", [SEMBA_EXE, "-i", CASE],
                         {"OMP_NUM_THREADS": str(t), "OMP_PROC_BIND": "close", "OMP_PLACES": "cores"}))
    if args.only != "omp":
        for n in RANKS:
            plan.append(("mpi", f"mpi{n:02d}", [MPIRUN or "mpirun", *MPI_FLAGS, "-n", str(n), SEMBA_EXE_MPI,
                                                "-i", CASE, *SEMBA_ARGS_MPI],
                         {"OMP_NUM_THREADS": MPI_OMP}))

    for exe, name in [(SEMBA_EXE, "serial/OpenMP"), (SEMBA_EXE_MPI, "MPI")]:
        used = any(g == ("omp" if name != "MPI" else "mpi") for g, *_ in plan)
        if used and not args.dry_run and not Path(exe).exists():
            sys.exit(f"{name} executable not found: {exe} (set SEMBA_EXE / SEMBA_EXE_MPI)")
    if any(g == "mpi" for g, *_ in plan) and not args.dry_run and not MPIRUN:
        sys.exit("mpirun not found in PATH (set MPIRUN)")
    if not (HERE / CASE).exists():
        sys.exit(f"{CASE} must be in {HERE}")

    # %% 2. run
    results, ref = [], None
    for group, label, cmd, env in plan:
        print(f"\n[{label}] {' '.join(cmd)}   ({', '.join(f'{k}={v}' for k, v in env.items())})")
        r = best_of(label, cmd, env, STEPS, args.dry_run)
        if r is None:
            continue
        r["group"] = group
        if ref is None:
            ref = r["probes"]
            r["diff"], r["nfiles"] = 0.0, len(ref)
        else:
            r["diff"], r["nfiles"] = compare(ref, r["probes"])
        r["cores"] = int(re.sub(r"\D", "", label))
        results.append(r)
        print(f"    wall {hms(r['wall'])} ({r['wall']:.1f} s)   {r['mcells_s_wall']:.1f} Mcells/s")
    if args.dry_run:
        return

    # %% 3. table + csv
    base = {}
    for g in ("omp", "mpi"):
        rows = [r for r in results if r["group"] == g]
        if rows:
            base[g] = min(rows, key=lambda r: r["cores"])
    print("\n" + "=" * 86)
    print(f"{'config':8s} {'wall [s]':>9s} {'Mcells/s':>9s} {'(report)':>9s} {'speedup':>8s} {'effic.':>7s}"
          f" {'max diff':>10s}  load")
    for r in results:
        b = base[r["group"]]
        sp = b["wall"] / r["wall"] * 1.0
        eff = sp / (r["cores"] / b["cores"])
        rep = f"{r['mcells_s_report']:.1f}" if r["mcells_s_report"] else "-"
        diff = "inf" if math.isinf(r["diff"]) else f"{r['diff']:.1e}"
        print(f"{r['label']:8s} {r['wall']:9.1f} {r['mcells_s_wall']:9.1f} {rep:>9s} {sp:8.2f} {eff:7.0%}"
              f" {diff:>10s}  {r['load0']:.1f}")
    print("=" * 86)
    print("speedup/efficiency are relative to the smallest run of the same sweep; 'max diff' is the largest probe\n"
          "difference against the first run, relative to its peak (float32: ~1e-6 is round-off, 1e-3 is not).\n"
          "Wall time includes the start-up (reading the case, allocation, ngspice), so short runs underestimate\n"
          "the speed: the '(report)' column is the solver's own average when the run lasted long enough to log it.")

    with open(RESULTS_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "group", "cores", "steps", "mcells", "wall_s", "mcells_per_s_wall",
                    "mcells_per_s_report", "max_rel_diff", "load_before"])
        for r in results:
            w.writerow([r["label"], r["group"], r["cores"], r["steps"], r["mcells"], f"{r['wall']:.2f}",
                        f"{r['mcells_s_wall']:.2f}", r["mcells_s_report"] or "", f"{r['diff']:.3e}",
                        f"{r['load0']:.2f}"])
    print(f"\nwritten {RESULTS_CSV}")
    best = max(results, key=lambda r: r["mcells_s_wall"])
    full = 38944 * best["mcells"] / best["mcells_s_wall"]
    print(f"best configuration: {best['label']}  ->  estimated full c3 run (38944 steps): {hms(full)}")


if __name__ == "__main__":
    main()
