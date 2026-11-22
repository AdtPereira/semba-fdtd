"""Run semba-fdtd (through pyWrapper) with a live progress bar, and load the two probe currents.

How semba-fdtd decides the length of a run and reports it (OpenSEMBA/fdtd, commit cd069a6; see solver_time_plan in
c1_common and the notes at the end of run_plan):
  * total steps = int(dt_json / dt_corrected * numberOfSteps), dt_corrected = 0.8 dt_CFL when the JSON dt is larger;
  * progress is printed to <case>_Report.txt (flushed) about every 60 s of wall time (REPORTINGSECONDS, progress.F90):
    "Simulated: n / N steps", "Mcells/sec : speed (n0 to n)", "Mins. until end";
  * the solver's own ETA is (N - n) * Total_Mcells / average_Mcells_per_second.
pyWrapper.FDTD.run() waits in silence (subprocess.run with capture_output), which in a Jupyter / VS Code interactive
cell means nothing is shown for the whole run. run_solver starts the same command (solver.run_command) with Popen,
sends its stdout to <case>_stdout.log and follows <case>_Report.txt, drawing a bar:

    c1  [##########..................................]  24.0 %   step ~1245/5191   t ~9.6/40.0 ns
        elapsed 0:01:30   ETA 0:04:45 (about 16:41)   54.2 Mcells/s   last report 17 s ago

Between two reports (60 s) the step is extrapolated with the speed of the last report ('~'). Interrupt (the button of
the interactive window, or Ctrl-C) asks semba-fdtd to stop cleanly by writing the 'stop' file, which also flushes the
probe data; it can take up to a minute because the solver looks for that file only at each report.
"""
import glob
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from sys import platform
from types import SimpleNamespace

import numpy as np

from c1_common import find_file, solver_time_plan

POLL = 1.0                      # seconds between screen updates of the monitor
BAR_WIDTH = 40
STOP_GRACE = 150                # seconds to wait for a clean stop (about two report periods) before terminating


def setNgspice(folder, spinit_folder):
    """MTLN generators are ngspice XSPICE sources: spinit and code models must be in the run folder."""
    sys_name = "windows" if platform == "win32" else "linux"
    os.environ["SPICE_SCRIPTS"] = "./"
    for f in ["spinit", "analog.cm", "digital.cm", "spice2poly.cm", "table.cm", "xtradev.cm", "xtraevt.cm"]:
        for src in glob.glob(str(Path(spinit_folder) / sys_name / f)):
            shutil.copy2(src, Path(folder) / os.path.basename(src))


# ---------------------------------------------------------------------------------------------------------------
# Plan of the run, before launching it
# ---------------------------------------------------------------------------------------------------------------
def hms(seconds):
    s = int(max(0, round(seconds)))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def speed_hint(case):
    """Mcells/s to use for an a-priori estimate: C1_MCELLS_PER_S, else the average of the last Report.txt found."""
    if os.environ.get("C1_MCELLS_PER_S"):
        return float(os.environ["C1_MCELLS_PER_S"]), "C1_MCELLS_PER_S"
    f = find_file(case.WORK, "*Report.txt")
    if f is not None:
        st = parse_report(f.read_text(errors="ignore"))
        if st.speed_avg:
            return st.speed_avg, f.name
    return None, None


def mpi_settings():
    """MPI launch from the environment (scalability tests of 2026-11-17 on ewing, case c3, MPI build):
      C1_MPI_RANKS   number of ranks; 1 or unset = the serial run, as before. 8 was the best point (2.9x the serial build);
                     16 was slower than 8, and one MPI rank is 1.8x slower than the serial build.
      C1_MPI_DIR     cut direction, default x. The default of semba-fdtd (z) aborts with MPI_ERR_RANK when a wire is present;
                     x and y work. The wire of case A runs along z, like those of c3, so x or y.
      MPIRUN         mpirun to use (default 'mpirun'; on ewing the one in /usr/local/bin, built with Slurm support)
      C1_MPI_FLAGS   extra words for the mpirun command (before the executable)
    SEMBA_EXE must then point to the MPI build (build-mpi/bin/semba-fdtd). Returns (mpi_command, flags, ranks, direction)."""
    ranks = int(os.environ.get("C1_MPI_RANKS", "1"))
    if ranks <= 1:
        return None, None, 1, None
    direction = os.environ.get("C1_MPI_DIR", "x").lower()
    assert direction in ("x", "y", "z"), f"C1_MPI_DIR must be x, y or z, not {direction!r}"
    cmd = f"{os.environ.get('MPIRUN', 'mpirun')} -n {ranks} {os.environ.get('C1_MPI_FLAGS', '')}".split()
    return " ".join(cmd), f"-mpidir {direction}", ranks, direction


def run_mode(semba_exe, ranks, direction, mpi_command):
    """Short tag of how the solver is launched, shown in the plan, on the progress bar and in the final line."""
    if ranks > 1:
        return f"MPI {ranks} ranks, -mpidir {direction}"
    return "serial (1 process)"


def run_plan(case, ranks=1, direction=None, semba_exe=None, mpi_command=None):
    """Print and return what semba-fdtd will run: steps, simulated time, size, and the time it should take."""
    p = case.plan
    mcells = float(np.prod(case.ncell)) / 1e6                  # semba reports the exact 'Total Mcells' at the start
    speed, src = speed_hint(case)
    plan = SimpleNamespace(**vars(p), mcells=mcells, speed=speed, mode=run_mode(semba_exe, ranks, direction, mpi_command))
    print("\n=========== semba-fdtd run plan (rules of OpenSEMBA/fdtd, checked on commit cd069a6) ===========")
    print(f"  JSON    : dt = {p.dt_json:.4e} s x {p.n_json} steps = {p.t_json * 1e9:.4f} ns")
    print(f"  CFL     : dt_CFL = {p.dt_cfl:.4e} s (dx = {case.dx * 1e3:g} mm); semba-fdtd accepts dt <= 0.8 dt_CFL "
          f"= {p.dt_limit:.4e} s")
    if p.corrected:
        print(f"  CORRECTED: dt = {p.dt:.4e} s and steps = int(dt_json/dt x N) = {p.n_final} "
              f"-> simulated time {p.t_final * 1e9:.4f} ns")
    else:
        print(f"  dt kept : {p.n_final} steps -> simulated time {p.t_final * 1e9:.4f} ns")
    print(f"  size    : {' x '.join(str(n) for n in case.ncell)} = {mcells:.2f} Mcells")
    exe = str(semba_exe or "?")
    is_mpi_build = "mpi" in exe.lower()
    print(f"  MODE    : {plan.mode}")
    print(f"  solver  : {exe}  ({'MPI build' if is_mpi_build else 'serial build'})")
    if ranks > 1:
        found = shutil.which((mpi_command or "mpirun").split()[0])
        print(f"  launcher: {mpi_command} {'-> ' + found if found else '-> NOT FOUND in PATH'}")
        print(f"  threads : OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS', '?')} per rank; "
              f"{os.cpu_count()} logical CPUs on this machine")
        if not is_mpi_build:
            print("  WARNING : C1_MPI_RANKS > 1 but SEMBA_EXE does not look like the MPI build (build-mpi/bin/semba-fdtd)")
        if not found:
            print("  WARNING : the launcher was not found; set MPIRUN to the full path of mpirun")
    else:
        print(f"  threads : OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS', 'not set')}; the serial build does not "
              f"scale with OpenMP (ewing, 2026-11-17)")
        if is_mpi_build:
            print("  NOTE    : the MPI build is running on 1 process, about 1.8x slower than the serial build")
        else:
            print("  NOTE    : serial run. For MPI: SEMBA_EXE = build-mpi/bin/semba-fdtd and C1_MPI_RANKS=8 "
                  "(about 2.9x faster on c3)")
    if ranks > 1:
        per_rank = case.ncell["xyz".index(direction)] / ranks
        print(f"  MPI     : {ranks} ranks, cut in {direction}: {per_rank:.0f} cells per rank "
              f"(the 16-rank run of c3, about 10 cells per rank, was slower than 8); OMP_NUM_THREADS = 1")
        if direction == "z":
            print("  WARNING : -mpidir z aborts with MPI_ERR_RANK when wires are present (c3, 2026-11-17); use x or y")
        if per_rank < 16:
            print("  WARNING : fewer than 16 cells per rank: the exchange of borders may cost more than it saves")
        print("            the speed hint below comes from the last Report.txt: a serial one underestimates the MPI run")
    if speed:
        t_run = p.n_final * mcells / speed
        print(f"  expected: {hms(t_run)} at {speed:.1f} Mcells/s (from {src}; "
              f"finish about {(datetime.now() + timedelta(seconds=t_run)):%H:%M})")
    else:
        print("  expected: unknown until the first report (set C1_MCELLS_PER_S for an estimate now)")
    print("  reports : <case>_Report.txt, about every 60 s of wall time (the first one after ~60 s); "
          "the bar extrapolates between reports")
    return plan


# ---------------------------------------------------------------------------------------------------------------
# _Report.txt parser
# ---------------------------------------------------------------------------------------------------------------
_F = r"([0-9]*\.?[0-9]+(?:[EeDd][+-]?\d+)?)"


def parse_report(text):
    """State of the run from the text of <case>_Report.txt (as written by InitTiming and Timing in progress.F90)."""
    st = SimpleNamespace(n0=None, n_final=None, t0=None, t_final=None, mcells=None, n=None, speed_inst=None,
                         speed_avg=None, t_now=None)
    m = re.search(r"Simulation from n=\s*(\d+),\s*t=\s*" + _F + r"\s*to n=\s*(\d+),\s*t=\s*" + _F, text)
    if m:
        st.n0, st.t0, st.n_final, st.t_final = int(m[1]), float(m[2].replace("D", "E")), int(m[3]), \
            float(m[4].replace("D", "E"))
    m = re.search(r"Total Mcells:\s*" + _F, text)
    if m:
        st.mcells = float(m[1])
    sims = list(re.finditer(r"Simulated:\s*(\d+)\s*/\s*(\d+)\s*steps", text))
    if sims:
        last = sims[-1]
        st.n, st.n_final = int(last[1]), int(last[2])          # the denominator is the final step really in use
        block = text[last.start():]
        speeds = re.findall(r"Mcells/sec\s*:\s*" + _F + r"\s*\(\s*(\d+)\s*to\s*(\d+)", block)
        if speeds:
            st.speed_avg = float(speeds[-1][0])                  # last line is the average since the start
            st.speed_inst = float(speeds[0][0]) if len(speeds) > 1 else None
        m = re.search(r"Time=\s*" + _F, block)
        if m:
            st.t_now = float(m[1].replace("D", "E"))
    else:                                                        # no report yet: maybe a finished older run
        speeds = re.findall(r"Mcells/sec\s*:\s*" + _F, text)
        if speeds:
            st.speed_avg = float(speeds[-1])
    return st


# ---------------------------------------------------------------------------------------------------------------
# Screen: Jupyter / VS Code interactive window (clear_output), terminal (carriage return) or plain log lines
# ---------------------------------------------------------------------------------------------------------------
def _in_notebook():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


class Screen:
    def __init__(self):
        self.mode = "notebook" if _in_notebook() else "tty" if sys.stdout.isatty() else "log"
        self._handle = None                                      # notebook: one display object, updated in place

    def update(self, lines, new_report=False):
        l1, l2, compact = lines
        if self.mode == "notebook":                              # only the bar is redrawn; the run plan above stays
            self._notebook(l1 + "\n" + l2)
        elif self.mode == "tty":                                 # one short line, cut to the width of the terminal
            width = shutil.get_terminal_size((100, 20)).columns - 1
            sys.stdout.write("\r\033[K" + compact[:width])
            sys.stdout.flush()
        elif new_report:                                         # redirected to a file: one line per report
            print(l1.strip() + " | " + l2.strip(), flush=True)

    def _notebook(self, text):
        from IPython.display import Pretty, display
        if self._handle is None:
            self._handle = display(Pretty(text), display_id=True)
        else:
            self._handle.update(Pretty(text))

    def finish(self, lines):                                     # lines: the two final lines
        if self.mode == "notebook":
            self._notebook("\n".join(lines))
            return
        if self.mode == "tty":
            sys.stdout.write("\r\033[K")
        print("\n".join(lines), flush=True)


def _bar(frac, width=BAR_WIDTH):
    k = int(round(width * min(max(frac, 0.0), 1.0)))
    return "[" + "#" * k + "." * (width - k) + "]"


def _lines(name, plan, st, t0_wall, rep, tick):
    """(line 1, line 2, compact line) of the bar. rep = (n, wall time) of the last report, or None before the first."""
    now = time.time()
    elapsed = now - t0_wall
    n_final = st.n_final or plan.n_final
    if rep is None:                                              # waiting for the first report: dotted slots
        dots = "." * (tick % 4)
        return (f"{name}  {_bar(0)}   0.0 %   step 0/{n_final}   t 0/{plan.t_final * 1e9:.1f} ns",
                f"    {plan.mode}   elapsed {hms(elapsed)}   waiting for the first semba report (they come every ~60 s){dots:<3}",
                f"{name} {_bar(0, 20)}  0.0 %  el {hms(elapsed)}  waiting 1st report{dots:<3}")
    n_rep, w_rep = rep
    mcells = st.mcells or plan.mcells
    speed = st.speed_inst or st.speed_avg
    steps_s = speed / mcells if speed else 0.0
    n_est = min(n_final, n_rep + steps_s * (now - w_rep))        # extrapolated between reports
    frac = n_est / n_final
    t_final = (st.t_final or plan.t_final)
    eta = ((n_final - n_est) * mcells / st.speed_avg) if st.speed_avg else None   # the solver's own formula
    tail = f"ETA {hms(eta)} (about {(datetime.now() + timedelta(seconds=eta)):%H:%M})" if eta is not None else "ETA ?"
    spd = f"{speed:.1f} Mcells/s" + (f" (avg {st.speed_avg:.1f})" if st.speed_avg and st.speed_inst else "")
    return (f"{name}  {_bar(frac)} {100 * frac:5.1f} %   step ~{int(n_est)}/{n_final}   "
            f"t ~{t_final * frac * 1e9:.1f}/{t_final * 1e9:.1f} ns",
            f"    {plan.mode}   elapsed {hms(elapsed)}   {tail}   {spd}   last report {hms(now - w_rep)} ago",
            f"{name} {_bar(frac, 20)} {100 * frac:5.1f} %  n~{int(n_est)}/{n_final}  el {hms(elapsed)}  "
            f"{tail.replace('(about ', '(').replace('ETA ', 'ETA ')}")


# ---------------------------------------------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------------------------------------------
def _report_path(solver, t_launch):
    f = Path(solver.getFolder()) / f"{solver.getCaseName()}_Report.txt"
    if f.exists() and f.stat().st_mtime >= t_launch - 2:
        return f
    hits = [p for p in Path(solver.getFolder()).glob("*_Report.txt") if p.stat().st_mtime >= t_launch - 2]
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None


def _stop_cleanly(proc, folder):
    """Ask for a clean stop ('stop' file: probes are flushed), wait, and terminate if it does not come."""
    stop = Path(folder) / "stop"
    stop.touch()
    print(f"\nInterrupted: asked semba-fdtd to stop (file '{stop.name}'); it checks at each report, up to ~1 min. "
          f"Interrupt again to terminate it now.")
    try:
        proc.wait(timeout=STOP_GRACE)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    stop.unlink(missing_ok=True)


def run_solver(case, repo, semba_exe, spinit_folder, progress=True):
    """Copy the ngspice files, run semba-fdtd with a live progress bar; returns the pyWrapper solver object."""
    sys.path[:0] = [str(Path(repo) / "src_pyWrapper")]
    from pyWrapper import FDTD                       # noqa: E402

    setNgspice(case.WORK, spinit_folder)
    mpi_command, flags, ranks, direction = mpi_settings()
    if ranks > 1:
        os.environ["OMP_NUM_THREADS"] = "1"          # one thread per rank (as in the scalability tests)
    solver = FDTD(input_filename=case.case_path.name, path_to_exe=semba_exe, flags=flags, mpi_command=mpi_command)
    solver.cleanUp()
    plan = run_plan(case, ranks, direction, semba_exe, mpi_command)
    print(f"  command : {' '.join(solver.run_command)}")

    folder = solver.getFolder()
    log_path = Path(folder) / f"{solver.getCaseName()}_stdout.log"
    t_launch = time.time()
    screen, name, tick, rep, st, seen = Screen(), case.case_path.stem.split(".")[0], 0, None, parse_report(""), None
    with open(log_path, "wb") as out:
        # own session: the interrupt of Jupyter / Ctrl-C reaches Python only, so the 'stop' file can end the run cleanly
        proc = subprocess.Popen(solver.run_command, cwd=folder, stdout=out, stderr=subprocess.STDOUT,
                                start_new_session=(os.name == "posix"))
        try:
            while proc.poll() is None:
                time.sleep(POLL)
                tick += 1
                rp = _report_path(solver, t_launch)
                new = False
                if rp is not None:
                    size = rp.stat().st_size
                    if size != seen:                                       # the report grew: read its tail
                        seen = size
                        with open(rp, "rb") as fh:
                            fh.seek(max(0, size - 60000))
                            txt = fh.read().decode(errors="ignore")
                        with open(rp, errors="ignore") as fh:
                            head = parse_report(fh.read(20000))        # 'Simulation from', 'Total Mcells'
                        st = parse_report(txt)
                        st.n_final, st.t_final, st.mcells = st.n_final or head.n_final, st.t_final or head.t_final, \
                            st.mcells or head.mcells
                        if st.n is not None and (rep is None or st.n != rep[0]):
                            rep, new = (st.n, time.time()), True
                if progress:
                    screen.update(_lines(name, plan, st, t_launch, rep, tick), new)
        except KeyboardInterrupt:
            _stop_cleanly(proc, folder)
            raise
        except BaseException:                                  # a bug here must not leave the solver running alone
            proc.terminate()
            raise
    rc = proc.returncode
    out_txt = log_path.read_bytes()[-200000:]
    solver.output = subprocess.CompletedProcess(solver.run_command, rc, stdout=out_txt, stderr=b"")
    solver._hasRun = True
    wall = time.time() - t_launch
    if rc != 0:
        raise RuntimeError(out_txt.decode(errors="ignore")[-2000:])
    rp = _report_path(solver, t_launch)
    st = parse_report(rp.read_text(errors="ignore")) if rp else st
    n_done = st.n or plan.n_final
    screen.finish((f"{name}  {_bar(1.0)} 100.0 %   step {n_done}/{st.n_final or plan.n_final}   "
                   f"t {(st.t_final or plan.t_final) * 1e9:.3f} ns",
                   f"    {plan.mode}   finished in {hms(wall)}   average {st.speed_avg or 0:.1f} Mcells/s   "
                   f"(solver log: {log_path.name}, report: {rp.name if rp else '?'})"))
    return solver


def _key(s):
    return re.sub(r"[^0-9a-z]", "", s.lower())


def probe_dat(root, name):
    """Elemwave/semba text probe of a run already made: header line, then time and current columns.
    The file is found by the probe name itself, ignoring case, spaces and underscores (newest match wins); if the name
    ends in a number and nothing matches, the older pattern '*Wire*probe<n>*.dat' is tried."""
    key = _key(name)
    ok = lambda p: key in _key(p.stem) and "energy" not in p.stem.lower()
    hits = [p for p in Path(root).glob("*.dat") if ok(p)] or [p for p in Path(root).rglob("*.dat") if ok(p)]
    if not hits and re.search(r"(\d+)$", name):
        n = re.search(r"(\d+)$", name).group(1)
        hits = sorted(Path(root).rglob(f"*Wire*probe{n}*.dat"))
    assert hits, f"no .dat file for probe '{name}' under {root}; run without --no-run or point to the results folder"
    f = max(hits, key=lambda p: p.stat().st_mtime)
    a = np.loadtxt(f, skiprows=1)
    return a[:, 0], a[:, 1]


def load_currents(case, solver=None):
    """(t, I_near, I_far) on one uniform time axis; positive along the polyline, near end -> far end."""
    raw = {}
    for end, name in case.PROBES.items():
        if solver is not None:
            from pyWrapper import Probe              # noqa: E402
            p = Probe(solver.getSolvedProbeFolders(name)[0])
            raw[end] = (p["time"].to_numpy(), p["current_0"].to_numpy())
        else:
            raw[end] = probe_dat(case.HERE, name)
    t = raw["near"][0]
    dt = np.diff(t)
    if not np.allclose(dt, dt[0], rtol=1e-3):        # the FFTs and the echo fit need a uniform axis
        t = np.arange(t[0], t[-1], np.median(dt))
    return t, np.interp(t, *raw["near"]), np.interp(t, *raw["far"])
