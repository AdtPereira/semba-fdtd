"""Optional checks on what semba-fdtd wrote: Energy.dat (decay), Report.txt (dt correction, speed, run time) and Warnings.txt."""
import re

import numpy as np

from paul_d28_common import find_file


def energy_check(root, log):
    f_en = find_file(root, "*Energy.dat")
    if f_en is None:
        return
    en = np.atleast_2d(np.loadtxt(f_en))
    if len(en) < 2:
        return
    dec = 10 * np.log10(en[:, 1].max() / en[-1, 1])
    log.info("energy", f"{len(en)} reports (one per ~11500 steps), {dec:.1f} dB lower at the last one (10 log10: energy is "
                       f"quadratic); the 200 ns record ends with the line still ringing, so no 60 dB criterion here "
                       f"(cases A-C, 300 ns: 95 to 117 dB)")
    log.check(en[-1, 1] < en[0, 1] and np.all(np.isfinite(en)),
              "energy decays after the source has ended (absorbed by the terminals and the PML), no growth",
              f"{en[0, 1]:.4g} -> {en[-1, 1]:.4g}")


def report_check(root, dt_solver, log):
    f_rep = find_file(root, "*Report.txt")
    if f_rep is None:
        return
    rep = f_rep.read_text(errors="ignore")
    m_dt = re.search(r"Original dt:\s*([0-9.Ee+-]+)\s*New dt:\s*([0-9.Ee+-]+)", rep)
    m_steps = re.search(r"Original Final Time Step=\s*(\d+)\s*Corrected Final Time Step=\s*(\d+)", rep)
    if m_dt:
        log.info("solver corrected the time step", f"{float(m_dt[1]):.3e} -> {float(m_dt[2]):.3e} s"
                 + (f", steps {m_steps[1]} -> {m_steps[2]}" if m_steps else ""))
    rate = re.findall(r"Mcells/sec\s*:\s*([0-9.]+)\s*\(\s*0\s+to\s+(\d+)", rep)
    mc = re.search(r"Total Mcells:\s*([0-9.]+)", rep)
    if rate and mc:
        r_ = float(rate[-1][0])
        n_run = int(m_steps[2]) if m_steps else 200e-9 / (float(m_dt[2]) if m_dt else dt_solver)
        log.info("speed", f"{r_:.1f} Mcells/s; {float(mc[1]):.3f} Mcells (PML included), {n_run:.0f} steps -> "
                          f"~ {float(mc[1]) * n_run / r_ / 60:.1f} min")
        log.check(r_ > 20, "solver speed above 20 Mcells/s (small grids run slower; plug the laptop in)", f"{r_:.1f} Mcells/s", "WARN")
    m_t = re.search(r"BEGUN .*? at (\S+) , (\S+).*?ENDED .*? at (\S+) , (\S+)", rep, re.S)
    if m_t:
        log.info("run date", f"{m_t[1]} {m_t[2]} -> {m_t[4]}")
    m_g = re.search(r"git commit:\s*(\w+)", rep)
    if m_g:
        log.info("solver", f"git {m_g[1]}, " + ("single precision" if "Single precision" in rep else "double precision"))
    f_w = find_file(root, "*Warnings.txt")
    if f_w is not None:
        w = f_w.read_text(errors="ignore")
        bad = [ln for ln in w.splitlines() if ln.strip() and "create_symlink" not in ln and "Closing warning file" not in ln
               and "____" not in ln]
        log.check(not bad, "no solver warning other than the benign create_symlink (same source and target file)",
                  "; ".join(bad[:3]) if bad else "only create_symlink" if "create_symlink" in w else "none", "WARN")
