"""Optional checks on what semba-fdtd wrote: Energy.dat (decay) and Report.txt (dt correction, speed, case C estimate)."""
import re

import numpy as np

from c2_common import find_file


def energy_check(root, log, min_db=50.0):
    f_en = find_file(root, "*Energy.dat")
    if f_en is None:
        return
    en = np.atleast_2d(np.loadtxt(f_en))
    # Energy.dat has one row per solver report (about every 60 s of wall time): a 3.5 min run leaves only 3 rows, the
    # last one well before the end of the window, so min_db must be lowered for short runs (c2.py passes 30 for the matched case).
    if len(en) >= 2:
        dec = 10 * np.log10(en[:, 1].max() / en[-1, 1])
        log.info("energy", f"{len(en)} reports, last at {en[-1, 0] * 1e9:.1f} ns; decays {dec:.1f} dB from its "
                           f"maximum to the last report (10 log10: energy is quadratic)")
        log.check(dec > min_db, f"energy decays by more than {min_db:g} dB (absorbed by the terminals)",
                  f"{dec:.1f} dB", "WARN")
    else:
        log.info("energy", "fewer than 2 reports in Energy.dat: decay not checked")


def border_check(root, bnd_xy, log):
    """The Report must list the absorbing borders that the JSON asks for (a misspelt type is ignored without an error)."""
    f_rep = find_file(root, "*Report.txt")
    if f_rep is None or not bnd_xy:
        return
    rep = f_rep.read_text(errors="ignore")
    kinds = {v.lower() for v in bnd_xy.values()}
    want_pml, want_mur = "pml" in kinds, "mur" in kinds
    has_pml = re.search(r"there are CPML Borders", rep) is not None
    has_mur = re.search(r"there are Mur Borders", rep) is not None
    log.check((not want_pml or has_pml) and (not want_mur or has_mur),
              "the Report lists the absorbing borders requested in the JSON",
              f"JSON asks {sorted(set(bnd_xy.values()))}; Report: CPML {'yes' if has_pml else 'NO'}, "
              f"Mur {'yes' if has_mur else 'NO'}")


def warnings_check(root, log):
    """Solver warnings written next to the case; the MTLN symlink warning for the .exc file is known and harmless."""
    f_w = find_file(root, "*Warnings.txt")
    if f_w is None:
        return
    lines = [ln.strip() for ln in f_w.read_text(errors="ignore").splitlines() if "WARN" in ln or "ERROR" in ln]
    benign = [ln for ln in lines if "create_symlink" in ln]
    other = [ln for ln in lines if ln not in benign]
    if benign:
        log.info("solver warnings", f"{len(benign)} known harmless (create_symlink of the .exc onto itself)")
    log.check(not other, "no solver warnings other than the known one", "; ".join(other[:3]), "WARN")


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
        m_n = re.search(r"Simulation from n=\s*\d+,\s*t=\s*\S+\s*to n=\s*(\d+)", rep)      # the steps this run really has
        n_run = int(m_n[1]) if m_n else int(m_steps[2]) if m_steps else 300e-9 / (float(m_dt[2]) if m_dt else dt_solver)
        log.info("speed", f"{r_:.1f} Mcells/s; this run ({n_run:.0f} steps) ~ {float(mc[1]) * n_run / r_ / 3600:.2f} h, "
                          f"case C (2x cells, same window) ~ {2 * float(mc[1]) * n_run / r_ / 3600:.2f} h")
        log.check(r_ > 50, "solver speed above 50 Mcells/s (plug the laptop in, high-performance power plan)",
                  f"{r_:.1f} Mcells/s", "WARN")
