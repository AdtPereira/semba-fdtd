"""Optional checks on what semba-fdtd wrote: Energy.dat (decay) and Report.txt (dt correction, speed, run time)."""
import re

import numpy as np

from c3_common import find_file


def energy_check(root, log):
    f_en = find_file(root, "*Energy.dat")
    if f_en is None:
        return
    en = np.atleast_2d(np.loadtxt(f_en))
    if len(en) > 3:
        dec = 10 * np.log10(en[:, 1].max() / en[-1, 1])
        log.info("energy", f"decays {dec:.1f} dB from its maximum to the last report (10 log10: energy is quadratic; "
                           f"case B: 117 dB)")
        log.check(dec > 60, "energy decays by more than 60 dB (absorbed by the terminals)", f"{dec:.1f} dB", "WARN")


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
        n_run = int(m_steps[2]) if m_steps else 300e-9 / (float(m_dt[2]) if m_dt else dt_solver)
        log.info("speed", f"{r_:.1f} Mcells/s; this run ({n_run:.0f} steps, {float(mc[1]):.2f} Mcells) took "
                          f"~ {float(mc[1]) * n_run / r_ / 3600:.2f} h (case B: 33 min at 120 Mcells/s)")
        log.check(r_ > 50, "solver speed above 50 Mcells/s (plug the laptop in, high-performance power plan)",
                  f"{r_:.1f} Mcells/s", "WARN")
