"""First echo at the near end and late-time residual of the semba-fdtd currents.

The analytic line of case A is matched (no echo), so the echo of semba-fdtd is the signature of the mismatch between the
terminals and the wire as semba-fdtd models it. It is measured, not fitted: residual = I_near - (main pulse), main pulse =
the source scaled to the first-pass peak.
"""
from types import SimpleNamespace

import numpy as np


def echo_analysis(case, cal, t, I_n, I_f, log):
    c = case
    t_echo = c.t0 + cal.delay * 2                    # centre of the echo (round trip measured from the pulse centres)
    i_main = cal.Ip_n * c.source(t) / c.s_pk
    resid = I_n - i_main
    gam_l = (c.R_L - c.Zc_r) / (c.R_L + c.Zc_r)       # analytic line: the echo at the near end is -Gamma_L of the main pulse
    an_echo = "none (matched)" if abs(gam_l) < 0.005 else f"{-100 * gam_l:+.2f} % of the main peak (Gamma_L = {gam_l:+.3f})"
    # Shape of the near-end current before the echo can return: resid ~ C_eq dV_S/dt (a small shunt capacitance at the line input).
    dV = np.gradient(c.source(t), t)                 # source voltage, V (peak 1 V), so the fitted coefficient is a capacitance (F)
    fit_win = (t > c.t0 - 5 * c.sigma) & (t < t_echo - 3 * c.sigma)
    if fit_win.sum() > 20:
        c_eq = float(np.sum(resid[fit_win] * dV[fit_win]) / np.sum(dV[fit_win] ** 2))
        rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
        before, after = rms(resid[fit_win]), rms(resid[fit_win] - c_eq * dV[fit_win])
        log.info("near-end shape before the echo (residual ~ C dV_S/dt)",
                 f"C = {c_eq * 1e15:.1f} fF; RMS of the residual {100 * before / cal.Ip_n:.3f} % -> {100 * after / cal.Ip_n:.3f} % of the peak "
                 f"after removing it, window {1e9 * t[fit_win][0]:.1f} to {1e9 * t[fit_win][-1]:.1f} ns (a fit, not a model)")
    else:
        c_eq = 0.0
    win = (t > t_echo - 4 * c.sigma) & (t < min(t_echo + 4 * c.sigma, t[-1]))
    if win.sum() > 20 and t[-1] > t_echo + 2 * c.sigma:
        mx, mn = resid[win].max(), resid[win].min()
        log.info("near-end echo (residual after the main pulse)",
                 f"max {mx * 1e3:+.4f} mA ({100 * mx / cal.Ip_n:+.2f} %), min {mn * 1e3:+.4f} mA "
                 f"({100 * mn / cal.Ip_n:+.2f} %); peak {20 * np.log10(abs(resid[win]).max() / cal.Ip_n):.1f} dB re main peak; "
                 f"analytic line: {an_echo}")
        log.check(abs(resid[win]).max() / cal.Ip_n < 0.10, "near-end echo below 10 % of the main peak",
                  f"{100 * abs(resid[win]).max() / cal.Ip_n:.2f} %", "WARN")
    else:
        log.info("echo", "window too short for the echo analysis")

    late = t > t_echo + 4 * c.sigma
    if late.any():
        late_n = 20 * np.log10(abs(I_n[late]).max() / cal.Ip_n + 1e-30)
        late_f = 20 * np.log10(abs(I_f[late]).max() / cal.Ip_f + 1e-30)
        log.info("late-time residual (after the echo)", f"near {late_n:.1f} dB, far {late_f:.1f} dB re peak")
        log.check(late_n < -40 and late_f < -40, "late-time residual below -40 dB (no ringing, no instability)",
                  f"{late_n:.1f} / {late_f:.1f} dB")
    return SimpleNamespace(t_echo=t_echo, resid=resid, win=win, fit=c_eq * dV, fit_win=fit_win, c_eq=c_eq)
