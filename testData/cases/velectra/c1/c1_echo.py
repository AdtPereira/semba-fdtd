"""First echo at the near end: resistive part (monopole) and reactive part (doublet), plus late-time residual.

echo(t) = alpha * i_main(t - 2T) + beta * d/dt i_main(t - 2T),   i_main = incident pulse as seen by the probe
alpha = -Gamma_L (resistive mismatch); beta = Zc*C/2 (shunt C at the far end) or -L/(2 Zc) (series L)
"""
import math
from types import SimpleNamespace

import numpy as np


def echo_analysis(case, cal, t, I_n, I_f, log):
    c = case
    t_echo = c.t0 + cal.delay * 2                    # centre of the echo (round trip measured from the peaks)
    i_main = cal.Ip_n * c.source(t) / c.s_pk
    resid = I_n - i_main
    win = (t > t_echo - 4 * c.sigma) & (t < min(t_echo + 4 * c.sigma, t[-1]))
    alpha = beta = None
    fit = np.zeros_like(t)
    if win.sum() > 20 and t[-1] > t_echo + 2 * c.sigma:
        m = np.interp(t - 2 * cal.delay, t, i_main, left=0.0, right=0.0)
        dm = np.gradient(m, t)
        A = np.column_stack([m[win], dm[win]])
        (alpha, beta), *_ = np.linalg.lstsq(A, resid[win], rcond=None)
        fit = alpha * m + beta * dm
        rms_fit = math.sqrt(np.mean((resid[win] - fit[win]) ** 2)) / cal.Ip_n
        reactive = (f"shunt C = {2 * beta / cal.Zc_meas * 1e12:.2f} pF" if beta > 0
                    else f"series L = {-2 * beta * cal.Zc_meas * 1e9:.2f} nH")
        g100 = 2 * math.pi * 1e8 * abs(beta)         # |Gamma| of the reactive part at 100 MHz
        log.info("echo (residual after the main pulse)",
                 f"max {resid[win].max() * 1e3:+.4f} mA ({100 * resid[win].max() / cal.Ip_n:+.2f} %), "
                 f"min {resid[win].min() * 1e3:+.4f} mA ({100 * resid[win].min() / cal.Ip_n:+.2f} %); "
                 f"peak {20 * np.log10(abs(resid[win]).max() / cal.Ip_n):.1f} dB re main peak")
        log.info("echo fit", f"resistive alpha = {alpha * 100:+.2f} % of the pulse "
                             f"(-Gamma_L = {-cal.Gamma_pred * 100:+.2f} %), reactive beta = {beta * 1e12:+.1f} ps "
                             f"-> {reactive}; |Gamma| at 100 MHz ~ {g100 * 100:.1f} %; "
                             f"fit error {rms_fit * 100:.2f} % of peak")
        log.check(abs(alpha + cal.Gamma_pred) < 0.005, "resistive echo equals -Gamma_L within 0.5 point",
                  f"{alpha * 100:+.2f} % vs {-cal.Gamma_pred * 100:+.2f} %")
        log.check(g100 < 0.05, "reactive end effect below 5 % reflection at 100 MHz", f"{g100 * 100:.1f} %", "WARN")
        log.check(rms_fit < 0.003, "echo explained by monopole + doublet (fit error below 0.3 % of peak)",
                  f"{rms_fit * 100:.2f} %", "WARN")
    else:
        log.info("echo", "window too short for the echo analysis")

    late = t > t_echo + 4 * c.sigma
    if late.any():
        late_n = 20 * np.log10(abs(I_n[late]).max() / cal.Ip_n + 1e-30)
        late_f = 20 * np.log10(abs(I_f[late]).max() / cal.Ip_f + 1e-30)
        log.info("late-time residual (after the echo)", f"near {late_n:.1f} dB, far {late_f:.1f} dB re peak")
        log.check(late_n < -40 and late_f < -40, "late-time residual below -40 dB (no ringing, no instability)",
                  f"{late_n:.1f} / {late_f:.1f} dB")
    return SimpleNamespace(t_echo=t_echo, resid=resid, win=win, fit=fit, alpha=alpha, beta=beta)
