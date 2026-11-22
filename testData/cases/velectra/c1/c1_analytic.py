"""Analytic reference for a single wire over a plane, with the PHYSICAL radius of the wire.

Lossless line of impedance Zc = eta0/(2 pi) acosh(h/r) and delay T = L/c, resistive terminals R_S and R_L (bounce diagram):
   i_f(t) = V_S(t)/(R_S+Zc) + G_S G_L i_f(t-2T);   I_near = i_f - G_L i_f(t-2T);   I_far = (1-G_L) i_f(t-T)
With R_S = R_L = Zc of the physical radius the line is matched: no echo, I_near = V_S/(2 Zc), I_far = I_near(t - T).
This is the first of the two curves of case A; the second one is semba-fdtd, with the terminals as they are in the JSON.
"""
import math

import numpy as np

from c1_calibration import peak_in, pulse_time


def line_response(source, Zc, R_s, R_l, tt, Td):
    sh = lambda x, dly: np.interp(tt - dly, tt, x, left=0.0, right=0.0)
    gs, gl = (R_s - Zc) / (R_s + Zc), (R_l - Zc) / (R_l + Zc)
    a = source(tt) / (R_s + Zc)
    i_f = a.copy()
    for _ in range(int(tt[-1] / (2 * Td)) + 2):
        i_f = a + gs * gl * sh(i_f, 2 * Td)
    return i_f - gl * sh(i_f, 2 * Td), (1 - gl) * sh(i_f, Td)


def compare_with_line(case, cal, t, I_n, I_f, log):
    """semba-fdtd against the analytic line (physical radius); returns {"analytic": (I_near, I_far)}."""
    c = case
    a, b = line_response(c.source, c.Zc_r, c.R_S, c.R_L, t, c.T)
    w_n, w_f = (0.0, c.t0 + c.T), (c.t0 + 0.5 * c.T, c.t0 + 1.5 * c.T)           # first-pass windows (no echo yet)
    Ia_n, Ia_f = peak_in(t, a, *w_n)[1], peak_in(t, b, *w_f)[1]
    d_an = pulse_time(t, b, *w_f)[0] - pulse_time(t, a, *w_n)[0]
    rn = math.sqrt(np.mean((I_n - a) ** 2)) / cal.Ip_n
    rf = math.sqrt(np.mean((I_f - b) ** 2)) / cal.Ip_f
    mx = (abs(I_n - a).max() / cal.Ip_n, abs(I_f - b).max() / cal.Ip_f)
    print("\nsemba-fdtd vs analytic line (physical radius, Zc = %.1f ohm, T = L/c)" % c.Zc_r)
    print(f"  RMS difference    near {100 * rn:5.2f} %   far {100 * rf:5.2f} %   [% of the semba-fdtd peak]")
    print(f"  max |difference|  near {100 * mx[0]:5.2f} %   far {100 * mx[1]:5.2f} %")
    log.info("peak current, semba-fdtd vs analytic",
             f"near {cal.Ip_n * 1e3:.4f} vs {Ia_n * 1e3:.4f} mA ({100 * (cal.Ip_n / Ia_n - 1):+.2f} %), "
             f"far {cal.Ip_f * 1e3:.4f} vs {Ia_f * 1e3:.4f} mA ({100 * (cal.Ip_f / Ia_f - 1):+.2f} %)")
    log.info("delay far - near, semba-fdtd vs analytic",
             f"{cal.delay * 1e9:.3f} vs {d_an * 1e9:.3f} ns ({(cal.delay - d_an) * 1e12:+.0f} ps)")
    log.info("RMS difference to the analytic line", f"{100 * rn:.2f} % near, {100 * rf:.2f} % far of the peak")
    # sanity band, not a fit: it only catches a wrong geometry, a wrong source or a broken run
    log.check(max(rn, rf) < 0.05, "semba-fdtd within 5 % RMS of the analytic line", f"{100 * rn:.2f} / {100 * rf:.2f} %",
              "WARN")
    log.check(max(rn, rf) < 0.15, "semba-fdtd within 15 % RMS of the analytic line (gross-error band)",
              f"{100 * rn:.2f} / {100 * rf:.2f} %")
    return {"analytic": (a, b)}
