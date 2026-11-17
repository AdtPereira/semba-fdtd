"""Analytic reference: bounce diagram of the lossless line (single wire over a plane).

i_f(t) = V_S(t)/(R_S+Zc) + G_S G_L i_f(t-2T);   I_near = i_f - G_L i_f(t-2T);   I_far = (1-G_L) i_f(t-T)
"""
import math

import numpy as np


def line_response(source, Zc, R_s, R_l, tt, Td):
    sh = lambda x, dly: np.interp(tt - dly, tt, x, left=0.0, right=0.0)
    gs, gl = (R_s - Zc) / (R_s + Zc), (R_l - Zc) / (R_l + Zc)
    a = source(tt) / (R_s + Zc)
    i_f = a.copy()
    for _ in range(int(tt[-1] / (2 * Td)) + 2):
        i_f = a + gs * gl * sh(i_f, 2 * Td)
    return i_f - gl * sh(i_f, 2 * Td), (1 - gl) * sh(i_f, Td)


def compare_with_line(case, cal, t, I_n, I_f, log):
    """FDTD against the analytic line for three Zc references; returns {name: (I_near, I_far)}."""
    c = case
    refs = {"radius r": c.Zc_r, "r_eff": c.Zc_reff, "Zc measured": cal.Zc_meas}
    curves = {k: line_response(c.source, z, c.R_S, c.R_L, t, c.T) for k, z in refs.items()}
    print("\nFDTD vs analytic line (T = L/c)            RMS near   RMS far   [% of the FDTD peak]   max|diff| near/far")
    for k, (a, b) in curves.items():
        rn, rf = math.sqrt(np.mean((I_n - a) ** 2)) / cal.Ip_n, math.sqrt(np.mean((I_f - b) ** 2)) / cal.Ip_f
        mx = (abs(I_n - a).max() / cal.Ip_n, abs(I_f - b).max() / cal.Ip_f)
        print(f"  Zc = {refs[k]:6.1f} ohm ({k:12s})        {100 * rn:6.2f}   {100 * rf:6.2f}"
              f"                      {100 * mx[0]:5.2f} / {100 * mx[1]:5.2f} %")
    a, b = curves["r_eff"]
    log.check(math.sqrt(np.mean((I_n - a) ** 2)) / cal.Ip_n < 0.01
              and math.sqrt(np.mean((I_f - b) ** 2)) / cal.Ip_f < 0.01,
              "FDTD currents match the analytic line (r_eff) within 1 % RMS of the peak")
    return curves
