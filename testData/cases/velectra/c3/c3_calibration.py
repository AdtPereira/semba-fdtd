"""Calibration of the aggressor from the first pass of the pulse: peaks, delay, impedance seen by the source, charge.

Same measures as c1 and c2 (half-maximum pulse centres for the delay, effective L' and C'), now inside the closed cylinder.
With 50 ohm on a line of about 300 ohm (Gamma_L ~ -0.72) the near-end echo at t0 + 2T is LARGER than the incident
pulse and the far-end current is (1 - Gamma_L) times it, so every measure is taken inside the first pass only.
"""
import math
from types import SimpleNamespace

import numpy as np

from c3_common import C0, ETA0, trapz


def peak(t, i):
    """Maximum with a 3-point parabolic refinement (sub-sample time)."""
    k = int(np.argmax(i))
    if k == 0 or k == len(i) - 1:                    # maximum on the edge of the window: no parabola (i[k - 1] would wrap)
        return t[k], i[k]
    y0, y1, y2 = i[k - 1], i[k], i[k + 1]
    den = y0 - 2 * y1 + y2
    off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    return t[k] + off * (t[1] - t[0]), y1 - 0.25 * (y0 - y2) * off


def peak_in(t, i, lo, hi):
    m = (t >= lo) & (t <= hi)
    return peak(t[m], i[m])


def pulse_time(t, x, lo, hi):
    """(centre, FWHM) of the pulse in [lo, hi]: midpoint of the two half-maximum crossings, linear interpolation.
    Stable where the peak time is not: on a Gaussian of sigma = 2.5 ns a 0.4 % amplitude change moves the PEAK by
    0.2 ns, but the half-maximum midpoint by a few ps."""
    m = (t >= lo) & (t <= hi)
    tt, xx = t[m], x[m] / x[m].max()
    k = int(np.argmax(xx))
    a = np.where(xx[:k] < 0.5)[0][-1]
    b = k + np.where(xx[k:] < 0.5)[0][0]
    ta = tt[a] + (0.5 - xx[a]) / (xx[a + 1] - xx[a]) * (tt[a + 1] - tt[a])
    tb = tt[b - 1] + (0.5 - xx[b - 1]) / (xx[b] - xx[b - 1]) * (tt[b] - tt[b - 1])
    return 0.5 * (ta + tb), tb - ta


def calibrate(case, t, I, log):
    c = case
    log.check(all(np.all(np.isfinite(v)) and not np.all(v == 0.0) for v in I.values()), "currents finite and non-zero")
    log.check(t[-1] >= c.window_req * 0.999, "result covers the simulated window the case needs",
              f"last sample {t[-1] * 1e9:.1f} ns, needed {c.window_req * 1e9:.1f} ns")

    w_n, w_f = (0.0, c.t0 + c.T), (c.t0 + 0.5 * c.T, c.t0 + 1.5 * c.T)           # first-pass windows (no echo yet)
    tp_n, Ip_n = peak_in(t, I["agg_near"], *w_n)                                   # incident pulse
    tp_f, Ip_f = peak_in(t, I["agg_far"], *w_f)                                    # pulse arriving at the far end
    tc_s = pulse_time(t, c.source(t), 0.0, c.t0 + 0.5 * c.T)[0]                    # the source itself, same measure
    tc_n, fw_n = pulse_time(t, I["agg_near"], *w_n)
    tc_f, fw_f = pulse_time(t, I["agg_far"], *w_f)
    delay = tc_f - tc_n                              # half-maximum centres: the peak time of a flat Gaussian is ill-conditioned
    Zc_meas = float(c.source(tp_n)) / Ip_n - c.R[0]                                # V_S / I - R_S while only the incident wave exists
    ratio = Ip_f / Ip_n                              # = 1 - Gamma_L
    Q_n, Q_f = trapz(I["agg_near"], t), trapz(I["agg_far"], t)
    Q_ideal = trapz(c.source(t), t) / (c.R[0] + c.R[1])  # DC limit of a lossless line: independent of Zc

    log.info("aggressor near end (first pass)", f"peak {Ip_n * 1e3:.4f} mA at {tp_n * 1e9:.3f} ns, "
                                                f"pulse centre (half max) {tc_n * 1e9:.3f} ns, FWHM {fw_n * 1e9:.3f} ns")
    log.info("aggressor far end (first pass)", f"peak {Ip_f * 1e3:.4f} mA at {tp_f * 1e9:.3f} ns, "
                                               f"pulse centre (half max) {tc_f * 1e9:.3f} ns, FWHM {fw_f * 1e9:.3f} ns")
    log.info("delay", f"{delay * 1e9:.3f} ns (L/c = {c.T * 1e9:.3f} ns): v = {c.T / delay:.4f} c")
    log.info("impedance seen by the source (V_S/I - R_S)",
             f"{Zc_meas:.1f} ohm in semba-fdtd (victim present); analytic, physical radius, one wire in the cylinder: {c.Zc_r:.1f} ohm")
    # Effective per-unit-length parameters of the aggressor as semba-fdtd models it (lossless line: Zc = sqrt(L'/C'), v = 1/sqrt(L'C'))
    v_eff = c.length / delay
    L_eff, C_eff = Zc_meas / v_eff, 1.0 / (Zc_meas * v_eff)
    L_an = c.L[c.primary][0, 0]                      # one wire inside the cylinder (self term of the pair), physical radius
    C_an = 1.0 / (L_an * C0 ** 2)
    log.info("effective L' (Zc / v)", f"{L_eff * 1e6:.4f} uH/m; analytic, one wire in the cylinder: {L_an * 1e6:.4f} uH/m "
                                      f"({100 * (L_eff / L_an - 1):+.2f} %)")
    log.info("effective C' (1 / (Zc v))", f"{C_eff * 1e12:.4f} pF/m; analytic, one wire in the cylinder: {C_an * 1e12:.4f} pF/m "
                                          f"({100 * (C_eff / C_an - 1):+.2f} %)")
    log.info("L'C' c^2 (1 = vacuum line)", f"{L_eff * C_eff * C0 ** 2:.4f} (case A, h = 35 mm, one wire over a plane: see c1_readme)")
    log.info("far/near peak ratio", f"{ratio:.4f} = 1 - Gamma_L; Gamma_L = {1 - ratio:+.4f} "
                                    f"(analytic, physical radius: {(c.R[1] - c.Zc_r) / (c.R[1] + c.Zc_r):+.4f})")
    log.info("charge", f"near {Q_n:.4e} C, far {Q_f:.4e} C, ideal {Q_ideal:.4e} C")

    log.check(abs(tc_n - tc_s) < 0.1e-9, "aggressor near-end pulse centred on the source pulse (source and .exc calibrated)",
              f"{tc_n * 1e9:.3f} ns vs {tc_s * 1e9:.3f} ns (half-maximum centres)")
    log.check(abs(delay / c.T - 1) < 0.05, "aggressor delay equals L/c within 5 % (no velocity error)",
              f"{delay * 1e9:.3f} ns vs {c.T * 1e9:.3f} ns")
    log.check(abs(Q_n / Q_ideal - 1) < 0.01 and abs(Q_f / Q_ideal - 1) < 0.01,
              "charge conservation: integral of I = integral of V_S / (R_S + R_L) within 1 %",
              f"{Q_n / Q_ideal:.4f} / {Q_f / Q_ideal:.4f}", "WARN")
    log.check(Ip_n > 0 and Ip_f > 0, "probe sign: positive aggressor current from near to far end")
    return SimpleNamespace(tp_n=tp_n, Ip_n=Ip_n, tp_f=tp_f, Ip_f=Ip_f, tc_n=tc_n, tc_f=tc_f, delay=delay,
                           Zc_meas=Zc_meas, ratio=ratio, L_eff=L_eff, C_eff=C_eff)
