"""Calibration numbers from the first pass of the pulse: peak times, delay, Zc effective, Gamma_L, charge."""
from types import SimpleNamespace

import numpy as np

from c1_common import trapz


def peak(t, i):
    """Maximum with a 3-point parabolic refinement (sub-sample time)."""
    k = int(np.argmax(i))
    y0, y1, y2 = i[k - 1], i[k], i[k + 1]
    den = y0 - 2 * y1 + y2
    off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    return t[k] + off * (t[1] - t[0]), y1 - 0.25 * (y0 - y2) * off


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


def peak_in(t, i, lo, hi):
    m = (t >= lo) & (t <= hi)
    return peak(t[m], i[m])


def calibrate(case, t, I_n, I_f, log):
    c = case
    log.check(all(np.all(np.isfinite(v)) and not np.all(v == 0.0) for v in (I_n, I_f)), "currents finite and non-zero")
    log.check(t[-1] >= c.window_need, "result covers the first echo", f"last sample at {t[-1] * 1e9:.1f} ns")

    w_n, w_f = (0.0, c.t0 + c.T), (c.t0 + 0.5 * c.T, c.t0 + 1.5 * c.T)           # first-pass windows (no echo yet)
    tp_n, Ip_n = peak_in(t, I_n, *w_n)               # incident pulse at the near end
    tp_f, Ip_f = peak_in(t, I_f, *w_f)               # pulse arriving at the far end
    tc_s = pulse_time(t, c.source(t), 0.0, c.t0 + 0.5 * c.T)[0]                     # the source itself, same measure
    tc_n, fw_n = pulse_time(t, I_n, *w_n)
    tc_f, fw_f = pulse_time(t, I_f, *w_f)
    delay = tc_f - tc_n                              # half-maximum centres: the peak time of a flat Gaussian is ill-conditioned
    Zin = float(c.source(tp_n)) / Ip_n               # V_S / I at the near end while only the incident wave exists
    Zc_meas = Zin - c.R_S
    ratio = Ip_f / Ip_n                              # = 1 - Gamma_L
    Gamma_meas = 1.0 - ratio
    Gamma_pred = (c.R_L - c.Zc_reff) / (c.R_L + c.Zc_reff)
    Ip_n_pred = 1.0 / (c.R_S + c.Zc_reff)
    Q_n, Q_f = trapz(I_n, t), trapz(I_f, t)
    Q_ideal = trapz(c.source(t), t) / (c.R_S + c.R_L)  # DC limit of a lossless line: independent of Zc

    log.info("near end", f"peak {Ip_n * 1e3:.4f} mA at {tp_n * 1e9:.3f} ns, "
                         f"pulse centre (half max) {tc_n * 1e9:.3f} ns, FWHM {fw_n * 1e9:.3f} ns")
    log.info("far end", f"peak {Ip_f * 1e3:.4f} mA at {tp_f * 1e9:.3f} ns, "
                        f"pulse centre (half max) {tc_f * 1e9:.3f} ns, FWHM {fw_f * 1e9:.3f} ns")
    log.info("delay", f"{delay * 1e9:.3f} ns (L/c = {c.T * 1e9:.3f} ns): v = {c.T / delay:.4f} c")
    log.info("Zc effective (V_S/I - R_S)",
             f"{Zc_meas:.2f} ohm (r_eff prediction {c.Zc_reff:.1f}, radius r {c.Zc_r:.1f})")
    log.info("far/near peak ratio",
             f"{ratio:.4f} = 1 - Gamma_L; Gamma_L = {Gamma_meas:+.4f} (prediction {Gamma_pred:+.4f})")
    log.info("charge", f"near {Q_n:.4e} C, far {Q_f:.4e} C, ideal {Q_ideal:.4e} C")

    log.check(abs(tc_n - tc_s) < 0.1e-9, "near-end pulse centred on the source pulse (source and .exc calibrated)",
              f"{tc_n * 1e9:.3f} ns vs {tc_s * 1e9:.3f} ns (half-maximum centres)")
    log.check(abs(delay / c.T - 1) < 0.01, "delay equals L/c within 1 %",
              f"{delay * 1e9:.3f} ns vs {c.T * 1e9:.3f} ns")
    log.check(abs(Zc_meas / c.Zc_reff - 1) < 0.02, "Zc effective equals the r_eff prediction within 2 %",
              f"{Zc_meas:.1f} vs {c.Zc_reff:.1f} ohm")
    log.check(abs(Ip_n / Ip_n_pred - 1) < 0.01, "near-end peak current equals 1/(R_S + Zc_eff) within 1 %",
              f"{Ip_n * 1e3:.4f} vs {Ip_n_pred * 1e3:.4f} mA")
    log.check(abs(Gamma_meas - Gamma_pred) < 0.01, "far/near ratio equals 1 - Gamma_L within 0.01",
              f"Gamma_L {Gamma_meas:+.4f} vs {Gamma_pred:+.4f}")
    log.check(abs(Q_n / Q_ideal - 1) < 0.005 and abs(Q_f / Q_ideal - 1) < 0.005,
              "charge conservation: integral of I = integral of V_S / (R_S + R_L) within 0.5 %",
              f"{Q_n / Q_ideal:.4f} / {Q_f / Q_ideal:.4f}")
    log.check(Ip_n > 0 and Ip_f > 0, "probe sign: positive current from near to far end")
    return SimpleNamespace(tp_n=tp_n, Ip_n=Ip_n, tp_f=tp_f, Ip_f=Ip_f, delay=delay, Zc_meas=Zc_meas,
                           Gamma_pred=Gamma_pred)
