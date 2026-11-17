"""Calibration of the aggressor from the first pass of the pulse: peak times, delay, Zc effective, probe sign.

With 50 ohm on a line of about 300 ohm (Gamma_L ~ -0.72) the near-end echo at t0 + 2T is LARGER than the incident
pulse and the far-end current is (1 - Gamma_L) times it, so the peaks are searched inside the first pass only.
Case C: the expected Zc effective is about 308 ohm (closed cylinder, r_eff from case A); case B measured 311.8 ohm.
"""
from types import SimpleNamespace

import numpy as np

from c3_common import EXPECTED, ZC_A_MEAS


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


def calibrate(case, t, I, log):
    c = case
    log.check(all(np.all(np.isfinite(v)) and not np.all(v == 0.0) for v in I.values()), "currents finite and non-zero")
    log.check(t[-1] >= EXPECTED["window"] * 0.999, "result covers the 300 ns window", f"last sample {t[-1] * 1e9:.1f} ns")

    w_n, w_f = (0.0, c.t0 + c.T), (c.t0 + 0.5 * c.T, c.t0 + 1.5 * c.T)           # first-pass windows
    tp_n, Ip_n = peak_in(t, I["agg_near"], *w_n)                                    # incident pulse, before any echo
    tp_f, Ip_f = peak_in(t, I["agg_far"], *w_f)                                     # pulse arriving at the far end
    tc_s = pulse_time(t, c.source(t), 0.0, c.t0 + 0.5 * c.T)[0]                     # the source itself, same measure
    tc_n, fw_n = pulse_time(t, I["agg_near"], *w_n)
    tc_f, fw_f = pulse_time(t, I["agg_far"], *w_f)
    delay = tc_f - tc_n                                                             # half-maximum delay
    Zc_meas = float(c.source(tp_n)) / Ip_n - c.R[0]                                 # V_S / I - R_S
    log.info("aggressor near end (first pass)", f"peak {Ip_n * 1e3:.4f} mA at {tp_n * 1e9:.3f} ns; "
             f"pulse centre (half max) {tc_n * 1e9:.3f} ns, FWHM {fw_n * 1e9:.3f} ns")
    log.info("aggressor far end (first pass)", f"peak {Ip_f * 1e3:.4f} mA at {tp_f * 1e9:.3f} ns; "
             f"pulse centre (half max) {tc_f * 1e9:.3f} ns, FWHM {fw_f * 1e9:.3f} ns")
    log.info("delay / Zc effective", f"{delay * 1e9:.3f} ns (L/c = {c.T * 1e9:.3f} ns, v = {c.T / delay:.4f} c; "
             f"near end {1e12 * (tc_n - tc_s):+.0f} ps from the source, far end {1e12 * (tc_f - tc_s - c.T):+.0f} ps "
             f"from source + L/c); Zc = V_S/I - R_S = {Zc_meas:.1f} ohm (case A: {ZC_A_MEAS} ohm, case B: 311.8 ohm; "
             f"cylinder, single wire, r_eff: {c.zc_single:.1f} ohm)")
    log.check(abs(tc_n - tc_s) < 0.1e-9, "aggressor near-end pulse centre within 0.1 ns of the source (half maximum)",
              f"{tc_n * 1e9:.3f} ns vs {tc_s * 1e9:.3f} ns ({1e12 * (tc_n - tc_s):+.0f} ps)")
    log.check(abs(delay / c.T - 1) < 0.01, "aggressor delay equals L/c within 1 % (half-maximum centres)",
              f"{delay * 1e9:.3f} ns vs {c.T * 1e9:.3f} ns ({100 * (delay / c.T - 1):+.2f} %)")
    log.check(Ip_n > 0 and Ip_f > 0, "probe sign: positive aggressor current from near to far end")
    log.check(abs(Zc_meas / c.zc_single - 1) < 0.02, "Zc effective within 2 % of the cylinder value (about 308 ohm, "
              "not the 311.8 ohm of the plane)", f"{Zc_meas:.1f} ohm vs {c.zc_single:.1f} ohm "
              f"({100 * (Zc_meas / c.zc_single - 1):+.2f} %)", "WARN")
    return SimpleNamespace(tp_n=tp_n, Ip_n=Ip_n, tp_f=tp_f, Ip_f=Ip_f, delay=delay, Zc_meas=Zc_meas)
