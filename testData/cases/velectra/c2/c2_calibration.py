"""Calibration of the aggressor from the first pass of the pulse: peak times, delay, Zc effective, probe sign.

With 50 ohm on a line of about 300 ohm (Gamma_L ~ -0.72) the near-end echo at t0 + 2T is LARGER than the incident
pulse and the far-end current is (1 - Gamma_L) times it, so the peaks are searched inside the first pass only.
"""
from types import SimpleNamespace

import numpy as np

from c2_common import EXPECTED, ZC_A_MEAS


def peak(t, i):
    """Maximum with a 3-point parabolic refinement (sub-sample time)."""
    k = int(np.argmax(i))
    y0, y1, y2 = i[k - 1], i[k], i[k + 1]
    den = y0 - 2 * y1 + y2
    off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    return t[k] + off * (t[1] - t[0]), y1 - 0.25 * (y0 - y2) * off


def peak_in(t, i, lo, hi):
    m = (t >= lo) & (t <= hi)
    return peak(t[m], i[m])


def calibrate(case, t, I, log):
    c = case
    log.check(all(np.all(np.isfinite(v)) and not np.all(v == 0.0) for v in I.values()), "currents finite and non-zero")
    log.check(t[-1] >= EXPECTED["window"] * 0.999, "result covers the 300 ns window", f"last sample {t[-1] * 1e9:.1f} ns")

    tp_n, Ip_n = peak_in(t, I["agg_near"], 0.0, c.t0 + c.T)                         # incident pulse, before any echo
    tp_f, Ip_f = peak_in(t, I["agg_far"], c.t0 + 0.5 * c.T, c.t0 + 1.5 * c.T)       # pulse arriving at the far end
    delay = tp_f - tp_n
    Zc_meas = float(c.source(tp_n)) / Ip_n - c.R[0]                                 # V_S / I - R_S
    log.info("aggressor near end (first pass)", f"peak {Ip_n * 1e3:.4f} mA at {tp_n * 1e9:.3f} ns")
    log.info("aggressor far end (first pass)", f"peak {Ip_f * 1e3:.4f} mA at {tp_f * 1e9:.3f} ns")
    log.info("delay / Zc effective", f"{delay * 1e9:.3f} ns (L/c = {c.T * 1e9:.3f} ns, v = {c.T / delay:.4f} c); "
             f"Zc = V_S/I - R_S = {Zc_meas:.1f} ohm (case A: {ZC_A_MEAS} ohm; here the victim is present)")
    log.check(abs(tp_n - c.t0) < 0.05e-9, "aggressor near-end peak at the source peak time",
              f"{tp_n * 1e9:.3f} ns vs {c.t0 * 1e9:.3f} ns")
    log.check(abs(delay / c.T - 1) < 0.01, "aggressor delay equals L/c within 1 %",
              f"{delay * 1e9:.3f} ns vs {c.T * 1e9:.3f} ns")
    log.check(Ip_n > 0 and Ip_f > 0, "probe sign: positive aggressor current from near to far end")
    return SimpleNamespace(tp_n=tp_n, Ip_n=Ip_n, tp_f=tp_f, Ip_f=Ip_f, delay=delay, Zc_meas=Zc_meas)
