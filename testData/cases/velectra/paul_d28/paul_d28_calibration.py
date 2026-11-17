"""Calibration of the aggressor from the first pass of the trapezoid: edge times, delay, input impedance, probe sign.

The source is a 12.5 / 7.5 / 12.5 ns trapezoid, so the first pass is read from its two 50 % edge crossings (rising at 6.25 ns,
falling at 26.25 ns) and from the plateau (12.5 to 20 ns). With 50 ohm on a line of about 285 ohm (Gamma_L about -0.70) the
near-end echo returns at 2T = 31.2 ns and the far-end current is the first pass until 3T = 46.8 ns, so every window below
stays inside the first pass.

Input impedance seen by the source on the first pass: the victim is loaded by R_NE, so Zin = Zc11 - Zc12^2 / (Zc22 + R_NE)
(285.1 ohm for r_eff = 0.756 r; Zc11 alone, 291.9 ohm, would be 2.4 % off).
"""
import math
from types import SimpleNamespace

import numpy as np

from paul_d28_common import C0, EXPECTED


def peak(t, i):
    """Maximum with a 3-point parabolic refinement (sub-sample time)."""
    k = int(np.argmax(i))
    y0, y1, y2 = i[k - 1], i[k], i[k + 1]
    den = y0 - 2 * y1 + y2
    off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    return t[k] + off * (t[1] - t[0]), y1 - 0.25 * (y0 - y2) * off


def plateau(t, x, lo, hi):
    """Mean of x on [lo, hi]."""
    m = (t >= lo) & (t <= hi)
    return float(x[m].mean())


def crossing(t, x, level, lo, hi, rising):
    """Time at which x crosses 'level' in [lo, hi] (first crossing, linear interpolation)."""
    m = (t >= lo) & (t <= hi)
    tt, xx = t[m], x[m]
    s = xx - level
    idx = np.where((s[:-1] < 0) & (s[1:] >= 0))[0] if rising else np.where((s[:-1] > 0) & (s[1:] <= 0))[0]
    assert len(idx), f"no {'rising' if rising else 'falling'} crossing of {level:g} in [{lo * 1e9:.1f}, {hi * 1e9:.1f}] ns"
    k = idx[0]
    return tt[k] + (level - xx[k]) / (xx[k + 1] - xx[k]) * (tt[k + 1] - tt[k])


def edge_times(t, x, shift, tr, tf_, flat):
    """(t_rise, t_fall, plateau) of a trapezoid-shaped signal x whose first edge starts about 'shift' after the source's:
    the 50 % crossings of each edge, with 50 % of the plateau level. tr, tf_, flat are the source rise, fall, flat top."""
    p = plateau(t, x, shift + tr + 1.0e-9, shift + tr + flat - 1.0e-9)
    tr_ = crossing(t, x, 0.5 * p, shift, shift + tr + 1.5e-9, True)
    tf = crossing(t, x, 0.5 * p, shift + tr + flat - 1.5e-9, shift + tr + flat + tf_ + 1.5e-9, False)
    return tr_, tf, p


def zin_pred(L, R_ne):
    """Input impedance of the aggressor on the first pass (victim terminated by R_ne at the near end)."""
    Zc = C0 * np.asarray(L)
    return float(Zc[0, 0] - Zc[0, 1] ** 2 / (Zc[1, 1] + R_ne))


def calibrate(case, t, I, log):
    c = case
    tr, flat, tf_ = c.rise, c.flat, c.fall
    log.check(all(np.all(np.isfinite(v)) and not np.all(v == 0.0) for v in I.values()), "currents finite and non-zero")
    log.check(t[-1] >= c.window * 0.999, f"result covers the {c.window * 1e9:.0f} ns window", f"last sample {t[-1] * 1e9:.1f} ns")

    src = c.source(t)
    ts_r, ts_f, ps = edge_times(t, src, 0.0, tr, tf_, flat)                          # the source itself, same measure
    tn_r, tn_f, pn = edge_times(t, I["agg_near"], 0.0, tr, tf_, flat)                  # near end, first pass
    tf_r, tf_f, pf = edge_times(t, I["agg_far"], c.T, tr, tf_, flat)                   # far end, first pass
    delay_r, delay_f = tf_r - tn_r, tf_f - tn_f
    delay = 0.5 * (delay_r + delay_f)
    off_n = 0.5 * ((tn_r - ts_r) + (tn_f - ts_f))                                      # near end against the source
    off_f = 0.5 * ((tf_r - ts_r) + (tf_f - ts_f)) - c.T                                # far end against source + L/c
    Zin = ps / pn - c.R[0]                                                             # V_S / I - R_S on the plateau
    # the plateau must really be flat: V_S / I - R_S on it is a pure line impedance. (On the edges the riser inductance,
    # about 15 nH, makes V_S / I move by a few ohm: that is real and is in the primary reference.)
    zr = src / I["agg_near"] - c.R[0]
    m1, m2 = (t >= 13.5e-9) & (t < 16.25e-9), (t >= 16.25e-9) & (t <= 19e-9)
    zspread = abs(float(zr[m1].mean() - zr[m2].mean()))                       # drift between the two halves of the plateau
    zp = {k: zin_pred(ref.L, c.R[2]) for k, ref in c.refs.items()}
    # effective radius from this run: the radius that gives the measured Zin (coupling included), for the record
    from paul_d28_analytic import lmat
    lo_, hi_ = 0.05 * c.radius, 1.5 * c.radius
    for _ in range(80):
        mid = 0.5 * (lo_ + hi_)
        if zin_pred(lmat(c.pts, mid), c.R[2]) > Zin:
            lo_ = mid
        else:
            hi_ = mid
    r_meas = 0.5 * (lo_ + hi_)
    log.info("aggressor near end (first pass)", f"plateau {pn * 1e3:.4f} mA; 50 % edges at {tn_r * 1e9:.3f} / {tn_f * 1e9:.3f} ns "
             f"(source {ts_r * 1e9:.3f} / {ts_f * 1e9:.3f} ns)")
    log.info("aggressor far end (first pass)", f"plateau {pf * 1e3:.4f} mA; 50 % edges at {tf_r * 1e9:.3f} / {tf_f * 1e9:.3f} ns")
    log.info("delay / Zin", f"{delay * 1e9:.3f} ns (rising {delay_r * 1e9:.3f}, falling {delay_f * 1e9:.3f}; L/c = "
             f"{c.T * 1e9:.3f} ns, v = {c.T / delay:.4f} c); near end {off_n * 1e12:+.0f} ps from the source (the 20 mm riser "
             f"is in the path: 67 ps at c), far end {off_f * 1e12:+.0f} ps from source + L/c; Zin = V_S/I - R_S = {Zin:.1f} ohm "
             f"(prediction: {', '.join(f'{v:.1f}' for v in zp.values())} ohm for the references in order)")
    log.info("effective radius of this run", f"r_eff = {r_meas * 1e3:.4f} mm = {r_meas / c.radius:.4f} r from the measured Zin "
             f"(case A: {c.r_eff_factor:.4f} r; the nominal radius would need Zin = {zp[c.refs_names[0]]:.1f} ohm)")
    log.check(max(abs(tn_r - ts_r), abs(tn_f - ts_f)) < 0.15e-9, "near-end edges within 0.15 ns of the source (50 % crossings; "
              "the riser adds about 0.07 ns)", f"rise {1e12 * (tn_r - ts_r):+.0f} ps, fall {1e12 * (tn_f - ts_f):+.0f} ps", "WARN")
    log.check(abs(delay / c.T - 1) < 0.01, "aggressor delay equals L/c within 1 % (50 % crossings, both edges)",
              f"{delay * 1e9:.3f} ns vs {c.T * 1e9:.3f} ns ({100 * (delay / c.T - 1):+.2f} %)")
    log.check(pn > 0 and pf > 0, "probe sign: positive aggressor current from near to far end")
    log.check(abs(Zin / zp[c.primary] - 1) < 0.02, "Zin within 2 % of the primary reference (r_eff = case A factor x r; "
              "the thin-wire effect of cases A-C carries over)", f"{Zin:.1f} ohm vs {zp[c.primary]:.1f} ohm "
              f"({100 * (Zin / zp[c.primary] - 1):+.2f} %)", "WARN")
    log.check(zspread < 0.01 * Zin, "V_S / I - R_S is constant on the plateau (13.5 to 19 ns): a pure line impedance",
              f"drift between the two halves {zspread:.2f} ohm over {Zin:.1f} ohm", "WARN")
    return SimpleNamespace(Ip_n=pn, Ip_f=pf, delay=delay, Zin=Zin, r_meas=r_meas, edges=(tn_r, tn_f, tf_r, tf_f),
                           src_edges=(ts_r, ts_f), off_n=off_n)
