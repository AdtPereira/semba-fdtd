"""Victim voltages (NE, FE), their extremes, plateaus and polarity, the transfer function and the late-time tail.

V_NE = -R_NE * I_vic,near and V_FE = +R_FE * I_vic,far (currents positive along the polyline, near -> far end).
Paul's first plateaus: NE positive (+23.3 mV, 12.5 to 20 ns) and FE negative (-32.0 mV, about 28 to 35 ns), for the nominal radius.
"""
import math
from types import SimpleNamespace

import numpy as np

from paul_d28_analytic import fft_setup
from paul_d28_calibration import peak, plateau
from paul_d28_common import BAND, LATE_START, ROLES, SRC_FLOOR_DB


def crosstalk(case, t, I, log):
    c = case
    V = {"NE": -c.R[2] * I["vic_near"], "FE": c.R[3] * I["vic_far"]}
    tp_ne, Vp_ne = peak(t, V["NE"])
    tp_fe, Vp_fe = peak(t, -V["FE"])
    Vp_fe = -Vp_fe
    plat = (plateau(t, V["NE"], 14e-9, 19e-9), plateau(t, V["FE"], c.T + 14e-9, c.T + 19e-9))
    log.info("NE", f"maximum {Vp_ne * 1e3:+.3f} mV at {tp_ne * 1e9:.2f} ns; first plateau (14-19 ns) {plat[0] * 1e3:+.3f} mV; "
                   f"minimum {V['NE'].min() * 1e3:+.3f} mV")
    log.info("FE", f"minimum {Vp_fe * 1e3:+.3f} mV at {tp_fe * 1e9:.2f} ns; first plateau ({(c.T + 14e-9) * 1e9:.1f}-"
                   f"{(c.T + 19e-9) * 1e9:.1f} ns) {plat[1] * 1e3:+.3f} mV; maximum {V['FE'].max() * 1e3:+.3f} mV")
    log.check(Vp_ne > 0 and Vp_fe < 0 and plat[0] > 0 and plat[1] < 0,
              "polarity: NE first plateau positive, FE negative (V_NE = -R*I_near, V_FE = +R*I_far)",
              f"NE {plat[0] * 1e3:+.2f} mV, FE {plat[1] * 1e3:+.2f} mV")
    return SimpleNamespace(V=V, Vp=(Vp_ne, Vp_fe), tp=(tp_ne, tp_fe), plat=plat)


def transfer_function(case, t, xt):
    """H(f) = FFT(V_probe) / FFT(source) on the zero-padded axis, restricted to the band where the source has energy
    (3 to 100 MHz, source above -40 dB; the trapezoid has nulls at multiples of 50 and 80 MHz, which the floor removes) and away
    from the line resonances (see compare_transfer). Also H relative to the line-input node, as in Paul Fig. D.30."""
    c = case
    nfft, f = fft_setup(t)
    S = np.fft.rfft(c.source(t), nfft)
    sdb = 20 * np.log10(np.abs(S) / np.abs(S).max() + 1e-30)
    H = {n: np.fft.rfft(v, nfft) / np.where(S == 0, 1e-30, S) for n, v in xt.V.items()}
    band = (f >= BAND[0]) & (f <= BAND[1]) & (sdb > SRC_FLOOR_DB)
    return SimpleNamespace(f=f, H=H, sdb=sdb, band=band, band_all=band.copy(), S=S, nfft=nfft)


def band_info(tf, log):
    """Frequency stretches of 3-100 MHz where the source spectrum is above the floor: what the dB comparison really covers."""
    idx = np.flatnonzero(tf.band_all)
    cuts = np.flatnonzero(np.diff(idx) > 1)
    parts = np.split(idx, cuts + 1)
    log.info("usable band (source above -40 dB)", "; ".join(f"{tf.f[p[0]] / 1e6:.1f}-{tf.f[p[-1]] / 1e6:.1f} MHz" for p in parts)
             + f" ({int(tf.band_all.sum())} bins)")


def node_transfer(case, t, I, xt, tf):
    """FDTD V/V_node: the node voltage after R_S is V_S(t) - R_S * I_agg,near(t); the same quantity as Paul Fig. D.30."""
    c = case
    Vn = c.source(t) - c.R[0] * I["agg_near"]
    Fn = np.fft.rfft(Vn, tf.nfft)
    ok = np.abs(Fn) > np.abs(Fn).max() * 10 ** (SRC_FLOOR_DB / 20)
    band = tf.band_all & ok
    H = {n: np.fft.rfft(v, tf.nfft) / np.where(Fn == 0, 1e-30, Fn) for n, v in xt.V.items()}
    return SimpleNamespace(H=H, band=band)


def late_time(case, t, I, xt, curves, log):
    """The mismatched line (Gamma_L about -0.70) is still ringing at the end of the book's 200 ns record, so the absolute
    -40 dB test of the Velectra cases does not apply. What must hold: no growth, and the tail (last 50 ns) equals the MTL tail."""
    c = case
    late = t > LATE_START
    sig = {**{f"I_{r}": I[r] for r in ROLES}, "V_NE": xt.V["NE"], "V_FE": xt.V["FE"]}
    cur = curves[c.primary]
    mtl = {**{f"I_{r}": cur[r] for r in ROLES}, "V_NE": -c.R[2] * cur["vic_near"], "V_FE": c.R[3] * cur["vic_far"]}
    dbl = {k: 20 * math.log10(abs(x[late]).max() / abs(x).max() + 1e-30) for k, x in sig.items()}
    dbm = {k: 20 * math.log10(abs(x[late]).max() / abs(x).max() + 1e-30) for k, x in mtl.items()}
    log.info(f"tail after {LATE_START * 1e9:.0f} ns, FDTD | MTL (re own peak)",
             "; ".join(f"{k} {dbl[k]:.1f} | {dbm[k]:.1f} dB" for k in sig))
    grow = all(abs(x[late]).max() < abs(x[(t > 60e-9) & (t < 100e-9)]).max() for x in sig.values())
    log.check(grow, "no late growth: the tail is below the stretch 60-100 ns in every current and voltage")
    d = max(100 * math.sqrt(np.mean((sig[k][late] - mtl[k][late]) ** 2)) / abs(sig[k]).max() for k in sig)
    log.check(d < 3.0, f"tail after {LATE_START * 1e9:.0f} ns equals the MTL tail (RMS below 3 % of the own peak)",
              f"worst {d:.2f} %", "WARN")
    if t[-1] < 350e-9:
        log.info("record length", f"{t[-1] * 1e9:.0f} ns: the NE voltage is still at {dbl['V_NE']:.0f} dB at the end; a record "
                                  f"of about 500 ns (paul_d28_make_exc.py --window 500e-9, Solver final time 5e-7 s, about 5 "
                                  f"min) lets the transfer function be compared with the infinite-record MTL and Paul Fig. D.30")
