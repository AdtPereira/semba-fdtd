"""Victim voltages (NE, FE), their peaks and polarity, the transfer function and the late-time ring-down.

V_NE = -R_NE * I_vic,near and V_FE = +R_FE * I_vic,far (currents positive along the polyline, near -> far).
"""
import math
from types import SimpleNamespace

import numpy as np

from c2_analytic import fft_setup
from c2_calibration import peak
from c2_common import BAND, LATE_START, ROLES, SRC_FLOOR_DB


def crosstalk(case, t, I, log):
    c = case
    V = {"NE": -c.R[2] * I["vic_near"], "FE": c.R[3] * I["vic_far"]}
    tp_ne, Vp_ne = peak(t, V["NE"])
    tp_fe, Vp_fe = peak(t, -V["FE"])
    Vp_fe = -Vp_fe
    log.info("NE peak", f"{Vp_ne * 1e3:+.3f} mV at {tp_ne * 1e9:.3f} ns")
    log.info("FE peak", f"{Vp_fe * 1e3:+.3f} mV at {tp_fe * 1e9:.3f} ns")
    log.check(Vp_ne > 0 and Vp_fe < 0, "polarity: NE positive, FE negative (V_NE = -R*I_near, V_FE = +R*I_far)")
    return SimpleNamespace(V=V, Vp=(Vp_ne, Vp_fe), tp=(tp_ne, tp_fe))


def transfer_function(case, t, xt):
    """H(f) = FFT(V_probe) / FFT(source) on the zero-padded axis, restricted to the band where the source has energy
    (3 to 100 MHz, source above -40 dB) and away from the line resonances (deep narrow nulls, see c2_analytic)."""
    c = case
    nfft, f = fft_setup(t)
    S = np.fft.rfft(c.source(t), nfft)
    sdb = 20 * np.log10(np.abs(S) / np.abs(S).max() + 1e-30)
    H = {n: np.fft.rfft(v, nfft) / np.where(S == 0, 1e-30, S) for n, v in xt.V.items()}
    band = (f >= BAND[0]) & (f <= BAND[1]) & (sdb > SRC_FLOOR_DB)
    return SimpleNamespace(f=f, H=H, sdb=sdb, band=band, band_all=band.copy())


def late_time(case, t, I, xt, log):
    """The mismatched line must ring down: no growth, below -40 dB after the echo (about 150 to 160 ns)."""
    late = t > LATE_START
    sig = {**{f"I_{r}": I[r] for r in ROLES}, "V_NE": xt.V["NE"], "V_FE": xt.V["FE"]}
    if late.any():
        dbl = {k: 20 * math.log10(abs(x[late]).max() / abs(x).max() + 1e-30) for k, x in sig.items()}
        log.info(f"late-time residual after {LATE_START * 1e9:.0f} ns (re own peak)",
                 ", ".join(f"{k} {v:.1f} dB" for k, v in dbl.items()))
        log.check(max(dbl.values()) < -40, "late-time residual below -40 dB (no ringing, no instability)",
                  f"worst {max(dbl.values()):.1f} dB", "WARN")
