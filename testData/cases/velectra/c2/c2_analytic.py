"""Analytic reference: two lossless lines over a PEC plane (MTL, exact in frequency; Appendix B of the tutorial).

lmat      inductance matrix of the wire pair
mtl_ends  currents at the four ends for 1 V open-circuit on the aggressor, at one frequency
mtl_time  time-domain currents of the four ends (FFT of the source x line response)
mtl_transfer  |V_NE/V_S| and |V_FE/V_S| (complex)
compare_with_mtl / compare_transfer  FDTD against the reference, with the pass/fail checks
"""
import math

import numpy as np

from c2_calibration import peak
from c2_common import BAND, C0, MU0, NULL_HALF_WIDTH, ROLES


def lmat(pts, r):
    """Inductance matrix (H/m) of thin wires of radius r over a plane at y = 0; pts = [(x, height), ...]."""
    n = len(pts)
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pi, pj = np.array(pts[i], float), np.array(pts[j], float)
            d = np.linalg.norm(pi - np.array([pj[0], -pj[1]]))          # distance to the image of wire j
            dd = r if i == j else np.linalg.norm(pi - pj)
            L[i, j] = MU0 / (2 * math.pi) * math.log(d / dd)
    return L


def mtl_ends(L, Lw, Z, f):
    """Currents (positive along z) at the four ends, 1 V open-circuit source on the aggressor.
    Z = (R_S, R_L, R_NE, R_FE). Returns I0 (near: aggressor, victim) and IL (far: aggressor, victim)."""
    Rs, Rl, Rne, Rfe = Z
    Zc = C0 * L
    Yc = np.linalg.inv(Zc)
    one = np.eye(2)
    b = 2 * math.pi * f * Lw / C0
    A, B, Cm = math.cos(b) * one, -1j * math.sin(b) * Zc, -1j * math.sin(b) * Yc
    Zn, Zf, Vs = np.diag([Rs, Rne]), np.diag([Rl, Rfe]), np.array([1.0, 0.0])
    I0 = np.linalg.solve((B - A @ Zn) - Zf @ (A - Cm @ Zn), Zf @ (Cm @ Vs) - A @ Vs)
    V0 = Vs - Zn @ I0
    return I0, Cm @ V0 + A @ I0


def fft_setup(t, n_min_time=1e-6):
    """Uniform grid t -> zero-padded FFT length (>= 1 us and >= 2 samples) and its frequency axis."""
    dt = t[1] - t[0]
    nfft = 1 << int(math.ceil(math.log2(max(2 * len(t), n_min_time / dt))))
    return nfft, np.fft.rfftfreq(nfft, dt)


def mtl_time(case, L, t, fmax=5e8):
    """Time-domain currents of the four ends: FFT of the source times the exact line response."""
    nfft, f = fft_setup(t)
    S = np.fft.rfft(case.source(t), nfft)
    spec = {k: np.zeros(len(f), complex) for k in ROLES}
    for m in range(int(np.searchsorted(f, fmax))):
        I0, IL = mtl_ends(L, case.length, case.R, f[m])
        for k, v in zip(ROLES, (I0[0], IL[0], I0[1], IL[1])):
            spec[k][m] = v * S[m]
    return {k: np.fft.irfft(v, nfft)[:len(t)] for k, v in spec.items()}


def mtl_transfer(case, L, f):
    """Analytic V_NE/V_S and V_FE/V_S (complex) on the frequency axis f; shape (2, len(f))."""
    H = np.zeros((2, len(f)), complex)
    for m, fm in enumerate(f):
        I0, IL = mtl_ends(L, case.length, case.R, fm)
        H[0, m], H[1, m] = -case.R[2] * I0[1], case.R[3] * IL[1]
    return H


def compare_with_mtl(case, cal, xt, t, I, log):
    """FDTD against the analytic MTL for the three references; returns {name: four currents}."""
    c = case
    Rne, Rfe = c.R[2], c.R[3]
    curves = {k: mtl_time(c, L, t) for k, L in c.L.items()}
    pk, rms_p = {}, None
    rms = lambda x, y, p: 100 * math.sqrt(np.mean((x - y) ** 2)) / abs(p)
    print("\nFDTD vs analytic MTL                     NE peak (mV)  t (ns)   FE peak (mV)  t (ns)   "
          "RMS agg near/far, NE, FE [% of own peak]")
    print(f"  {'FDTD':38s} {xt.Vp[0] * 1e3:+9.3f}   {xt.tp[0] * 1e9:6.2f}   {xt.Vp[1] * 1e3:+9.3f}   {xt.tp[1] * 1e9:6.2f}")
    for k, cur in curves.items():
        vne, vfe = -Rne * cur["vic_near"], Rfe * cur["vic_far"]
        a, b = peak(t, vne), peak(t, -vfe)
        pk[k] = (a[0], a[1], b[0], -b[1])                       # t_NE, V_NE, t_FE, V_FE
        r = (rms(I["agg_near"], cur["agg_near"], cal.Ip_n), rms(I["agg_far"], cur["agg_far"], cal.Ip_f),
             rms(xt.V["NE"], vne, xt.Vp[0]), rms(xt.V["FE"], vfe, xt.Vp[1]))
        print(f"  {k:38s} {pk[k][1] * 1e3:+9.3f}   {pk[k][0] * 1e9:6.2f}   {pk[k][3] * 1e3:+9.3f}   "
              f"{pk[k][2] * 1e9:6.2f}   {r[0]:5.2f} {r[1]:5.2f} {r[2]:5.2f} {r[3]:5.2f}")
        if k == c.primary:
            rms_p = r
    tne_a, vne_a, tfe_a, vfe_a = pk[c.primary]
    log.check(abs(xt.Vp[0] / vne_a - 1) < 0.10, f"NE peak within 10 % of the analytic MTL ({c.primary})",
              f"{xt.Vp[0] * 1e3:+.3f} vs {vne_a * 1e3:+.3f} mV ({100 * (xt.Vp[0] / vne_a - 1):+.1f} %)")
    log.check(abs(xt.Vp[1] / vfe_a - 1) < 0.10, f"FE peak within 10 % of the analytic MTL ({c.primary})",
              f"{xt.Vp[1] * 1e3:+.3f} vs {vfe_a * 1e3:+.3f} mV ({100 * (xt.Vp[1] / vfe_a - 1):+.1f} %)")
    log.check(abs(xt.tp[0] - tne_a) < 0.2e-9 and abs(xt.tp[1] - tfe_a) < 0.2e-9,
              "NE and FE peak times within 0.2 ns of the MTL",
              f"NE {xt.tp[0] * 1e9:.2f} vs {tne_a * 1e9:.2f} ns, FE {xt.tp[1] * 1e9:.2f} vs {tfe_a * 1e9:.2f} ns", "WARN")
    log.check(max(rms_p[:2]) < 2.0, f"aggressor currents match the MTL ({c.primary}) within 2 % RMS of the peak",
              f"near {rms_p[0]:.2f} %, far {rms_p[1]:.2f} %", "WARN")
    log.check(max(rms_p[2:]) < 10.0, f"NE/FE waveforms match the MTL ({c.primary}) within 10 % RMS of their peak",
              f"NE {rms_p[2]:.2f} %, FE {rms_p[3]:.2f} %", "WARN")
    return curves


def compare_transfer(case, tf, log):
    """|H_NE| and |H_FE| of the FDTD against the MTL between 3 and 100 MHz.
    The line resonances n*c/(2L) (75 MHz in band) have deep narrow nulls; a 0.6 % longer delay, as measured in
    case A, shifts them by ~0.5 MHz and a dB difference there is not a model error, so +-NULL_HALF_WIDTH is skipped."""
    c = case
    f1 = C0 / (2 * c.length)
    band = tf.band.copy()
    for n in range(1, int(BAND[1] / f1) + 2):
        band &= abs(tf.f - n * f1) > NULL_HALF_WIDTH
    tf.band = band
    dB = lambda x: 20 * np.log10(np.abs(x) + 1e-30)
    f = tf.f[band]
    Href = mtl_transfer(c, c.L[c.primary], f)
    for i, n in enumerate(("NE", "FE")):
        d = dB(tf.H[n][band]) - dB(Href[i])
        log.info(f"|H_{n}| FDTD - MTL", f"{BAND[0] / 1e6:g}-{BAND[1] / 1e6:g} MHz: mean {d.mean():+.2f} dB, "
                 f"rms {np.sqrt(np.mean(d ** 2)):.2f} dB, max {abs(d).max():.2f} dB at {f[np.argmax(abs(d))] / 1e6:.1f} MHz")
        log.check(abs(d).max() < 2.0, f"|H_{n}| within 2 dB of the MTL between {BAND[0] / 1e6:g} and "
                  f"{BAND[1] / 1e6:g} MHz (resonance nulls skipped)", f"max {abs(d).max():.2f} dB", "WARN")
    for fm in (1e6, 10e6, 30e6, 100e6):
        k = int(np.argmin(abs(tf.f - fm)))
        h = mtl_transfer(c, c.L[c.primary], np.array([fm]))[:, 0]
        log.info(f"|NE| / |FE| at {fm / 1e6:g} MHz (FDTD | MTL)",
                 f"{dB(tf.H['NE'][k]):.1f} / {dB(tf.H['FE'][k]):.1f} | {dB(h[0]):.1f} / {dB(h[1]):.1f} dB")
