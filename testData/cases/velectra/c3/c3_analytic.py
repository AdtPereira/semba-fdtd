"""Analytic reference: two lossless lines inside a closed PEC cylinder (MTL, exact in frequency; Appendix B of the tutorial) or, for the
B -> C comparison, over a PEC plane, with the
PHYSICAL radius of the wires (no fitted r_eff: case A showed that the solver's own Zc at 35 mm, 313 ohm, is not a radius change).
Plane: pts = [(x, height)]; cylinder (R given): pts = [(x, y)] inside the tube.

lmat      inductance matrix of the wire pair
mtl_ends  currents at the four ends for 1 V open-circuit on the aggressor, at one frequency
mtl_time  time-domain currents of the four ends (FFT of the source x line response)
mtl_transfer  |V_NE/V_S| and |V_FE/V_S| (complex)
compare_with_mtl / compare_transfer  FDTD against the reference: the differences are INFO (they ARE the result);
              only gross-error bands are pass/fail
"""
import math

import numpy as np

from c3_calibration import peak
from c3_common import BAND, C0, MU0, NULL_HALF_WIDTH, ROLES


def lmat(pts, r, R=None):
    """Inductance matrix (H/m) of thin wires of radius r.
    R is None: PEC plane at y = 0, pts = [(x, height), ...].
    R given:   circular PEC tube of radius R centred on the origin, pts = [(x, y), ...] inside it.
               Image of p is R^2 p / (p.p); the distance term is |pi - img_j| |pj| / R (tutorial, Appendix B)."""
    n = len(pts)
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pi, pj = np.array(pts[i], float), np.array(pts[j], float)
            if R is None:
                d = np.linalg.norm(pi - np.array([pj[0], -pj[1]]))      # distance to the image of wire j
            else:
                img = R ** 2 * pj / pj.dot(pj)
                d = np.linalg.norm(pi - img) * np.linalg.norm(pj) / R
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


def xcorr_lag(t, x, ref, window=None):
    """Lag (s) of x against ref (positive = x arrives later), from the peak of the cross-correlation, parabolic refinement."""
    if window is not None:
        m = (t >= window[0]) & (t <= window[1])
        x, ref = x[m], ref[m]
    n = len(x)
    cc = np.fft.irfft(np.fft.rfft(x, 2 * n) * np.conj(np.fft.rfft(ref, 2 * n)), 2 * n)
    cc = np.concatenate((cc[-n + 1:], cc[:n]))                    # lags -(n-1) .. n-1
    k = int(np.argmax(cc))
    off = 0.0
    if 0 < k < len(cc) - 1:
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    return (k - (n - 1) + off) * (t[1] - t[0])


# gross-error bands: they only catch a wrong geometry, a wrong source or a broken run, not the model difference itself
GROSS_PEAK = 0.25          # FAIL: NE or FE peak more than 25 % from the analytic one
GROSS_RMS_AGG = 0.15       # FAIL: aggressor currents RMS above 15 % of the peak (WARN at 5 %)
GROSS_RMS_XT = 0.30        # FAIL: NE/FE waveforms RMS above 30 % of their peak


def compare_with_mtl(case, cal, xt, t, I, log):
    """FDTD against the analytic MTL (physical radius); returns {name: four currents}.
    Matched case (terminals = Zc): an ideal homogeneous lossless line has NO far-end cross-talk then (the analytic FE is
    ~0), so a percentage of the FE peak means nothing; FE is compared in mV and as a fraction of the analytic NE peak, and
    the waveform RMS of FE is taken against the NE peak."""
    c = case
    Rne, Rfe = c.R[2], c.R[3]
    curves = {k: mtl_time(c, L, t) for k, L in c.L.items()}
    cur = curves[c.primary]
    vne, vfe = -Rne * cur["vic_near"], Rfe * cur["vic_far"]
    (tne_a, vne_a), b = peak(t, vne), peak(t, -vfe)
    tfe_a, vfe_a = b[0], -b[1]
    ref_fe = abs(xt.Vp[0]) if c.matched else abs(xt.Vp[1])            # what the FE RMS is a percentage of
    rms = lambda x, y, p: 100 * math.sqrt(np.mean((x - y) ** 2)) / abs(p)
    r = (rms(I["agg_near"], cur["agg_near"], cal.Ip_n), rms(I["agg_far"], cur["agg_far"], cal.Ip_f),
         rms(xt.V["NE"], vne, xt.Vp[0]), rms(xt.V["FE"], vfe, ref_fe))
    w_near = (c.t0 - 4 * c.sigma, c.t0 + c.T + 6 * c.sigma)             # the first pass of the pulse, seen at the near end
    w_far = (c.t0 + c.T - 4 * c.sigma, c.t0 + c.T + 6 * c.sigma)        # and at the far end
    lag = {"agg_near": xcorr_lag(t, I["agg_near"], cur["agg_near"], w_near),
           "agg_far": xcorr_lag(t, I["agg_far"], cur["agg_far"], w_far),
           "NE": xcorr_lag(t, xt.V["NE"], vne, w_near)}
    if not c.matched:
        lag["FE"] = xcorr_lag(t, xt.V["FE"], vfe, w_far)
    unit = "of the NE peak" if c.matched else "of the own peak"
    print(f"\nsemba-fdtd vs analytic MTL, {c.primary} (r = {c.radius * 1e3:g} mm, Zc = {c.Zc_r:.1f} ohm, delay L/c)")
    print(f"  {'':16s} NE peak (mV)  t (ns)   FE peak (mV)  t (ns)")
    print(f"  {'semba-fdtd':16s} {xt.Vp[0] * 1e3:+9.3f}   {xt.tp[0] * 1e9:6.2f}   {xt.Vp[1] * 1e3:+9.3f}   {xt.tp[1] * 1e9:6.2f}")
    print(f"  {'analytic':16s} {vne_a * 1e3:+9.3f}   {tne_a * 1e9:6.2f}   {vfe_a * 1e3:+9.3f}   {tfe_a * 1e9:6.2f}")
    print(f"  RMS [% of the semba-fdtd peak]: aggressor near {r[0]:.2f}, far {r[1]:.2f}; NE {r[2]:.2f}, FE {r[3]:.2f} ({unit} for FE)")
    log.info("NE peak, semba-fdtd vs analytic", f"{xt.Vp[0] * 1e3:+.3f} vs {vne_a * 1e3:+.3f} mV ({100 * (xt.Vp[0] / vne_a - 1):+.1f} %)")
    if c.matched:
        log.info("FE peak, semba-fdtd vs analytic",
                 f"{xt.Vp[1] * 1e3:+.3f} vs {vfe_a * 1e3:+.3f} mV: {100 * abs(xt.Vp[1]) / abs(xt.Vp[0]):.1f} % vs "
                 f"{100 * abs(vfe_a) / abs(vne_a):.1f} % of the NE peak (an ideal homogeneous matched line has no FE: "
                 f"the FDTD value is the imbalance between its inductive and capacitive coupling)")
        log.info("NE peak time, semba-fdtd vs analytic", f"{xt.tp[0] * 1e9:.2f} vs {tne_a * 1e9:.2f} ns")
    else:
        log.info("FE peak, semba-fdtd vs analytic", f"{xt.Vp[1] * 1e3:+.3f} vs {vfe_a * 1e3:+.3f} mV ({100 * (xt.Vp[1] / vfe_a - 1):+.1f} %)")
        log.info("NE / FE peak time, semba-fdtd vs analytic",
                 f"NE {xt.tp[0] * 1e9:.2f} vs {tne_a * 1e9:.2f} ns, FE {xt.tp[1] * 1e9:.2f} vs {tfe_a * 1e9:.2f} ns "
                 f"(the peak of a flat Gaussian is ill-conditioned: see the lags)")
    log.info("lag of semba-fdtd against the analytic line (cross-correlation, first pass)",
             ", ".join(f"{k} {v * 1e12:+.0f} ps" for k, v in lag.items())
             + "  (positive = semba-fdtd later; case A: delay +41 ps at 35 mm)")
    log.info("RMS difference to the analytic MTL",
             f"aggressor {r[0]:.2f} % near / {r[1]:.2f} % far; NE {r[2]:.2f} %, FE {r[3]:.2f} % ({unit} for FE)")
    d_ne = xt.Vp[0] / vne_a - 1
    if c.matched:
        d_fe = (xt.Vp[1] - vfe_a) / abs(vne_a)                          # mV difference as a fraction of the analytic NE peak
        log.check(abs(d_ne) < GROSS_PEAK and abs(d_fe) < GROSS_PEAK,
                  f"NE peak within {GROSS_PEAK * 100:.0f} % of the analytic MTL and FE difference below {GROSS_PEAK * 100:.0f} % of its NE peak "
                  f"(gross-error band)", f"NE {100 * d_ne:+.1f} %, FE {100 * d_fe:+.1f} % of NE")
    else:
        d_fe = xt.Vp[1] / vfe_a - 1
        log.check(abs(d_ne) < GROSS_PEAK and abs(d_fe) < GROSS_PEAK,
                  f"NE and FE peaks within {GROSS_PEAK * 100:.0f} % of the analytic MTL (gross-error band)",
                  f"NE {100 * d_ne:+.1f} %, FE {100 * d_fe:+.1f} %")
    log.check(max(r[:2]) < 5.0, "aggressor currents within 5 % RMS of the analytic MTL", f"near {r[0]:.2f} %, far {r[1]:.2f} %", "WARN")
    log.check(max(r[:2]) < 100 * GROSS_RMS_AGG, f"aggressor currents within {GROSS_RMS_AGG * 100:.0f} % RMS (gross-error band)",
              f"near {r[0]:.2f} %, far {r[1]:.2f} %")
    log.check(max(r[2:]) < 100 * GROSS_RMS_XT, f"NE/FE waveforms within {GROSS_RMS_XT * 100:.0f} % RMS of their peak (gross-error band)",
              f"NE {r[2]:.2f} %, FE {r[3]:.2f} %")
    return curves


def compare_transfer(case, tf, log):
    """|H_NE| and |H_FE| of the FDTD against the MTL between 3 and 100 MHz (differences are INFO).
    The line resonances n*c/(2L) (75 MHz in band) have deep narrow nulls; a delay excess of a few tens of ps, as measured
    in case A, shifts them by ~0.5 MHz and a dB difference there is not a model error, so +-NULL_HALF_WIDTH is skipped."""
    c = case
    f1 = C0 / (2 * c.length)
    # matched line: |H_NE| is proportional to |sin(pi f / f1)|, a true zero at n*f1, so the null is much sharper than in the 50 ohm case and
    # a null that sits ~2.5 MHz away in the FDTD (measured: 77.5 vs 75 MHz) gives +16/-13 dB there; skip +-8 MHz instead of +-4 MHz.
    hw = 2 * NULL_HALF_WIDTH if c.matched else NULL_HALF_WIDTH
    band = tf.band.copy()
    for n in range(1, int(BAND[1] / f1) + 2):
        band &= abs(tf.f - n * f1) > hw
    tf.band = band
    dB = lambda x: 20 * np.log10(np.abs(x) + 1e-30)
    f = tf.f[band]
    Href = mtl_transfer(c, c.L[c.primary], f)
    for i, n in enumerate(("NE", "FE")):
        d = dB(tf.H[n][band]) - dB(Href[i])
        if c.matched and n == "FE":                       # the analytic FE of a matched homogeneous line is ~0 (-80 dB or less)
            log.info("|H_FE| FDTD, matched case", f"{BAND[0] / 1e6:g}-{BAND[1] / 1e6:g} MHz: mean {dB(tf.H['FE'][band]).mean():.1f} dB "
                     f"(|H_NE| mean {dB(tf.H['NE'][band]).mean():.1f} dB, analytic |H_FE| mean {dB(Href[i]).mean():.1f} dB): "
                     f"no dB comparison, the reference is zero")
            continue
        log.info(f"|H_{n}| FDTD - MTL", f"{BAND[0] / 1e6:g}-{BAND[1] / 1e6:g} MHz: mean {d.mean():+.2f} dB, "
                 f"rms {np.sqrt(np.mean(d ** 2)):.2f} dB, max {abs(d).max():.2f} dB at {f[np.argmax(abs(d))] / 1e6:.1f} MHz")
        log.check(abs(d).max() < 3.0, f"|H_{n}| within 3 dB of the MTL between {BAND[0] / 1e6:g} and "
                  f"{BAND[1] / 1e6:g} MHz (resonance nulls skipped)", f"max {abs(d).max():.2f} dB", "WARN")
        log.check(abs(d).max() < 6.0, f"|H_{n}| within 6 dB of the MTL (gross-error band)", f"max {abs(d).max():.2f} dB")
    for fm in (1e6, 10e6, 30e6, 100e6):
        k = int(np.argmin(abs(tf.f - fm)))
        h = mtl_transfer(c, c.L[c.primary], np.array([fm]))[:, 0]
        log.info(f"|NE| / |FE| at {fm / 1e6:g} MHz (FDTD | MTL)",
                 f"{dB(tf.H['NE'][k]):.1f} / {dB(tf.H['FE'][k]):.1f} | {dB(h[0]):.1f} / {dB(h[1]):.1f} dB")
