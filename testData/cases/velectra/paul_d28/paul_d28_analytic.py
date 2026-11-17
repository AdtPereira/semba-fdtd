"""Analytic reference of Paul D.28: two lossless wires over a PEC plane in a homogeneous medium, exact in frequency.

lmat           inductance matrix of the wire pair over the plane (wide-separation formulas of the book, exact for this pair)
riser_L        series inductance of one 20 mm riser (thin vertical wire between the plane and the line)
mtl_ends       currents at the four ends for 1 V open-circuit on the aggressor, at one frequency
mtl_time       time-domain currents of the four ends (FFT of the source x line response)
mtl_transfer   V_NE/V_S and V_FE/V_S (complex), relative to the generator or to the line-input node
modal_bounce   independent time-domain solution of the book's SPICE subcircuit (two modes, reflection series)
book_check     L, C, modal Z0 / TD against SPICEMTL_D28.OUT, and frequency domain against the bounce solution
compare_with_mtl / compare_transfer   FDTD against the references, with the pass/fail checks
"""
import math

import numpy as np

from paul_d28_calibration import peak, plateau
from paul_d28_common import (BAND, BOOK, C0, EPS0, MU0, NULL_HALF_WIDTH, ROLES, TOL_DB, TOL_LAG, TOL_PEAK_PCT, TOL_PLATEAU_PCT,
                              TOL_RMS_AGG, TOL_RMS_VIC)


def lmat(pts, r):
    """Inductance matrix (H/m) of thin wires of radius r over a PEC plane at height 0; pts = [(y, height), ...].
    L_ii = mu0/(2 pi) ln(2 h_i / r), L_ij = mu0/(4 pi) ln(1 + (2h/s)^2) for equal heights (image method)."""
    n = len(pts)
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pi, pj = np.array(pts[i], float), np.array(pts[j], float)
            d = np.linalg.norm(pi - np.array([pj[0], -pj[1]]))          # distance to the image of wire j
            dd = r if i == j else np.linalg.norm(pi - pj)
            L[i, j] = MU0 / (2 * math.pi) * math.log(d / dd)
    return L


def riser_L(h, r):
    """Inductance (H) of a straight wire of length h and radius r (Rosa): mu0 h / (2 pi) (ln(2 h / r) - 1), about 15 nH for
    20 mm and r_eff = 0.31 mm. It is in series with every terminal: the FDTD wire goes up 20 mm from the plane to the line."""
    return MU0 * h / (2 * math.pi) * (math.log(2 * h / r) - 1)


def terms(ref, R, f):
    """Terminal impedances (R_S, R_L, R_NE, R_FE) at frequency f: the resistor in series with the riser inductance."""
    return tuple(complex(r_, 2 * math.pi * f * ref.Lr) for r_ in R)


def mtl_ends(L, Lw, Z, f):
    """Currents (positive along the line) at the four ends, 1 V open-circuit source on the aggressor.
    Z = (R_S, R_L, R_NE, R_FE), may be complex. Returns I0 (near: aggressor, victim) and IL (far: aggressor, victim)."""
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


def mtl_time(case, ref, t, fmax=5e8):
    """Time-domain currents of the four ends: FFT of the source times the exact line response."""
    nfft, f = fft_setup(t)
    S = np.fft.rfft(case.source(t), nfft)
    spec = {k: np.zeros(len(f), complex) for k in ROLES}
    for m in range(int(np.searchsorted(f, fmax))):
        I0, IL = mtl_ends(ref.L, case.length, terms(ref, case.R, f[m]), f[m])
        for k, v in zip(ROLES, (I0[0], IL[0], I0[1], IL[1])):
            spec[k][m] = v * S[m]
    return {k: np.fft.irfft(v, nfft)[:len(t)] for k, v in spec.items()}


def mtl_transfer(case, ref, f, node=False):
    """Analytic V_NE/V_S and V_FE/V_S (complex), shape (2, len(f)). V_S is the generator voltage, or with node=True the
    voltage at the line input after R_S (V_S - R_S I), the quantity of Paul Fig. D.30."""
    H = np.zeros((2, len(f)), complex)
    for m, fm in enumerate(f):
        I0, IL = mtl_ends(ref.L, case.length, terms(ref, case.R, fm), fm)
        den = 1.0 - case.R[0] * I0[0] if node else 1.0
        H[0, m], H[1, m] = -case.R[2] * I0[1] / den, case.R[3] * IL[1] / den
    return H


def xcorr_lag(t, x, y, lo, hi):
    """Time by which x lags y (s, positive = x later): maximum of the cross-correlation in [lo, hi], parabolic."""
    m = (t >= lo) & (t <= hi)
    a, b = x[m], y[m]
    cc = np.correlate(a, b, "full")
    k = int(np.argmax(cc))
    y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
    return (k - (len(a) - 1) + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)) * (t[1] - t[0])


# ---------------------------------------------------------------------------------------------------------------
# Independent check of the reference itself
# ---------------------------------------------------------------------------------------------------------------
def modal_bounce(case, t, L, R):
    """Time-domain currents of the four ends from the book's SPICE subcircuit: with equal terminations the pair splits into
    an even mode (Z0 = c (L11 + L12)) and an odd mode (c (L11 - L12)), each a line with its own Z0 and delay T. For a mode:
    forward wave a(t) = K Vs_m(t) + G_S G_L a(t - 2T), K = Z0/(R + Z0), G = (R - Z0)/(R + Z0); V(0) = a(t) + G_L a(t - 2T),
    I(0) = (a(t) - G_L a(t - 2T)) / Z0, I(l) = (1 - G_L) a(t - T) / Z0. Valid for equal real R at the four ends."""
    assert len({R[0], R[1], R[2], R[3]}) == 1, "modal_bounce needs equal terminations"
    r = R[0]
    T = case.length / C0
    vs = case.source(t)
    Im0, ImL = [], []
    for sign in (+1, -1):
        Z0 = C0 * (L[0, 0] + sign * L[0, 1])
        K, G = Z0 / (r + Z0), (r - Z0) / (r + Z0)
        vm = vs / math.sqrt(2)
        a = np.zeros_like(t)
        for k in range(0, int(t[-1] / (2 * T)) + 2):
            a += K * (G * G) ** k * np.interp(t - 2 * k * T, t, vm, left=0.0, right=0.0)
        a_d = lambda d: np.interp(t - d, t, a, left=0.0, right=0.0)
        Im0.append((a - G * a_d(2 * T)) / Z0)
        ImL.append((1 - G) * a_d(T) / Z0)
    s2 = math.sqrt(2)
    return {"agg_near": (Im0[0] + Im0[1]) / s2, "vic_near": (Im0[0] - Im0[1]) / s2,
            "agg_far": (ImL[0] + ImL[1]) / s2, "vic_far": (ImL[0] - ImL[1]) / s2}


def book_check(case, t, log):
    """The reference against the book: the L, C matrices and the two modes of SPICEMTL_D28.OUT, and the exact frequency-domain
    solver against an independent time-domain solution of the same subcircuit (nominal radius, no risers)."""
    nominal = case.refs[case.refs_names[0]]
    L = nominal.L
    C = MU0 * EPS0 * np.linalg.inv(L)
    rel = lambda a, b: abs(a / b - 1)
    ok = max(rel(L[0, 0], BOOK["L11"]), rel(L[0, 1], BOOK["L12"]), rel(C[0, 0], BOOK["C11"]), rel(C[0, 1], BOOK["C12"])) < 2e-5
    log.check(ok, "reference L and C match the book (SPICEMTL_D28.OUT) to 5 digits",
              f"L11 {L[0, 0] * 1e6:.6f}, L12 {L[0, 1] * 1e6:.6f} uH/m; C11 {C[0, 0] * 1e12:.4f}, C12 {C[0, 1] * 1e12:.5f} pF/m")
    z0 = (C0 * (L[0, 0] + L[0, 1]), C0 * (L[0, 0] - L[0, 1]))
    td = case.length / C0
    log.check(max(rel(z0[0], BOOK["Z0"][0]), rel(z0[1], BOOK["Z0"][1]), rel(td, BOOK["TD"][0])) < 2e-5,
              "modal impedances and delay match the book subcircuit", f"Z0 {z0[0]:.4f} / {z0[1]:.4f} ohm, T {td * 1e9:.5f} ns")
    b = modal_bounce(case, t, L, case.R)
    a = mtl_time(case, nominal, t)
    h10 = 20 * math.log10(abs(mtl_transfer(case, nominal, np.array([1e4]), node=True)[0, 0]))
    log.check(abs(h10 - BOOK["NE_over_S_10kHz_dB"]) < 0.1, "|V_NE / V_S node| at 10 kHz equals the book's -66.2 dB (Fig. D.30)",
              f"{h10:.2f} dB")
    d = max(100 * math.sqrt(np.mean((a[k] - b[k]) ** 2)) / np.abs(b[k]).max() for k in ROLES)
    log.check(d < 0.5, "frequency-domain reference agrees with the independent time-domain (modal bounce) solution",
              f"worst RMS {d:.3f} % of the own peak, over the four ends")


# ---------------------------------------------------------------------------------------------------------------
# FDTD against the references
# ---------------------------------------------------------------------------------------------------------------
def _plateaus(case, t, vne, vfe):
    return (plateau(t, vne, 14e-9, 19e-9), plateau(t, vfe, case.T + 14e-9, case.T + 19e-9))


def compare_with_mtl(case, cal, xt, t, I, log):
    """FDTD against the analytic MTL for every reference; returns {name: four currents}."""
    c = case
    Rne, Rfe = c.R[2], c.R[3]
    curves = {k: mtl_time(c, ref, t) for k, ref in c.refs.items()}
    pk, rms_p, pl = {}, None, {}
    rms = lambda x, y: 100 * math.sqrt(np.mean((x - y) ** 2)) / np.abs(y).max()
    print("\nFDTD vs analytic MTL                                     NE max (mV)  FE min (mV)   NE / FE plateau (mV)   "
          "RMS agg near/far, NE, FE [% of own peak]")
    print(f"  {'FDTD':55s} {xt.Vp[0] * 1e3:+8.3f}   {xt.Vp[1] * 1e3:+8.3f}     {xt.plat[0] * 1e3:+7.3f} / {xt.plat[1] * 1e3:+7.3f}")
    for k, cur in curves.items():
        vne, vfe = -Rne * cur["vic_near"], Rfe * cur["vic_far"]
        a, b = peak(t, vne), peak(t, -vfe)
        pk[k] = (a[0], a[1], b[0], -b[1])
        pl[k] = _plateaus(c, t, vne, vfe)
        r = (rms(I["agg_near"], cur["agg_near"]), rms(I["agg_far"], cur["agg_far"]), rms(xt.V["NE"], vne), rms(xt.V["FE"], vfe))
        print(f"  {k:55s} {pk[k][1] * 1e3:+8.3f}   {pk[k][3] * 1e3:+8.3f}     {pl[k][0] * 1e3:+7.3f} / {pl[k][1] * 1e3:+7.3f}"
              f"     {r[0]:5.2f} {r[1]:5.2f} {r[2]:5.2f} {r[3]:5.2f}")
        if k == c.primary:
            rms_p = r
    tne_a, vne_a, tfe_a, vfe_a = pk[c.primary]
    log.check(abs(xt.Vp[0] / vne_a - 1) < TOL_PEAK_PCT / 100, f"NE maximum within {TOL_PEAK_PCT:g} % of the primary MTL",
              f"{xt.Vp[0] * 1e3:+.3f} vs {vne_a * 1e3:+.3f} mV ({100 * (xt.Vp[0] / vne_a - 1):+.1f} %)")
    log.check(abs(xt.Vp[1] / vfe_a - 1) < TOL_PEAK_PCT / 100, f"FE minimum within {TOL_PEAK_PCT:g} % of the primary MTL",
              f"{xt.Vp[1] * 1e3:+.3f} vs {vfe_a * 1e3:+.3f} mV ({100 * (xt.Vp[1] / vfe_a - 1):+.1f} %)")
    pp = pl[c.primary]
    log.check(max(abs(xt.plat[0] / pp[0] - 1), abs(xt.plat[1] / pp[1] - 1)) < TOL_PLATEAU_PCT / 100,
              f"first plateaus (NE, FE) within {TOL_PLATEAU_PCT:g} % of the primary MTL",
              f"NE {xt.plat[0] * 1e3:+.2f} vs {pp[0] * 1e3:+.2f} mV ({100 * (xt.plat[0] / pp[0] - 1):+.1f} %), "
              f"FE {xt.plat[1] * 1e3:+.2f} vs {pp[1] * 1e3:+.2f} mV ({100 * (xt.plat[1] / pp[1] - 1):+.1f} %)", "WARN")
    # the thin-wire deficit against the book is predicted by the effective radius: the same effect as cases A to C
    nom, eff = c.refs_names[0], c.refs_names[1]
    dp = [pl[eff][i] / pl[nom][i] - 1 for i in (0, 1)]                   # predicted (analytic r_eff against nominal r)
    do = [xt.plat[i] / pl[nom][i] - 1 for i in (0, 1)]                   # observed (FDTD against the book)
    log.info("FDTD against the book (nominal radius)", f"first plateaus NE {do[0] * 100:+.1f} %, FE {do[1] * 100:+.1f} % "
             f"(book: {BOOK['NE_plateau'] * 1e3:+.2f} / {BOOK['FE_plateau'] * 1e3:+.2f} mV)")
    log.check(max(abs(do[0] - dp[0]), abs(do[1] - dp[1])) < 0.03,
              "the deficit against the book is the thin-wire effect: it equals the analytic change from r to r_eff within 3 points",
              f"observed NE {do[0] * 100:+.1f} / FE {do[1] * 100:+.1f} %, predicted {dp[0] * 100:+.1f} / {dp[1] * 100:+.1f} %")
    lim = c.T * 6 + 20e-9                                                # the whole record, up to the first echoes
    lag = {n: xcorr_lag(t, x, y, 0.0, min(lim, t[-1])) for n, x, y in
           (("NE", xt.V["NE"], -Rne * curves[c.primary]["vic_near"]), ("FE", xt.V["FE"], Rfe * curves[c.primary]["vic_far"]))}
    log.check(max(abs(v) for v in lag.values()) < TOL_LAG, f"NE and FE waveforms in time with the MTL "
              f"(cross-correlation lag below {TOL_LAG * 1e9:.2f} ns)", f"NE {lag['NE'] * 1e12:+.0f} ps, FE {lag['FE'] * 1e12:+.0f} ps "
              f"(positive = FDTD later)", "WARN")
    log.check(max(rms_p[:2]) < TOL_RMS_AGG, f"aggressor currents match the primary MTL within {TOL_RMS_AGG:g} % RMS of the peak",
              f"near {rms_p[0]:.2f} %, far {rms_p[1]:.2f} %", "WARN")
    log.check(max(rms_p[2:]) < TOL_RMS_VIC, f"NE/FE waveforms match the primary MTL within {TOL_RMS_VIC:g} % RMS of their peak",
              f"NE {rms_p[2]:.2f} %, FE {rms_p[3]:.2f} %", "WARN")
    # what the risers explain: the same reference without them
    noris = c.refs_names[1]
    r0 = (rms(xt.V["NE"], -Rne * curves[noris]["vic_near"]), rms(xt.V["FE"], Rfe * curves[noris]["vic_far"]),
          rms(I["agg_near"], curves[noris]["agg_near"]), rms(I["agg_far"], curves[noris]["agg_far"]))
    log.info("effect of the risers (series inductance in the MTL)", f"RMS without -> with: NE {r0[0]:.2f} -> {rms_p[2]:.2f} %, "
             f"FE {r0[1]:.2f} -> {rms_p[3]:.2f} %, aggressor near {r0[2]:.2f} -> {rms_p[0]:.2f} %, far {r0[3]:.2f} -> {rms_p[1]:.2f} % "
             f"(L_riser = {c.refs[c.primary].Lr * 1e9:.1f} nH per terminal, from the formula, not fitted)")
    log.check(sum(rms_p) <= sum(r0) + 1e-9, "the riser inductance brings the MTL closer to the FDTD (not further)",
              f"sum of the four RMS: {sum(r0):.2f} -> {sum(rms_p):.2f} %", "WARN")
    return curves


def compare_transfer(case, tf, t, curves, log):
    """|H_NE| and |H_FE| of the FDTD against the primary MTL between 3 and 100 MHz, both truncated to the same record: the
    record is only 200 ns and the mismatched line is still ringing at its end (about -14 dB for the NE voltage), so the
    infinite-time response is not what the FDTD spectrum contains. The MTL time response is cut at the same instant, and the
    two are divided by the same source spectrum. The line resonances n*c/(2L) (32 MHz, 64, 96) have deep narrow nulls and
    +-NULL_HALF_WIDTH around them is skipped."""
    c = case
    f1 = C0 / (2 * c.length)
    band = tf.band.copy()
    for n in range(1, int(BAND[1] / f1) + 2):
        band &= abs(tf.f - n * f1) > NULL_HALF_WIDTH
    tf.band = band
    dB = lambda x: 20 * np.log10(np.abs(x) + 1e-30)
    cur = curves[c.primary]
    nfft, _ = fft_setup(t)
    S = np.fft.rfft(c.source(t), nfft)
    Hm = {"NE": np.fft.rfft(-c.R[2] * cur["vic_near"], nfft) / np.where(S == 0, 1e-30, S),
          "FE": np.fft.rfft(c.R[3] * cur["vic_far"], nfft) / np.where(S == 0, 1e-30, S)}
    tf.Hm = Hm
    f = tf.f[band]
    for n in ("NE", "FE"):
        d = dB(tf.H[n][band]) - dB(Hm[n][band])
        log.info(f"|H_{n}| FDTD - MTL (same 200 ns record)", f"{BAND[0] / 1e6:g}-{BAND[1] / 1e6:g} MHz: mean {d.mean():+.2f} dB, "
                 f"rms {np.sqrt(np.mean(d ** 2)):.2f} dB, max {abs(d).max():.2f} dB at {f[np.argmax(abs(d))] / 1e6:.1f} MHz")
        log.check(abs(d).max() < TOL_DB, f"|H_{n}| within {TOL_DB:g} dB of the MTL between {BAND[0] / 1e6:g} and "
                  f"{BAND[1] / 1e6:g} MHz (resonance nulls skipped)", f"max {abs(d).max():.2f} dB", "WARN")
    ref = c.refs[c.primary]
    for fm in (1e6, 10e6, 20e6, 30e6, 40e6):
        k = int(np.argmin(abs(tf.f - fm)))
        h = mtl_transfer(c, ref, np.array([fm]))[:, 0]
        log.info(f"|NE| / |FE| at {fm / 1e6:g} MHz (FDTD | MTL same record | MTL infinite record)",
                 f"{dB(tf.H['NE'][k]):.1f} / {dB(tf.H['FE'][k]):.1f} | {dB(Hm['NE'][k]):.1f} / {dB(Hm['FE'][k]):.1f} | "
                 f"{dB(h[0]):.1f} / {dB(h[1]):.1f} dB")
