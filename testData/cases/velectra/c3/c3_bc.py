"""B -> C comparison: what changes when the ground plane of case B becomes the closed PEC cylinder of case C.

The case B FDTD numbers come from the c2 case with the same name (50ohm -> c2/50ohm/c2_results.npz, matched ->
c2/matched/c2_results_matched.npz; keys t, V_NE, V_FE, I_*), both from PML runs with the same grid step, wires and source.
The reference is the change of the analytic line: the same wires (physical radius) over the plane of case B and inside the
cylinder of case C, each with its own terminals. The offset of the thin-wire model against the analytic line (about +6 % in L'
at 35 mm) is the same in B and C, so it cancels in this difference: the B -> C change is the clean test of the wall.
Tutorial 5.2 asks for 'a few percent': BC_WARN_PCT points on the peak change and BC_WARN_DB dB on |H| are WARN, the FAIL limits
(BC_FAIL_PCT, BC_FAIL_DB) only catch a leak in the shell or a wall at the wrong height.
Matched case: the analytic FE is about 0 in B and in C, so only NE is compared; FE is reported in mV.
"""
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from c3_analytic import fft_setup, mtl_time, mtl_transfer
from c3_calibration import peak, peak_in
from c3_common import B_RESULTS, BAND, BC_FAIL_DB, BC_FAIL_PCT, BC_WARN_DB, BC_WARN_PCT, C0, H_PLANE, find_file, zc_wire_plane


def find_b(here, case):
    """Where case B left its results: C2_RESULTS (file path), else ../c2/<case>/ next to the c3 folder (testData/cases/velectra/c2),
    else c2/<case>/ inside it, else the case folder itself."""
    env = os.environ.get("C2_RESULTS")
    if env and Path(env).is_file():
        return Path(env)
    here, name = Path(here), B_RESULTS[case.matched]
    for root in (here.parent / "c2" / case.name, here / "c2" / case.name, case.WORK):
        if root.is_dir():
            f = find_file(root, name)
            if f is not None:
                return f
    return None


def load_b(here, case, log):
    """Case B results of the same case name, or None (WARN) if they are found nowhere (see find_b)."""
    f = find_b(here, case)
    if f is None:
        log.check(False, f"case B results ({B_RESULTS[case.matched]}) found for the B -> C comparison",
                  f"not found in {Path(here).parent / 'c2' / case.name}, {Path(here) / 'c2' / case.name} or C2_RESULTS; "
                  f"set C2_RESULTS=/path/to/{B_RESULTS[case.matched]}", "WARN")
        return None
    log.info("case B results", f"{f}")
    d = np.load(f)
    return SimpleNamespace(path=f, t=d["t"], V_NE=d["V_NE"], V_FE=d["V_FE"], I_agg_near=d["I_agg_near"])


def _vpeaks(R, cur, t):
    vne, vfe = -R[2] * cur["vic_near"], R[3] * cur["vic_far"]
    return peak(t, vne)[1], -peak(t, -vfe)[1]


def compare_with_b(case, cal, xt, tf, t, b, log):
    """Peak change B -> C, seen impedance, and transfer-function change, FDTD against the analytic line.
    Returns None without B. Call after compare_transfer: tf.band then excludes the line-resonance nulls."""
    if b is None:
        return None
    c = case
    dB = lambda x: 20 * np.log10(np.abs(x) + 1e-30)
    # case B terminals: 50 ohm in the real experiment (the same as C); in the matched case each geometry has its own Zc
    R_B = c.R if not c.matched else (round(zc_wire_plane(H_PLANE, c.radius), 2),) * 4
    cb = SimpleNamespace(source=c.source, length=c.length, R=R_B)
    Lb, Lc = c.L_plane[c.primary], c.L[c.primary]

    # --- peaks
    b_ne, b_fe = peak(b.t, b.V_NE)[1], -peak(b.t, -b.V_FE)[1]
    f_ne, f_fe = xt.Vp[0] / b_ne - 1, xt.Vp[1] / b_fe - 1
    pb, pc = _vpeaks(R_B, mtl_time(cb, Lb, b.t), b.t), _vpeaks(c.R, mtl_time(c, Lc, t), t)
    a_ne, a_fe = pc[0] / pb[0] - 1, pc[1] / pb[1] - 1
    print("\nB -> C change                         NE peak B -> C (mV)        change     FE peak B -> C (mV)        change")
    print(f"  {'semba-fdtd':34s} {b_ne * 1e3:+8.3f} -> {xt.Vp[0] * 1e3:+8.3f}   {100 * f_ne:+7.2f} %   "
          f"{b_fe * 1e3:+8.3f} -> {xt.Vp[1] * 1e3:+8.3f}   {100 * f_fe:+7.2f} %")
    print(f"  {'analytic, ' + c.primary:34s} {pb[0] * 1e3:+8.3f} -> {pc[0] * 1e3:+8.3f}   {100 * a_ne:+7.2f} %   "
          f"{pb[1] * 1e3:+8.3f} -> {pc[1] * 1e3:+8.3f}   {100 * a_fe:+7.2f} %")
    log.info("B -> C change of the NE peak", f"FDTD {b_ne * 1e3:+.3f} -> {xt.Vp[0] * 1e3:+.3f} mV ({100 * f_ne:+.2f} %); analytic "
             f"{pb[0] * 1e3:+.3f} -> {pc[0] * 1e3:+.3f} mV ({100 * a_ne:+.2f} %); difference {100 * (f_ne - a_ne):+.2f} points")
    d_ne = abs(f_ne - a_ne) * 100
    log.check(d_ne < BC_WARN_PCT, f"B -> C change of the NE peak within {BC_WARN_PCT:g} points of the analytic change",
              f"FDTD {100 * f_ne:+.2f} % vs analytic {100 * a_ne:+.2f} % (difference {100 * (f_ne - a_ne):+.2f} points)", "WARN")
    log.check(d_ne < BC_FAIL_PCT, f"B -> C change of the NE peak within {BC_FAIL_PCT:g} points of the analytic change (gross-error band)",
              f"difference {100 * (f_ne - a_ne):+.2f} points")
    if c.matched:
        log.info("FE peak in B and in C (matched: the analytic FE is about 0)",
                 f"FDTD {b_fe * 1e3:+.3f} -> {xt.Vp[1] * 1e3:+.3f} mV; analytic {pb[1] * 1e3:+.3f} -> {pc[1] * 1e3:+.3f} mV")
    else:
        log.info("B -> C change of the FE peak", f"FDTD {b_fe * 1e3:+.3f} -> {xt.Vp[1] * 1e3:+.3f} mV ({100 * f_fe:+.2f} %); analytic "
                 f"{pb[1] * 1e3:+.3f} -> {pc[1] * 1e3:+.3f} mV ({100 * a_fe:+.2f} %); difference {100 * (f_fe - a_fe):+.2f} points")
        d_fe = abs(f_fe - a_fe) * 100
        log.check(d_fe < BC_WARN_PCT, f"B -> C change of the FE peak within {BC_WARN_PCT:g} points of the analytic change",
                  f"FDTD {100 * f_fe:+.2f} % vs analytic {100 * a_fe:+.2f} % (difference {100 * (f_fe - a_fe):+.2f} points)", "WARN")
        log.check(d_fe < BC_FAIL_PCT, f"B -> C change of the FE peak within {BC_FAIL_PCT:g} points of the analytic change (gross-error band)",
                  f"difference {100 * (f_fe - a_fe):+.2f} points")

    # --- impedance seen by the source (V_S / I - R_S): the change B -> C, as in the analytic Zc of one wire
    tp, Ip = peak_in(b.t, b.I_agg_near, 0.0, c.t0 + c.T)
    zc_b = float(c.source(tp)) / Ip - R_B[0]
    zc_c = cal.Zc_meas
    z_an = Lc[0, 0] / Lb[0, 0]
    log.info("impedance seen by the source, B -> C",
             f"semba-fdtd {zc_b:.1f} -> {zc_c:.1f} ohm ({100 * (zc_c / zc_b - 1):+.2f} %); analytic one wire "
             f"{C0 * Lb[0, 0]:.1f} -> {C0 * Lc[0, 0]:.1f} ohm ({100 * (z_an - 1):+.2f} %)")

    # --- transfer function
    nfb = fft_setup(b.t)[0]
    fb = fft_setup(b.t)[1]
    S = np.fft.rfft(c.source(b.t), nfb)
    HB = {"NE": np.fft.rfft(b.V_NE, nfb) / np.where(S == 0, 1e-30, S), "FE": np.fft.rfft(b.V_FE, nfb) / np.where(S == 0, 1e-30, S)}
    band = tf.band
    f = tf.f[band]
    Hb_an, Hc_an = mtl_transfer(cb, Lb, f), mtl_transfer(c, Lc, f)
    for i, n in enumerate(("NE", "FE")):
        if c.matched and n == "FE":
            continue
        d_fd = dB(tf.H[n][band]) - np.interp(f, fb, dB(HB[n]))
        d_an = dB(Hc_an[i]) - dB(Hb_an[i])
        dev = d_fd - d_an
        log.info(f"B -> C change of |H_{n}| (dB)", f"{BAND[0] / 1e6:g}-{BAND[1] / 1e6:g} MHz: FDTD mean {d_fd.mean():+.2f}, "
                 f"analytic mean {d_an.mean():+.2f}; FDTD minus analytic: mean {dev.mean():+.2f}, max {abs(dev).max():.2f} dB at "
                 f"{f[np.argmax(abs(dev))] / 1e6:.1f} MHz")
        log.check(abs(dev).max() < BC_WARN_DB, f"B -> C change of |H_{n}| within {BC_WARN_DB:g} dB of the analytic change "
                  f"({BAND[0] / 1e6:g}-{BAND[1] / 1e6:g} MHz, resonance nulls skipped)", f"max {abs(dev).max():.2f} dB", "WARN")
        log.check(abs(dev).max() < BC_FAIL_DB, f"B -> C change of |H_{n}| within {BC_FAIL_DB:g} dB of the analytic change (gross-error band)",
                  f"max {abs(dev).max():.2f} dB")
    return SimpleNamespace(t=b.t, V_NE=b.V_NE, V_FE=b.V_FE, f=fb, H=HB, dpk=(f_ne, f_fe), dpk_mtl=(a_ne, a_fe))
