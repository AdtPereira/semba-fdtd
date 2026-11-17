"""B -> C comparison: what changes when the ground plane becomes the closed PEC cylinder.

The case B FDTD numbers come from c2_results.npz (keys t, V_NE, V_FE, I_*). The expected change is the analytic one:
same wires (r_eff from case A), plane of case B against the cylinder of case C. Tutorial 5.2: about -10 % in NE and
FE (7.10 / 7.89 mV and 10.22 / 11.42 mV with r_eff = 0.75 r); a larger difference points at the staircase wall or at a
leak in the shell. 'A few percent' is taken here as BC_TOL_PCT = 3 points on the change, and BC_TOL_DB = 1.5 dB on
the transfer function.
"""
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from c3_analytic import fft_setup, mtl_time, mtl_transfer
from c3_calibration import peak
from c3_common import B_RESULTS, BAND, BC_TOL_DB, BC_TOL_PCT, find_file


def find_b(here):
    """Where case B left c2_results.npz: C2_RESULTS (file path), else the c3 folder, else the sibling folder ../c2/
    (testData/cases/velectra/c2 next to .../c3), so that no copy is needed."""
    env = os.environ.get("C2_RESULTS")
    if env and Path(env).is_file():
        return Path(env)
    here = Path(here)
    for root in (here, here.parent / "c2"):
        if root.is_dir():
            f = find_file(root, B_RESULTS)
            if f is not None:
                return f
    return None


def load_b(here, log):
    """Case B results, or None (WARN) if c2_results.npz is found nowhere (see find_b)."""
    f = find_b(here)
    if f is None:
        log.check(False, f"case B results ({B_RESULTS}) found for the B -> C comparison",
                  f"not found in {here}, in {Path(here).parent / 'c2'} or in C2_RESULTS; "
                  f"set C2_RESULTS=/path/to/c2_results.npz", "WARN")
        return None
    log.info("case B results", f"{f}")
    d = np.load(f)
    return SimpleNamespace(path=f, t=d["t"], V_NE=d["V_NE"], V_FE=d["V_FE"])


def _vpeaks(case, cur, t):
    vne, vfe = -case.R[2] * cur["vic_near"], case.R[3] * cur["vic_far"]
    return peak(t, vne)[1], -peak(t, -vfe)[1]


def compare_with_b(case, xt, tf, t, b, log):
    """Peak change B -> C and transfer-function change, FDTD against the analytic MTL. Returns None without B.
    Call after compare_transfer: tf.band then excludes the line-resonance nulls."""
    if b is None:
        return None
    c = case
    dB = lambda x: 20 * np.log10(np.abs(x) + 1e-30)
    # --- peaks
    b_ne, b_fe = peak(b.t, b.V_NE)[1], -peak(b.t, -b.V_FE)[1]
    f_ne, f_fe = xt.Vp[0] / b_ne - 1, xt.Vp[1] / b_fe - 1
    print("\nB -> C change                                 NE peak B -> C (mV)         change     FE peak B -> C (mV)         change")
    print(f"  {'FDTD':40s} {b_ne * 1e3:+8.3f} -> {xt.Vp[0] * 1e3:+8.3f}   {100 * f_ne:+7.2f} %   "
          f"{b_fe * 1e3:+8.3f} -> {xt.Vp[1] * 1e3:+8.3f}   {100 * f_fe:+7.2f} %")
    an = {}
    for k in c.refs:
        pb = _vpeaks(c, mtl_time(c, c.L_plane[k], t), t)
        pc = _vpeaks(c, mtl_time(c, c.L[k], t), t)
        an[k] = (pb, pc, pc[0] / pb[0] - 1, pc[1] / pb[1] - 1)
        print(f"  MTL {k:36s} {pb[0] * 1e3:+8.3f} -> {pc[0] * 1e3:+8.3f}   {100 * an[k][2]:+7.2f} %   "
              f"{pb[1] * 1e3:+8.3f} -> {pc[1] * 1e3:+8.3f}   {100 * an[k][3]:+7.2f} %")
    a_ne, a_fe = an[c.primary][2], an[c.primary][3]
    log.check(abs(f_ne - a_ne) * 100 < BC_TOL_PCT, f"B -> C change of the NE peak matches the analytic change within "
              f"{BC_TOL_PCT:g} points ({c.primary})", f"FDTD {100 * f_ne:+.2f} % vs MTL {100 * a_ne:+.2f} % "
              f"(difference {100 * (f_ne - a_ne):+.2f} points)")
    log.check(abs(f_fe - a_fe) * 100 < BC_TOL_PCT, f"B -> C change of the FE peak matches the analytic change within "
              f"{BC_TOL_PCT:g} points ({c.primary})", f"FDTD {100 * f_fe:+.2f} % vs MTL {100 * a_fe:+.2f} % "
              f"(difference {100 * (f_fe - a_fe):+.2f} points)")
    log.check(-0.14 < f_ne < -0.06 and -0.14 < f_fe < -0.06, "B -> C change is about -10 % in NE and FE (tutorial 5.2)",
              f"NE {100 * f_ne:+.1f} %, FE {100 * f_fe:+.1f} % (expected about -10.0 / -10.5 %)", "WARN")

    # --- transfer function
    fb, nfb = fft_setup(b.t)[1], fft_setup(b.t)[0]
    S = np.fft.rfft(c.source(b.t), nfb)
    HB = {"NE": np.fft.rfft(b.V_NE, nfb) / np.where(S == 0, 1e-30, S), "FE": np.fft.rfft(b.V_FE, nfb) / np.where(S == 0, 1e-30, S)}
    band = tf.band
    f = tf.f[band]
    Hb_an, Hc_an = mtl_transfer(c, c.L_plane[c.primary], f), mtl_transfer(c, c.L[c.primary], f)
    for i, n in enumerate(("NE", "FE")):
        d_fd = dB(tf.H[n][band]) - np.interp(f, fb, dB(HB[n]))
        d_an = dB(Hc_an[i]) - dB(Hb_an[i])
        dev = d_fd - d_an
        log.info(f"B -> C change of |H_{n}| (dB)", f"{BAND[0] / 1e6:g}-{BAND[1] / 1e6:g} MHz: FDTD mean {d_fd.mean():+.2f}, "
                 f"MTL mean {d_an.mean():+.2f}; FDTD minus MTL: mean {dev.mean():+.2f}, max {abs(dev).max():.2f} dB at "
                 f"{f[np.argmax(abs(dev))] / 1e6:.1f} MHz")
        log.check(abs(dev).max() < BC_TOL_DB, f"B -> C change of |H_{n}| matches the analytic change within {BC_TOL_DB:g} dB "
                  f"({BAND[0] / 1e6:g}-{BAND[1] / 1e6:g} MHz, resonance nulls skipped)", f"max {abs(dev).max():.2f} dB", "WARN")
    return SimpleNamespace(t=b.t, V_NE=b.V_NE, V_FE=b.V_FE, f=fb, H=HB, dpk=(f_ne, f_fe), dpk_mtl=(a_ne, a_fe))
