"""Two figures of case B, each with two vertical panels (same style as c1):
plot_time: aggressor currents (mA) and NE/FE voltages (mV) against the analytic MTL.
plot_db:   transfer functions |V/V_S| (dB) against the MTL, and the ring-down of the four currents (dB).
Style in every panel: semba-fdtd = solid line (near / NE C0, far / FE C1); analytic MTL, physical radius = dotted
(near / NE black, far / FE gray). The ring-down panel has no reference: the four currents are C0..C3."""
import matplotlib.pyplot as plt
import numpy as np

from c2_analytic import mtl_transfer
from c2_common import LATE_START, ROLES


def plot_time(case, cal, xt, curves, t, I, out_png):
    c = case
    prim = curves[c.primary]
    xmax = min(80.0, 1e9 * t[-1])
    fig, ax = plt.subplots(2, 1, figsize=(8, 7))

    a = ax[0]
    a.plot(1e9 * t, 1e3 * I["agg_near"], "C0", lw=1.5, label="semba-fdtd near")
    a.plot(1e9 * t, 1e3 * I["agg_far"], "C1", lw=1.5, label="semba-fdtd far")
    a.plot(1e9 * t, 1e3 * prim["agg_near"], "k:", lw=2, label=f"analytic near")
    a.plot(1e9 * t, 1e3 * prim["agg_far"], ":", color="gray", lw=2, label="analytic far")
    a.set_ylabel("I (mA)")
    a.set_title(f"Case B, {c.name}: aggressor currents, R = {c.R[0]:g} ohm; "
                f"delay {cal.delay * 1e9:.2f} ns, Zc = {cal.Zc_meas:.1f} ohm", fontsize=10)

    a = ax[1]
    a.plot(1e9 * t, 1e3 * xt.V["NE"], "C0", lw=1.5, label="semba-fdtd NE")
    a.plot(1e9 * t, 1e3 * xt.V["FE"], "C1", lw=1.5, label="semba-fdtd FE")
    a.plot(1e9 * t, -1e3 * c.R[2] * prim["vic_near"], "k:", lw=2, label="analytic NE")
    a.plot(1e9 * t, 1e3 * c.R[3] * prim["vic_far"], ":", color="gray", lw=2, label="analytic FE")
    a.set_ylabel("V (mV)")
    a.set_title(f"Victim: NE = {xt.Vp[0] * 1e3:+.2f} mV, FE = {xt.Vp[1] * 1e3:+.2f} mV", fontsize=10)

    for a_ in ax:
        a_.set_xlim(0, xmax)
        a_.set_xlabel("t (ns)")
        a_.grid(True, which="both", alpha=0.3)
        a_.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.show()


def plot_db(case, tf, curves, t, I, out_png):
    c = case
    dB = lambda x: 20 * np.log10(np.abs(x) + 1e-30)
    fig, ax = plt.subplots(2, 1, figsize=(8, 7))

    a = ax[0]
    b = tf.band_all
    a.plot(1e-6 * tf.f[b], dB(tf.H["NE"][b]), "C0", lw=1.5, label="semba-fdtd NE")
    a.plot(1e-6 * tf.f[b], dB(tf.H["FE"][b]), "C1", lw=1.5, label="semba-fdtd FE")
    fr = np.linspace(1e6, 200e6, 400)
    Hk = mtl_transfer(c, c.L[c.primary], fr)
    a.plot(1e-6 * fr, dB(Hk[0]), "k:", lw=2, label="analytic NE")
    a.plot(1e-6 * fr, dB(Hk[1]), ":", color="gray", lw=2, label="analytic FE")
    a.set_xscale("log")
    a.set_xlim(1, 200)
    a.set_ylim(-70, -10)
    a.set_xlabel("f (MHz)")
    a.set_ylabel("|V / V_S| (dB)")
    a.set_title("Transfer function (source spectrum above -40 dB)", fontsize=10)

    a = ax[1]
    for k, col in zip(ROLES, ("C0", "C1", "C2", "C3")):
        a.plot(1e9 * t, dB(I[k] / np.abs(I[k]).max() + 1e-12), col, lw=1.5, label=f"semba-fdtd I {k}")
    if not c.matched:
        a.axvline(1e9 * LATE_START, color="gray", ls=":")
    a.set_ylim(-100, 3)
    a.set_xlabel("t (ns)")
    a.set_ylabel("20 log10(|I| / own peak) (dB)")
    a.set_title("Ring-down of the mismatched line" if not c.matched
                else "Currents re own peak (matched ends: no ring-down)", fontsize=10)

    for a_ in ax:
        a_.grid(True, which="both", alpha=0.3)
        a_.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.show()
