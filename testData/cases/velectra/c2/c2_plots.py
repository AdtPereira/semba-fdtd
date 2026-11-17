"""Four-panel figure: aggressor currents, NE/FE voltages, transfer functions against the MTL, ring-down."""
import matplotlib.pyplot as plt
import numpy as np

from c2_analytic import mtl_transfer
from c2_common import LATE_START, ROLES


def plot_case(case, cal, xt, tf, curves, t, I, out_png):
    c = case
    prim = curves[c.primary]
    styles = dict(zip(c.refs, (":", "-.", "-")))
    dB = lambda x: 20 * np.log10(np.abs(x) + 1e-30)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    a = ax[0, 0]
    a.plot(1e9 * t, 1e3 * prim["agg_near"], "k", lw=2, label="MTL aggressor near")
    a.plot(1e9 * t, 1e3 * prim["agg_far"], "gray", lw=2, label="MTL aggressor far")
    a.plot(1e9 * t, 1e3 * I["agg_near"], "C0--", label="semba-fdtd near")
    a.plot(1e9 * t, 1e3 * I["agg_far"], "C1--", label="semba-fdtd far")
    a.set_xlim(0, 80)
    a.set_ylabel("I (mA)")
    a.set_title(f"Aggressor currents, R = {c.R[0]:g} ohm; delay {cal.delay * 1e9:.2f} ns, Zc = {cal.Zc_meas:.1f} ohm")

    a = ax[0, 1]
    for k, ls in styles.items():
        a.plot(1e9 * t, -1e3 * c.R[2] * curves[k]["vic_near"], "k", ls=ls, lw=1.5, label=f"MTL NE, {k}")
        a.plot(1e9 * t, 1e3 * c.R[3] * curves[k]["vic_far"], "gray", ls=ls, lw=1.5, label=f"MTL FE, {k}")
    a.plot(1e9 * t, 1e3 * xt.V["NE"], "C0--", label="semba-fdtd NE")
    a.plot(1e9 * t, 1e3 * xt.V["FE"], "C1--", label="semba-fdtd FE")
    a.set_xlim(0, 80)
    a.set_ylabel("V (mV)")
    a.set_title(f"NE = {xt.Vp[0] * 1e3:+.2f} mV, FE = {xt.Vp[1] * 1e3:+.2f} mV")

    a = ax[1, 0]
    fr = np.linspace(1e6, 200e6, 400)
    for k, ls in styles.items():
        Hk = mtl_transfer(c, c.L[k], fr)
        a.plot(1e-6 * fr, dB(Hk[0]), "k", ls=ls, label=f"MTL NE, {k}")
        a.plot(1e-6 * fr, dB(Hk[1]), "gray", ls=ls, label=f"MTL FE, {k}")
    b = tf.band_all
    a.plot(1e-6 * tf.f[b], dB(tf.H["NE"][b]), "C0--", label="semba-fdtd NE")
    a.plot(1e-6 * tf.f[b], dB(tf.H["FE"][b]), "C1--", label="semba-fdtd FE")
    a.set_xscale("log")
    a.set_xlim(1, 200)
    a.set_ylim(-70, -20)
    a.set_ylabel("|V / V_S| (dB)")
    a.set_xlabel("f (MHz)")
    a.set_title("Transfer function (source spectrum above -40 dB)")

    a = ax[1, 1]
    for k, col in zip(ROLES, ("C0", "C1", "C2", "C3")):
        a.plot(1e9 * t, dB(I[k] / np.abs(I[k]).max() + 1e-12), col, label=f"I {k}")
    a.axvline(1e9 * LATE_START, color="gray", ls=":")
    a.set_ylim(-100, 3)
    a.set_ylabel("20 log10 |I| / own peak (dB)")
    a.set_title("Ring-down of the mismatched line")

    for i, a_ in enumerate(ax.flat):
        a_.grid(True, which="both", alpha=0.3)
        a_.legend(fontsize=7)
        if i != 2:
            a_.set_xlabel("t (ns)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.show()
