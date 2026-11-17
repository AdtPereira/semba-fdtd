"""Six-panel figure: aggressor currents, NE/FE voltages, residual against the references, transfer function (to the generator
and to the line-input node, Paul Fig. D.30) and the ring-down."""
import matplotlib

import matplotlib.pyplot as plt
import numpy as np

from paul_d28_analytic import mtl_transfer
from paul_d28_common import BAND, LATE_START, ROLES


def plot_case(case, cal, xt, tf, tfn, curves, t, I, out_png, show=True):
    c = case
    names = c.refs_names
    book, eff, prim = names
    cur_b, cur_e, cur_p = curves[book], curves[eff], curves[prim]
    dB = lambda x: 20 * np.log10(np.abs(x) + 1e-30)
    vne = lambda cur: -1e3 * c.R[2] * cur["vic_near"]
    vfe = lambda cur: 1e3 * c.R[3] * cur["vic_far"]
    tn = 1e9 * t
    fig, ax = plt.subplots(6, 1, figsize=(8, 20))

    a = ax[0]
    a.plot(tn, 1e3 * cur_p["agg_near"], "k", lw=2, label="MTL near (primary)")
    a.plot(tn, 1e3 * cur_p["agg_far"], "gray", lw=2, label="MTL far (primary)")
    a.plot(tn, 1e3 * cur_b["agg_near"], "k:", lw=1.2, label="MTL near, book (nominal r)")
    a.plot(tn, 1e3 * I["agg_near"], "C0--", label="semba-fdtd near")
    a.plot(tn, 1e3 * I["agg_far"], "C1--", label="semba-fdtd far")
    a.set_ylabel("I (mA)")
    a.set_title(f"Generator-wire currents; delay {cal.delay * 1e9:.2f} ns, Zin = {cal.Zin:.1f} ohm (r_eff = {cal.r_meas / c.radius:.3f} r)")

    a = ax[1]
    a.plot(tn, vne(cur_b), "k:", lw=1.5, label="MTL NE, book (nominal r)")
    a.plot(tn, vfe(cur_b), ":", color="gray", lw=1.5, label="MTL FE, book (nominal r)")
    a.plot(tn, vne(cur_p), "k", lw=1.5, label="MTL NE, r_eff + risers (primary)")
    a.plot(tn, vfe(cur_p), "gray", lw=1.5, label="MTL FE, r_eff + risers (primary)")
    a.plot(tn, 1e3 * xt.V["NE"], "C0--", label="semba-fdtd NE")
    a.plot(tn, 1e3 * xt.V["FE"], "C1--", label="semba-fdtd FE")
    a.set_ylabel("V (mV)")
    a.set_title(f"NE {xt.plat[0] * 1e3:+.1f} mV, FE {xt.plat[1] * 1e3:+.1f} mV on the first plateaus (book: +23.3 / -32.0)")

    a = ax[2]
    a.plot(tn, 1e3 * xt.V["NE"] - vne(cur_p), "C0", label="NE: FDTD - primary MTL")
    a.plot(tn, 1e3 * xt.V["FE"] - vfe(cur_p), "C1", label="FE: FDTD - primary MTL")
    a.plot(tn, 1e3 * xt.V["NE"] - vne(cur_e), "C0", ls=":", lw=1.2, label="NE: FDTD - MTL without risers")
    a.plot(tn, 1e3 * xt.V["FE"] - vfe(cur_e), "C1", ls=":", lw=1.2, label="FE: FDTD - MTL without risers")
    a.axhline(0, color="k", lw=0.5)
    a.set_ylabel("difference (mV)")
    a.set_title("Residual of the FDTD against the MTL (the dotted curves show what the risers explain)")

    a = ax[3]
    fr = np.linspace(1e6, 200e6, 500)
    Hp, Hb = mtl_transfer(c, c.refs[prim], fr), mtl_transfer(c, c.refs[book], fr)
    a.plot(1e-6 * fr, dB(Hb[0]), "k:", lw=1.2, label="MTL NE, book (infinite record)")
    a.plot(1e-6 * fr, dB(Hb[1]), ":", color="gray", lw=1.2, label="MTL FE, book (infinite record)")
    a.plot(1e-6 * fr, dB(Hp[0]), "k", lw=1.2, label="MTL NE, primary (infinite record)")
    a.plot(1e-6 * fr, dB(Hp[1]), "gray", lw=1.2, label="MTL FE, primary (infinite record)")
    b = tf.band_all
    a.plot(1e-6 * tf.f[b], dB(tf.Hm["NE"][b]), "C2-.", lw=1, label="MTL NE, primary, same 200 ns record")
    a.plot(1e-6 * tf.f[b], dB(tf.Hm["FE"][b]), "C3-.", lw=1, label="MTL FE, primary, same 200 ns record")
    a.plot(1e-6 * tf.f[b], dB(tf.H["NE"][b]), "C0--", label="semba-fdtd NE")
    a.plot(1e-6 * tf.f[b], dB(tf.H["FE"][b]), "C1--", label="semba-fdtd FE")
    a.set_xscale("log")
    a.set_xlim(1, 200)
    a.set_ylim(-75, -20)
    a.set_ylabel("|V / V_S| (dB)")
    a.set_title("Transfer function to the generator (source spectrum above -40 dB)")

    a = ax[4]
    fr = np.logspace(3, 8, 500)
    Hpn, Hbn = mtl_transfer(c, c.refs[prim], fr, node=True), mtl_transfer(c, c.refs[book], fr, node=True)
    a.plot(1e-6 * fr, dB(Hbn[0]), "k:", lw=1.5, label="MTL NE, book")
    a.plot(1e-6 * fr, dB(Hbn[1]), ":", color="gray", lw=1.5, label="MTL FE, book")
    a.plot(1e-6 * fr, dB(Hpn[0]), "k", lw=1.2, label="MTL NE, primary")
    a.plot(1e-6 * fr, dB(Hpn[1]), "gray", lw=1.2, label="MTL FE, primary")
    b = tfn.band
    a.plot(1e-6 * tf.f[b], dB(tfn.H["NE"][b]), "C0--", label="semba-fdtd NE")
    a.plot(1e-6 * tf.f[b], dB(tfn.H["FE"][b]), "C1--", label="semba-fdtd FE")
    a.set_xscale("log")
    a.set_xlim(1e-3, 100)
    a.set_ylim(-110, -20)
    a.set_ylabel("|V / V_S node| (dB)")
    a.set_title("Relative to the line-input node, as Paul Fig. D.30 (book: -66.2 dB at 10 kHz)")

    a = ax[5]
    for k, col in zip(ROLES, ("C0", "C1", "C2", "C3")):
        a.plot(tn, dB(I[k] / np.abs(I[k]).max() + 1e-12), col, label=f"I {k}")
    a.axvline(1e9 * LATE_START, color="gray", ls=":")
    a.set_ylim(-100, 3)
    a.set_ylabel("20 log10 |I| / own peak (dB)")
    a.set_title("Ring-down of the mismatched line (the 200 ns record ends before it has died)")

    for i, a_ in enumerate(ax.flat):
        a_.grid(True, which="both", alpha=0.3)
        a_.legend(fontsize=7)
        a_.set_xlabel("f (MHz)" if i in (3, 4) else "t (ns)")
        if i in (0, 1, 2, 5):
            a_.set_xlim(0, 200)
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    if show:
        plt.show()
    plt.close(fig)
