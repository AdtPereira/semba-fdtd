"""Two figures of case A.

plot_case:     two panels - semba-fdtd currents against the analytic line, and |I| in dB.
plot_near_end: three panels - near-end current against the main pulse, the same with a vertical zoom, and their difference
               (echo and the small shunt-capacitance effect before it).
Style in every panel: semba-fdtd = solid line (near C0, far C1); analytic / reference = dotted (near black, far gray).
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_case(case, cal, echo, t, I_n, I_f, curves, out_png):
    c = case
    fig, ax = plt.subplots(2, 1, figsize=(8, 7))
    ref_n, ref_f = curves["analytic"]
    ax[0].plot(1e9 * t, 1e3 * I_n, "C0", lw=1.5, label="semba-fdtd near")
    ax[0].plot(1e9 * t, 1e3 * I_f, "C1", lw=1.5, label="semba-fdtd far")
    ax[0].plot(1e9 * t, 1e3 * ref_n, "k:", lw=2, label=f"analytic, physical radius (Zc = {c.Zc_r:.1f} ohm) near")
    ax[0].plot(1e9 * t, 1e3 * ref_f, ":", color="gray", lw=2, label="analytic far")
    ax[0].set_ylabel("I (mA)")
    ax[0].set_title(f"Case A (h = {c.h * 1e3:g} mm): R_S = R_L = {c.R_S:g} ohm; "
                    f"delay {cal.delay * 1e9:.2f} ns (L/c = {c.T * 1e9:.2f} ns)", fontsize=10)

    ax[1].plot(1e9 * t, 20 * np.log10(np.abs(I_n) / cal.Ip_n + 1e-12), "C0", lw=1.5, label="semba-fdtd near")
    ax[1].plot(1e9 * t, 20 * np.log10(np.abs(I_f) / cal.Ip_f + 1e-12), "C1", lw=1.5, label="semba-fdtd far")
    ax[1].plot(1e9 * t, 20 * np.log10(np.abs(ref_n) / cal.Ip_n + 1e-12), "k:", lw=2, label="analytic near")
    ax[1].plot(1e9 * t, 20 * np.log10(np.abs(ref_f) / cal.Ip_f + 1e-12), ":", color="gray", lw=2, label="analytic far")
    ax[1].set_ylim(-100, 3)
    ax[1].set_ylabel("20 log10(|I| / I_peak) (dB)")
    for a_ in ax:
        a_.grid(True, which="both", alpha=0.3)
        a_.legend()
        a_.set_xlabel("t (ns)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.show()


def plot_near_end(case, cal, echo, t, I_n, out_png):
    """Near-end current of semba-fdtd against the main pulse (the source scaled to the first-pass peak), and their difference."""
    c = case
    tn = 1e9 * t
    main = I_n - echo.resid                          # = Ip_n * source / s_pk (see echo_analysis)
    fig, ax = plt.subplots(3, 1, figsize=(8, 10))
    ax[0].plot(tn, 1e3 * I_n, "C0", lw=1.5, label="semba-fdtd near, I(t)")
    ax[0].plot(tn, 1e3 * main, "k:", lw=2, label="main pulse = source scaled to the semba peak")
    ax[0].set_ylabel("I (mA)")
    ax[0].set_title("Near-end current and main pulse: they overlap at this scale", fontsize=10)

    ax[1].plot(tn, 1e3 * I_n, "C0", lw=1.5, label="semba-fdtd near")
    ax[1].plot(tn, 1e3 * main, "k:", lw=2, label="main pulse")
    ax[1].set_ylim(-0.03, 0.25 * 1e3 * cal.Ip_n)
    ax[1].set_ylabel("I (mA), vertical zoom")
    ax[1].set_title("Same curves, zoomed on the flanks: the echo opens a gap after the main pulse", fontsize=10)

    ax[2].plot(tn, 100 * echo.resid / cal.Ip_n, "C0", lw=1.5, label="difference: semba-fdtd - main pulse")
    if echo.c_eq:
        ax[2].plot(tn, 100 * echo.fit / cal.Ip_n, "k:", lw=2, label=f"C dV_S/dt fitted before the echo (C = {echo.c_eq * 1e15:.0f} fF)")
        ax[2].axvspan(tn[echo.fit_win][0], tn[echo.fit_win][-1], color="C2", alpha=0.07)
        ax[2].text(tn[echo.fit_win][0] + 0.3, -2.7, "fitted window", color="C2", fontsize=8)
    ax[2].axvline(1e9 * echo.t_echo, color="gray", ls=":")
    ax[2].text(1e9 * echo.t_echo + 0.2, -2.7, "t0 + 2 delay", color="gray", fontsize=8)
    ax[2].set_ylim(-3, 5)
    ax[2].set_ylabel("% of the main peak")
    ax[2].set_title("Difference only, magnified", fontsize=10)
    for a_ in ax:
        a_.grid(True, which="both", alpha=0.3)
        a_.set_xlabel("t (ns)")
    ax[0].legend(loc="upper right")
    ax[1].legend(loc="upper right")
    ax[2].legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.show()
