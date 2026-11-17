"""Three-panel figure: currents against the analytic line, |I| in dB, near-end echo with its monopole + doublet fit."""
import matplotlib.pyplot as plt
import numpy as np


def plot_case(case, cal, echo, t, I_n, I_f, curves, out_png):
    c = case
    fig, ax = plt.subplots(3, 1, figsize=(8, 10))
    ref_n, ref_f = curves["r_eff"]
    ax[0].plot(1e9 * t, 1e3 * ref_n, "k", lw=2, label="line, Zc(r_eff) near")
    ax[0].plot(1e9 * t, 1e3 * ref_f, "gray", lw=2, label="line, Zc(r_eff) far")
    ax[0].plot(1e9 * t, 1e3 * I_n, "C0--", label="semba-fdtd near")
    ax[0].plot(1e9 * t, 1e3 * I_f, "C1--", label="semba-fdtd far")
    ax[0].set_ylabel("I (mA)")
    ax[0].set_title(f"Case A: R_S = {c.R_S:g}, R_L = {c.R_L:g} ohm; Zc_eff = {cal.Zc_meas:.1f} ohm, "
                    f"delay {cal.delay * 1e9:.2f} ns")

    ax[1].plot(1e9 * t, 20 * np.log10(np.abs(I_n) / cal.Ip_n + 1e-12), "C0", label="near")
    ax[1].plot(1e9 * t, 20 * np.log10(np.abs(I_f) / cal.Ip_f + 1e-12), "C1", label="far")
    ax[1].plot(1e9 * t, 20 * np.log10(np.abs(ref_n) / cal.Ip_n + 1e-12), "k:", label="line near")
    ax[1].set_ylim(-100, 3)
    ax[1].set_ylabel("20 log10 |I| / I_peak (dB)")

    ax[2].plot(1e9 * t, 100 * echo.resid / cal.Ip_n, "C0", label="near-end residual (FDTD - main pulse)")
    if echo.alpha is not None:
        ax[2].plot(1e9 * t[echo.win], 100 * echo.fit[echo.win] / cal.Ip_n, "r--", label="monopole + doublet fit")
    ax[2].axvline(1e9 * echo.t_echo, color="gray", ls=":")
    ax[2].set_xlim(1e9 * max(t[0], echo.t_echo - 6 * c.sigma), 1e9 * t[-1])
    ax[2].set_ylim(-3, 5)
    ax[2].set_ylabel("% of the main peak")
    for a_ in ax:
        a_.grid(True, which="both", alpha=0.3)
        a_.legend()
        a_.set_xlabel("t (ns)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.show()
