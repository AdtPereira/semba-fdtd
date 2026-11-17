# %% Velectra cross-talk experiment, case C (c3): SMOKE TEST before the long run (about 7 min instead of about 1 h).
#
# Why: a defective case C (wall at the wrong cell, leak in the shell, wrong sign, wrong wire impedance, ...) shows up in
# the FIRST 35 ns: the aggressor pulse, the NE peak (15 ns) and the FE peak (21.7 ns) are all over by about 32 ns. This
# script writes a copy of c3.fdtd.json with only the first 35 ns (c3_smoke.fdtd.json, same dt, same everything else),
# runs it, and checks what is already known at that time against the same references as the full run:
#   * aggressor first pass: pulse centre on the source, delay = L/c, Zc effective (about 308 ohm), probe sign;
#   * NE positive and FE negative, their peaks against the analytic cylinder MTL (within 10 %, WARN above 5 %);
#   * waveforms in time with the MTL (cross-correlation lag), RMS against the MTL;
#   * the B -> C change of the NE and FE peaks against the analytic change (within 3 points; about -10 %).
# What it does NOT cover: late-time instability, the ring-down after 175 ns and the transfer function. For those,
# watch the full run (the 'stop' file stops it cleanly and the probes are flushed) or read its first 100 ns.
#
# Usage (from anywhere, or cell by cell in VS Code):
#   python c3_smoke.py             check c3.fdtd.json, write the 35 ns copy, run it, analyse, give the verdict
#   python c3_smoke.py --no-run    analyse a smoke run already made (the .dat files of c3_smoke.fdtd)
# Environment: SEMBA_EXE, REPO_ROOT, C3_NO_RUN=1 (same as --no-run), C3_MCELLS_PER_S, C2_RESULTS (case B results),
#              C3_SMOKE_NS (length of the smoke run in ns, default 35; the FE pulse needs at least 32).
# Needs next to it: c3.fdtd.json, gauss_sigma2p5ns_t15ns.exc, the c3_*.py modules and c2_results.npz (see c3_bc.py).
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE))                         # the c3_*.py modules

from c3_analytic import mtl_time, xcorr_lag
from c3_bc import load_b
from c3_calibration import peak, peak_in, pulse_time
from c3_common import BC_TOL_PCT, ROLES, Log, repo_paths
from c3_input import load_case

FULL_CASE = "c3.fdtd.json"
SMOKE_CASE = "c3_smoke.fdtd.json"
SMOKE_NS = float(os.environ.get("C3_SMOKE_NS", "35"))
NO_RUN = "--no-run" in sys.argv or os.environ.get("C3_NO_RUN") == "1"
solver = None        # stays None when cell 2 is skipped or --no-run: the probes are then read from the .dat files


def _key(s):
    return re.sub(r"[^0-9a-z]", "", s.lower())


def make_smoke_json(case, ns=SMOKE_NS):
    """Copy of the case with numberOfSteps cut to ns nanoseconds (same timeStep, same everything else)."""
    d = json.loads(json.dumps(case.inp))
    dt_json = d["general"]["timeStep"]
    d["general"]["numberOfSteps"] = int(math.ceil(ns * 1e-9 / dt_json))
    path = case.WORK / SMOKE_CASE
    path.write_text(json.dumps(d, indent=2))
    return path, d["general"]["numberOfSteps"]


def load_smoke_currents(case, solver=None):
    """(t, {role: current}) on one uniform axis. With a solver object: through pyWrapper (c3_solver). Without: the
    .dat files whose name starts with the smoke case ('c3_smoke.fdtd_'), so that a full run of c3.fdtd never mixes in."""
    if solver is not None:
        from c3_solver import load_currents
        return load_currents(case, solver)
    prefix, raw = case.case_path.stem + "_", {}
    for role, name in case.PROBES.items():
        hits = [p for p in Path(case.HERE).rglob("*.dat")
                if p.name.startswith(prefix) and _key(name) in _key(p.stem[len(prefix):]) and "energy" not in p.name.lower()]
        assert hits, (f"no .dat file of the smoke run for probe '{name}' (expected {prefix}*.dat) under {case.HERE}; "
                      f"run without --no-run first")
        a = np.loadtxt(max(hits, key=lambda p: p.stat().st_mtime), skiprows=1)
        raw[role] = (a[:, 0], a[:, 1])
    t = raw[ROLES[0]][0]
    if not np.allclose(np.diff(t), np.diff(t)[0], rtol=1e-3):
        t = np.arange(t[0], t[-1], np.median(np.diff(t)))
    return t, {k: np.interp(t, *raw[k]) for k in ROLES}


def analyse_smoke(case, t, I, b, log):
    """Checks of the first ~35 ns; returns a namespace-like dict with the numbers for the plot."""
    c = case
    need = c.t0 + c.T + 4 * c.sigma                                   # the FE pulse (centre t0 + T) has ended
    log.check(t[-1] >= need, "smoke record holds the NE and FE pulses", f"{t[-1] * 1e9:.1f} ns, needed {need * 1e9:.1f} ns")
    log.check(all(np.all(np.isfinite(v)) and not np.all(v == 0.0) for v in I.values()), "currents finite and non-zero")

    # --- aggressor, first pass (same windows and measures as c3_calibration)
    w_n, w_f = (0.0, c.t0 + c.T), (c.t0 + 0.5 * c.T, c.t0 + 1.5 * c.T)
    tp_n, Ip_n = peak_in(t, I["agg_near"], *w_n)
    tp_f, Ip_f = peak_in(t, I["agg_far"], *w_f)
    tc_s = pulse_time(t, c.source(t), 0.0, c.t0 + 0.5 * c.T)[0]
    tc_n, fw_n = pulse_time(t, I["agg_near"], *w_n)
    tc_f, fw_f = pulse_time(t, I["agg_far"], *w_f)
    delay = tc_f - tc_n
    Zc = float(c.source(tp_n)) / Ip_n - c.R[0]
    log.info("aggressor first pass", f"near peak {Ip_n * 1e3:.4f} mA, far peak {Ip_f * 1e3:.4f} mA, pulse centres "
             f"{tc_n * 1e9:.3f} / {tc_f * 1e9:.3f} ns, delay {delay * 1e9:.3f} ns (L/c = {c.T * 1e9:.3f} ns), "
             f"Zc = {Zc:.1f} ohm (cylinder, r_eff: {c.zc_single:.1f} ohm; case B: 311.8 ohm)")
    log.check(abs(tc_n - tc_s) < 0.1e-9, "aggressor near-end pulse centre within 0.1 ns of the source",
              f"{tc_n * 1e9:.3f} vs {tc_s * 1e9:.3f} ns ({1e12 * (tc_n - tc_s):+.0f} ps)")
    log.check(abs(delay / c.T - 1) < 0.01, "aggressor delay equals L/c within 1 %", f"{100 * (delay / c.T - 1):+.2f} %")
    log.check(Ip_n > 0 and Ip_f > 0, "probe sign: positive aggressor current from near to far end")
    log.check(abs(Zc / c.zc_single - 1) < 0.05, "Zc effective within 5 % of the cylinder value (wire or wall wrong if not)",
              f"{Zc:.1f} vs {c.zc_single:.1f} ohm ({100 * (Zc / c.zc_single - 1):+.2f} %)")
    log.check(abs(Zc / c.zc_single - 1) < 0.02, "Zc effective within 2 % of the cylinder value",
              f"{100 * (Zc / c.zc_single - 1):+.2f} %", "WARN")

    # --- victim voltages
    V = {"NE": -c.R[2] * I["vic_near"], "FE": c.R[3] * I["vic_far"]}
    tp_ne, vne = peak(t, V["NE"])
    tp_fe, vfe = peak(t, -V["FE"])
    vfe = -vfe
    log.info("victim", f"NE {vne * 1e3:+.3f} mV at {tp_ne * 1e9:.2f} ns, FE {vfe * 1e3:+.3f} mV at {tp_fe * 1e9:.2f} ns")
    log.check(vne > 0 and vfe < 0, "polarity: NE positive, FE negative")

    # --- analytic cylinder MTL on the same record (causal, so a short record is exact)
    cur = mtl_time(c, c.L[c.primary], t)
    a_ne, a_fe = peak(t, -c.R[2] * cur["vic_near"])[1], -peak(t, -(c.R[3] * cur["vic_far"]))[1]
    log.info("analytic cylinder MTL", f"NE {a_ne * 1e3:+.3f} mV, FE {a_fe * 1e3:+.3f} mV ({c.primary})")
    log.check(abs(vne / a_ne - 1) < 0.10, "NE peak within 10 % of the analytic cylinder MTL",
              f"{vne * 1e3:+.3f} vs {a_ne * 1e3:+.3f} mV ({100 * (vne / a_ne - 1):+.1f} %)")
    log.check(abs(vfe / a_fe - 1) < 0.10, "FE peak within 10 % of the analytic cylinder MTL",
              f"{vfe * 1e3:+.3f} vs {a_fe * 1e3:+.3f} mV ({100 * (vfe / a_fe - 1):+.1f} %)")
    log.check(abs(vne / a_ne - 1) < 0.05 and abs(vfe / a_fe - 1) < 0.05, "NE and FE peaks within 5 % (case B: 1.1 / 1.4 %)",
              f"NE {100 * (vne / a_ne - 1):+.1f} %, FE {100 * (vfe / a_fe - 1):+.1f} %", "WARN")
    ref_ne, ref_fe = -c.R[2] * cur["vic_near"], c.R[3] * cur["vic_far"]
    lag = {n: xcorr_lag(t, x, y, 0.0, t[-1]) for n, x, y in (("NE", V["NE"], ref_ne), ("FE", V["FE"], ref_fe))}
    log.check(max(abs(v) for v in lag.values()) < 0.15e-9, "NE and FE in time with the MTL (cross-correlation lag below 0.15 ns)",
              f"NE {lag['NE'] * 1e12:+.0f} ps, FE {lag['FE'] * 1e12:+.0f} ps (case B: +68 / -133 ps)", "WARN")
    rms = lambda x, y, p: 100 * math.sqrt(np.mean((x - y) ** 2)) / abs(p)
    r = (rms(I["agg_near"], cur["agg_near"], Ip_n), rms(I["agg_far"], cur["agg_far"], Ip_f), rms(V["NE"], ref_ne, vne),
         rms(V["FE"], ref_fe, vfe))
    log.check(max(r[:2]) < 2.0, "aggressor currents within 2 % RMS of the MTL", f"near {r[0]:.2f} %, far {r[1]:.2f} %", "WARN")
    log.check(max(r[2:]) < 10.0, "NE/FE waveforms within 10 % RMS of the MTL", f"NE {r[2]:.2f} %, FE {r[3]:.2f} %", "WARN")

    # --- B -> C: the wall of the cylinder against the plane of case B, on the same record length
    dpk = None
    if b is not None:
        m = b.t <= t[-1]
        b_ne, b_fe = peak(b.t[m], b.V_NE[m])[1], -peak(b.t[m], -b.V_FE[m])[1]
        f_ne, f_fe = vne / b_ne - 1, vfe / b_fe - 1
        curp = mtl_time(c, c.L_plane[c.primary], t)
        p_ne, p_fe = peak(t, -c.R[2] * curp["vic_near"])[1], -peak(t, -(c.R[3] * curp["vic_far"]))[1]
        g_ne, g_fe = a_ne / p_ne - 1, a_fe / p_fe - 1
        print(f"\nB -> C (first {t[-1] * 1e9:.0f} ns)       NE peak B -> C (mV)        change     FE peak B -> C (mV)        change")
        print(f"  {'FDTD':26s} {b_ne * 1e3:+8.3f} -> {vne * 1e3:+8.3f}   {100 * f_ne:+7.2f} %   {b_fe * 1e3:+8.3f} -> {vfe * 1e3:+8.3f}   {100 * f_fe:+7.2f} %")
        print(f"  {'MTL (plane -> cylinder)':26s} {p_ne * 1e3:+8.3f} -> {a_ne * 1e3:+8.3f}   {100 * g_ne:+7.2f} %   {p_fe * 1e3:+8.3f} -> {a_fe * 1e3:+8.3f}   {100 * g_fe:+7.2f} %")
        log.check(abs(f_ne - g_ne) * 100 < BC_TOL_PCT, f"B -> C change of the NE peak within {BC_TOL_PCT:g} points of the analytic change",
                  f"FDTD {100 * f_ne:+.2f} % vs MTL {100 * g_ne:+.2f} %")
        log.check(abs(f_fe - g_fe) * 100 < BC_TOL_PCT, f"B -> C change of the FE peak within {BC_TOL_PCT:g} points of the analytic change",
                  f"FDTD {100 * f_fe:+.2f} % vs MTL {100 * g_fe:+.2f} %")
        log.check(-0.14 < f_ne < -0.06 and -0.14 < f_fe < -0.06, "B -> C change is about -10 % in NE and FE (tutorial 5.2)",
                  f"NE {100 * f_ne:+.1f} %, FE {100 * f_fe:+.1f} %", "WARN")
        dpk = (f_ne, f_fe, g_ne, g_fe)
    return dict(V=V, ref=(ref_ne, ref_fe), cur=cur, dpk=dpk)


def plot_smoke(case, t, I, out, b):
    import matplotlib.pyplot as plt
    c, V, ref, cur = case, out["V"], out["ref"], out["cur"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for k, col in zip(("agg_near", "agg_far"), ("C0", "C1")):
        ax[0].plot(1e9 * t, 1e3 * cur[k], "k", lw=2 if k == "agg_near" else 1, alpha=.6)
        ax[0].plot(1e9 * t, 1e3 * I[k], col + "--", label=f"FDTD {k}")
    ax[0].set(xlabel="t (ns)", ylabel="I (mA)", title="Aggressor currents (black: analytic cylinder MTL)")
    for n, r_, col in (("NE", ref[0], "C0"), ("FE", ref[1], "C1")):
        ax[1].plot(1e9 * t, 1e3 * r_, "k", lw=2, alpha=.5)
        ax[1].plot(1e9 * t, 1e3 * V[n], col + "--", label=f"FDTD {n} (C)")
        if b is not None:
            m = b.t <= t[-1]
            ax[1].plot(1e9 * b.t[m], 1e3 * (b.V_NE if n == "NE" else b.V_FE)[m], col, lw=.8, alpha=.7, label=f"FDTD {n} (case B)")
    ax[1].set(xlabel="t (ns)", ylabel="V (mV)", title="NE and FE: C, B and the analytic cylinder (black)")
    for a in ax:
        a.grid(True, alpha=.3)
        a.legend(fontsize=7)
    plt.tight_layout()
    out_png = case.WORK / "c3_smoke.png"
    plt.savefig(out_png, dpi=120)
    plt.show()
    return out_png


# %% 0. Setup
REPO, SEMBA_EXE, SPINIT_FOLDER = repo_paths(HERE)
log = Log()

# %% 1. Gate 0: static check of the FULL c3.fdtd.json (grid, wall at j = 4 and closed, terminals, time plan, ...), then
# the 35 ns copy that will be run. The copy changes numberOfSteps only; its own static check is not repeated here.
full = load_case(HERE, FULL_CASE, log)
smoke_path, n_steps = make_smoke_json(full)
case = load_case(HERE, SMOKE_CASE, Log())
print(f"smoke case written: {smoke_path.name}, {n_steps} steps = {n_steps * full.inp['general']['timeStep'] * 1e9:.2f} ns "
      f"(full case: {full.inp['general']['numberOfSteps']} steps)")

# %% 2. Run the smoke case (about 7 min at 120 Mcells/s: 11.3 Mcells x 4545 steps), unless --no-run
from c3_solver import run_solver                      # noqa: E402
# solver = None if NO_RUN else run_solver(case, REPO, SEMBA_EXE, SPINIT_FOLDER)

# %% 3. Probe currents of the smoke run
t, I = load_smoke_currents(case, solver)

# %% 4. Analysis of the first 35 ns against the analytic cylinder and against case B
b = load_b(HERE, log)
out = analyse_smoke(case, t, I, b, log)

# %% 5. Summary and verdict
n_fail = log.summary(title="case C (c3) SMOKE TEST")
if n_fail == 0:
    print("\nSMOKE TEST PASSED: geometry, signs, Zc and the B -> C change look right. Run the full case (python c3.py) "
          "and watch it: the late-time behaviour is the only thing this test cannot see.")
else:
    print("\nSMOKE TEST FAILED: do NOT run the full case. Fix the failed items first (wall / leak / wire / signs).")

# %% 6. Time of the full run, from the speed measured here, and the figure
from c3_solver import parse_report                    # noqa: E402

rep = next(iter(sorted(Path(case.HERE).rglob(f"{case.case_path.stem}_Report.txt"))), None)
st = parse_report(rep.read_text(errors="ignore")) if rep else None
if st and st.speed_avg:
    mc = float(np.prod(full.ncell)) / 1e6
    n_full = full.plan.n_final
    print(f"\nmeasured speed {st.speed_avg:.1f} Mcells/s -> full run ({n_full} steps, {mc:.2f} Mcells) about "
          f"{n_full * mc / st.speed_avg / 3600:.2f} h")
print("figure:", plot_smoke(case, t, I, out, b))

# %%