# c1 - case A: one thin wire over a PEC ground plane

Pre- and post-processing for the first rung of the cross-talk validation ladder (A -> B -> C) of the Velectra project
(issue #5). Case A calibrates everything that case B (`c2`, two wires) and case C (closed cylinder) rely on: the source
and its `.exc` file, the delay, the effective characteristic impedance of the thin wire, the sign of the probes and the
boundary handling. If case A does not pass, do not move to case B.

All text in this document and in the scripts is in English.

## 1. Setup

| Item | Value |
|---|---|
| Grid | dx = 5 mm cubic cells, 168 x 88 x 400 cells (about 5.91 Mcells) |
| Wire | radius 0.5 mm, parallel to z, h = 35 mm above the plane, L = 2 m |
| Plane | PEC cell interval at constant j, spanning the whole grid in x and z |
| Boundaries | z faces PEC (the wire ends sit on them), x/y faces Mur |
| Terminals | series resistors, one at each end; near end = source resistance R_S, far end = load R_L (313.1 ohm in the tutorial) |
| Source | 1 V Gaussian generator in series with the near-end terminal, sigma = 2.5 ns, t0 = 15 ns, from `gauss_sigma2p5ns_t15ns.exc` |
| Probes | one current probe at each end, positive from near end to far end |
| Window | 40 ns, about 5190 steps (the pulse and the first complete echo end at t0 + 2T + 4 sigma = 38.3 ns; the tutorial first used 50 ns, which is not needed) |

Reference theory: the thin-wire model of semba-fdtd behaves as a wire of effective radius r_eff = 0.75 r (as in Paul,
example D.28). The expected characteristic impedance is Zc = eta0 / (2 pi) * acosh(h / r_eff), about 313.5 ohm; with the
nominal radius it would be about 296.3 ohm.

## 2. Files

Keep all of them in one folder (the case folder also holds `c1.fdtd.json` and the `.exc` file).

| File | Role |
|---|---|
| `c1.py` | Orchestrator, written as VS Code / Jupyter cells (`# %%`). Contains no logic of its own. |
| `c1_common.py` | Constants, the PASS/WARN/FAIL `Log`, `find_file`, `repo_paths`, and `solver_time_plan` (how semba-fdtd turns the JSON time step and step count into the real run). |
| `c1_input.py` | Static check of `c1.fdtd.json` and the `.exc` file; no solver needed. |
| `c1_solver.py` | ngspice files, run plan, run with live progress bar and ETA, clean stop, probe loading. |
| `c1_calibration.py` | Peaks, pulse centres (half maximum), delay, effective Zc, Gamma_L, charge. |
| `c1_echo.py` | First echo at the near end as a monopole (resistive) plus a doublet (reactive); late-time residual. |
| `c1_analytic.py` | Bounce-diagram reference of the lossless line, for three Zc references. |
| `c1_solver_report.py` | Checks of `Energy.dat` and `Report.txt`. |
| `c1_plots.py` | Three-panel figure. |

## 3. Usage

```
python c1.py             # check the input, run semba-fdtd with a progress bar, analyse the result
python c1.py --no-run    # check the input and analyse a run already made (takes seconds)
```

In the VS Code / Jupyter interactive window run the cells in order. To avoid repeating a long run:

- set `os.environ["C1_NO_RUN"] = "1"` before cell 1, or
- simply do not run cell 2. `solver` is `None` from the setup cell, so cell 3 reads the probes from the `.dat` files in
  the case folder.

Environment variables:

| Variable | Meaning |
|---|---|
| `SEMBA_EXE` | Path of the semba-fdtd executable (default `<repo>/build/bin/semba-fdtd`) |
| `REPO_ROOT` | Repository root (default: nearest parent folder that holds `src_pyWrapper`) |
| `C1_NO_RUN=1` | Same as `--no-run` |
| `C1_MCELLS_PER_S` | Solver speed for the a-priori time estimate; otherwise the average of the last `Report.txt` is used |

Cells of `c1.py`:

1. `load_case`: static check of the input.
2. `run_solver`: run plan, then the run (skipped with `--no-run`).
3. `load_currents`: near and far currents on one uniform time axis.
4. `calibrate`: first pass of the pulse.
5. `echo_analysis`: first echo and late-time residual.
6. `compare_with_line`: analytic reference.
7. `energy_check`, `report_check`: optional solver diagnostics.
8. `log.summary()`: all PASS / WARN / FAIL / INFO lines.
9. `plot_case`: writes `c1_prepost.png`.

## 4. How the run is planned and followed

semba-fdtd rules (OpenSEMBA/fdtd, checked on commit `cd069a6`):

- dt_CFL = 1 / (c sqrt(1/dx^2 + 1/dy^2 + 1/dz^2)); for dx = 5 mm it is 9.629e-12 s.
- If the JSON `timeStep` is 0 or larger than 0.8 dt_CFL (7.703e-12 s), the solver uses dt = 0.8 dt_CFL and runs
  `int(dt_json / dt * numberOfSteps)` steps. The simulated window is kept to within one step.
- Progress is written to `<case>_Report.txt` about every 60 s of wall time ("Simulated: n / N steps", "Mcells/sec",
  "Mins. until end").

`run_plan` prints these numbers before the run. `run_solver` then starts the same command that pyWrapper would, but
with `Popen`, and redraws a bar from the report. Between two reports (60 s) the step is extrapolated with the speed of
the last report (marked `~`). Before the first report the bar shows dotted slots.

```
c1  [##########..................................]  22.4 %   step ~1163/5191   t ~9.0/40.0 ns
    elapsed 0:01:12   ETA 0:04:08 (about 16:41)   54.2 Mcells/s   last report 0:00:17 ago
```

Display modes: Jupyter / VS Code interactive window (one updatable output, the run plan above stays visible), terminal
(one compact line with `\r`), redirected output (one line per report). The solver's stdout goes to
`<case>_stdout.log`.

Interrupt (the stop button or Ctrl-C): the solver runs in its own session, so the interrupt reaches only Python. Python
writes the `stop` file; semba-fdtd looks for it at each report (up to about a minute), stops cleanly and flushes the
probes. A second interrupt, or 150 s without a stop, terminates it.

## 5. Checks and thresholds

`FAIL` blocks the case; `WARN` is reported but not fatal; `INFO` is for reading.

Input (`c1_input.py`):

- Terminals are pure resistors; cells are cubic; grid, wire and plane as in the tutorial (WARN if not).
- Wire parallel to z; plane found and spanning the grid; r / dx <= 0.11 (thin-wire stability limit).
- z faces PEC; generator is a voltage source at the near end; one current probe at each end.
- Excitation: 1 V peak, starts at about 0 V (no step); spectrum levels are listed.
- Time step: WARN if semba-fdtd will correct it, with the exact corrected dt, step count and window.
- Window holds the first echo; terminals match Zc_eff within 2 % (otherwise a resistive echo appears).

Calibration (`c1_calibration.py`). Peak searches use the first-pass windows (near: 0 to t0 + T; far: t0 + T/2 to
t0 + 1.5 T), so an echo can never be taken as the incident pulse.

| Check | Threshold |
|---|---|
| Currents finite and non-zero; result covers the first echo | - |
| Near-end pulse centred on the source pulse (half-maximum centres) | within 0.1 ns |
| Delay between half-maximum centres equals L/c | within 1 % |
| Zc effective = V_S(t_peak) / I_peak - R_S equals the r_eff prediction | within 2 % |
| Near-end peak current equals 1 / (R_S + Zc_eff) | within 1 % |
| Far/near peak ratio equals 1 - Gamma_L | within 0.01 |
| Charge: integral of I equals integral of V_S / (R_S + R_L) | within 0.5 % |
| Probe sign: positive near -> far | - |

Echo (`c1_echo.py`): the residual after the main pulse is fitted as alpha * i(t - 2T) + beta * di/dt(t - 2T).
Checks: alpha equals -Gamma_L within 0.5 point; reactive part below 5 % reflection at 100 MHz (WARN); fit error below
0.3 % of the peak (WARN); late-time residual below -40 dB (no ringing, no instability).

Analytic (`c1_analytic.py`): FDTD currents match the bounce diagram with Zc(r_eff) within 1 % RMS of the peak. The RMS
table also shows Zc from the nominal radius and the measured Zc, so you can see which reference the FDTD follows.

Solver report (`c1_solver_report.py`): energy decays by more than 60 dB (WARN); solver speed above 50 Mcells/s
(WARN: plug the laptop in, high-performance power plan); the report also gives time estimates for cases B and C from
the measured speed.

## 6. Outputs

- `c1_prepost.png`: three panels - currents against the analytic line, |I| in dB, near-end echo with its fit.
- The summary printed by `log.summary()` and the solver files (`c1.fdtd_Report.txt`, `_Energy.dat`, probe `.dat`).
- The value Zc_eff measured here (313.1 ohm in the tutorial run) is the reference that `c2_common.ZC_A_MEAS` uses.

## 7. Notes and pitfalls

- The peak time of a flat-topped Gaussian is ill-conditioned (a 0.4 % amplitude change moves it by about 0.2 ns), which
  is why timing checks use half-maximum pulse centres, not peak times. Peak times are still printed as INFO.
- The `.dat` files are found by probe name, ignoring case, spaces and underscores (newest match wins). Files named
  `*Energy*` are ignored. The old pattern `*Wire*probe<n>*.dat` is tried when the name ends in a number and nothing
  else matches.
- Probe samples are stamped at (n + 0.5) dt; the time axis is made uniform automatically if needed.
- Check in the Solver dialog that the wire model is Holland with MTLN off. The report line "Wire model: MTLN wires" is
  printed by any build compiled with MTLN, so it does not prove which model ran.
- Running again overwrites the previous `.dat` files in the case folder. Copy the folder first if you need to keep a run.
