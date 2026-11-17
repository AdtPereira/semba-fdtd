# c2 - case B: two thin wires over the same PEC ground plane (cross-talk)

Second rung of the cross-talk validation ladder (A -> B -> C), Velectra project issue #5. Case B validates the FDTD
cross-talk (near-end NEXT and far-end FEXT) against the analytic multiconductor transmission line (MTL) before case C
(closed PEC cylinder). Case A (`c1`) already calibrated the source, the delay, the effective impedance and the probe
sign.

All text in this document and in the scripts is in English.

## 1. Setup

| Item | Value |
|---|---|
| Grid | dx = 5 mm cubic cells, 168 x 88 x 400 cells (about 5.91 Mcells) |
| Wires | two, radius 0.5 mm, parallel to z, h = 35 mm above the plane, separation 70 mm, L = 2 m |
| Plane | PEC cell interval at constant j, spanning the whole grid in x and z |
| Boundaries | z faces PEC (end walls), x/y faces Mur |
| Terminals | 50 ohm series resistors on all four ends |
| Source | 1 V Gaussian generator (sigma = 2.5 ns, t0 = 15 ns) in series with the aggressor near-end terminal (= R_S), from the `.exc` file |
| Probes | four current probes: aggressor near ("Source probe") and far ("Load probe"), victim near ("NE probe") and far ("FE probe"), positive along z |
| Window | 300 ns, because the mismatch (Zc about 300 ohm against 50 ohm, Gamma_L about -0.72) rings for about 150 ns |

Roles (aggressor, victim, which probe is where) are found from positions in the JSON, not from names: the aggressor
is the cable that starts at the generator node.

Voltage conventions (current positive along z, near -> far):

- V_NE = -R_NE * I_vic,near
- V_FE = +R_FE * I_vic,far

so NE is expected positive and FE negative. Because the line is mismatched, the near-end echo is larger than the
incident pulse; the aggressor calibration therefore uses first-pass windows only.

## 2. Files

Keep all of them in one folder with `c2.fdtd.json` and the `.exc` file. The modules do not depend on `c1`.

| File | Role |
|---|---|
| `c2.py` | Orchestrator, cell by cell (`# %%`). No logic of its own. |
| `c2_common.py` | Constants and tolerances, `Log`, `find_file`, `repo_paths`, `solver_time_plan`. |
| `c2_input.py` | Static check of `c2.fdtd.json` and the `.exc` file; roles from positions; analytic inductance matrices. |
| `c2_solver.py` | Run plan, run with progress bar and ETA, clean stop, probe loading. |
| `c2_calibration.py` | Aggressor on the first pass: peaks, half-maximum pulse centres, delay, Zc effective, probe sign. |
| `c2_crosstalk.py` | NE/FE voltages, peaks, polarity, transfer function, late-time ring-down. |
| `c2_analytic.py` | MTL reference (image-method inductance matrix, chain matrix exact in frequency) and the comparison checks. |
| `c2_solver_report.py` | Checks of `Energy.dat` and `Report.txt`; time estimate for case C. |
| `c2_plots.py` | Four-panel figure. |

## 3. Usage

```
python c2.py             # check the input, run semba-fdtd with a progress bar, analyse the result
python c2.py --no-run    # check the input and analyse a run already made (takes seconds)
```

A full run takes about 33 minutes at 120 Mcells/s (38944 steps), roughly seven times case A (same cells, about 5190
steps for A).

In the VS Code / Jupyter interactive window run the cells in order. To avoid repeating the solver:

- set `os.environ["C2_NO_RUN"] = "1"` before cell 1, or
- do not run cell 2. `solver` is `None` from the setup cell, so cell 3 reads the probes from the `.dat` files.

For `--no-run` the case folder must already hold the four probe `.dat` files; `Energy.dat` and `Report.txt` are optional
(missing ones are skipped by the diagnostics).

Environment variables:

| Variable | Meaning |
|---|---|
| `SEMBA_EXE` | Path of the semba-fdtd executable (default `<repo>/build/bin/semba-fdtd`) |
| `REPO_ROOT` | Repository root (default: nearest parent folder that holds `src_pyWrapper`) |
| `C2_NO_RUN=1` | Same as `--no-run` |
| `C2_MCELLS_PER_S` | Solver speed for the a-priori time estimate; otherwise the average of the last `Report.txt` |

Cells of `c2.py`:

1. `load_case`: static check of the input.
2. `run_solver`: run plan, then the run (skipped with `--no-run`).
3. `load_currents`: the four currents on one uniform time axis.
4. `calibrate`: aggressor first pass.
5. `crosstalk`: NE and FE voltages.
6. `transfer_function`, `late_time`: H = FFT(V) / FFT(source) and the ring-down.
7. `compare_with_mtl`, `compare_transfer`: analytic reference.
8. `energy_check`, `report_check`: optional solver diagnostics.
9. `log.summary()`.
10. Saves `c2_results.npz`.
11. `plot_case`: writes `c2.png`.

Create `log = Log()` once, before cell 1, and pass that same object to every check. If the summary prints "0 checks ran
-> NOTHING WAS VALIDATED", the log was empty (for example `load_case` was not run on it): an empty summary is never a
pass.

## 4. How the run is planned and followed

semba-fdtd rules (OpenSEMBA/fdtd, commit `cd069a6`):

- dt_CFL = 1 / (c sqrt(1/dx^2 + 1/dy^2 + 1/dz^2)); HEURCFL = 0.8.
- If the JSON `timeStep` is 0 or larger than 0.8 dt_CFL, dt becomes 0.8 dt_CFL and the final step is
  `int(dt_json / dt * numberOfSteps)` (truncated).
- For case B as first written: dt 8.66e-12 s -> 7.7033e-12 s, steps 34642 -> 38944, simulated time 299.9986 ns. Set
  `timeStep` 7.7033e-12 s and 38944 steps in the JSON and nothing is corrected (this is what case C does).
- Progress goes to `<case>_Report.txt` about every 60 s (REPORTINGSECONDS in `progress.F90`).

`run_plan` prints this before the run. `run_solver` starts the solver command with `Popen`, sends its stdout to
`<case>_stdout.log`, follows the report and redraws a bar. Between reports the step is extrapolated (`~`); the ETA uses
the solver's own formula, (N - n) * Total_Mcells / average speed.

```
c2  [##########..................................]  22.4 %   step ~8723/38944   t ~67.2/300.0 ns
    elapsed 0:09:12   ETA 0:31:55 (about 16:41)   54.2 Mcells/s   last report 0:00:17 ago
```

Modes: Jupyter / VS Code interactive window (updatable output, run plan stays), terminal (compact `\r` line),
redirected (one line per report).

Interrupt: the solver runs in its own session, so the interrupt reaches only Python. Python writes the `stop` file;
semba-fdtd checks it at each report (up to about a minute), stops cleanly and flushes the probes. A second interrupt or
150 s without a stop terminates it. Any other exception in the monitor also terminates the child.

## 5. Checks and thresholds

`FAIL` blocks the case; `WARN` is reported but not fatal; `INFO` is for reading.

Input (`c2_input.py`): terminals are pure 50 ohm resistors; plane, wires and separation as in the tutorial; z faces PEC
and x/y faces Mur; `.exc` has 1 V peak, t0 = 15 ns, sigma = 2.5 ns and covers the whole window; WARN if semba-fdtd will
correct the time step (the exact dt, steps and window are printed).

Aggressor calibration (`c2_calibration.py`), first-pass windows only (near: 0 to t0 + T; far: t0 + T/2 to t0 + 1.5 T):

| Check | Threshold |
|---|---|
| Near-end pulse centre equals the source pulse centre (half maximum) | within 0.1 ns |
| Delay between half-maximum centres equals L/c | within 1 % |
| Probe sign | positive near -> far |

Zc effective is source(t_peak) / I_peak - R_S.

Cross-talk (`c2_crosstalk.py`): NE positive and FE negative; transfer function H = FFT(V) / FFT(source) is used from
3 to 100 MHz where the source spectrum is above -40 dB; late-time residual after 175 ns below -40 dB (WARN).

Analytic comparison (`c2_analytic.py`). Three references are computed: nominal radius r, 0.75 r, and r_eff from case A
Zc (313.1 ohm). The last one (`case.primary`) is used for the pass/fail checks.

| Check | Threshold |
|---|---|
| NE and FE peak against the MTL | within 10 % |
| NE and FE waveforms in time (cross-correlation lag) | below 0.15 ns |
| Aggressor currents, RMS of the peak | below 2 % (WARN) |
| NE and FE waveforms, RMS of their peak | below 10 % (WARN) |
| Transfer functions H_NE and H_FE against the MTL over 3-100 MHz | max 2 dB (WARN); bins within 4 MHz of n c / (2 L) are skipped |

Solver report (`c2_solver_report.py`): energy decays more than 60 dB (WARN); speed above 50 Mcells/s (WARN); the
estimate for case C is 2x the cells over the same window (the real case C run took 1.00 h at 122 Mcells/s).

## 6. Outputs

- `c2.png`: aggressor currents; NE/FE voltages; transfer functions; ring-down of the mismatched line.
- `c2_results.npz`, for the B -> C comparison: `t`, `V_NE`, `V_FE`, `I_agg_near`, `I_agg_far`, `I_vic_near`, `I_vic_far`.
  `c3_bc.py` finds it through `C2_RESULTS`, the c3 folder or `../c2/`; a copy is also kept in the Project. Case C gave
  -9.2 % (NE) and -9.6 % (FE) against it, as the analytic model predicts (about -10 %).
- The printed summary and the solver files.

## 7. Result of the real run (33 min, 38944 steps, 120 Mcells/s)

Result: 41 passed, 1 warning, 0 failed.

| Quantity | FDTD | MTL (r_eff from case A) | Difference |
|---|---|---|---|
| NE peak | +7.82 mV | +7.91 mV | -1.1 % |
| FE peak | -11.28 mV | -11.44 mV | -1.4 % |
| H_NE / H_FE over 3-100 MHz | max 1.54 dB (at 99 MHz) / 0.38 dB | | |
| Aggressor currents, RMS | 0.12 % near / 0.04 % far | | |
| NE / FE waveform RMS | 0.94 % / 0.65 % | | |

Other numbers: delay 6.715 ns (v = 0.9934 c); Zc_eff 311.8 ohm; late-time worst -41.7 dB (thin margin to -40); energy
decay 117 dB; NE lag +68 ps, FE lag -133 ps. The only warning is the solver correcting the time step.

The FDTD follows the r_eff = 0.756 r curves (0.3778 mm, from case A), not the nominal radius: the thin-wire behaviour
already seen in case A and in Paul D.28. The 1.5 dB NE deviation near 100 MHz is unexplained; it is not the 0.7 % extra
delay. In case C the same deviation is 1.10 dB (NE) at 99 MHz.

## 8. Notes and pitfalls

- Peak times are ill-conditioned on a flat-topped Gaussian, so timing checks use half-maximum centres and
  cross-correlation lag. The near-end pulse is 33 ps early (0.5 % of the line delay), the same offset as in case A.
- Probe `.dat` files are found by probe name ("Source probe" and so on), ignoring case, spaces and underscores; `Energy`
  files are ignored; the newest match wins. Samples are stamped at (n + 0.5) dt.
- "Wire model: MTLN wires" in `Report.txt` is hard-coded in any MTLN build (`semba_fdtd.F90`, l.504) and does not prove
  which model ran. Zc of 313 ohm points to the FDTD thin-wire model. Confirm Holland with MTLN off in the Solver
  dialog.
- The `create_symlink` warning in `Warnings.txt` is harmless when source and target are the same file in the same folder.
- A new run overwrites the `.dat` files. Copy the case folder first if you want to keep case B; `c2_results.npz`
  already holds its numbers.
