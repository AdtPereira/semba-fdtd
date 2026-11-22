# c1: one thin wire over a PEC ground plane

Pre- and post-processing for the first rung of the cross-talk validation ladder (c1 -> c2 -> c3) of the Velectra project
(issue #5). Case `c1` checks the common ground of cases `c2` (two wires) and `c3` (closed cylinder): the source and its
`.exc` file, the delay, the sign of the probes and the boundary handling, and it shows how ``semba-fdtd`` differs from the
textbook model of a wire over a plane.

**Two curves are compared, nothing else:**

1. **semba-fdtd**, with the terminals exactly as they are in `c1.fdtd.json` (R_S = R_L = 359.2 ohm at both ends). They are not tuned to
   the wire as the solver models it, so any mismatch shows up in the result.
2. **The analytic lossless line with the physical radius of the wire**: $Zc = \mu_0 / (2 \pi) acosh(h / r) = 359.2 \ \Omega$ at
   h = 100 mm, delay T = L / c, resistive terminals (bounce diagram). With R_S = R_L = 359.2 ohm this line is matched: no
   echo, I_near = V_S / (2 Zc), I_far = I_near(t - T).

The script reports the differences (peak current, delay,
RMS, echo) and only fails on gross errors.

One copy of the code serves three **cases**, one subfolder each: `h100` (the default test), `h35` (the height of cases B and C) and `h100_dx10` (the same as `h100` on a coarser grid); section 2. Choose one with `C1_CASE`. All use `pml` borders and terminals equal to the analytic Zc of the physical radius. Five more cases were run during the study and then removed from the folder; their results stay in the tables and how to rebuild them is in "Cases removed" at the end. The numbers quoted in sections 1 and 5 are those of `h100`.

****

- [c1: one thin wire over a PEC ground plane](#c1-one-thin-wire-over-a-pec-ground-plane)
  - [1. Setup](#1-setup)
  - [2. Files](#2-files)
  - [3. Usage](#3-usage)
  - [4. How the run is planned and followed](#4-how-the-run-is-planned-and-followed)
  - [5. Checks and thresholds](#5-checks-and-thresholds)
  - [6. Outputs](#6-outputs)
  - [7. Notes and pitfalls](#7-notes-and-pitfalls)
- [Review notes, h = 100 mm (2026-11-21)](#review-notes-h--100-mm-2026-11-21)
  - [Scope of this version](#scope-of-this-version)
  - [Changes by module](#changes-by-module)
  - [Run of 2026-11-21, 8 MPI ranks on ewing (168 x 88 x 400, working PML): 31 PASS, 0 WARN, 0 FAIL](#run-of-2026-11-21-8-mpi-ranks-on-ewing-168-x-88-x-400-working-pml-31-pass-0-warn-0-fail)
  - [What went wrong in two earlier runs (data deleted)](#what-went-wrong-in-two-earlier-runs-data-deleted)
  - [Wire model (OpenSEMBA/fdtd, commit cd069a6)](#wire-model-opensembafdtd-commit-cd069a6)
  - [Reading of the 2026-11-21 differences](#reading-of-the-2026-11-21-differences)
  - [Layout by case (2026-11-22)](#layout-by-case-2026-11-22-one-code-folder-one-subfolder-per-case)
  - [Open points](#open-points)


## 1. Setup

Case `h100` (the default). The other cases differ as listed in section 2; height, grid, boundaries and terminals are read from the `c1.fdtd.json` of the chosen case, and only the length (2 m), the radius (0.5 mm) and dx (5 mm) are fixed in `c1_common.EXPECTED`.

| Item | Value |
|---|---|
| Grid | dx = 5 mm cubic cells, 168 x 88 x 400 cells (5.9 Mcells; 7.7 Mcells with the 8 PML layers) |
| Wire | radius 0.5 mm, parallel to z, **h = 100 mm** above the plane (wire at j = 24, plane at j = 4), L = 2 m |
| Plane | PEC cell interval at constant j, spanning the whole grid in x and z |
| Boundaries | z faces PEC (the wire ends sit on them); x/y faces open. `c1.fdtd.json` uses lowercase **"pml"** (8 layers added outside the grid by default). Clearance from the wire: 3.7 h to x_low, 4.7 h to x_high, 3.2 h to y_high. Use the same type in A, B and C |
| Terminals | series resistors, one at each end; near end = source resistance R_S, far end = load R_L; both 359.24 ohm = Zc of the physical radius at h = 100 mm |
| Source | 1 V Gaussian generator in series with the near-end terminal, sigma = 2.5 ns, t0 = 15 ns, from `gauss_sigma2p5ns_t15ns.exc` |
| Probes | one current probe at each end, positive from near end to far end |
| Window | 50 ns, 6490 steps of 7.7033 ps (the pulse and the first complete echo end at t0 + 2T + 4 sigma = 38.3 ns, so 40 ns, about 5190 steps, is the minimum). The longer window also holds the second bounce at t0 + 4T = 41.7 ns |

Result of the run of 2026-11-21 (168 x 88 x 400, working PML), semba-fdtd against the analytic line:

| Quantity | semba-fdtd | analytic (physical radius) |
|---|---|---|
| Peak current, near end | 1.3619 mA | 1.3918 mA (-2.15 %) |
| Peak current, far end | 1.3861 mA | 1.3918 mA (-0.41 %) |
| Delay far - near (half-maximum centres) | 6.819 ns | 6.671 ns = L/c (+148 ps) |
| Impedance seen by the source, V_S / I - R_S | 374.9 ohm | 359.2 ohm |
| Echo at the near end | +3.69 % / -0.98 % of the main peak (-28.7 dB) | none (matched) |
| Late-time residual, near / far | -77 / -66 dB | - |
| Charge, near / far | 1.0000 / 1.0000 of the ideal value | - |

RMS difference between the two curves: 1.26 % (near) and 0.90 % (far) of the peak, maximum difference 3.69 % / 2.62 %. The same
run, with 8 MPI ranks cutting in x on ewing (build-mpi, 7.65 Mcells including the PML, 6490 steps), averaged 394 Mcells/s and took about
2 to 3 minutes; the serial run on the desktop had given 137.6 Mcells/s. The static and post-processing summary of that run: 31 PASS,
0 WARN, 0 FAIL.

Pitfall found on the way: semba-fdtd reads the boundary type with a case-sensitive match, so "PML" is silently ignored (the Report then
says "no CPML Borders found" and the x/y faces are not absorbing). Write "pml" or "mur" in lowercase; `c1_input` and `border_check` check it.

## 2. Files

The code files sit together in one folder; each case has its own subfolder with its `c1.fdtd.json`, the `.exc` file and every output of its runs.

```
c1/   c1.py, c1_*.py, c1_readme.md          one copy of the code
      h100/      h = 100 mm, pml, terminals 359.24 ohm (the default test; results of the 2026-11-21 run)
      h35/   h = 35 mm (wire at [77, 11], plane at j = 4), pml, terminals 296.29 ohm (physical Zc), 7.7033 ps / 6490 steps: only the height differs from h100
      h100_dx10/ h = 100 mm, pml, terminals 359.24 ohm, dx = 10 mm: 84 x 44 x 200 cells (0.74 Mcells), 15.406 ps / 3245 steps (same 50 ns window), same box in metres,
                 plane at j = 2, wire at [37, 12]; r/dx = 0.05. A coarser grid at the same height; 17 s with 4 MPI ranks
```

| File | Role |
|---|---|
| `c1.py` | Orchestrator, written as VS Code / Jupyter cells (`# %%`). Contains no logic of its own. Picks the case folder from `C1_CASE`. |
| `c1_common.py` | Constants, the PASS/WARN/FAIL `Log`, `find_file`, `repo_paths`, and `solver_time_plan` (how semba-fdtd turns the JSON time step and step count into the real run). |
| `c1_input.py` | Static check of the case's `c1.fdtd.json` and `.exc` file; no solver needed. |
| `c1_solver.py` | ngspice files, MPI settings, run plan (mode, solver, command), run with live progress bar and ETA, clean stop, probe loading. |
| `c1_calibration.py` | Peaks, pulse centres (half maximum), delay, impedance seen by the source, charge (semba-fdtd only). |
| `c1_echo.py` | First echo at the near end (measured, not fitted); late-time residual; the shape of the near-end current before the echo (INFO fit of a shunt capacitance). |
| `c1_analytic.py` | Analytic lossless line with the physical radius (bounce diagram) and its difference to semba-fdtd. |
| `c1_solver_report.py` | Checks of `Energy.dat`, `Warnings.txt` and `Report.txt`. |
| `c1_plots.py` | Two figures: `plot_case` (currents against the analytic line, and |I| in dB) and `plot_near_end` (near-end current against the main pulse, vertical zoom, and their difference). |

## 3. Usage

```
python c1.py                        # case h100: check the input, run semba-fdtd with a progress bar, analyse the result
python c1.py --no-run               # check the input and analyse a run already made (takes seconds)
C1_CASE=h35 python c1.py        # h = 35 mm with PML and matched terminals
C1_MPI_RANKS=4 C1_CASE=h100_dx10 python c1.py   # h100 with dx = 10 mm (0.74 Mcells; 84 cells in x allow 4 ranks of 21 cells, or 1 for serial)
```

If `C1_CASE` names a folder that has no `c1.fdtd.json`, `c1.py` stops and lists the case folders next to the code.

In the VS Code / Jupyter interactive window run the cells in order. To avoid repeating a long run:

- set `os.environ["C1_NO_RUN"] = "1"` before cell 1, or
- simply do not run cell 2. `solver` is `None` from the setup cell, so cell 3 reads the probes from the `.dat` files in
  the case folder.

Environment variables:

| Variable | Meaning |
|---|---|
| `SEMBA_EXE` | Path of the semba-fdtd executable. Default: `<repo>/build-mpi/bin/semba-fdtd` when `C1_MPI_RANKS` > 1, else `<repo>/build/bin/semba-fdtd` |
| `REPO_ROOT` | Repository root (default: nearest parent folder that holds `src_pyWrapper`) |
| `C1_CASE` | Case: name of a subfolder of the code folder, or a path. Default `h100`. Set it before cell 1 |
| `C1_NO_RUN=1` | Same as `--no-run` |
| `C1_MCELLS_PER_S` | Solver speed for the a-priori time estimate; otherwise the average of the last `Report.txt` is used |
| `C1_MPI_RANKS` | Number of MPI ranks. **Default 8** (set in `c1.py`); 1 = serial run with the serial build |
| `C1_MPI_DIR` | Direction of the MPI cut, default `x`. Never `z`: the solver aborts with `MPI_ERR_RANK` when a wire is present |
| `MPIRUN`, `C1_MPI_FLAGS` | Launcher (default `/usr/local/bin/mpirun` if it exists) and extra words for it |

Cells of `c1.py`:

1. `load_case`: static check of the input (case folder `C1_CASE`).
2. `run_solver`: run plan, then the run (skipped with `--no-run`).
3. `load_currents`: near and far currents on one uniform time axis.
4. `calibrate`: first pass of the pulse (semba-fdtd).
5. `echo_analysis`: first echo and late-time residual (semba-fdtd).
6. `compare_with_line`: analytic line with the physical radius and its difference to semba-fdtd.
7. `energy_check`, `warnings_check`, `report_check`: optional solver diagnostics.
8. `log.summary()`: all PASS / WARN / FAIL / INFO lines.
9. `plot_case` and `plot_near_end`: write `c1_prepost.png` and `c1_near_end_shape.png` in the case folder.

Create `log = Log()` once, before cell 1, and pass that same object to `load_case` and the later checks. If the summary
prints "0 checks ran -> NOTHING WAS VALIDATED", the log was empty (for example `load_case` was not run on it): an empty
summary is never a pass.

## 4. How the run is planned and followed

semba-fdtd rules (OpenSEMBA/fdtd, checked on commit `cd069a6`):

- dt_CFL = 1 / (c sqrt(1/dx^2 + 1/dy^2 + 1/dz^2)); for dx = 5 mm it is 9.629e-12 s.
- If the JSON `timeStep` is 0 or larger than 0.8 dt_CFL (7.703e-12 s), the solver uses dt = 0.8 dt_CFL and runs
  `int(dt_json / dt * numberOfSteps)` steps. The simulated window is kept to within one step.
- Progress is written to `<case>_Report.txt` about every 60 s of wall time ("Simulated: n / N steps", "Mcells/sec",
  "Mins. until end").

MPI (ewing scalability tests of 2026-11-17, case c3, 11.3 Mcells): OpenMP of the serial build does not scale (169 Mcells/s from 1 to 16
threads); MPI with 8 ranks cutting in x reached 493 Mcells/s (2.9 x the serial build); 16 ranks were slower than 8; one MPI rank is
1.8 x slower than the serial build. `c1.py` therefore launches `mpirun -n 8 build-mpi/bin/semba-fdtd -i c1.fdtd.json -mpidir x`
with `OMP_NUM_THREADS = 1`. With 168 cells in x that is 21 cells per rank. The run plan prints `MODE` (serial or MPI with the
number of ranks), the solver path and whether it is the MPI build, the launcher, the threads and the full command, with warnings for
`-mpidir z`, fewer than 16 cells per rank, a missing launcher or an executable that is not the MPI build; the progress bar and
the final line repeat the mode. The speed hint of the plan comes from the last `Report.txt`, so a serial one underestimates an MPI run.

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

- Terminals are pure resistors; cells are cubic with dx = 5 mm (WARN if not); wire length and radius as expected (`c1_common.EXPECTED`; WARN if not). Grid size and wire height are read from the JSON and printed as INFO.
- x/y faces open, exactly lowercase `mur` or `pml`, not PEC; the clearance from the wire to the boundaries is printed in units of h.
- Wire parallel to z; plane found and spanning the grid; r / dx <= 0.11 (thin-wire stability limit, FAIL). The MTLN wire of this solver appears to share it (not isolated): the dx = 2.5 mm attempt with r = 0.5 mm (r/dx = 0.2) blew up at once (energy `NaN` from the first report, near-end current alternating in sign and growing by about 3.3 per step from the first step, `NaN` after 78 steps; record in "Cases removed"). A check was briefly a WARN on 2026-11-22 and is a FAIL again.
- z faces PEC; generator is a voltage source at the near end; one current probe at each end.
- Excitation: 1 V peak, starts at about 0 V (no step); spectrum levels are listed.
- Time step: WARN if semba-fdtd will correct it, with the exact corrected dt, step count and window.
- Window holds the first echo.
- Both terminals equal the analytic Zc of the physical radius within 2 % (PASS: the analytic line is matched). Otherwise an INFO gives the
  reflection coefficients Gamma_S and Gamma_L ("the analytic line is NOT matched ... it has its own echoes", as in the removed 313.1 ohm run `h35_313ohm`, Gamma = +0.028). Whatever
  the solver does with that is not corrected: it is the result.

Calibration (`c1_calibration.py`, semba-fdtd only). Peak searches use the first-pass windows (near: 0 to t0 + T; far: t0 + T/2 to
t0 + 1.5 T), so an echo can never be taken as the incident pulse.

| Check | Threshold |
|---|---|
| Currents finite and non-zero; result covers the first echo | - |
| Near-end pulse centred on the source pulse (half-maximum centres) | within 0.1 ns |
| Delay between half-maximum centres equals L/c (no velocity error) | FAIL above 5 % |
| Charge: integral of I equals integral of V_S / (R_S + R_L) | within 0.5 % |
| Probe sign: positive near -> far | - |

Printed as INFO: peaks and pulse centres, delay and v/c, the impedance seen by the source V_S(t_peak) / I_peak - R_S, the effective per-unit-length L' = Zc / v and
C' = 1 / (Zc v) with their analytic values (physical radius) and the product L'C' c^2, the far/near peak ratio, the charges.

Echo (`c1_echo.py`): residual after the main pulse = I_near minus the source scaled to the first-pass peak; maximum, minimum and level
in dB are printed (the analytic line has none when it is matched; otherwise its expected echo, -100 Gamma_L % of the main peak, is printed). WARN above 10 % of the main peak. Before the echo the residual is fitted to C dV_S/dt
over t0 - 5 sigma ... t_echo - 3 sigma (INFO: C in fF and the RMS before and after; a fit, not a model). Late-time residual below -40 dB (no ringing, no
instability).

Analytic (`c1_analytic.py`): the RMS and maximum differences between semba-fdtd and the analytic line, the difference in peak current and
in delay, all as INFO. Two sanity bands only: WARN above 5 % RMS of the peak, FAIL above 15 %. They catch a wrong geometry, source or
broken run; they are not a calibration of the solver.

Solver report (`c1_solver_report.py`): energy decays by more than `min_db` (WARN; `c1.py` passes 30 dB because a 50 ns run leaves only a few
reports; the default is 60 dB for the long B and C runs); the Report lists the borders requested in the JSON; solver warnings other than the known
harmless `create_symlink` of the `.exc` (WARN); solver speed above 50 Mcells/s (WARN); time estimates for cases B and C from the measured speed.

## 6. Outputs

- Per case, in its folder (`h100/`, `h35/`, `h100_dx10/`): `c1_prepost.png`: two panels - semba-fdtd currents against the analytic line (top), |I| / I_peak in dB (bottom). Style: semba-fdtd solid, analytic dotted; near C0 / black, far C1 / gray.
- `c1_near_end_shape.png` (same folder): three panels - near-end current against the main pulse (the source scaled to the first-pass peak), the same with a vertical zoom, and their difference in % of the peak. The difference has two parts: the small oscillation before the echo, which follows C dV_S/dt (about 56 fF in the 2026-11-21 run, fitted between t0 - 5 sigma and t_echo - 3 sigma; a fit, not a model), and the echo of the mismatch (+3.69 % / -0.97 %).
- The summary printed by `log.summary()` and the solver files (`c1.fdtd_Report.txt`, `_Energy.dat`, probe `.dat`), all in the case folder.
- Cases B and C (h = 35 mm) have their own documents and are not changed by this version of case A. Note that `c2_common.ZC_A_MEAS` (313.1 ohm, measured in the 35 mm run) is still the reference that case C uses to derive its cylinder radius; that value is not produced by this version of `c1.py`.

## 7. Notes and pitfalls

- The peak time of a flat-topped Gaussian is ill-conditioned (a 0.4 % amplitude change moves it by about 0.2 ns), which
  is why timing checks use half-maximum pulse centres, not peak times. Peak times are still printed as INFO.
- The `.dat` files are found by probe name, ignoring case, spaces and underscores (newest match wins). Files named
  `*Energy*` are ignored. The old pattern `*Wire*probe<n>*.dat` is tried when the name ends in a number and nothing
  else matches.
- Probe samples are stamped at (n + 0.5) dt; the time axis is made uniform automatically if needed.
- Wire model (OpenSEMBA/fdtd, commit cd069a6): with MTLN compiled, a `wire` material goes through the MTLN path (the Report shows
  "Init MTLN Wires..."); the Holland code is not used. The line "Wire model: MTLN wires" is printed by any build compiled with MTLN.
  The in-cell inductance comes from `computeLCParametersFromRadius` (`mtl.F90`), so the impedance the solver gives the wire is a result of
  the grid and the model, not an input; case A reports it (INFO) and does not predict it.
- The `create_symlink` warning in `Warnings.txt` is harmless when source and target are the same file.
- Running again overwrites the previous `.dat` files in the case folder. Copy the case folder first if you need to keep a run.

# Review notes, h = 100 mm (2026-11-21)

Reference for the original input (h = 35 mm, Mur, 313.1 ohm terminals, 8.66 ps / 4618 steps): branch `velectra-xtalk` of
AdtPereira/semba-fdtd, commit 52cbf48, `testData/cases/velectra/c1` (inputs and scripts only; no results are stored there).

## Scope of this version

Case A compares two curves only: semba-fdtd with the terminals as they are in the JSON (359.24 ohm), and the analytic lossless line with
the physical radius (Zc = 359.2 ohm, matched). The effective-radius model (r_eff = 0.75 r), the fitted line with end capacitance and every
check built on them were removed. The differences are reported; two sanity bands (5 % WARN, 15 % FAIL RMS) catch gross errors.

## Changes by module

| Module | Change |
|---|---|
| `c1_common` | `EXPECTED`: h = 0.100, grid 168 x 88 x 400; `R_EFF_FACTOR` and `r_eff_factor` removed; `zc_wire_plane` kept |
| `c1_input` | exact lowercase `mur` / `pml` check (the solver ignores other spellings); clearance in units of h; time-step message follows `plan.corrected`; terminals compared with the Zc of the physical radius (PASS), and kept as they are |
| `c1_calibration` | `peak()` guard at a window edge; r_eff checks removed; the 1 % delay WARN removed (FAIL at 5 % stays); impedance seen by the source is INFO |
| `c1_echo` | monopole + doublet fit removed; echo measured (max, min, dB), WARN above 10 %; late-time residual kept |
| `c1_analytic` | only the bounce diagram with the physical radius; `line_response_fd`, `fit_end_capacitance` removed; differences in peak, delay and RMS as INFO; bands 5 % / 15 % |
| `c1_solver_report` | `min_db` for the energy check; `warnings_check`; `border_check` (the Report must list CPML / Mur as requested) |
| `c1_solver` | `probe_dat` looks in the case folder first; MPI settings (`mpi_settings`), run plan with MODE, solver, launcher, threads and command; the mode is repeated on the progress bar |
| `c1_plots` | two figures: `plot_case` (currents against the analytic line + |I| in dB; semba-fdtd solid, analytic dotted) and `plot_near_end` (near-end current vs main pulse, zoom, difference with the C dV_S/dt fit) |
| `c1_echo` | INFO fit of the near-end shape before the echo: C = 56 fF in the 2026-11-21 run, RMS of the residual 0.574 % -> 0.073 % of the peak |
| `c1_calibration` | INFO: effective L' and C' (Zc / v, 1 / (Zc v)) against the analytic values and L'C' c^2 = 1.0447 |
| `c1.py` | header and cell titles for the new scope; MPI by default (8 ranks, `-mpidir x`, `build-mpi`) |

## Run of 2026-11-21, 8 MPI ranks on ewing (168 x 88 x 400, working PML): 31 PASS, 0 WARN, 0 FAIL

Launch: `mpirun -n 8 build-mpi/bin/semba-fdtd -i c1.fdtd.json -mpidir x` (Report: "MPI Processes: 8", 7.65 Mcells including the PML, 6490 steps,
average 394 Mcells/s, about 2 to 3 minutes; the serial desktop run gave 137.6 Mcells/s). The Report lists the CPML borders and the MTLN wire path.

semba-fdtd against the analytic line (physical radius, Zc = 359.2 ohm): peaks 1.3619 mA (near) and 1.3861 mA (far) against 1.3918 mA (-2.15 % and
-0.41 %); delay 6.819 ns against L/c = 6.671 ns (+148 ps); V_S / I - R_S = 374.9 ohm; echo +3.69 % / -0.98 % of the main peak (-28.7 dB);
late-time residual -77 / -66 dB; charge 1.0000 / 1.0000; RMS difference 1.26 % (near), 0.90 % (far). These numbers are identical, to the digits
printed, to the run of the 250 x 150 x 400 box and to the serial run, so the smaller box (3.2 h to the nearest PML) is enough and MPI does not
change the result.

## What went wrong in two earlier runs (data deleted)

semba-fdtd compares the boundary type with a case-sensitive `select case` (`labelToBoundaryType`, `smbjson.F90`). The JSON had "PML": not
recognised, the Report said "no CPML Borders found" and the x/y faces were not absorbing. Those runs gave wrong impedances, ringing and,
in one of them, rising energy, and were discarded. One of them also had the ground plane stopping at x = 168 of a larger grid.

## Wire model (OpenSEMBA/fdtd, commit cd069a6)

With MTLN compiled, a `wire` material is an MTLN unshielded multiwire (`readMTLNCable`); the Holland code is not used and the Report shows
"Init MTLN Wires...". The in-cell L' comes from `computeLCParametersFromRadius` (`mtl.F90`) and C' = 1 / (L' c^2). The default PML adds 8
layers outside the grid.

## Reading of the 2026-11-21 differences

- Delay +148 ps (v = 0.978 c): L' +6.7 %, C' -2.1 %, so L'C' c^2 = 1.045. A change of radius moves L' and C' in opposite ways and cannot produce it. The in-cell
  `computeLCParametersFromRadius` keeps L'C' = 1/c^2, so the excess comes from the coupling between the MTLN wire and the 3D field (not isolated).
- Near-end residual before the echo (about +0.9 % / -1.1 % at t0 -/+ sigma) follows C dV_S/dt with C about 56 fF, of the order of one cell (C' dz = 45 fF).
- Echo +3.69 % / -0.97 % of the peak: the impedance seen by the source (374.9 ohm) is not the terminal (359.24 ohm).
- Possible check: repeat with dx = 2.5 mm and see whether L'C' c^2 goes towards 1.

## Layout by case (2026-11-22): one code folder, one subfolder per case

```
c1/   c1.py, c1_*.py                  one copy of the code
      h100/      h = 100 mm, PML, terminals 359.24 ohm (default; results of 2026-11-21)
      h35/   h = 35 mm, PML, terminals 296.29 ohm, same grid, step and window as h100
      h100_dx10/ h = 100 mm, PML, terminals 359.24 ohm, dx = 10 mm
```

Kept on 2026-11-22 after the study: `h100` (the standard test), `h35` (the height of cases B and C; with `h100` it carries the height result) and `h100_dx10` (the grid check; it runs in seconds). The other five cases were removed from the folder; they are recorded below.

Choose the case with `C1_CASE` (subfolder name or path; default `h100`), set before cell 1 or on the command line. Each subfolder holds its `c1.fdtd.json`,
the `.exc` file and the outputs of its runs (including `c1_prepost.png` and `c1_near_end_shape.png`). Code changes for this: `c1.py` (`CASE_DIR`, message listing
the available folders), `c1_common.EXPECTED` (only length, radius, dx are fixed; height, grid and terminals come from each JSON), `c1_input` (the terminal check is
a PASS when the terminals equal the analytic Zc within 2 %, otherwise an INFO with Gamma_S and Gamma_L), `c1_calibration` and `c1_echo` (the analytic line is called
"matched" only when it is; otherwise the analytic echo -Gamma_L is printed). The h100 post-processing was re-run in the new layout: identical, 31 PASS / 0 WARN / 0 FAIL.
Static check of the kept cases: `h100` 17 PASS / 0 WARN / 0 FAIL; `h35` 17 PASS / 0 WARN / 0 FAIL; `h100_dx10` 16 PASS / 1 WARN (dx differs from 5 mm) / 0 FAIL.

Runs of 2026-11-22 (columns marked "removed" are of cases no longer in the folder; post-processed with `--no-run`; h100_dx10_r1: 27 PASS, 2 WARN = dx and radius differ from `c1_common.EXPECTED`, 0 FAIL; h35_mur (Mur, 296.29 ohm): 28 PASS, 1 WARN = time step corrected by the solver, 0 FAIL; h100_dx10: 28 PASS, 1 WARN = dx differs from 5 mm, 0 FAIL, 4 MPI ranks, 17 s; h35_313ohm (313.1 ohm): 27 PASS, 1 WARN = time step corrected by the solver, 0 FAIL; h35: 31 PASS, 0 WARN, 0 FAIL; h70: 31 PASS, 0 WARN, 0 FAIL, 388 Mcells/s; the three PML cases: 7.65 Mcells, 6490 steps):

| Quantity | h35_mur (Mur, 296.29 ohm; removed) | h35 (PML) | h70 (PML; removed) | h100 (PML) | h100_dx10 (PML, dx = 10 mm) | h100_dx10_r1 (PML, dx = 10 mm, r = 1 mm; removed) | h35_313ohm (Mur, 313.1 ohm; removed) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Terminals R_S = R_L | 296.29 ohm | 296.29 ohm | 337.85 ohm | 359.24 ohm | 359.24 ohm | 317.68 ohm | 313.1 ohm |
| Delay far - near | 6.712 ns (+40 ps, v = 0.994 c) | 6.712 ns (+41 ps, v = 0.994 c) | 6.772 ns (+100 ps, v = 0.985 c) | 6.819 ns (+148 ps, v = 0.978 c) | 6.789 ns (+118 ps, v = 0.983 c) | 6.802 ns (+131 ps, v = 0.981 c) | 6.710 ns (+39 ps, v = 0.994 c) |
| Impedance seen by the source | 313.1 ohm (analytic 296.3) | 313.3 ohm (analytic 296.3) | 354.3 ohm (analytic 337.9) | 374.9 ohm (analytic 359.2) | 374.4 ohm (analytic 359.2) | 333.6 ohm (analytic 317.7) | 313.1 ohm (analytic 296.3) |
| Effective L' / analytic | +6.31 % | +6.40 % | +6.45 % | +6.66 % | +6.04 % | +7.06 % | +6.30 % |
| Effective C' / analytic | -4.80 % | -4.86 % | -3.21 % | -2.05 % | -2.34 % | -2.89 % | -4.82 % |
| L'C' c^2 | 1.0121 | 1.0123 | 1.0303 | 1.0447 | 1.0356 | 1.0397 | 1.0117 |
| Peak near / far vs analytic | -2.76 % / -0.08 % | -2.80 % / -0.14 % | -2.39 % / -0.25 % | -2.15 % / -0.41 % | -2.08 % / -0.39 % | -2.46 % / -0.48 % | -2.69 % / +0.07 % |
| Echo at the near end (max / min) | +3.29 % / -0.15 % | +3.32 % / -0.17 % | +3.46 % / -0.61 % | +3.69 % / -0.98 % | +3.53 % / -0.86 % | +4.40 % / -1.20 % | +1.07 % / -1.13 % |
| RMS difference near / far | 1.39 % / 0.33 % | 1.27 % / 0.29 % | 1.22 % / 0.63 % | 1.26 % / 0.90 % | 1.20 % / 0.70 % | 1.44 % / 0.86 % | 1.37 % / 0.33 % |
| Shunt C fitted to the near-end shape | 8.7 fF | 11.9 fF | 37.5 fF | 56 fF | 48.9 fF | 46.3 fF | 6.8 fF |

Boundary test with matched terminals (`h35_mur` against `h35`, both h = 35 mm and 296.29 ohm, only Mur against PML differs): delay 6.712 ns in both (+40 and +41 ps), L'C' c^2 1.0121 and 1.0123, echo +3.29 % / -0.15 % and +3.32 % / -0.17 %, peaks -2.76 % / -0.08 % and -2.80 % / -0.14 %, RMS 1.39 / 0.33 % and 1.27 / 0.29 %. The boundary does not matter here; only the fitted shunt C differs (8.7 against 11.9 fF). With Mur the Report lists "CPML NO, Mur yes" (checked) and `Energy.dat` has fewer than 2 reports, so no energy decay is checked.

Dependence on dx, r and r/dx (three PML cases at h = 100 mm): (dx = 5 mm, r = 0.5 mm, r/dx = 0.1) `h100`; (10 mm, 0.5 mm, 0.05) `h100_dx10`; (10 mm, 1 mm, 0.1) `h100_dx10_r1`. The excess L'C' c^2 - 1 is 0.0447, 0.0356 and 0.0397, and the delay excess +148, +118 and +131 ps. Doubling dx at the same r (`h100` to `h100_dx10`) lowers the excess by about 20 %; doubling r at the same dx (`h100_dx10` to `h100_dx10_r1`) raises it by about 11 %; doubling both (same r/dx, `h100` to `h100_dx10_r1`) lowers it by about 11 %, which is the product of the other two (0.80 x 1.11). So the excess does not follow r/dx alone (the two cases with r/dx = 0.1 differ by 11 %) nor dx alone, and it changes only by 10 to 20 % over a factor 2 in each; it stays between 3.6 % and 4.5 % and shows no sign of going to 1. Three points cannot fix the exponents, and in the two cases with the same dx, r and r/dx are not separable. The impedance seen by the source is 333.6 ohm against 317.7 ohm of the terminals (+5.0 %), hence an echo of +4.40 % / -1.20 %, larger than in `h100` (+3.69 %); L' is +7.06 % and C' -2.89 % from the analytic values.

Reading: `h35` and the old `h35_313ohm` give the same delay (6.712 and 6.710 ns) although the boundary (PML or Mur) and the terminals differ, and both reproduce the ~6.70 to 6.715 ns of the original runs; `h35` differs from `h100` only in the height (and the terminals, which do not set the delay). So the delay excess depends on the height (+39 ps at h = 35 mm, +148 ps at h = 100 mm) and not on the boundary. The effective L' is about +6.5 % above the analytic value in every case, while the C' deficit falls from -4.8 % to -2.1 % as h grows, which is what raises L'C' c^2 from 1.012 to 1.045. The three PML heights (35, 70, 100 mm) give an excess L'C' c^2 - 1 of 0.012, 0.030 and 0.045 and a delay excess of +41, +100 and +148 ps: roughly linear in h (about 1.6 ps/mm), with L' nearly constant (+6.4 %, +6.45 %, +6.7 %) and the C' deficit falling (-4.9 %, -3.2 %, -2.1 %). Grid test (`h100_dx10`, same case with dx = 10 mm and r/dx = 0.05): the excess falls when the cell doubles, not when it shrinks. L'C' c^2 goes from 1.0447 (dx = 5 mm) to 1.0356 (dx = 10 mm), the delay excess from +148 to +118 ps (both about -20 %), L' from +6.66 % to +6.04 % and the C' deficit from -2.05 % to -2.34 %; echo (+3.53 %), RMS (1.20 / 0.70 %) and impedance seen (374.4 ohm) hardly move. So the excess is not ordinary numerical dispersion of the 3D grid (that would make the coarse grid slower, not less slow), and it does not vanish at the finer cell tried. With two cell sizes and r/dx changing together (0.05 and 0.1) the dependence on dx cannot be separated from a dependence on r/dx, and no extrapolation to dx -> 0 is justified. Three heights are a trend, not a law; the mechanism is still the coupling between the MTLN wire and the 3D field (not isolated). The impedance seen by the source is 313 ohm at 35 mm in both runs, i.e. the 313.1 ohm that cases B and C use as `ZC_A_MEAS` is the solver's own value, not the analytic one (296.3 ohm). In `h35_313ohm` the terminals (313.1) match the solver's line, so the echo is small (+-1.1 %); in `h35` they match the analytic line, so the echo is +3.32 % (Gamma_L of the solver's line). The Mur run (`h35_313ohm`) left `Energy.dat` empty (fewer than 2 reports), so no energy decay is checked there.

Status: the study of case A is closed with these runs. `h35_mur` (Mur and matched terminals) confirmed that the boundary does not matter (above); the finer-grid test with r = 0.5 mm was unstable (dx = 2.5 mm, r/dx = 0.2) and one with r = 0.25 mm (dx = 2.5 mm, r/dx = 0.1, Zc about 400.7 ohm) was not prepared. Cases B and C keep their own folders.

## Open points

1. Confirm in the Report of the 35 mm runs (cases B and C) that they ran the MTLN path and had a working Mur.
2. Apply the lowercase boundary check (`border_check`) to cases B and C, which use `mur`.
3. The tutorial and the READMEs of c2, c3 and D.28 still assume Holland with MTLN off, and cases B and C may still rely on a measured Zc
   from the 35 mm run (`c2_common.ZC_A_MEAS`); they were not touched here.

## Cases removed (2026-11-22)

The kept `h35` was called `h35_pml` during the study (PML, 296.29 ohm); the removed Mur case with the same terminals is `h35_mur` here. Their numbers are in the table above. Each one can be rebuilt from a kept case by changing only what is listed (copy the folder, edit `c1.fdtd.json`, keep the `.exc` file; the checks of `c1_input` then tell whether the edit is consistent).

| Case | What it was | Rebuild from | Result (see table) |
|---|---|---|---|
| `h35_mur` | the original input (branch `velectra-xtalk`, commit 52cbf48) with the terminals set to the physical Zc: h = 35 mm, `mur` borders, 296.29 ohm (terminal id 5 "Terminal296p29ohm"), time step 8.66 ps / 4618 steps (the solver corrects it to 7.7033 ps / 5191, WARN) | `h35`: boundaries `mur` instead of `pml`, `timeStep` 8.66e-12, `numberOfSteps` 4618 | 28 PASS, 1 WARN; same delay and echo as `h35`: the boundary does not matter |
| `h35_313ohm` | the same original input with the terminals as in the commit, 313.1 ohm (the Zc of the solver's own wire at 35 mm), `mur` | as `h35_mur` with 313.1 ohm | 27 PASS, 1 WARN; echo +1.07 % / -1.13 %, delay 6.710 ns; analytic line not matched (Gamma = +0.028) |
| `h70` | h = 70 mm, `pml`, terminals 337.85 ohm, same grid, step and window as `h100` | `h100`: wire at [74, 18, 0] to [74, 18, 400] (plane at j = 4), terminals 337.85 ohm | 31 PASS; delay +100 ps, L'C' c^2 1.0303; 388 Mcells/s |
| `h100_dx10_r1` | `h100_dx10` with radius 1 mm (r/dx = 0.1) and terminals 317.68 ohm (terminal id 3 "Terminal317p68ohm") | `h100_dx10`: `radius` 0.001, `resistance` 317.68 | 27 PASS, 2 WARN (dx and radius differ from `c1_common.EXPECTED`); delay +131 ps, L'C' c^2 1.0397, echo +4.40 % |
| `h100_dx2p5` (unstable) | `h100` with dx = 2.5 mm and r = 0.5 mm (r/dx = 0.2) | not worth rebuilding | blew up at once: energy `NaN` from the first report (1.77 ns), near-end current alternating in sign and growing by about 3.3 per step from the first step, `NaN` after 78 steps (about 0.3 ns); the check r/dx <= 0.11 (FAIL) now stops it |
