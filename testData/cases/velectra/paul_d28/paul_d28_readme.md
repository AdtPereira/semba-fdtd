# paul_d28 - Paul, Example 9.2 / Fig. D.28: two wires over a ground plane (NEXT / FEXT)

The textbook benchmark of the cross-talk validation ladder (Velectra project, issue #5). Paul, *Introduction to Electromagnetic
Compatibility*, 3rd ed., Appendix D.4.1, Fig. D.28. It was the first case built (October 2026, "Exemplo 9.2"); this version brings
its tests, validation and graphics to the level of the cases A, B and C (`c1`, `c2`, `c3`). It replaces `paul_d28_prepost.py`
(a run-and-plot script with no checks).

All text in this document and in the scripts is in English.

## 1. Setup

| Item | Value |
|---|---|
| Wires | two bare #20 AWG, r = 16 mil = 0.4064 mm, h = s = 20 mm, line 4.674 m (x = 102 to 4776 mm, y = 190 and 210 mm) |
| Risers | each wire goes up 20 mm from the plane at both ends (polyline: base near, top near, top far, base far) |
| Plane | PEC cell interval at k = 0 over the whole x-y footprint, lower z face PEC |
| Grid | graded, 247 x 30 x 15 cells (0.278 Mcells with the PML): 6 mm along the line near the ends, growing to 24 mm; 5 mm across near the wires |
| Boundaries | PML on the five other faces (4 to 5 h of margin) |
| Terminals | 50 ohm series resistors at the four bases |
| Source | 1 V trapezoid, 12.5 / 7.5 / 12.5 ns (book: `PULSE (0 1 0 12.5N 12.5N 7.5N 1000N)`), file `paul_d28.exc`, 0 to 200 ns, 0.05 ns step |
| Probes | four wire-current probes at the tops of the risers: generator wire near / far (`Wire probe001`, `003`), receptor wire near / far (`002`, `004`) |
| Window | 200 ns (book: `.TRAN .2N 200N`) |
| Wire model | MTLN wires: the only wire model of this solver build (confirmed 2026-11-17) |

Roles (generator wire, receptor wire, which probe is where) are found from positions in the JSON, not from names. Voltage
conventions (current positive along the polyline): `V_NE = -R_NE * I_vic,near`, `V_FE = +R_FE * I_vic,far`.

## 2. Files

Keep all of them in one folder with `paul_d28.fdtd.json` and `paul_d28.exc`.

| File | Role |
|---|---|
| `paul_d28.py` | Orchestrator, cell by cell (`# %%`). No logic of its own. |
| `paul_d28_common.py` | Constants and tolerances, the book's numbers, `Log`, `find_file`, `repo_paths`, `solver_time_plan`. |
| `paul_d28_input.py` | Static check of the JSON and the `.exc`; roles from positions; the three analytic references. |
| `paul_d28_solver.py` | Run plan, run with progress bar and ETA, clean stop, probe loading (same code as c3). |
| `paul_d28_calibration.py` | Generator wire on the first pass: 50 % edges, delay, Zin, effective radius, probe sign. |
| `paul_d28_crosstalk.py` | NE / FE voltages, extremes, plateaus, polarity, transfer functions, late-time tail. |
| `paul_d28_analytic.py` | MTL reference (exact in frequency), independent modal-bounce solution, book check, comparisons. |
| `paul_d28_solver_report.py` | Checks of `Energy.dat`, `Report.txt` and `Warnings.txt`. |
| `paul_d28_plots.py` | Six-panel figure. |
| `paul_d28_make_exc.py` | Writes the trapezoid `.exc` (200 ns as in the book, or longer with `--window`). |
| `paul_d28_selftest.py` | Offline tests (section 6). |

## 3. Usage

```
python paul_d28.py             # check the input, run semba-fdtd (about 2 to 3 min), analyse the result
python paul_d28.py --no-run    # check the input and analyse a run already made (takes seconds)
python paul_d28_selftest.py    # offline tests, no solver
```

For `--no-run` the folder must already hold the four probe `.dat` files; `Energy.dat`, `Report.txt` and `Warnings.txt` are
optional. Environment variables: `SEMBA_EXE`, `REPO_ROOT`, `D28_NO_RUN=1`, `D28_MCELLS_PER_S` (speed for the a-priori estimate).
Create `log = Log()` once and pass it to every check; "0 checks ran -> NOTHING WAS VALIDATED" means the log was empty.

Cells of `paul_d28.py`: 1 `load_case`; 2 `run_solver`; 3 `load_currents`; 4 `book_check`; 5 `calibrate`; 6 `crosstalk`;
7 `compare_with_mtl`; 8 `transfer_function`, `compare_transfer`, `late_time`; 9 `energy_check`, `report_check`;
10 `log.summary()`; 11 saves `paul_d28_results.npz`; 12 writes `paul_d28.png`.

## 4. The references

Three analytic references, all lossless, homogeneous medium, exact in frequency (chain matrix) and brought to time by FFT:

| Reference | Radius | Terminals | Use |
|---|---|---|---|
| book | 0.4064 mm | ideal 50 ohm | the textbook answer (+23.3 / -32.0 mV); information |
| r_eff, ideal | 0.7555 r = 0.3070 mm | ideal 50 ohm | isolates the thin-wire effect |
| **primary**: r_eff + risers | 0.3070 mm | 50 ohm + 15.5 nH in series at each end | pass / fail |

- **r_eff = 0.7555 r** is the factor measured in case A (Zc 313.1 ohm for r = 0.5 mm, h = 35 mm). It is a factor of the radius,
  not a fixed length: here the radius is different (0.4064 mm) and the measured Zin gives 0.758 r.
- **Riser inductance** `L = mu0 h / (2 pi) (ln(2h/r_eff) - 1)` = 15.5 nH, from the formula, not fitted. The risers are in the
  FDTD model and not in the book's ideal line.

## 5. Checks and thresholds

`FAIL` blocks the case; `WARN` is reported but not fatal; `INFO` is for reading.

Input (`paul_d28_input.py`): grid lists match `numberOfCells`; adjacent cell ratio at most 1.3; polylines are base-near /
top-near / top-far / base-far; both wires have the same extent and height; lossless wire of equal radius; r/dy at most 0.11;
four pure 50 ohm terminals; PEC plane covers the whole footprint at k = 0 and all four ends sit on it; margin to the
absorbing faces at least 4 h (WARN); lower z face PEC, the other five Mur or PML; generator is a voltage source at the
generator-wire near base; four wire-current probes mapped to roles; excitation is the book's PULSE (12.5 / 7.5 / 12.5 ns, 1 V,
uniform axis); geometry equals the book; WARN if semba-fdtd will correct the time step (the exact dt, steps and window are printed);
window 200 ns covered by the `.exc`.

Reference (`book_check`): L and C equal `SPICEMTL_D28.OUT` to 5 digits; modal Z0 (323.42 / 226.92 ohm) and delay equal the
book's subcircuit; `|V_NE / V_S node|` at 10 kHz is -66.2 dB (Fig. D.30); the frequency-domain solver agrees with an
independent time-domain (modal bounce) solution within 0.5 % RMS.

Generator wire (`paul_d28_calibration.py`), first-pass windows only:

| Check | Threshold |
|---|---|
| Near-end 50 % edges against the source | within 0.15 ns (the riser adds about 0.07 ns); WARN |
| Delay far - near (both edges) equals L/c | within 1 % |
| Probe sign | positive near -> far |
| Zin = V_S / I - R_S on the plateau against the primary reference | within 2 % (WARN) |
| V_S / I - R_S flat on the plateau | drift below 1 % (WARN) |

`Zin = Zc11 - Zc12^2 / (Zc22 + R_NE)`: the receptor wire, loaded by R_NE, lowers the input impedance (285.2 ohm; Zc11 alone
would be 292.0 ohm).

Cross-talk and comparison (`paul_d28_crosstalk.py`, `paul_d28_analytic.py`):

| Check | Threshold |
|---|---|
| NE first plateau positive, FE negative | FAIL |
| NE maximum and FE minimum against the primary MTL | within 10 % (FAIL) |
| First plateaus against the primary MTL | within 5 % (WARN) |
| Deficit against the book equals the analytic change from r to r_eff | within 3 points |
| Cross-correlation lag of NE, FE | below 0.15 ns (WARN) |
| Generator-wire currents, RMS of the own peak | below 2 % (WARN) |
| NE and FE waveforms, RMS of the own peak | below 3 % (WARN) |
| The riser inductance does not worsen the agreement | WARN |
| `|H_NE|`, `|H_FE|` against the MTL cut at the same instant, 3 to 100 MHz | max 2 dB (WARN); bins within 4 MHz of n c / (2 L) (32, 64, 96 MHz) skipped |
| No late growth; tail after 150 ns equals the MTL tail | RMS below 3 % of the own peak (WARN) |

**Why the transfer function is compared on the same record.** The book's 200 ns window ends while the mismatched line
(Gamma_L about -0.70) is still ringing: the NE voltage is at -14 dB of its peak. The FDTD spectrum is that of the truncated
record, so the MTL time response is cut at the same instant and divided by the same source spectrum. The infinite-record MTL and
Paul Fig. D.30 (relative to the line-input node) are drawn in the figure for reference. The trapezoid has spectral nulls at
multiples of 50 and 80 MHz, so the band where the source is above -40 dB is only 3.8-48.8, 51.6-76.0 and 86.4-92.0 MHz.

**For a longer record:** `python paul_d28_make_exc.py --window 500e-9`, set the Solver final time to 5e-7 s and the file in the
generator's Magnitude. About 5 minutes. The NE voltage then rings down below -40 dB and the transfer function can be compared with
the infinite-record MTL.

Solver report (`paul_d28_solver_report.py`): energy lower at the end than at the first report (no 60 dB criterion on a record that
ends mid-ringing); speed above 20 Mcells/s (small grids run slower than the 120 Mcells/s of case B); no solver warning other than the
benign `create_symlink`.

## 6. Self-test (`paul_d28_selftest.py`)

16 tests, no solver: the `.exc` writer reproduces the delivered file; the delivered JSON passes with only the time-step WARN; the
reference matches the book; synthetic data (the primary MTL with 0.3 % noise) gives 0 FAIL and 0 WARN; a flipped current, a
15 % wrong victim amplitude, a 0.5 ns shift and the book's nominal-radius curves posing as FDTD are caught; and eight tampered
copies of the JSON (75 ohm terminal, PEC upper face, missing probe, 600 steps, wire moved up, 10 mm cell breaking the grading,
Mur lower face, wrong radius) plus a trapezoid with a 20 ns flat top are caught.

## 7. Result of the real run (12/11/2026, 3 min, 24602 steps, 53 Mcells/s, solver git 18691417)

Re-analysed from the raw files with this pipeline: **55 passed, 1 warning (the solver corrects the time step), 0 failed.**

| Quantity | FDTD | Primary MTL | Book (nominal r) |
|---|---|---|---|
| NE maximum | +21.23 mV | +21.11 mV (+0.6 %) | +23.4 mV |
| FE minimum | -29.90 mV | -29.63 mV (+0.9 %) | -32.1 mV |
| NE / FE first plateau | +21.22 / -29.77 mV | +21.04 / -29.53 mV (+0.8 % both) | +23.33 / -31.99 mV |
| Deficit against the book | NE -9.0 %, FE -6.9 % | predicted by r_eff: -9.8 % / -7.7 % | |
| Delay, generator wire | 15.626 ns (+0.22 % over L/c) | | 15.591 ns |
| Zin | 285.0 ohm | 285.2 ohm (-0.07 %) | 268.0 ohm |
| Effective radius from Zin | 0.7580 r | 0.7555 r (case A) | 1 r |
| RMS, generator wire near / far | 0.16 % / 0.11 % | | |
| RMS, NE / FE | 0.73 % / 0.54 % | | |
| Lag, NE / FE | 0 / -22 ps | | |
| `|H_NE|`, `|H_FE|` (same record) | max 1.36 dB (45.1 MHz), 0.72 dB (92.0 MHz) | rms 0.32 / 0.21 dB | |

What the numbers say:

- The deficit of about -9 % in NE and -7 % in FE against the book is **not discretisation**. The first tutorial reading (four cells
  across h and s) was wrong: the FDTD follows the curve with r_eff = 0.7555 r, the same factor as in cases A to C, and the
  measured Zin gives 0.758 r. That the deficit is smaller in FE than in NE is also what the analytic change predicts.
- Without the riser inductance the RMS against the MTL is 1.16 % (NE), 0.83 % (FE), 0.87 % and 0.61 % (generator wire); with
  15.5 nH per end (from the formula) it is 0.73, 0.54, 0.16 and 0.11 %. The aggressor edges show it: the near-end 50 % crossings
  are 59 ps after the source, close to the 67 ps the riser takes at c.
- The tail after 150 ns equals the MTL tail to 0.44 %: the ringing is physical.
- The reference reproduces the book: L and C to 5 digits, Z0 = 323.42 / 226.92 ohm, T = 15.5908 ns, and -66.22 dB at 10 kHz
  (Fig. D.30: -66.2 dB).

## 8. Where the first tutorial is out of date

`claude/tutorial_paul_D28_crosstalk_EN.md` still describes the first attempt. Items to correct:

| Tutorial | Now |
|---|---|
| Wires flavor HOLLAND, MTLN off (section 4, step 10) | This build has only MTLN wires; the effective radius is 0.7555 r. |
| Boundaries Mur, margin about 10 h (section 4, step 12) | The run uses PML on five faces, margin 4 to 5 h. |
| Grid options A (813 x 80 x 40) and B (813 x 66 x 37) | The run used a graded 247 x 30 x 15 (6 mm to 24 mm along the line). |
| Step 10: leave the time step empty | The JSON holds 1.5407e-10 s and the solver corrects it; set 8.128e-12 s and 24606 steps (200.0 ns; the check prints the numbers). |
| Probes at Vertex1 / Vertex4 (z = 0) | The probes sit at the tops of the risers (z = 20 mm). |
| Section 7.1: the -10 % plateau points to under-resolved mutual coupling | It is the thin-wire effective radius; refining does not remove it. |
| Section 7.2: run 2 pending, decision rule | Closed by the run of 12/11 (section 7 above). |

## 9. Notes and pitfalls

- A new run overwrites the `.dat` files. Copy the folder first to keep the old one; `paul_d28_results.npz` holds the numbers of the
  last analysed run.
- `Report.txt` prints the 12/11 run's `Total Mcells` = 0.278, more than 247 x 30 x 15 = 0.111 because of the PML layers; the run
  plan uses the number of an earlier Report when it exists.
- The `create_symlink` warning in `Warnings.txt` is harmless when source and target are the same file in the same folder.
- A flat top of 20 ns is the 50 % width of the pulse (12.5/2 + 7.5 + 12.5/2), not the flat top; the book's card gives 7.5 ns.
