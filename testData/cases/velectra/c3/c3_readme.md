# c3: two thin wires inside the closed PEC cylinder (cross-talk)

Third rung of the cross-talk validation ladder (c1 -> c2 -> c3) of the Velectra project (issue #5). Case C is case B (`c2`)
with **one** change: the ground plane becomes the closed PEC cylinder (R = 400 mm, L = 2 m). Case A (`c1`) showed that the
thin-wire behaviour of ``semba-fdtd`` (L' about +6 %, C' about -5 % at 35 mm) is a property of its wire model and not of the
boundary or of a radius change, and case B found the same offset with two wires.

All text in this document and in the scripts is in English.

**Three curves are compared, nothing else:**

1. **semba-fdtd**, with the terminals exactly as they are in `c3.fdtd.json`.
2. **The analytic lossless MTL of the wire pair inside the cylinder, with the physical radius** (0.5 mm; circular image method,
   chain matrix exact in frequency). There is no fitted `r_eff`.
3. **The semba-fdtd result of case B** (the plane), from the `c2` case with the same name.

The main test is the **change B -> C**: the FDTD change of NE, FE, the impedance and |H| against the change of the analytic line.
The offset of the thin-wire model is the same in B and C, so it cancels in this difference and what is left is the effect of the
wall. The differences to the analytic line itself (peaks, lags, RMS, transfer function) are results and are reported as INFO;
only gross errors fail.

One copy of the code serves two **cases**, one subfolder each (section 2): `50ohm` (the real experiment, the default) and
`matched` (same wires, grid and wall, terminals equal to the analytic Zc). Both use `pml` borders. Choose one with `C3_CASE`.

## 1. Setup

Common to both cases (grid, boundaries, terminals, time step and window are read from the `c3.fdtd.json` of the case; only the
expected geometry is fixed in `c3_common.EXPECTED`):

| Item | Value |
|---|---|
| Grid | dx = 5 mm cubic cells, **168 x 168 x 400** cells (11.3 Mcells; 13.5 Mcells with the 8 PML layers on each x/y face), origin (-0.42, -0.42, 0) |
| Wall | `Cylinder` (Part -> Cylinder), radius 400 mm, height 2000 mm, axis +z through the origin, PEC, meshed body (staircase) |
| Wires | those of case B, same indices: aggressor [77, 11, 0] -> [77, 11, 400], victim [91, 11, 0] -> [91, 11, 400]; radius 0.5 mm, separation 70 mm, L = 2 m; the meshed wall sits at j = 4 under both wires, so h = 35 mm |
| Boundaries | z faces PEC (end walls); x/y faces lowercase **"pml"** (physically irrelevant behind the closed shell; kept for the convention of c1 and c2) |
| Time step | 7.7033e-12 s in the JSON (= 0.8 dt_CFL, so semba-fdtd does not correct it, and the same value as c2, so the samples of B and C coincide) |
| Source | 1 V Gaussian generator (sigma = 2.5 ns, t0 = 15 ns) in series with the aggressor near-end terminal (= R_S), from `gauss_sigma2p5ns_t15ns.exc` |
| Probes | four current probes: aggressor near ("Source probe") and far ("Load probe"), victim near ("NE probe") and far ("FE probe"), positive along z |

The two cases:

| Case | Terminals (all four ends) | Window | Compared with | Why |
|---|---|---|---|---|
| `50ohm` | 50 ohm | 300 ns, 38944 steps | `c2/50ohm` | The real experiment. Gamma = -0.71 at every end, the line rings for about 150 ns |
| `matched` | 290.80 ohm = Zc of one wire in the cylinder (physical radius) | 40 ns, 5192 steps | `c2/matched` | No ring-down. The analytic FE is about 0, so the FE of the solver measures the imbalance between its inductive and capacitive coupling inside the cylinder |

Analytic reference with the physical radius (`lmat` with the circular image, points (+-0.035, -0.365) m, R = 0.4 m):

| | Plane (case B) | Cylinder (case C) |
|---|---|---|
| l11 / l12 | 0.9883 / 0.0693 uH/m | 0.9700 / 0.0606 uH/m |
| Zc of one wire | 296.29 ohm | 290.80 ohm |
| NE / FE peak, 50 ohm | +8.695 / -12.350 mV | +7.840 / -11.066 mV |

The analytic change B -> C is therefore **-9.84 % in NE and -10.40 % in FE** (50 ohm). In the `matched` case each geometry has its
own terminal (296.29 ohm in B, 290.80 ohm in C), and the analytic reference uses the right one in each.

Roles (aggressor, victim, which probe is where) are found from positions in the JSON, not from names. Voltage conventions:
V_NE = -R_NE * I_vic,near and V_FE = +R_FE * I_vic,far, so NE is expected positive and FE negative (in `matched` only NE is checked).

Before a long run, run cell 1 of `c3.py` on the case JSON (static check, seconds): grid, origin, wire indices, ring of the wall, the
wall at j = 4 under both wires, closed shell, terminals, pml, time plan. If the wall lands at j = 3 or 5, h becomes 40 or 30 mm and the
result is shifted by the staircase, not by physics.

## 2. Files

One code folder, one subfolder per case; each case folder holds its `c3.fdtd.json`, the `.exc` file and, after a run, its results.

```
c3/
  c3.py  c3_common.py  c3_input.py  c3_solver.py  c3_calibration.py  c3_crosstalk.py
  c3_analytic.py  c3_bc.py  c3_solver_report.py  c3_plots.py  c3_readme.md
  50ohm/     c3.fdtd.json  gauss_sigma2p5ns_t15ns.exc
  matched/   c3.fdtd.json  gauss_sigma2p5ns_t15ns.exc
  old/       history only, delete when not needed:
               old_mur/    the run of 2026-11-16 (Mur borders, r_eff), its JSON, results and the code before this version
               c3_smoke/   the 35 ns smoke test (no longer used)
               bench/      c3_bench.py, bench_runs/ and the csv of the MPI/OpenMP scaling study
```

For the B -> C cell, the results of case B of the same name are looked for in `../c2/<case>/` (next to the c3 folder, e.g.
`testData/cases/velectra/c2/50ohm/c2_results.npz`; `c2/matched/c2_results_matched.npz`), then in `c2/<case>/` inside the c3 folder,
then in the case folder, or in the file named by `C2_RESULTS`. They must come from PML runs of c2. If they are found nowhere there is
one WARN and the B -> C checks are skipped.

| File | Role |
|---|---|
| `c3.py` | Orchestrator, cell by cell (`# %%`). No logic of its own. |
| `c3_common.py` | Constants, `EXPECTED`, B -> C limits, `Log`, `find_file`, `repo_paths`, `solver_time_plan`, `zc_wire_plane`. |
| `c3_input.py` | Static check of the case JSON and the `.exc`; the cylinder wall (ring at k = 200, wall at j = 4, closed shell); roles from positions; analytic inductance matrices (cylinder and plane). |
| `c3_solver.py` | Run plan (PML layers included in the cell count), MPI launch, run with progress bar and ETA, clean stop, probe loading. |
| `c3_calibration.py` | Aggressor first pass: peaks, half-maximum centres, delay, Zc seen by the source, effective L' and C', charge. |
| `c3_crosstalk.py` | NE/FE voltages, peaks, polarity, transfer function, late-time ring-down (50 ohm case). |
| `c3_analytic.py` | MTL reference (`lmat` for the plane or the circular tube) and its difference to semba-fdtd; cross-correlation lags. |
| `c3_bc.py` | B -> C comparison against the c2 results of the same case. |
| `c3_solver_report.py` | `Energy.dat`, absorbing borders listed in the Report, `Warnings.txt`, `Report.txt`. |
| `c3_plots.py` | Two figures, two vertical panels each (`plot_time`, `plot_db`). |

## 3. Usage

```
python c3.py                       # case 50ohm: check the input, run semba-fdtd (MPI), analyse the result
C3_CASE=matched python c3.py       # case matched
python c3.py --no-run              # check the input and analyse a run already made (seconds)
```

MPI is the default, as in `c1` and `c2`: 8 ranks cutting in x (23 cells per rank with the PML), `SEMBA_EXE` defaults to
`<repo>/build-mpi/bin/semba-fdtd`, `mpirun` is `/usr/local/bin/mpirun` when it exists. On a machine without the MPI build set
`C3_MPI_RANKS=1`. Expected duration on ewing (c2 measured 380 Mcells/s with 8 ranks and PML): about 23 min for `50ohm`
(13.5 Mcells x 38944 steps) and 3 min for `matched`. The old serial run of 2026-11-16 took 1.0 h at 122 Mcells/s.

| Variable | Meaning |
|---|---|
| `C3_CASE` | Case subfolder (or a path); default `50ohm` |
| `C2_RESULTS` | Full path of the case B results file (see section 2) |
| `SEMBA_EXE`, `REPO_ROOT` | Executable (default `<repo>/build/bin/semba-fdtd`, or `build-mpi` when ranks > 1) and repository root (nearest parent holding `src_pyWrapper`) |
| `C3_NO_RUN=1` | Same as `--no-run` |
| `C3_MCELLS_PER_S` | Speed for the a-priori time estimate; otherwise the average of the last `Report.txt` |
| `C3_MPI_RANKS`, `C3_MPI_DIR`, `C3_MPI_FLAGS`, `MPIRUN` | MPI launch (default 8, `x`, none, `/usr/local/bin/mpirun`); never `z` with wires |

Cells of `c3.py`: 1 `load_case`; 2 `run_solver`; 3 `load_currents`; 4 `calibrate`; 5 `crosstalk`; 6 `transfer_function`, `late_time`;
7 `compare_with_mtl`, `compare_transfer`; 7b `compare_with_b`; 8 `energy_check`, `border_check`, `warnings_check`, `report_check`;
9 `log.summary()`; 10 saves the results; 11 `plot_time` and `plot_db`.

In the VS Code / Jupyter interactive window run the cells in order; to avoid repeating the solver set
`os.environ["C3_NO_RUN"] = "1"` before cell 1 or skip cell 2. Create `log = Log()` once, before cell 1: a summary that prints
"0 checks ran" is never a pass. A new run overwrites the `.dat` files of the case folder: copy the folder first if you need to keep a run.

## 4. Checks and thresholds

`FAIL` blocks the case, `WARN` is reported but not fatal, `INFO` is for reading. Everything of case B stays (terminals, `.exc`,
boundaries, time plan, calibration windows, NE/FE polarity, late-time, gross-error bands against the analytic line, solver files).
What is new or changed:

| Check | Limit | Level |
|---|---|---|
| Grid 168 x 168 x 400 of 5 mm, origin (-0.42, -0.42, 0) | exact | FAIL |
| Wire indices identical to B | exact | FAIL |
| PEC cross-section at k = 200 is a ring: mean radius within 1 cell of 400 mm, centre within 1 cell of the axis | 5 mm | FAIL |
| Meshed wall under both wires at j = 4 (h = 35 mm) | exact | FAIL |
| Wall closed: free-node flood fill from the grid corner cannot reach the wires | | FAIL |
| B -> C change of the NE peak (and of FE in `50ohm`) against the analytic change | 3 points / 10 points | WARN / FAIL |
| B -> C change of \|H_NE\| (and \|H_FE\| in `50ohm`) against the analytic change, 3-100 MHz, resonance nulls skipped | 1.5 dB / 6 dB | WARN / FAIL |
| Impedance seen by the source, B -> C, FDTD against analytic | | INFO |

The 3 points and 1.5 dB ("within a few percent", tutorial 5.2) are chosen here (`BC_WARN_PCT`, `BC_WARN_DB` in `c3_common.py`);
the FAIL limits only catch a leak in the shell or a wall at the wrong height.

`matched`: the analytic FE is about 0 in B and in C, so only NE is compared in the B -> C cell; the FE peaks are reported in mV.
As in c2, the bins within 8 MHz of n c / (2 L) are skipped in |H| for this case (|H_NE| has a true zero there).

## 5. Outputs

- `c3_time.png` in the case folder (two vertical panels): aggressor currents (mA) and NE/FE voltages (mV), with case B.
- `c3_db.png` (two vertical panels): transfer functions |V/V_S| in dB (with case B) and ring-down of the four currents in dB.
  Style as in c1 and c2: semba-fdtd solid (near / NE C0, far / FE C1), analytic MTL dotted (black / gray); case B thin solid C4 (NE) and C5 (FE).
- `c3_results.npz` (case `50ohm`) and `c3_results_matched.npz` (case `matched`), in the case folder: `t`, `V_NE`, `V_FE`,
  `I_agg_near`, `I_agg_far`, `I_vic_near`, `I_vic_far`.
- The printed summary and the solver files.

## 6. Expected, from the old run (confirmed in section 7)

Before the PML runs, the run of 2026-11-16 (Mur borders, serial, 1.0 h) was analysed again with this version, against the physical radius and against
the PML run of c2/50ohm, to show what to expect. The PML run of section 7 gave the same numbers.

| Quantity (old run, Mur) | semba-fdtd | analytic, physical radius | Difference |
|---|---|---|---|
| NE peak | +7.104 mV | +7.840 mV | -9.4 % |
| FE peak | -10.196 mV | -11.066 mV | -7.9 % |
| B -> C change, NE | -10.08 % | -9.84 % | -0.24 points |
| B -> C change, FE | -11.18 % | -10.40 % | -0.78 points |
| Impedance seen by the source, B -> C | 312.0 -> 306.3 ohm (-1.84 %) | 296.3 -> 290.8 ohm (-1.85 %) | |
| B -> C change of \|H\|, FDTD minus analytic, 3-100 MHz | | | max 0.24 dB (NE), 0.14 dB (FE) |
| \|H_NE\|, \|H_FE\| against the MTL | | | max 1.86 dB / 0.94 dB |
| Lag against the analytic line: aggressor near / far, NE, FE | | | -6 / 0, +131, -74 ps |
| Delay | 6.707 ns (v = 0.9946 c) | 6.671 ns | |

The offsets against the analytic line (NE -9 %, FE -8 %, Zc +5 %) are those of case B (-9.1 %, -7.1 %, +5.3 %); the change B -> C is
reproduced by the analytic line within a point. The old README compared the same data with a fitted r_eff and found -0.2 % / -0.5 %.

## 7. Results of the runs with PML

Runs of 2026-11-22 on ewing: MPI, 8 ranks cutting in x, 13.54 Mcells (grid plus PML layers), about 394 Mcells/s (`50ohm`, 38944 steps, 0.37 h)
and 390 Mcells/s (`matched`, 5192 steps, 3 min). Borders `pml` (Report: CPML yes, Mur no). B -> C uses the PML runs of c2 of the same case.

| Case | Result of `c3.py` |
|---|---|
| `50ohm` | **59 PASS, 0 WARN, 0 FAIL** |
| `matched` | **52 PASS, 1 WARN, 0 FAIL** (the WARN is \|H_NE\| 3.25 dB at 83.2 MHz, see below) |

**The change B -> C (the main test, the thin-wire offset cancels in it):**

| Quantity | semba-fdtd B -> C | analytic B -> C | Difference |
|---|---|---|---|
| `50ohm`, NE peak | +7.900 -> +7.104 mV (-10.08 %) | +8.695 -> +7.840 mV (-9.84 %) | -0.24 points |
| `50ohm`, FE peak | -11.479 -> -10.196 mV (-11.18 %) | -12.350 -> -11.066 mV (-10.40 %) | -0.78 points |
| `matched`, NE peak | +16.592 -> +14.719 mV (-11.29 %) | +17.554 -> +15.627 mV (-10.98 %) | -0.31 points |
| Impedance seen by the source, `50ohm` | 312.0 -> 306.3 ohm (-1.84 %) | 296.3 -> 290.8 ohm (-1.85 %) | |
| Impedance seen by the source, `matched` | 312.6 -> 306.7 ohm (-1.88 %) | 296.3 -> 290.8 ohm (-1.85 %) | |
| \|H\| change, FDTD minus analytic, 3-100 MHz | `50ohm`: max 0.24 dB (NE), 0.14 dB (FE) | | `matched`: max 0.60 dB (NE) |

The wall moves the FDTD result as the analytic cylinder says, within a point on the peaks and a few tenths of a dB on |H|. Limits were 3 points and 1.5 dB.
The staircase wall (about 1.5 mm farther from the wires than the true circle) is not separated from these -0.2 to -0.8 points; it is of that order.

**Against the analytic line itself (INFO; the offset of the thin-wire model, as in B):**

| Quantity | `50ohm` | `matched` |
|---|---|---|
| NE peak | +7.104 vs +7.840 mV (-9.4 %) | +14.719 vs +15.627 mV (-5.8 %) |
| FE peak | -10.196 vs -11.066 mV (-7.9 %) | -2.141 mV (analytic -0.030 mV) = 14.5 % of the NE peak |
| Case B for comparison | NE -9.1 %, FE -7.1 % | NE -5.5 %, FE -2.814 mV (17.0 % of NE) |
| Lag, aggressor near / far, NE, FE | -6 / 0, +131, -74 ps | -3 / +26, +39 ps (NE) |
| RMS, aggressor near / far, NE, FE | 0.77 / 0.53 %, 1.82 %, 1.32 % | 1.38 / 0.24 %, 3.93 %, 5.08 % (of the NE peak) |
| \|H_NE\|, \|H_FE\| against the MTL, max | 1.86 dB (43.6 MHz), 0.94 dB | 3.25 dB at 83.2 MHz (WARN); \|H_FE\| mean -53.4 dB, no reference |
| Delay | 6.707 ns (v = 0.9946 c) | 6.700 ns (v = 0.9957 c) |
| Effective L' / C' against the cylinder | +5.90 % / -4.54 % | +5.94 % / -4.78 % |
| Impedance seen by the source | 306.3 ohm (analytic 290.8) | 306.7 ohm |
| Charge conservation (near / far) | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| Late-time residual after 175 ns | worst -42.7 dB (victim near end); others -48 to -68 dB | not checked |
| Energy decay | 91.8 dB (22 reports) | 73.3 dB (3 reports) |

Reading:

- **Case C reproduces what case B showed, with the same offset.** L' about +6 %, C' about -4.5 to -4.8 %, delay excess about +29 to +36 ps,
  Zc seen about 5 % above the analytic one. It is the solver's wire model, not the cylinder; the cylinder changes everything else as the image method predicts.
- **`50ohm` gives the same numbers as the run of 2026-11-16 (Mur, serial), to the printed precision:** +7.104 / -10.196 mV, same lags, residuals and B -> C changes.
  With the shell closed, the x/y borders are physically irrelevant, so Mur against PML is not expected to change the result; this agrees with a closed shell without leaks.
  The numbers of section 6 are now confirmed by the PML run.
- **`matched`:** the NE peak is 5.8 % low (B: 5.5 %), the B -> C change is within 0.31 points. The FE of the solver is -2.14 mV (14.5 % of the NE peak)
  against about 0 for the ideal line: the imbalance between the inductive and capacitive coupling of the wire model. It is 24 % smaller than in B (-2.814 mV),
  while NE fell by 11 %. The Gamma_L seen at the far end is -0.026 (analytic 0).
- **The one WARN** is |H_NE| 3.25 dB at 83.2 MHz, next to the true zero at 75 MHz of the matched line (c2 found the FDTD null near 77.5 MHz); in c2/matched the same effect gave 3.84 dB at 83 MHz.
  It is consistent with the small delay excess moving the null, not with a wall effect: the B -> C change of |H_NE| in the band is at most 0.60 dB. Left visible, as in c2.
- **Late-time margin:** -42.7 dB against the -40 dB limit, as in the Mur run (thin but stable; no cavity ringing: the first modes, 220 and 232 MHz, are above the spectrum).
- **Wall:** the PEC ring has 640 nodes, mean radius 400.0 mm (min 396.5, max 403.3), the wall sits at j = 4 under both wires (h = 35 mm) and the shell is closed.

Ladder conclusion: A, B and C show the same thin-wire offset against the analytic line, and the changes plane -> cylinder (-10 % on NE and FE,
-1.9 % on the seen impedance, -1 dB on |H|) agree with the physical-radius analytic model within about one point, so the model is validated for the
change of geometry; the absolute level of NE and FE has the 5 to 9 % low offset of the solver's wire model.

## 8. Notes and pitfalls

- **Wall (staircase):** the meshed wall sits at y = -400 mm (j = 4) under both wires, while the true circle at x = +-35 mm is at
  y = -398.5 mm. The FDTD wall is therefore about 1.5 mm farther from the wires than the analytic cylinder, which lowers NE and FE a
  little (the previous README estimated about -2 %). It shows up as a small extra difference in the B -> C change, not as an error of the script.
- `c3_input.py` reads the PEC `cell` intervals that cross k = 200. If a later file stores the wall in another form (conformal
  mesher, triangles), the wall check degrades to a WARN that lists the PEC elements and does not stop the case; the analytic reference
  and the post-processing checks do not depend on it.
- Peak times are ill-conditioned on a flat-topped Gaussian, so timing uses half-maximum centres and cross-correlation lags.
- Probe `.dat` files are found by probe name, ignoring case, spaces and underscores; `Energy` files are ignored; the newest match
  wins. Samples are stamped at (n + 0.5) dt.
- Wire model (OpenSEMBA/fdtd, commit `cd069a6`): with MTLN compiled, a `wire` material goes through the MTLN path (the Report
  shows "Init MTLN Wires..."); the Holland code is not used. The line "Wire model: MTLN wires" is printed by any build compiled with
  MTLN. The in-cell inductance comes from `computeLCParametersFromRadius`, so the impedance the solver gives the wire is a result,
  not an input. Earlier versions of this README, of the case C handoff and of the tutorial said "Holland, MTLN off": that was wrong.
- The reactive end effect (about 0.29 pF, about 1 % of the pulse) is expected again.
- Not modelled: plate thickness and connectors. Usable FDTD band 3 to 100 MHz; below 3 MHz use the analytic MTL.
- The `create_symlink` warning in `Warnings.txt` is harmless when source and target are the same file in the same folder.
- The run plan printed by `c3.py` includes the PML layers in the cell count (13.5 Mcells), so its time estimate is realistic
  (the plan of c2 counts the grid only, 5.9 instead of 7.7 Mcells).
- `old/` is history. `old/old_mur/c3_results_mur_old.npz` is not the file that any script looks for.
