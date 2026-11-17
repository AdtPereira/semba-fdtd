# c3 - case C: two thin wires inside the closed PEC cylinder (cross-talk)

Third rung of the cross-talk validation ladder (A -> B -> C), Velectra project issue #5. Case C is case B with **one**
change: the ground plane becomes the closed PEC cylinder (R = 400 mm, L = 2 m). The pre/post-processing is the case B
pipeline (`c2_*`) with only the plane-dependent parts changed, plus a B -> C comparison.

Status: the real run of 2026-11-16 passed all checks, 51 PASS, 0 WARN, 0 FAIL (section 5).

All text in this document and in the scripts is in English.

## 1. Setup

| Item | Value |
|---|---|
| Grid | dx = 5 mm cubic cells, **168 x 168 x 400** cells (about 11.3 Mcells, twice B), origin (-0.42, -0.42, 0) |
| Wall | `Cylinder` (Part -> Cylinder): radius 400, height 2000, axis +Z, position (0, 0, 0); PEC; Mesh body |
| Wires, terminals, generator, probes, `.exc` | identical to B: aggressor [77,11,0] -> [77,11,400], victim [91,11,0] -> [91,11,400]; 50 ohm on all four ends; probes "Source probe", "Load probe", "NE probe", "FE probe" |
| Boundaries | PEC on the z faces, Mur on x and y (shielded by the cylinder) |
| Solver | **TimeStep 7.7033e-12 s, 38944 steps (299.9986 ns)**, Holland wire flavour, **MTLN off**; then semba-fdtd corrects nothing |
| Run time | 1.00 h at 122 Mcells/s in the real run (11.3 Mcells x 38944 steps); keep the laptop plugged in, high-performance power plan |

Before the long run, run cell 1 of `c3.py` on the generated `c3.fdtd.json` (the static check, no solver needed) and
confirm `numberOfCells` [168, 168, 400], `origin` [-0.42, -0.42, 0.0], the wire coordinates, and that the **meshed wall
sits at y = -400 (j = 4) at x = +-35**. If it lands at j = 3 or 5, h becomes 40 or 30 mm and the result is shifted by
the staircase, not by physics; use the conformal mesher if the dialog offers it.

## 2. Files

Keep all of them in one folder with `c3.fdtd.json` and the `.exc` file. For the B -> C cell, `c2_results.npz` (case B
numbers) is looked for, in this order, in `C2_RESULTS` (full path of the file), in the c3 folder, and in the sibling
folder `../c2/` (e.g. `testData/cases/velectra/c2/`), so no copy is needed; if it is found nowhere there is one WARN
and the B -> C checks are skipped. The modules do not depend on `c1` or `c2`.

| File | Role | Change from `c2_*` |
|---|---|---|
| `c3.py` | Orchestrator, cell by cell (`# %%`). No logic of its own. | new cell 7b (B -> C); saves `c3_results.npz`, `c3.png` |
| `c3_common.py` | Constants, tolerances, `Log`, `find_file`, `repo_paths`, `solver_time_plan` | `EXPECTED` cells [168,168,400], origin, wall j = 4, wire indices; `H_A`, `B_RESULTS`, `BC_TOL_*` |
| `c3_input.py` | Static check of `c3.fdtd.json` and the `.exc` | the plane check is replaced by the cylinder-wall checks (section 4); grid and wire indices are FAIL |
| `c3_analytic.py` | MTL reference | `lmat(pts, r, R=None)`: plane or circular tube (image of p is R^2 p/(p.p), distance term \|pi - img\| \|pj\| / R) |
| `c3_calibration.py` | Aggressor first pass | new WARN: Zc effective within 2 % of the cylinder value (about 308 ohm) |
| `c3_crosstalk.py` | NE/FE voltages, transfer function, ring-down | unchanged logic; messages quote case B |
| `c3_bc.py` | **new**: B -> C comparison | peak change and transfer-function change against `c2_results.npz` |
| `c3_solver.py` | Run plan, run with progress bar, probe loading | `C3_*` variables only |
| `c3_solver_report.py` | `Energy.dat`, `Report.txt` | the "case C estimate" is now "this run" |
| `c3_plots.py` | Four-panel figure | case B curves overlaid on panels 2 and 3 |
| `c3_smoke.py` | **new, separate**: 35 ns smoke test before the long run (own readme: `c3_smoke_readme.md`) | writes `c3_smoke.fdtd.json`, runs it, checks the first 35 ns |

## 3. Usage

```
python c3_smoke.py       # gate 2: 35 ns smoke run, about 7 min (see c3_smoke_readme.md)
python c3.py             # gate 3: check the input, run semba-fdtd with a progress bar, analyse the result
python c3.py --no-run    # check the input and analyse a run already made (takes seconds)
```

Gates before the full run: 0 = static check of `c3.fdtd.json` (seconds, cell 1 of `c3.py`); 2 = smoke test; 3 = the full
run, watched through the progress bar (the `stop` file ends it cleanly with the probes flushed).

Environment variables: `SEMBA_EXE`, `REPO_ROOT`, `C3_NO_RUN=1` (same as `--no-run`), `C3_MCELLS_PER_S` (speed for the
a-priori estimate), `C2_RESULTS` (path of `c2_results.npz`). In the VS Code / Jupyter interactive window run the cells
in order; skip cell 2 (`solver = None`) to avoid repeating the solver. A new run overwrites the `.dat` files of the case
folder: copy the folder of each case before rerunning.

Cells of `c3.py`: 1 `load_case`; 2 `run_solver`; 3 `load_currents`; 4 `calibrate`; 5 `crosstalk`; 6 `transfer_function`,
`late_time`; 7 `compare_with_mtl`, `compare_transfer`; 7b `compare_with_b`; 8 solver diagnostics; 9 summary; 10 saves
`c3_results.npz`; 11 `c3.png`.

Create `log = Log()` once, before cell 1, and pass that same object to every check. If the summary prints "0 checks ran
-> NOTHING WAS VALIDATED", the log was empty (for example `load_case` was not run on it): an empty summary is never a
pass. A valid summary for the case C file lists its checks (29 PASS at gate 0; 51 PASS after the full run).

## 4. Checks and thresholds

`FAIL` blocks the case; `WARN` is reported but not fatal; `INFO` is for reading. Everything of case B stays (terminals,
`.exc`, boundaries, time plan, calibration windows, NE/FE polarity, late-time, MTL comparison). What is new or changed:

| Check | Threshold | Level |
|---|---|---|
| Grid 168 x 168 x 400 of 5 mm, origin (-0.42, -0.42, 0) | exact | FAIL |
| Wire indices identical to B | exact | FAIL |
| PEC cross-section at k = 200 is a ring: mean radius within 1 cell of 400 mm, centre within 1 cell of the axis | 5 mm | FAIL |
| Meshed wall under both wires at j = 4 (h = 35 mm) | exact | FAIL |
| Wall closed: free-node flood fill from the grid corner cannot reach the wires (no leak) | | FAIL |
| `timeStep` not corrected by the solver (7.7033e-12 s, 38944 steps) | | WARN |
| Zc effective within 2 % of the cylinder value (about 308 ohm; B measured 311.8 ohm) | 2 % | WARN |
| NE and FE peaks against the analytic cylinder MTL | 10 % | FAIL |
| B -> C change of NE and FE peaks against the analytic B -> C change (analytic: -10.0 % NE, -10.5 % FE) | **3 points** | FAIL |
| B -> C change about -10 % (between -6 % and -14 %) | | WARN |
| B -> C change of \|H_NE\|, \|H_FE\| against the analytic change, 3-100 MHz, resonance nulls skipped | **1.5 dB** | WARN |
| \|H_NE\|, \|H_FE\| against the cylinder MTL, 3-100 MHz | 2 dB | WARN |
| Late-time residual after 175 ns | -40 dB (B: -41.7 dB, thin margin) | WARN |

"Within a few percent" (tutorial 5.2) is not a number in the tutorial: the 3 points and the 1.5 dB above were chosen
here (`BC_TOL_PCT`, `BC_TOL_DB` in `c3_common.py`); change them there.

The analytic reference uses r_eff from case A: the plane formula of case A (h = 35 mm) turns Zc = 313.1 ohm into
r = 0.3778 mm (0.756 r), and that radius is applied inside the cylinder. Reference numbers (tutorial 5.1, reproduced by
`lmat`): l11 / l12 = 1.028 / 0.061 uH/m (r_eff = 0.75 r), single-wire Zc 308.1 ohm, NE +7.10 mV, FE -10.22 mV; plane
(B): +7.89 / -11.42 mV. With r_eff from case A (Zc 313.1 ohm) the cylinder reference is NE +7.118 mV, FE -10.245 mV and
Zc 307.6 ohm.

## 5. Results of the real run (2026-11-16, 1.00 h, 38944 steps, 122 Mcells/s)

Result: **51 passed, 0 warnings, 0 failed -> case C validated**. Gate 0 on the real `c3.fdtd.json` gave 29 PASS (ring of
640 nodes, mean radius 400.0 mm, wall at j = 4, closed, dt = 7.70333e-12 s x 38944 steps, not corrected); the 35 ns smoke
run passed before the full run.

| Quantity | FDTD | Reference |
|---|---|---|
| NE peak | +7.104 mV | +7.118 mV (analytic cylinder, r_eff from case A): -0.2 % |
| FE peak | -10.196 mV | -10.245 mV: -0.5 % |
| B -> C change, NE | -9.16 % | MTL -9.97 % (difference +0.81 points) |
| B -> C change, FE | -9.65 % | MTL -10.47 % (difference +0.82 points) |
| Zc effective | 306.3 ohm | 307.6 ohm (cylinder, r_eff): -0.43 % (case A 313.1, case B 311.8) |
| Delay | 6.707 ns | L/c = 6.671 ns: +0.54 % (v = 0.9946 c) |
| NE / FE lag against the MTL | +92 / -56 ps | case B: +68 / -133 ps |
| RMS against the MTL | aggressor 0.11 / 0.03 %, NE 0.76 %, FE 0.29 % | |
| H_NE against the MTL, 3-100 MHz | rms 0.44 dB, max 1.10 dB at 99 MHz | case B max 1.54 dB |
| H_FE against the MTL, 3-100 MHz | rms 0.10 dB, max 0.31 dB at 70.3 MHz | case B max 0.38 dB |
| B -> C change of H, mean 3-100 MHz | NE -0.85 dB, FE -0.86 dB | MTL -0.90 / -0.94 dB |
| Late-time residual after 175 ns | worst -42.7 dB (victim near end); others -48 to -68 dB | limit -40 dB |
| Energy decay | 96.6 dB (case B 117 dB) | limit 60 dB |

Full numbers and the A -> B -> C conclusion are in `claude/c3_results_summary_EN.md`.

How the pipeline was tested before the real run: `lmat` with the circular image reproduces the tutorial section 5.1 table;
`python c3.py --no-run` on a synthetic case passed (49 PASS); negative cases were caught (wall one cell high, a gap in the
shell, case B grid, stale `timeStep`). `c3_input.py` reads the PEC `cell` intervals that cross k = 200; if a later file
stores the wall in another form (conformal mesher, triangles), the wall check degrades to a WARN that lists the PEC
elements and does not stop the case; the analytic reference and all post-processing checks are independent of it.

## 6. Open items and pitfalls

- Confirm Holland with MTLN off in the Solver dialog; "Wire model: MTLN wires" in `Report.txt` is hard-coded and proves nothing.
- The late-time margin is thin (B -41.7 dB, C -42.7 dB, limit -40). If it fails in another geometry, lengthen the window
  and the `.exc` file (it covers 300 ns).
- The NE transfer-function deviation near 99 MHz (1.54 dB in B, 1.10 dB in C) is still unexplained; since it appears in
  both, it is a property of the FDTD thin-wire model, not of the wall geometry.
- The reactive end effect (about 0.29 pF, about 1 % of the pulse) is expected again.
- Still unconfirmed physically: the central screw at 270 degrees (k0 = 48) and the aggressor = left hole assignment
  (tutorial section 7); plate thickness and connectors are not modelled; the snapping costs about -2 % on NE/FE.
- Usable FDTD band 3 to 100 MHz; below 3 MHz use the analytic MTL (NE/FE rise at 20 dB/decade, about -94 dB at 5 kHz).
- The `create_symlink` warning in `Warnings.txt` is harmless when source and target are the same file.
