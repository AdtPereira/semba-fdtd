# c3_smoke - short smoke test before the long case C run

Purpose: avoid spending about one hour of simulation on a defective case C. A wrong wall cell, a leak in the shell,
a wrong wire impedance or a wrong sign shows up in the first 35 ns, and that run takes about 7 minutes
(11.3 Mcells x 4545 steps at 120 Mcells/s). All text in this document and in the script is in English.

Status: used on the real case C on 2026-11-16 (7 min, 124 Mcells/s); it passed, and the full run that followed also
passed all checks (51 PASS). See "Result of the real smoke run" below.

## Gates before the full run

| Gate | What | Time | Command |
|---|---|---|---|
| 0 | Static check of the full `c3.fdtd.json`: grid, wires, terminals, time plan, cylinder wall (ring, j = 4, closed) | seconds | cell 1 of `c3.py`, or the first cell of `c3_smoke.py` |
| 1 | Wire model: Holland with MTLN off, confirmed in the Solver dialog (not automated: `Report.txt` prints "MTLN wires" in any MTLN build) | minutes | manual |
| 2 | Smoke run of the first 35 ns and its analysis | about 7 min | `python c3_smoke.py` |
| 3 | The full run, watched (progress bar; the `stop` file stops it cleanly with the probes flushed) | about 1 h | `python c3.py` |

An empty `Log` is not a pass: the summary prints "0 checks ran -> NOTHING WAS VALIDATED" when no check was recorded.
The gate-0 result is only valid if the summary lists the checks (29 PASS, 0 FAIL for the case C file written on 2026-11-16).

## What `c3_smoke.py` does

1. Runs the static check of `c3.fdtd.json` (gate 0) and writes `c3_smoke.fdtd.json`: the same file with only
   `numberOfSteps` cut to 35 ns (override with `C3_SMOKE_NS`; the FE pulse needs at least 32 ns).
2. Runs it with the same progress bar as `c3.py` (outputs are named `c3_smoke.fdtd_*`, so the full case is never mixed in).
3. Analyses what is already known at 35 ns against the same references as the full run, and prints a verdict.

| Check | Threshold | Level |
|---|---|---|
| Record holds the NE and FE pulses (t0 + T + 4 sigma = 31.7 ns) | | FAIL |
| Aggressor near-end pulse centre on the source (half maximum) | 0.1 ns | FAIL |
| Aggressor delay equals L/c | 1 % | FAIL |
| Probe sign (positive near -> far) and polarity (NE positive, FE negative) | | FAIL |
| Zc effective against the cylinder value (about 308 ohm) | 5 % (FAIL), 2 % (WARN) | FAIL / WARN |
| NE and FE peaks against the analytic cylinder MTL | 10 % (FAIL), 5 % (WARN) | FAIL / WARN |
| NE and FE in time with the MTL (cross-correlation lag) | 0.15 ns | WARN |
| Aggressor currents / NE-FE waveforms against the MTL (RMS of the peak) | 2 % / 10 % | WARN |
| B -> C change of the NE and FE peaks against the analytic change (about -10 % / -10.5 %) | 3 points | FAIL |
| B -> C change between -6 % and -14 % | | WARN |

The B -> C check is the one that catches a plane-like or leaking cylinder: with a wall that does not change the
field, NE and FE stay within 10 % of the analytic cylinder value but the change B -> C is about -1 % instead of -10 %.

It also prints the time of the full run from the speed measured in the smoke run.

## What the smoke test does not see

Late-time instability, the ring-down after 175 ns (case B ended at -41.7 dB, thin margin) and the transfer function
need the long run. During the full run, create the `stop` file (or use the interrupt of the interactive window) to
flush the probes, then read the `.dat` files; an exponential growth of the currents is the sign of instability. The wire
model (Holland, MTLN off) is gate 1, a manual check in the Solver dialog.

## Usage

```
python c3_smoke.py             # gate 0, write the 35 ns copy, run it, analyse, verdict
python c3_smoke.py --no-run    # analyse a smoke run already made
```

Environment: `SEMBA_EXE`, `REPO_ROOT`, `C3_NO_RUN=1`, `C3_MCELLS_PER_S`, `C2_RESULTS` (path of `c2_results.npz`),
`C3_SMOKE_NS`. Needs next to it: `c3.fdtd.json`, `gauss_sigma2p5ns_t15ns.exc`, the `c3_*.py` modules and `c2_results.npz`
(found as for `c3.py`). In the interactive window run the cells in order; skip cell 2 to avoid repeating the solver.

Cell 2 holds the line `solver = None if NO_RUN else run_solver(case, REPO, SEMBA_EXE, SPINIT_FOLDER)`. If that line is
commented out (as in the copy used on 2026-11-16), the script does not launch the solver: run semba-fdtd yourself on
`c3_smoke.fdtd.json` and the script reads the `.dat` files, exactly as with `--no-run`. Uncomment it to let the script
run the solver with the progress bar.

Create `log` once (cell 0) and do not create it again before cell 1; a new `Log()` after `load_case` would leave the
summary empty.

## Result of the real smoke run (2026-11-16, 35.008 ns, 4545 steps, 7 min, 124 Mcells/s)

Recomputed from the four probe `.dat` files of the run:

| Quantity | Smoke run | Reference |
|---|---|---|
| Aggressor near end, peak | 2.806 mA at 14.97 ns | pulse centred on the source (15 ns) |
| Aggressor far end, peak | 4.823 mA at 21.68 ns | L/c = 6.671 ns after the near end |
| NE peak | +7.10 mV at 15.2 ns | +7.10 mV (analytic cylinder, tutorial 5.1) |
| FE peak | -10.20 mV at 21.6 ns | -10.22 mV |
| B -> C change | NE -9.2 %, FE -9.6 % | about -10.0 % / -10.5 % (within 3 points) |

The figure `c3_smoke.png` shows the FDTD curves on the analytic cylinder, with case B above. The full run then gave the
same peaks (NE +7.104 mV, FE -10.196 mV). The total energy rises and falls as the pulse enters and leaves the line (no
late growth in the first 35 ns). The `create_symlink` warning of `Warnings.txt` is harmless.

## How it was tested

Before the real run, on synthetic probe files, with the analytic cylinder MTL as the "FDTD" result and the real case B
numbers (`c2_results.npz`): a correct case C passes (0 FAIL); a plane-like result fails on the B -> C change
(-0.9 / -0.6 % against -10 %); a 20 % weak cross-talk fails the peak checks; a 12 % low aggressor current (Zc 15 % high)
fails the Zc check. The solver path of the script was tested with a mock semba-fdtd (plan, progress bar and report);
on the real run the solver was started outside the script and the script analysed the `.dat` files.
