"""Writes the trapezoid source of Paul D.28 (the book's SPICE card  VS 1 0 PULSE (0 1 0 12.5N 12.5N 7.5N 1000N)).

    python paul_d28_make_exc.py                       -> paul_d28.exc, 200 ns, 0.05 ns step (the book's .TRAN .2N 200N)
    python paul_d28_make_exc.py --window 500e-9 --out paul_d28_500ns.exc

The longer file (zeros after the pulse) lets the line ring down: with Gamma_L about -0.70 the NE voltage is still at -14 dB at
200 ns, and about 500 ns brings it below -40 dB. Set the Solver final time to the same value (5e-7 s) and the file name in the
generator's Magnitude. Two columns, time (s) and voltage (V)."""
import argparse
from pathlib import Path

import numpy as np

RISE, FLAT, FALL, STEP = 12.5e-9, 7.5e-9, 12.5e-9, 0.05e-9


def trapezoid(window=200e-9, step=STEP):
    t = np.round(np.arange(0.0, window + 0.5 * step, step), 12)
    return t, np.interp(t, [0.0, RISE, RISE + FLAT, RISE + FLAT + FALL], [0.0, 1.0, 1.0, 0.0], right=0.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--window", type=float, default=200e-9, help="length of the file in s (default 200e-9)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "paul_d28.exc"))
    a = ap.parse_args()
    t, v = trapezoid(a.window)
    np.savetxt(a.out, np.column_stack([t, v]), fmt="%.6e")
    print(f"{a.out}: {len(t)} samples, 0 to {t[-1] * 1e9:g} ns, step {STEP * 1e9:g} ns, peak {v.max():g} V")
