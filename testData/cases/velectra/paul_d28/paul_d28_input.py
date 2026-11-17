"""Static check of paul_d28.fdtd.json + the .exc file (no solver needed). Returns everything the later steps use.

The role of every wire and probe is found from positions, not from names: the aggressor is the cable that starts at the
generator node; a probe belongs to the wire whose top-near or top-far corner (or whose near / far end on the plane) it sits on.
The model is the one Elemwave writes: a PEC plane at z = 0, two polylines with 20 mm risers (base near, top near, top far,
base far), 50 ohm series terminals at the four bases, a voltage generator at the aggressor near base, four wire-current probes.
The grid is graded (6 mm along the line, 5 mm across it near the wires, growing towards PML boundaries), so the physical
positions are read from the cumulative step lists, not from index x cell size.
"""
import json
import math
import os
from types import SimpleNamespace

import numpy as np

from paul_d28_analytic import lmat, riser_L
from paul_d28_common import (C0, EXPECTED, R_EFF_FACTOR, ROLES, ZC_A_MEAS, find_file, solver_time_plan)

TOL = 1e-6                       # m: positions are compared on the 1 um level (the steps carry float noise)


def load_case(here, case_name, log):
    case_path = find_file(here, case_name)
    assert case_path is not None, f"{case_name} not found in {here}"
    work = case_path.parent
    os.chdir(work)                                   # semba-fdtd runs and writes its outputs here
    inp = json.loads(case_path.read_text())
    c = SimpleNamespace(case_path=case_path, WORK=work, inp=inp, HERE=here)
    _geometry(c, log)
    _boundaries_and_probes(c, log)
    _excitation(c, log)
    _time_and_references(c, log)
    return c


def _geometry(c, log):
    inp = c.inp
    mats = {m["id"]: m for m in inp["materials"]}
    elements = {e["id"]: e for e in inp["mesh"]["elements"]}
    coords = {k["id"]: k["relativePosition"] for k in inp["mesh"]["coordinates"]}
    c.mats, c.elements, c.coords = mats, elements, coords
    grid = inp["mesh"]["grid"]
    c.ncell, c.origin = list(grid["numberOfCells"]), list(grid["origin"])
    steps = {a: np.array(grid["steps"][a], float) for a in "xyz"}
    c.steps = steps
    c.nodes = [c.origin[i] + np.concatenate([[0.0], np.cumsum(steps[a])]) for i, a in enumerate("xyz")]
    log.check(all(len(steps[a]) == n for a, n in zip("xyz", c.ncell)), "grid: numberOfCells equals the length of the step lists",
              f"{c.ncell}")
    pos = lambda idx: tuple(float(c.nodes[a][idx[a]]) for a in range(3))          # index -> metres
    c.pos = pos
    c.dmin = [float(steps[a].min()) for a in "xyz"]
    ratio = {a: float(np.max(np.maximum(steps[a][1:] / steps[a][:-1], steps[a][:-1] / steps[a][1:]))) for a in "xyz"}
    log.check(max(ratio.values()) <= 1.3, "graded grid: ratio of adjacent cells at most 1.3 (no spurious reflections)",
              ", ".join(f"{a} {v:.2f}" for a, v in ratio.items()))
    log.info("grid", f"{c.ncell[0]} x {c.ncell[1]} x {c.ncell[2]} = {np.prod(c.ncell) / 1e6:.3f} Mcells, extent "
                     f"{c.nodes[0][-1]:.3f} x {c.nodes[1][-1] - c.nodes[1][0]:.3f} x {c.nodes[2][-1]:.3f} m, smallest cells "
                     f"{c.dmin[0] * 1e3:.1f} / {c.dmin[1] * 1e3:.1f} / {c.dmin[2] * 1e3:.1f} mm")

    gen = inp["sources"][0]
    c.gen = gen
    c.gen_idx = coords[elements[gen["elementIds"][0]]["coordinateIds"][0]]
    cables = [a for a in inp["materialAssociations"] if a.get("type") == "cable"]
    log.check(len(cables) == 2, "two cables (generator wire and receptor wire)", f"{len(cables)} found")
    W = {}
    for cab in cables:
        ids = elements[cab["elementIds"][0]]["coordinateIds"]
        idx = [coords[i] for i in ids]
        w = SimpleNamespace(name=cab["name"], idx=idx, p=[pos(i) for i in idx], mat=mats[cab["materialId"]],
                            term_n=mats[cab["initialTerminalId"]], term_f=mats[cab["endTerminalId"]])
        W["agg" if idx[0] == c.gen_idx else "vic"] = w
    log.check(set(W) == {"agg", "vic"}, "one wire starts at the generator node (aggressor), the other is the victim",
              f"{ {k: (w.name, w.idx[0]) for k, w in W.items()} }")
    agg, vic = W["agg"], W["vic"]
    c.W = W
    for k, w in W.items():
        p = w.p
        shape = (len(p) == 4 and abs(p[0][0] - p[1][0]) < TOL and abs(p[2][0] - p[3][0]) < TOL
                 and abs(p[1][1] - p[2][1]) < TOL and abs(p[1][2] - p[2][2]) < TOL and abs(p[0][1] - p[1][1]) < TOL
                 and abs(p[2][1] - p[3][1]) < TOL and p[0][2] < p[1][2] and p[3][2] < p[2][2] and p[2][0] > p[1][0])
        log.check(shape, f"{k} ({w.name}): polyline base near, top near, top far, base far (two vertical risers and a horizontal "
                  f"wire along x)", f"{[tuple(round(v, 4) for v in q) for q in p]}")
    c.length = agg.p[2][0] - agg.p[1][0]                                  # horizontal run between the risers
    c.h = agg.p[1][2] - agg.p[0][2]
    c.riser = c.h
    c.sep = abs(vic.p[1][1] - agg.p[1][1])
    c.radius = agg.mat["radius"]
    c.y = (agg.p[1][1], vic.p[1][1])
    c.xends = (agg.p[1][0], agg.p[2][0])
    log.check(abs(vic.p[1][0] - agg.p[1][0]) < TOL and abs(vic.p[2][0] - agg.p[2][0]) < TOL and abs(vic.p[1][2] - agg.p[1][2]) < TOL,
              "both wires have the same x extent and height")
    log.check(vic.mat["radius"] == c.radius, "same wire radius", f"r = {c.radius * 1e3:g} mm")
    log.check(agg.mat.get("resistancePerMeter", 0) == 0 and agg.mat.get("inductancePerMeter", 0) == 0,
              "wire material is lossless (R = 0, L = 0 per metre)", f"{agg.mat.get('resistancePerMeter')}, "
              f"{agg.mat.get('inductancePerMeter')}")
    log.check(c.radius / min(c.dmin[1], c.dmin[2]) <= 0.11, "wire radius / transverse cell below the thin-wire stability limit 0.11",
              f"r/dy = {c.radius / c.dmin[1]:.3f}")
    nz = agg.idx[1][2] - agg.idx[0][2]
    ns = vic.idx[1][1] - agg.idx[1][1]
    log.info("resolution", f"{nz} cells across h and {abs(ns)} across s (tutorial option A: 4; option B, graded: 8). With the "
                           f"thin-wire model this sets r_eff (about 0.756 r), not an error to be removed by refining")

    def res(term):
        t_ = term["terminations"][0]
        return t_["resistance"], t_["type"] == "series" and t_.get("capacitance", 0) >= 1e20
    for name, term in {"generator near (R_S)": agg.term_n, "generator far (R_L)": agg.term_f, "receptor near (R_NE)": vic.term_n,
                       "receptor far (R_FE)": vic.term_f}.items():
        log.check(res(term)[1], f"{name} terminal {term['name']} is a pure resistor", f"{term['terminations'][0]}")
    c.R = (res(agg.term_n)[0], res(agg.term_f)[0], res(vic.term_n)[0], res(vic.term_f)[0])
    log.check(all(abs(r - EXPECTED["R"]) < 1e-9 for r in c.R), "all four ends are 50 ohm", f"R_S, R_L, R_NE, R_FE = {c.R}")

    # the ground plane: PEC cell interval at k = 0 covering the whole x-y footprint
    pec_ids = [e for a in inp["materialAssociations"] if mats[a["materialId"]]["type"] == "pec" for e in a["elementIds"]]
    nx, ny, _ = c.ncell
    cover = np.zeros((nx, ny), bool)
    for eid in pec_ids:
        for (a0, a1, a2), (b0, b1, b2) in elements[eid].get("intervals", []):
            if a2 == 0 and b2 == 0:
                cover[min(a0, b0):max(a0, b0), min(a1, b1):max(a1, b1)] = True
    log.check(bool(cover.all()), "PEC plane at z = 0 covers the whole x-y footprint of the grid (infinite plane)",
              f"{100 * cover.mean():.1f} % of the {nx} x {ny} cells covered")
    log.check(all(abs(w.p[0][2]) < TOL and abs(w.p[3][2]) < TOL for w in W.values()),
              "all four wire ends sit on the plane (z = 0), where the 50 ohm terminals connect to ground")
    mg = {"y low": c.y[0] - c.nodes[1][0], "y high": c.nodes[1][-1] - c.y[1], "z top": c.nodes[2][-1] - c.h,
          "x low": c.xends[0] - c.nodes[0][0], "x high": c.nodes[0][-1] - c.xends[1]}
    log.check(min(mg.values()) >= 4 * c.h - TOL, "margin to the absorbing faces at least 4 h (the tutorial used about 9.5 h)",
              ", ".join(f"{k} {v / c.h:.1f} h" for k, v in mg.items()), "WARN")


def _boundaries_and_probes(c, log):
    inp, elements, coords, W = c.inp, c.elements, c.coords, c.W
    bnd = inp["boundary"]
    log.check(bnd["zLower"]["type"] == "pec", "lower z face is PEC (ground plane)", f"{bnd['zLower']['type']}")
    others = {k: bnd[k]["type"] for k in ("xLower", "xUpper", "yLower", "yUpper", "zUpper")}
    log.check(all(v in ("mur", "pml") for v in others.values()), "the other five faces absorb (Mur or PML), not PEC (no metal box)",
              f"{others}")
    if all(v == "pml" for v in others.values()):
        log.info("boundaries", "PML on five faces (the tutorial's step 12 suggests PML as the option to Mur)")
    log.check(c.gen["field"] == "voltage" and c.gen_idx == W["agg"].idx[0],
              "generator is a voltage source at the aggressor near end (base)", f"field = {c.gen['field']}, node {c.gen_idx}")
    c.PROBES = {}                                    # role -> probe name, e.g. "vic_near" -> "Wire probe002"
    where = {}
    for pr in inp["probes"]:
        p = coords[elements[pr["elementIds"][0]]["coordinateIds"][0]]
        for key, w in W.items():
            for slot, role in ((0, "near"), (1, "near"), (2, "far"), (3, "far")):
                if p == w.idx[slot]:
                    c.PROBES[f"{key}_{role}"] = pr["name"]
                    where[f"{key}_{role}"] = "base" if slot in (0, 3) else "top of the riser"
    log.check(set(c.PROBES) == set(ROLES) and len(inp["probes"]) == 4
              and all(pr["field"] == "current" and pr["type"] == "wire" for pr in inp["probes"]),
              "four wire-current probes: generator wire near/far, receptor wire near/far", f"{c.PROBES}")
    log.info("probe position", ", ".join(f"{k} at the {v}" for k, v in where.items()) + ". The terminal current equals the "
             "riser current (continuity); the riser inductance sits between the resistor and the probe")
    log.info("wire model", "this solver build has only MTLN wires (confirmed by the user, 2026-11-17), so 'Wire model: MTLN wires' "
             "in Report.txt is correct; the tutorial's 'Holland, MTLN off' no longer applies")


def _excitation(c, log):
    gen = c.gen
    exc_path = c.WORK / gen["magnitudeFile"]
    if not exc_path.exists():
        exc_path = find_file(c.HERE, gen["magnitudeFile"])
    assert exc_path is not None, f"excitation file {gen['magnitudeFile']} not found next to {c.case_path.name}"
    exc = np.loadtxt(exc_path)
    c.exc, c.exc_path = exc, exc_path
    c.source = lambda t: np.interp(t, exc[:, 0], exc[:, 1], left=0.0, right=0.0)
    t, v = exc[:, 0], exc[:, 1]
    pk = v.max()
    top = np.where(v >= pk - 1e-9)[0]
    t_rise0 = t[np.where(v > 1e-9)[0][0] - 1]
    t_top0, t_top1 = t[top[0]], t[top[-1]]
    t_fall1 = t[top[-1] + np.where(v[top[-1]:] <= 1e-9)[0][0]]
    c.rise, c.flat, c.fall = t_top0 - t_rise0, t_top1 - t_top0, t_fall1 - t_top1
    ex = EXPECTED
    log.check(abs(pk - ex["amp"]) < 1e-9 and v[0] < 1e-9, "excitation: 1 V peak, starts at 0 (no step)",
              f"peak {pk:g} V, first sample {v[0]:.1e}")
    log.check(abs(c.rise - ex["rise"]) < ex["dt_exc"] and abs(c.flat - ex["flat"]) < ex["dt_exc"]
              and abs(c.fall - ex["fall"]) < ex["dt_exc"],
              "source is the book's PULSE (0 1 0 12.5N 12.5N 7.5N): rise 12.5, flat top 7.5, fall 12.5 ns",
              f"{c.rise * 1e9:.2f} / {c.flat * 1e9:.2f} / {c.fall * 1e9:.2f} ns (a flat top of 20 ns is the 50 % width, "
              f"not the flat top)")
    log.check(np.allclose(np.diff(t), t[1] - t[0], rtol=1e-6), "excitation time axis is uniform", f"step {1e9 * (t[1] - t[0]):.3f} ns")
    f_ax = np.fft.rfftfreq(1 << 16, t[1] - t[0])
    S = np.abs(np.fft.rfft(v, 1 << 16))
    sdb = 20 * np.log10(S / S[0] + 1e-30)
    f_at = lambda db: f_ax[np.argmax(sdb < -db)]
    log.info("excitation", f"{exc_path.name}: window {t[-1] * 1e9:.0f} ns; spectrum -3 dB at {f_at(3) / 1e6:.1f} MHz, -20 dB at "
             f"{f_at(20) / 1e6:.0f} MHz, -40 dB at {f_at(40) / 1e6:.0f} MHz; nulls at multiples of 50 and 80 MHz")


def _time_and_references(c, log):
    inp = c.inp
    c.T = c.length / C0
    dt_json, n_json = inp["general"]["timeStep"], inp["general"]["numberOfSteps"]
    window_json = dt_json * n_json
    c.plan = plan = solver_time_plan(dt_json, n_json, *c.dmin)
    c.dt_solver = plan.dt_limit
    c.window = EXPECTED["window"]
    scale = 10 ** (math.floor(math.log10(plan.dt_limit)) - 3)
    rec_dt = math.floor(plan.dt_limit / scale) * scale                       # 4 digits, just below the limit
    rec_n = round(EXPECTED["window"] / rec_dt)
    log.check(not plan.corrected, "timeStep within the solver limit (no correction, no truncation of the window)",
              f"json {dt_json:.4e} s x {n_json} steps; limit {plan.dt_limit:.4e} s (0.8 dt_CFL)"
              + (f"; semba-fdtd will correct dt to {plan.dt:.4e} s and run {plan.n_final} steps = {plan.t_final * 1e9:.3f} ns. "
                 f"To avoid it set timeStep {rec_dt:.4g} s and numberOfSteps {rec_n} in the Solver dialog "
                 f"({rec_n * rec_dt * 1e9:.3f} ns)" if plan.corrected else f" -> {plan.t_json * 1e9:.3f} ns"), "WARN")
    log.check(window_json >= c.window * 0.999, "simulated window is the 200 ns of the book (.TRAN .2N 200N)",
              f"{window_json * 1e9:.1f} ns")
    dt_exc = c.exc[1, 0] - c.exc[0, 0]
    log.check(c.exc[-1, 0] >= window_json - 2 * dt_exc, "the .exc file covers the whole simulated window",
              f"file ends at {c.exc[-1, 0] * 1e9:.1f} ns, window {window_json * 1e9:.1f} ns")
    log.check(abs(c.h - EXPECTED["height"]) < TOL and abs(c.length - EXPECTED["length"]) < TOL and abs(c.sep - EXPECTED["sep"]) < TOL
              and abs(c.radius - EXPECTED["radius"]) < 1e-9 and abs(c.xends[0] - EXPECTED["x_ends"][0]) < TOL
              and abs(c.xends[1] - EXPECTED["x_ends"][1]) < TOL and abs(c.y[0] - EXPECTED["y_wires"][0]) < TOL
              and abs(c.y[1] - EXPECTED["y_wires"][1]) < TOL,
              "geometry as in the book and the tutorial (h = s = 20 mm, line 4.674 m from x = 102 to 4776 mm, y = 190 / 210 mm, "
              "r = 16 mil)", f"h = {c.h * 1e3:g} mm, s = {c.sep * 1e3:g} mm, L = {c.length:g} m, r = {c.radius * 1e3:g} mm, "
              f"x = {c.xends[0] * 1e3:g}..{c.xends[1] * 1e3:g}, y = {c.y[0] * 1e3:g}/{c.y[1] * 1e3:g} mm")

    # the references: the book (nominal radius, ideal terminals) and the thin-wire model of cases A-C
    c.r_eff_factor = R_EFF_FACTOR
    c.pts = [(c.y[0], c.h), (c.y[1], c.h)]
    r_eff = R_EFF_FACTOR * c.radius
    mk = lambda r, Lr: SimpleNamespace(r=r, L=lmat(c.pts, r), Lr=Lr)
    names = ["book: nominal radius, ideal terminals", f"r_eff = {R_EFF_FACTOR:.4f} r (case A), ideal terminals",
             f"r_eff = {R_EFF_FACTOR:.4f} r (case A) + riser inductance (primary)"]
    c.refs_names = names
    c.refs = {names[0]: mk(c.radius, 0.0), names[1]: mk(r_eff, 0.0), names[2]: mk(r_eff, riser_L(c.riser, r_eff))}
    c.primary = names[2]
    log.info("input", f"R = {c.R} ohm, h = {c.h * 1e3:g} mm, s = {c.sep * 1e3:g} mm, L = {c.length:g} m, r = {c.radius * 1e3:g} mm, "
                      f"T = {c.T * 1e9:.3f} ns")
    for k, ref in c.refs.items():
        Z = C0 * ref.L
        log.info("analytic reference", f"{k}: r = {ref.r * 1e3:.4f} mm, Zc11 = {Z[0, 0]:.1f} ohm, Zc12 = {Z[0, 1]:.1f} ohm, "
                                       f"modes {Z[0, 0] + Z[0, 1]:.1f} / {Z[0, 0] - Z[0, 1]:.1f} ohm, riser {ref.Lr * 1e9:.1f} nH")
