"""Static check of the c3.fdtd.json + .exc input (no solver needed). Returns everything the later steps use.

Case C = case B with the ground plane replaced by the closed PEC cylinder (R = 400 mm, axis z through the grid
origin, grid 168 x 168 x 400). The role of every wire and probe is found from positions, not from names: the
aggressor is the cable that starts at the generator node, a probe belongs to the wire whose (i, j) it sits on, and it
is near/far by its k.

The cylinder wall is read from the PEC 'cell' intervals of the JSON, cut at the middle of the line (k = 200): that
cross-section must be a closed ring of radius about 400 mm centred on the origin, with its wall at j = 4 (y = -400 mm)
under both wires. If Elemwave writes the cylinder in another form (conformal mesher, triangles), the check says so (WARN)
and prints what the PEC elements contain; the analytic reference does not depend on it.
"""
import json
import math
import os
from types import SimpleNamespace

import numpy as np

from c3_analytic import lmat
from c3_common import C0, ETA0, EXPECTED, H_A, R_EFF_FACTOR, ROLES, ZC_A_MEAS, find_file, solver_time_plan


def load_case(here, case_name, log):
    case_path = find_file(here, case_name)
    assert case_path is not None, f"{case_name} not found in {here}"
    work = case_path.parent
    os.chdir(work)                                   # semba-fdtd runs and writes its outputs here
    inp = json.loads(case_path.read_text())
    c = SimpleNamespace(case_path=case_path, WORK=work, inp=inp, HERE=here)
    _geometry(c, log)
    _wall(c, log)
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
    dx, origin, ncell = grid["steps"]["x"][0], grid["origin"], grid["numberOfCells"]
    c.dx, c.origin, c.ncell = dx, origin, ncell
    log.check(grid["steps"]["y"][0] == dx and grid["steps"]["z"][0] == dx, "cubic cells", f"dx = {dx * 1e3:g} mm")
    log.check(abs(dx - EXPECTED["dx"]) < 1e-12 and list(ncell) == EXPECTED["cells"], "grid is 168 x 168 x 400 cells of 5 mm",
              f"dx = {dx * 1e3:g} mm, cells = {ncell} (case B was 168 x 88 x 400)")
    log.check(all(abs(a - b) < 1e-12 for a, b in zip(origin, EXPECTED["origin"])), "grid origin (-0.42, -0.42, 0)",
              f"{origin}")

    # wires: aggressor = the cable that starts at the generator node
    gen = inp["sources"][0]
    c.gen = gen
    c.gen_pos = coords[elements[gen["elementIds"][0]]["coordinateIds"][0]]
    cables = [a for a in inp["materialAssociations"] if a.get("type") == "cable"]
    log.check(len(cables) == 2, "two cables (aggressor and victim)", f"{len(cables)} found")
    W = {}
    for cab in cables:
        i0, i1 = elements[cab["elementIds"][0]]["coordinateIds"]
        p0, p1 = coords[i0], coords[i1]
        w = SimpleNamespace(name=cab["name"], p0=p0, p1=p1, mat=mats[cab["materialId"]],
                            term_n=mats[cab["initialTerminalId"]], term_f=mats[cab["endTerminalId"]])
        W["agg" if p0 == c.gen_pos else "vic"] = w
    log.check(set(W) == {"agg", "vic"}, "one wire starts at the generator node (aggressor), the other is the victim",
              f"{ {k: (w.name, w.p0) for k, w in W.items()} }")
    agg, vic = W["agg"], W["vic"]
    c.W = W
    for k, w in W.items():
        log.check(w.p0[0] == w.p1[0] and w.p0[1] == w.p1[1] and w.p1[2] > w.p0[2],
                  f"{k}: wire parallel to z, near end at the lower k", f"{w.p0} -> {w.p1}")
    log.check(agg.p0[2] == vic.p0[2] and agg.p1[2] == vic.p1[2] and agg.p0[1] == vic.p0[1],
              "both wires have the same ends in z and the same height j")
    log.check([list(agg.p0), list(agg.p1)] == [list(v) for v in EXPECTED["idx_agg"]]
              and [list(vic.p0), list(vic.p1)] == [list(v) for v in EXPECTED["idx_vic"]],
              "wire indices identical to case B: aggressor [77,11,0]->[77,11,400], victim [91,11,0]->[91,11,400]",
              f"aggressor {agg.p0}->{agg.p1}, victim {vic.p0}->{vic.p1}")
    c.length = (agg.p1[2] - agg.p0[2]) * dx
    c.sep = abs(vic.p0[0] - agg.p0[0]) * dx
    c.radius = agg.mat["radius"]
    log.check(vic.mat["radius"] == c.radius, "same wire radius", f"r = {c.radius * 1e3:g} mm")
    log.check(c.radius / dx <= 0.11, "wire radius / dx below the thin-wire stability limit 0.11",
              f"r/dx = {c.radius / dx:.3f}")

    # terminals: pure resistors, 50 ohm on all four ends
    def res(term):
        t_ = term["terminations"][0]
        return t_["resistance"], t_["type"] == "series" and t_.get("capacitance", 0) >= 1e20
    for name, term in {"agg near (R_S)": agg.term_n, "agg far (R_L)": agg.term_f, "vic near (NE)": vic.term_n,
                       "vic far (FE)": vic.term_f}.items():
        log.check(res(term)[1], f"{name} terminal {term['name']} is a pure resistor", f"{term['terminations'][0]}")
    c.R = (res(agg.term_n)[0], res(agg.term_f)[0], res(vic.term_n)[0], res(vic.term_f)[0])   # R_S, R_L, R_NE, R_FE
    log.check(all(abs(r - EXPECTED["R"]) < 1e-9 for r in c.R), "all four ends are 50 ohm (no leftover 296/313 ohm)",
              f"R_S, R_L, R_NE, R_FE = {c.R}")
    c.x = (origin[0] + agg.p0[0] * dx, origin[0] + vic.p0[0] * dx)         # wire x positions (m)
    c.yw = origin[1] + agg.p0[1] * dx                                        # wire y position (m), same for both


def _pec_cross_section(c):
    """Boolean map G[i, j] of the PEC nodes of the 'cell' intervals that cross the middle of the line (k = nz // 2),
    and a description of every PEC element (for the message when nothing usable is found)."""
    nx, ny, nz = c.ncell
    kmid = nz // 2
    G = np.zeros((nx + 1, ny + 1), bool)
    pec_ids = [e for a in c.inp["materialAssociations"] if c.mats[a["materialId"]]["type"] == "pec"
               for e in a["elementIds"]]
    desc = []
    for eid in pec_ids:
        el = c.elements[eid]
        ivs = el.get("intervals", [])
        desc.append(f"element {eid} '{el.get('name')}' type {el.get('type')}: {len(ivs)} intervals, "
                    f"keys {sorted(el)}")
        for (a0, a1, a2), (b0, b1, b2) in ivs:
            if min(a2, b2) <= kmid <= max(a2, b2):
                G[max(0, min(a0, b0)):min(nx, max(a0, b0)) + 1, max(0, min(a1, b1)):min(ny, max(a1, b1)) + 1] = True
    return G, desc


def _sealed(G, probes):
    """True if none of the 'probes' nodes can be reached from the grid corner through non-PEC nodes (4-connected):
    the wall is closed, nothing leaks from the wire region to the Mur faces."""
    nx, ny = G.shape
    free = ~G
    start = next(((i, j) for i, j in ((0, 0), (nx - 1, 0), (0, ny - 1), (nx - 1, ny - 1)) if free[i, j]), None)
    if start is None:
        return True
    seen = np.zeros_like(G)
    seen[start] = True
    stack = [start]
    while stack:
        i, j = stack.pop()
        for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if 0 <= a < nx and 0 <= b < ny and free[a, b] and not seen[a, b]:
                seen[a, b] = True
                stack.append((a, b))
    return not any(seen[p] for p in probes)


def _wall(c, log):
    """The closed PEC cylinder, from the PEC intervals cut at the middle of the line."""
    dx, o, W = c.dx, c.origin, c.W
    jw, iw = W["agg"].p0[1], (W["agg"].p0[0], W["vic"].p0[0])
    G, desc = _pec_cross_section(c)
    c.wall = None
    if G.sum() < 100:                                # not a cylinder in cell intervals
        log.check(False, "cylinder wall found in the PEC cell intervals of the JSON (cut at k = 200)",
                  f"only {int(G.sum())} PEC nodes there. PEC elements: {' | '.join(desc) or 'none'}. "
                  f"Open c3.fdtd.json and check that the wall sits at y = -400 (j = 4) at x = +-35 and that the shell "
                  f"is closed; the analytic reference below does not depend on this check", "WARN")
        c.h = (jw - EXPECTED["wall_j"]) * dx           # not read from the file: assume the intended wall at j = 4
        return
    ii, jj = np.nonzero(G)
    x, y = o[0] + ii * dx, o[1] + jj * dx
    r = np.hypot(x, y)
    cx, cy = 0.5 * (x.max() + x.min()), 0.5 * (y.max() + y.min())
    R = EXPECTED["cyl_R"]
    log.check(abs(r.mean() - R) <= dx and abs(cx) <= dx and abs(cy) <= dx,
              "PEC cross-section is a ring of radius about 400 mm centred on the axis (k = 200)",
              f"{int(G.sum())} PEC nodes, radius mean {r.mean() * 1e3:.1f} mm (min {r.min() * 1e3:.1f}, max "
              f"{r.max() * 1e3:.1f}), extent centre ({cx * 1e3:+.1f}, {cy * 1e3:+.1f}) mm")
    below = {}
    for name, i in zip(("aggressor", "victim"), iw):
        js = np.nonzero(G[i, :])[0]
        js = js[js < jw]
        below[name] = int(js.max()) if len(js) else None
    ok = all(v == EXPECTED["wall_j"] for v in below.values())
    c.h = (jw - below["aggressor"]) * dx if below["aggressor"] is not None else float("nan")
    log.check(ok, "meshed wall under both wires at j = 4 (y = -400 mm), so h = 35 mm",
              f"wall j under aggressor {below['aggressor']}, under victim {below['victim']}; wire j = {jw}; "
              f"h = {c.h * 1e3:.0f} mm (j = 3 or 5 gives h = 40 or 30 mm: the result would be shifted by the staircase, "
              f"not by physics; use the conformal mesher if offered)")
    sealed = _sealed(G, [(i, jw) for i in iw])
    log.check(sealed, "wall is closed: the wire region cannot reach the Mur faces through free nodes (no leak in the shell)",
              "closed ring" if sealed else "LEAK: free nodes connect the wires to the outside at k = 200")
    c.wall = SimpleNamespace(G=G, radius_mean=float(r.mean()), j_under=below)


def _boundaries_and_probes(c, log):
    inp, elements, coords, W = c.inp, c.elements, c.coords, c.W
    bnd = inp["boundary"]
    log.check(bnd["zLower"]["type"] == "pec" and bnd["zUpper"]["type"] == "pec",
              "z faces are PEC (end walls: wire ends sit on them)", f"{bnd['zLower']['type']}/{bnd['zUpper']['type']}")
    log.check(all(bnd[k]["type"] == "mur" for k in ("xLower", "xUpper", "yLower", "yUpper")),
              "x and y faces are Mur (shielded by the cylinder)")
    log.check(c.gen["field"] == "voltage" and c.gen_pos == W["agg"].p0,
              "generator is a voltage source at the aggressor near end", f"field = {c.gen['field']}, node {c.gen_pos}")
    c.PROBES = {}                                    # role -> probe name, e.g. "vic_near" -> "NE probe"
    for pr in inp["probes"]:
        p = coords[elements[pr["elementIds"][0]]["coordinateIds"][0]]
        for key, w in W.items():
            if p[0] == w.p0[0] and p[1] == w.p0[1] and p[2] in (w.p0[2], w.p1[2]):
                c.PROBES[f"{key}_{'near' if p[2] == w.p0[2] else 'far'}"] = pr["name"]
    log.check(set(c.PROBES) == set(ROLES) and len(inp["probes"]) == 4
              and all(pr["field"] == "current" for pr in inp["probes"]),
              "four current probes: aggressor near/far, victim near/far", f"{c.PROBES}")
    log.info("wire model", "the JSON does not say which wire model runs, and Report.txt prints 'MTLN wires' in any MTLN build. "
             "Confirm Holland with MTLN off in the Solver dialog before the long run (Zc about 308 ohm points to the thin wire)")


def _excitation(c, log):
    gen = c.gen
    exc_path = c.WORK / gen["magnitudeFile"]
    if not exc_path.exists():
        exc_path = find_file(c.HERE, gen["magnitudeFile"])
    assert exc_path is not None, f"excitation file {gen['magnitudeFile']} not found next to {c.case_path.name}"
    exc = np.loadtxt(exc_path)
    c.exc = exc
    c.source = lambda t: np.interp(t, exc[:, 0], exc[:, 1], left=0.0, right=0.0)
    c.s_pk = exc[:, 1].max()
    c.t0 = exc[np.argmax(exc[:, 1]), 0]
    c.sigma = math.sqrt(np.sum(exc[:, 1] * (exc[:, 0] - c.t0) ** 2) / np.sum(exc[:, 1]))
    log.check(abs(c.s_pk - 1.0) < 1e-9 and exc[0, 1] < 1e-6, "excitation: 1 V peak, starts at ~0 (no step)",
              f"peak {c.s_pk:g} V at {c.t0 * 1e9:.2f} ns, first sample {exc[0, 1]:.1e}")
    log.check(abs(c.t0 - 15e-9) < 1e-12 and abs(c.sigma - 2.5e-9) < 0.05e-9,
              "source is the tutorial pulse (t0 = 15 ns, sigma = 2.5 ns)", f"{c.t0 * 1e9:.2f} / {c.sigma * 1e9:.2f} ns",
              "WARN")
    log.info("excitation", f"{exc_path.name}: t0 = {c.t0 * 1e9:.2f} ns, sigma = {c.sigma * 1e9:.2f} ns, "
                           f"file length {exc[-1, 0] * 1e9:.0f} ns")
    f_ax = np.fft.rfftfreq(1 << 16, exc[1, 0] - exc[0, 0])
    spec_db = 20 * np.log10(np.abs(np.fft.rfft(exc[:, 1], 1 << 16)) / np.abs(np.fft.rfft(exc[:, 1], 1 << 16))[0] + 1e-30)
    f_at = lambda db: f_ax[np.argmax(spec_db < -db)]
    log.info("excitation spectrum",
             f"-3 dB at {f_at(3) / 1e6:.0f} MHz, -10 dB at {f_at(10) / 1e6:.0f} MHz, "
             f"-20 dB at {f_at(20) / 1e6:.0f} MHz, -40 dB at {f_at(40) / 1e6:.0f} MHz "
             f"(100 MHz: {np.interp(1e8, f_ax, spec_db):.1f} dB)")


def _time_and_references(c, log):
    """Time step / window against the solver limit, and the analytic references (inductance matrices)."""
    inp = c.inp
    c.T = c.length / C0                                          # one-way delay (s)
    dt_json, n_json = inp["general"]["timeStep"], inp["general"]["numberOfSteps"]
    window_json = dt_json * n_json
    c.plan = plan = solver_time_plan(dt_json, n_json, c.dx)      # what semba-fdtd will really run (c3_common)
    c.dt_solver = plan.dt_limit                                  # largest dt semba-fdtd accepts without correcting it
    log.check(not plan.corrected, "timeStep within the solver limit (set 7.7033e-12 s and 38944 steps in the Solver dialog)",
              f"json {dt_json:.4e} s x {n_json} steps; limit {plan.dt_limit:.4e} s (0.8 dt_CFL)"
              + (f"; semba-fdtd will correct dt to {plan.dt:.4e} s and run {plan.n_final} steps = "
                 f"{plan.t_final * 1e9:.4f} ns" if plan.corrected else f" -> {plan.t_json * 1e9:.4f} ns"), "WARN")
    log.check(window_json >= EXPECTED["window"] * 0.999, "simulated window is the 300 ns of the tutorial",
              f"{window_json * 1e9:.1f} ns")
    dt_exc = c.exc[1, 0] - c.exc[0, 0]
    log.check(c.exc[-1, 0] >= window_json - 2 * dt_exc,
              "the .exc file covers the whole simulated window (otherwise the source is cut or zero)",
              f"file ends at {c.exc[-1, 0] * 1e9:.1f} ns, window {window_json * 1e9:.1f} ns")
    log.check(abs(c.h - EXPECTED["height"]) < 1e-9 and abs(c.length - EXPECTED["length"]) < 1e-9
              and abs(c.sep - EXPECTED["sep"]) < 1e-9 and abs(c.radius - EXPECTED["radius"]) < 1e-12,
              "geometry as in the tutorial", f"h = {c.h * 1e3:g} mm, s = {c.sep * 1e3:g} mm, L = {c.length:g} m, "
              f"r = {c.radius * 1e3:g} mm", "WARN")

    # r_eff from case A: the plane formula of case A (h = 35 mm) turns the measured Zc into the radius of the thin wire
    c.refs = {"radius r": c.radius, f"r_eff = {R_EFF_FACTOR} r": R_EFF_FACTOR * c.radius}
    if ZC_A_MEAS:
        c.refs[f"r_eff from case A (Zc {ZC_A_MEAS:g} ohm)"] = H_A / math.cosh(2 * math.pi * ZC_A_MEAS / ETA0)
    c.primary = list(c.refs)[-1]                     # the reference the pass/fail checks use
    R = EXPECTED["cyl_R"]
    pts = [(c.x[0], c.yw), (c.x[1], c.yw)]           # wires inside the circle of radius R centred on the origin
    pts_plane = [(c.x[0], H_A), (c.x[1], H_A)]       # the same pair over the plane of case B (h = 35 mm)
    c.L = {k: lmat(pts, r, R=R) for k, r in c.refs.items()}
    c.L_plane = {k: lmat(pts_plane, r) for k, r in c.refs.items()}
    c.zc_single = C0 * c.L[c.primary][0, 0]          # self term of the aggressor in the pair (tutorial 5.1: 308.1 ohm)
    log.info("input", f"R = {c.R} ohm, h = {c.h * 1e3:g} mm, s = {c.sep * 1e3:g} mm, L = {c.length:g} m, "
                      f"r = {c.radius * 1e3:g} mm, T = {c.T * 1e9:.3f} ns, cylinder R = {R * 1e3:g} mm")
    for k, r in c.refs.items():
        log.info("analytic reference (cylinder)", f"{k}: r = {r * 1e3:.4f} mm, Zc = c l11 = {C0 * c.L[k][0, 0]:.1f} ohm, "
                                                  f"l11/l12 = {c.L[k][0, 0] * 1e6:.3f}/{c.L[k][0, 1] * 1e6:.3f} uH/m "
                                                  f"(plane: {c.L_plane[k][0, 0] * 1e6:.3f}/{c.L_plane[k][0, 1] * 1e6:.3f})")
