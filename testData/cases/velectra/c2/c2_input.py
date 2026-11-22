"""Static check of the c2.fdtd.json + .exc input (no solver needed). Returns everything the later steps use.

The role of every wire and probe is found from positions, not from names: the aggressor is the cable that starts at
the generator node, a probe belongs to the wire whose (i, j) it sits on, and it is near/far by its k.
"""
import json
import math
import os
from types import SimpleNamespace

import numpy as np

from c2_analytic import lmat
from c2_common import C0, EXPECTED, MATCH_TOL, ROLES, WINDOW_MISMATCHED, find_file, solver_time_plan, zc_wire_plane


def load_case(here, case_name, log):
    case_path = find_file(here, case_name)
    assert case_path is not None, f"{case_name} not found in {here}"
    work = case_path.parent
    os.chdir(work)                                   # semba-fdtd runs and writes its outputs here
    inp = json.loads(case_path.read_text())
    c = SimpleNamespace(case_path=case_path, WORK=work, inp=inp, HERE=here, name=work.name)
    log.info("case folder", str(work))
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
    dx, origin, ncell = grid["steps"]["x"][0], grid["origin"], grid["numberOfCells"]
    c.dx, c.origin, c.ncell = dx, origin, ncell
    log.check(grid["steps"]["y"][0] == dx and grid["steps"]["z"][0] == dx, "cubic cells", f"dx = {dx * 1e3:g} mm")
    log.check(abs(dx - EXPECTED["dx"]) < 1e-12 and list(ncell) == EXPECTED["cells"], "grid as expected (c2_common.EXPECTED)",
              f"dx = {dx * 1e3:g} mm, cells = {ncell}", "WARN")

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
    c.length = (agg.p1[2] - agg.p0[2]) * dx
    c.sep = abs(vic.p0[0] - agg.p0[0]) * dx
    c.radius = agg.mat["radius"]
    log.check(vic.mat["radius"] == c.radius, "same wire radius", f"r = {c.radius * 1e3:g} mm")
    log.check(c.radius / dx <= 0.11, "wire radius / dx below the thin-wire stability limit 0.11",
              f"r/dx = {c.radius / dx:.3f}")
    log.info("grid", f"{ncell[0]} x {ncell[1]} x {ncell[2]} cells = {ncell[0] * ncell[1] * ncell[2] / 1e6:.2f} Mcells "
                     f"(before the PML layers)")

    # terminals: pure resistors, one value for the four ends (50 ohm in the real experiment, Zc of the wire when matched)
    def res(term):
        t_ = term["terminations"][0]
        return t_["resistance"], t_["type"] == "series" and t_.get("capacitance", 0) >= 1e20
    for name, term in {"agg near (R_S)": agg.term_n, "agg far (R_L)": agg.term_f, "vic near (NE)": vic.term_n,
                       "vic far (FE)": vic.term_f}.items():
        log.check(res(term)[1], f"{name} terminal {term['name']} is a pure resistor", f"{term['terminations'][0]}")
    c.R = (res(agg.term_n)[0], res(agg.term_f)[0], res(vic.term_n)[0], res(vic.term_f)[0])   # R_S, R_L, R_NE, R_FE
    log.check(len(set(c.R)) == 1, "the four ends have the same resistance", f"R_S, R_L, R_NE, R_FE = {c.R}", "WARN")
    used = {w.term_n["id"] for w in W.values()} | {w.term_f["id"] for w in W.values()}
    spare = [m["name"] for m in inp["materials"] if m["type"] == "terminal" and m["id"] not in used]
    log.check(not spare, "no terminal material is left unused in the JSON (old 296/313 ohm leftovers)", f"unused: {spare}", "WARN")

    # ground plane
    pec_ids = [e for a in inp["materialAssociations"] if mats[a["materialId"]]["type"] == "pec" for e in a["elementIds"]]
    plane = [iv for eid in pec_ids for iv in elements[eid].get("intervals", []) if iv[0][1] == iv[1][1]]
    log.check(len(plane) >= 1, "ground plane found (PEC cell interval with constant j)")
    pl = plane[0]
    c.h = (agg.p0[1] - pl[0][1]) * dx
    span_ok = (min(pl[0][0], pl[1][0]) == 0 and max(pl[0][0], pl[1][0]) == ncell[0]
               and min(pl[0][2], pl[1][2]) == 0 and max(pl[0][2], pl[1][2]) == ncell[2])
    log.check(span_ok, "plane spans the whole grid in x and z", f"{pl} (grid {ncell})")
    log.check(abs(c.h - EXPECTED["height"]) < 1e-9 and abs(c.length - EXPECTED["length"]) < 1e-9
              and abs(c.sep - EXPECTED["sep"]) < 1e-9 and abs(c.radius - EXPECTED["radius"]) < 1e-12,
              "geometry as expected (c2_common.EXPECTED)", f"h = {c.h * 1e3:g} mm, s = {c.sep * 1e3:g} mm, L = {c.length:g} m, "
              f"r = {c.radius * 1e3:g} mm", "WARN")
    c.x = (c.origin[0] + agg.p0[0] * dx, c.origin[0] + vic.p0[0] * dx)         # wire x positions (m)
    # clearance of the wires to the lateral / top boundaries, in units of h (the quasi-static fringing field of a wire
    # over a plane reaches several h; an absorbing boundary closer than that can bias Zc_eff)
    c.clear = dict(x_low=agg.p0[0] * dx, x_high=(ncell[0] - vic.p0[0]) * dx, y_high=(ncell[1] - agg.p0[1]) * dx)
    log.info("clearance wires -> boundary", ", ".join(f"{k} = {v * 1e3:.0f} mm ({v / c.h:.1f} h)" for k, v in c.clear.items()))


def _boundaries_and_probes(c, log):
    inp, elements, coords, W = c.inp, c.elements, c.coords, c.W
    bnd = inp["boundary"]
    log.check(bnd["zLower"]["type"] == "pec" and bnd["zUpper"]["type"] == "pec",
              "z faces are PEC (end walls: wire ends sit on them)", f"{bnd['zLower']['type']}/{bnd['zUpper']['type']}")
    c.bnd_xy = {k: bnd[k]["type"] for k in ("xLower", "xUpper", "yLower", "yUpper")}
    # semba-fdtd (smbjson.F90, labelToBoundaryType) matches the type with a case-sensitive `select case`: "PML" or
    # "Pml" is silently not recognised and the face is NOT absorbing (the Report then says "no CPML Borders found").
    log.check(all(v in ("mur", "pml") for v in c.bnd_xy.values()),
              "x/y faces are open: exactly lowercase 'mur' or 'pml' (other spellings are ignored by the solver)", f"{c.bnd_xy}")
    log.check(all(v == "pml" for v in c.bnd_xy.values()), "x/y faces are PML (convention of c1, c2 and c3)",
              f"{c.bnd_xy}", "WARN")
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
    """Time step / window against the solver limit, and the analytic reference (inductance matrix, physical radius)."""
    inp = c.inp
    c.T = c.length / C0                                          # one-way delay (s)
    dt_json, n_json = inp["general"]["timeStep"], inp["general"]["numberOfSteps"]
    window_json = dt_json * n_json
    c.plan = plan = solver_time_plan(dt_json, n_json, c.dx)      # what semba-fdtd will really run (c2_common)
    c.dt_solver = plan.dt_limit                                  # largest dt semba-fdtd accepts without correcting it
    log.check(not plan.corrected, "timeStep within the solver limit",
              f"json {dt_json:.4e} s, limit {plan.dt_limit:.4e} s (0.8 dt_CFL); "
              + (f"semba-fdtd will correct dt to {plan.dt:.4e} s and run {plan.n_final} steps = "
                 f"{plan.t_final * 1e9:.4f} ns (json: {n_json} steps = {window_json * 1e9:.4f} ns)"
                 if plan.corrected else f"kept: {n_json} steps = {window_json * 1e9:.4f} ns"), "WARN")

    c.Zc_r = zc_wire_plane(c.h, c.radius)                        # analytic Zc with the PHYSICAL radius
    c.matched = all(abs(r - c.Zc_r) / c.Zc_r < MATCH_TOL for r in c.R)
    c.window_need = c.t0 + 2 * c.T + 4 * c.sigma                 # pulse + one round trip + tail
    need = c.window_need if c.matched else WINDOW_MISMATCHED
    c.window_req = need                                          # what the results must cover
    log.check(window_json >= need * 0.999,
              "simulated window holds the first echo" if c.matched else "simulated window is the 300 ns of the tutorial",
              f"{window_json * 1e9:.1f} ns, needed {need * 1e9:.1f} ns")
    dt_exc = c.exc[1, 0] - c.exc[0, 0]
    log.check(c.exc[-1, 0] >= window_json - 2 * dt_exc,
              "the .exc file covers the whole simulated window (otherwise the source is cut or zero)",
              f"file ends at {c.exc[-1, 0] * 1e9:.1f} ns, window {window_json * 1e9:.1f} ns")

    gam = (c.R[0] - c.Zc_r) / (c.R[0] + c.Zc_r)
    if c.matched:
        log.check(True, "terminals equal the analytic Zc of the physical radius within 2 % (the analytic line is then matched)",
                  f"R = {c.R[0]:g} ohm, Zc = {c.Zc_r:.1f} ohm")
    else:
        log.info("terminals vs analytic Zc (physical radius)",
                 f"R = {c.R[0]:g} ohm, Zc = {c.Zc_r:.1f} ohm: mismatched (Gamma = {gam:+.3f} at every end); "
                 f"the line rings and the analytic line has the same echoes")
    # only the physical radius: no fitted r_eff (case A showed that the solver's own Zc is 313 ohm at 35 mm, not a radius change)
    c.refs = {"physical radius": c.radius}
    c.primary = "physical radius"
    pts = [(c.x[0], c.h), (c.x[1], c.h)]
    c.L = {k: lmat(pts, r) for k, r in c.refs.items()}
    log.info("input", f"R = {c.R} ohm, h = {c.h * 1e3:g} mm, s = {c.sep * 1e3:g} mm, L = {c.length:g} m, "
                      f"r = {c.radius * 1e3:g} mm, T = {c.T * 1e9:.3f} ns")
    for k, r in c.refs.items():
        log.info("analytic reference", f"{k}: r = {r * 1e3:.4f} mm, Zc single = {c.Zc_r:.1f} ohm, "
                                       f"l11/l12 = {c.L[k][0, 0] * 1e6:.3f}/{c.L[k][0, 1] * 1e6:.3f} uH/m")
