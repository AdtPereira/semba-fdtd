"""Static check of the c1.fdtd.json + gauss.exc input (no solver needed). Returns everything the later steps use."""
import json
import math
import os
from types import SimpleNamespace

import numpy as np

from c1_common import C0, ETA0, EXPECTED, find_file, solver_time_plan, zc_wire_plane


def load_case(here, case_name, log):
    case_path = find_file(here, case_name)
    assert case_path is not None, f"{case_name} not found in {here}"
    work = case_path.parent
    os.chdir(work)                                   # semba-fdtd runs and writes its outputs here
    inp = json.loads(case_path.read_text())
    c = SimpleNamespace(case_path=case_path, WORK=work, inp=inp, HERE=here)
    log.info("case folder", str(work))
    _geometry(c, log)
    _boundaries_and_sources(c, log)
    _excitation(c, log)
    _time_and_references(c, log)
    return c


def _geometry(c, log):
    inp = c.inp
    mats = {m["id"]: m for m in inp["materials"]}
    elements = {e["id"]: e for e in inp["mesh"]["elements"]}
    coords = {k["id"]: k["relativePosition"] for k in inp["mesh"]["coordinates"]}
    c.mats, c.elements, c.coords = mats, elements, coords

    wire = next(m for m in inp["materials"] if m["type"] == "wire")
    c.radius = wire["radius"]
    cable = next(a for a in inp["materialAssociations"] if a.get("type") == "cable")
    terminal = lambda tid: mats[tid]["terminations"][0]
    c.R_S = terminal(cable["initialTerminalId"])["resistance"]     # near end = source resistance
    c.R_L = terminal(cable["endTerminalId"])["resistance"]         # far end = load
    for end_, tid in (("near", cable["initialTerminalId"]), ("far", cable["endTerminalId"])):
        t_ = terminal(tid)
        log.check(t_["type"] == "series" and t_.get("capacitance", 0) >= 1e20,
                  f"{end_}-end terminal {mats[tid]['name']} is a pure resistor", f"{t_}")

    grid = inp["mesh"]["grid"]
    d = grid["steps"]["x"][0]
    c.dx, c.ncell = d, grid["numberOfCells"]
    log.check(grid["steps"]["y"][0] == d and grid["steps"]["z"][0] == d, "cubic cells", f"dx = {d * 1e3:g} mm")
    origin, ncell = grid["origin"], grid["numberOfCells"]
    log.check(abs(d - EXPECTED["dx"]) < 1e-12, "cell size as expected (c1_common.EXPECTED)", f"dx = {d * 1e3:g} mm", "WARN")
    log.info("grid", f"{ncell[0]} x {ncell[1]} x {ncell[2]} cells = {ncell[0] * ncell[1] * ncell[2] / 1e6:.2f} Mcells (before the PML layers)")

    poly = next(e for e in elements.values() if e["type"] == "polyline")
    p0, p1 = (coords[k] for k in poly["coordinateIds"])
    c.p0 = p0
    log.check(p0[0] == p1[0] and p0[1] == p1[1], "wire is parallel to z", f"{p0} -> {p1}")
    c.length = abs(p1[2] - p0[2]) * d

    pec_ids = [e for a in inp["materialAssociations"] if mats[a["materialId"]]["type"] == "pec"
               for e in a["elementIds"]]
    plane = [iv for eid in pec_ids for iv in elements[eid].get("intervals", []) if iv[0][1] == iv[1][1]]
    log.check(len(plane) >= 1, "ground plane found (PEC cell interval with constant j)")
    c.h = abs(p0[1] - plane[0][0][1]) * d
    span_ok = (min(plane[0][0][0], plane[0][1][0]) == 0 and max(plane[0][0][0], plane[0][1][0]) == ncell[0]
               and min(plane[0][0][2], plane[0][1][2]) == 0 and max(plane[0][0][2], plane[0][1][2]) == ncell[2])
    log.check(span_ok, "plane spans the whole grid in x and z", f"{plane[0]} (grid {ncell})")
    log.check(abs(c.length - EXPECTED["length"]) < 1e-9 and abs(c.radius - EXPECTED["radius"]) < 1e-12,
              "wire length and radius as expected (c1_common.EXPECTED)",
              f"L = {c.length:g} m, r = {c.radius * 1e3:g} mm (height h = {c.h * 1e3:g} mm is read from the JSON)", "WARN")
    log.check(c.radius / d <= 0.11, "wire radius / dx below the thin-wire stability limit 0.11",
              f"r/dx = {c.radius / d:.3f}")
    # clearance of the wire to the lateral / top boundaries, in units of h (the quasi-static fringing field of a wire
    # over a plane reaches several h; an absorbing boundary closer than that can bias Zc_eff)
    c.clear = dict(x_low=p0[0] * d, x_high=(ncell[0] - p0[0]) * d, y_high=(ncell[1] - p0[1]) * d)
    log.info("clearance wire -> boundary", ", ".join(f"{k} = {v * 1e3:.0f} mm ({v / c.h:.1f} h)"
                                                      for k, v in c.clear.items()))


def _boundaries_and_sources(c, log):
    inp, elements, coords, p0 = c.inp, c.elements, c.coords, c.p0
    bnd = inp["boundary"]
    log.check(bnd["zLower"]["type"] == "pec" and bnd["zUpper"]["type"] == "pec",
              "z faces are PEC (end walls: wire ends sit on them)", f"{bnd['zLower']['type']}/{bnd['zUpper']['type']}")
    c.bnd_xy = {k: bnd[k]["type"] for k in ("xLower", "xUpper", "yLower", "yUpper")}
    # semba-fdtd (smbjson.F90, labelToBoundaryType) matches the type with a case-sensitive `select case`: "PML" or
    # "Pml" is silently not recognised and the face is NOT absorbing (the Report then says "no CPML Borders found").
    log.check(all(v in ("mur", "pml") for v in c.bnd_xy.values()),
              "x/y faces are open: exactly lowercase 'mur' or 'pml' (other spellings are ignored by the solver)",
              f"{c.bnd_xy}")
    log.info("x/y boundary type", "/".join(sorted(set(c.bnd_xy.values()))) +
             " (keep the same type in A, B and C for comparable results)")
    gen = inp["sources"][0]
    c.gen = gen
    gen_k = coords[elements[gen["elementIds"][0]]["coordinateIds"][0]][2]
    log.check(gen["field"] == "voltage" and gen_k == p0[2], "generator is a voltage source at the near end",
              f"field = {gen['field']}, k = {gen_k}")
    c.PROBES = {}                                    # near/far end by position along the wire
    for pr in inp["probes"]:
        k = coords[elements[pr["elementIds"][0]]["coordinateIds"][0]][2]
        c.PROBES["near" if k == p0[2] else "far"] = pr["name"]
    log.check(set(c.PROBES) == {"near", "far"} and all(pr["field"] == "current" for pr in inp["probes"]),
              "one current probe at each end", f"{c.PROBES}")


def _excitation(c, log):
    gen = c.gen
    exc_path = (c.WORK / gen["magnitudeFile"])
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
    log.info("excitation", f"t0 = {c.t0 * 1e9:.2f} ns, sigma = {c.sigma * 1e9:.2f} ns, "
                           f"file length {exc[-1, 0] * 1e9:.0f} ns")
    f_ax = np.fft.rfftfreq(1 << 16, exc[1, 0] - exc[0, 0])
    spec = np.abs(np.fft.rfft(exc[:, 1], 1 << 16))
    spec_db = 20 * np.log10(spec / spec[0] + 1e-30)
    f_at = lambda db: f_ax[np.argmax(spec_db < -db)]
    log.info("excitation spectrum",
             f"-3 dB at {f_at(3) / 1e6:.0f} MHz, -10 dB at {f_at(10) / 1e6:.0f} MHz, "
             f"-20 dB at {f_at(20) / 1e6:.0f} MHz, -40 dB at {f_at(40) / 1e6:.0f} MHz "
             f"(100 MHz: {np.interp(1e8, f_ax, spec_db):.1f} dB)")


def _time_and_references(c, log):
    """Time step / window against the solver limit, and the analytic Zc references."""
    inp = c.inp
    c.T = c.length / C0                                          # one-way delay (s)
    dt_json, n_json = inp["general"]["timeStep"], inp["general"]["numberOfSteps"]
    window_json = dt_json * n_json
    c.plan = plan = solver_time_plan(dt_json, n_json, c.dx)     # what semba-fdtd will really run (c1_common)
    c.dt_solver = plan.dt_limit                                  # largest dt semba-fdtd accepts without correcting it
    c.window_need = c.t0 + 2 * c.T + 4 * c.sigma     # pulse + one round trip + tail: the first echo is complete
    log.check(not plan.corrected, "timeStep within the solver limit",
              f"json {dt_json:.4e} s, limit {plan.dt_limit:.4e} s (0.8 dt_CFL); "
              + (f"semba-fdtd will correct dt to {plan.dt:.4e} s and run {plan.n_final} steps = "
                 f"{plan.t_final * 1e9:.4f} ns (json: {n_json} steps = {window_json * 1e9:.4f} ns)"
                 if plan.corrected else f"kept: {n_json} steps = {window_json * 1e9:.4f} ns"), "WARN")
    log.check(window_json >= c.window_need, "simulated window holds the first echo",
              f"{window_json * 1e9:.1f} ns, needed {c.window_need * 1e9:.1f} ns")

    c.Zc_r = zc_wire_plane(c.h, c.radius)                        # analytic Zc with the PHYSICAL radius
    log.info("input", f"R_S = {c.R_S:g} ohm, R_L = {c.R_L:g} ohm, h = {c.h * 1e3:g} mm, L = {c.length:g} m, "
                      f"r = {c.radius * 1e3:g} mm, T = {c.T * 1e9:.3f} ns")
    log.info("analytic Zc (physical radius)", f"{c.Zc_r:.1f} ohm = eta0/(2 pi) acosh(h/r)")
    gam = lambda r: (r - c.Zc_r) / (r + c.Zc_r)
    if abs(c.R_S - c.Zc_r) / c.Zc_r < 0.02 and abs(c.R_L - c.Zc_r) / c.Zc_r < 0.02:
        log.check(True, "terminals equal the analytic Zc of the physical radius within 2 % (the analytic line is then matched)",
                  f"R_S = {c.R_S:g}, R_L = {c.R_L:g}, Zc = {c.Zc_r:.1f} ohm")
    else:
        log.info("terminals vs analytic Zc (physical radius)",
                 f"R_S = {c.R_S:g}, R_L = {c.R_L:g}, Zc = {c.Zc_r:.1f} ohm: the analytic line is NOT matched "
                 f"(Gamma_S = {gam(c.R_S):+.3f}, Gamma_L = {gam(c.R_L):+.3f}); it has its own echoes, which the comparison includes")
    log.info("terminals as modelled", "kept as they are: the solver is not told to match its own wire; any mismatch "
                                      "seen in the results is part of the semba-fdtd answer")
