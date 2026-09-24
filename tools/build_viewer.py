"""Build the interactive 3D flight viewer.

Flies the nominal missions (calm and 6 m/s crosswind), packs them with the
Monte Carlo ground tracks into compact JSON, and inlines it into
viewer/template.html to produce viewer/flight_viewer.html: a single
self-contained page that opens straight from disk or can be published.

Run from the FlightSim6DOF folder, after examples/monte_carlo.py:
    python tools/build_viewer.py
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from flightsim.design import load_sized
from flightsim.montecarlo import ellipse
from flightsim.sim import SimConfig, Wind, simulate

VIEWER = ROOT / "viewer"
PHASES = ["On the rail", "Powered ascent", "Coast", "Drogue descent", "Main descent"]


def pack_flight(vehicle, res, label):
    t = res.t
    # Keep full rate while the attitude is interesting, thin out the long descent.
    keep = np.ones_like(t, dtype=bool)
    ascent = res.phase <= 2
    idx_coast = np.where((res.phase == 2) & (t > res.events["burnout"] + 5))[0]
    keep[idx_coast[1::3]] = keep[idx_coast[2::3]] = False
    t_max_thrust = vehicle.engine.data.thrust_vac
    speed = np.linalg.norm(res.v, axis=1)
    rows = []
    for i in np.where(keep)[0]:
        q = res.q[i] if ascent[i] else [None] * 4
        rows.append([
            round(float(t[i]), 3),
            *[round(float(c), 1) for c in res.r[i]],
            *([round(float(c), 4) for c in q] if ascent[i] else q),
            int(res.phase[i]),
            round(float(speed[i]), 1),
            None if np.isnan(res.mach[i]) else round(float(res.mach[i]), 3),
            None if np.isnan(res.alpha[i]) else round(math.degrees(res.alpha[i]), 2),
            round(float(res.thrust[i]) / t_max_thrust, 3) if not np.isnan(res.thrust[i]) else 0.0,
        ])
    e = res.events
    return {
        "label": label,
        "columns": ["t", "x", "y", "z", "qw", "qx", "qy", "qz", "phase", "speed", "mach", "alpha", "throttle"],
        "rows": rows,
        "events": [
            {"name": "Liftoff", "t": 0.0},
            {"name": "Rail exit", "t": e["rail_exit"]},
            {"name": "Burnout", "t": e["burnout"]},
            {"name": "Apogee", "t": e["apogee"]},
            {"name": "Main chute", "t": e["main_deploy"]},
            {"name": "Landing", "t": e["landing"]},
        ],
        "summary": {
            "apogee_ft": res.apogee_agl / 0.3048,
            "max_mach": float(np.nanmax(res.mach)),
            "landing_east_m": float(res.r[-1, 0]),
            "landing_north_m": float(res.r[-1, 1]),
        },
    }


def pack_monte_carlo():
    runs = json.loads((ROOT / "outputs" / "monte_carlo_runs.json").read_text())
    runs = [r for r in runs if r["ok"]]
    pts = np.array([[r["landing_east_m"], r["landing_north_m"]] for r in runs])
    summary = json.loads((ROOT / "outputs" / "monte_carlo_summary.json").read_text())
    return {
        "tracks": [[[round(c) for c in p] for p in r["track"][::2]] + [[round(r["landing_east_m"]),
                   round(r["landing_north_m"]), 0]] for r in runs],
        "apogees_ft": [round(r["apogee_ft"]) for r in runs],
        "ellipse99": ellipse(pts, 0.99, 120).round(0).tolist(),
        "ellipse90": ellipse(pts, 0.90, 120).round(0).tolist(),
        "apogee": summary["apogee_ft"],
    }


def main():
    spec, vehicle, site = load_sized()
    af = vehicle.airframe
    f = af.fins
    data = {
        "vehicle": {
            "length": af.length, "diameter": af.diameter, "nose_length": af.nose.length,
            "x_cg": vehicle.mass_properties(0.0).x_cg,
            "fin_root": f.root_chord, "fin_tip": f.tip_chord, "fin_span": f.span,
            "fin_sweep": f.sweep, "fin_thickness": f.thickness, "fins": f.n,
            "nozzle_exit_diameter": vehicle.engine.data.exit_diameter,
            "rail_length": vehicle.rail_length,
            "tanks": [[vehicle.ox_tank.station, vehicle.ox_tank.length],
                      [vehicle.fuel_tank.station, vehicle.fuel_tank.length]],
        },
        "site": {"name": site.name, "altitude_m": site.altitude,
                 "rail_elevation_deg": site.rail_elevation_deg, "rail_azimuth_deg": 0.0},
        "phases": PHASES,
        "flights": [
            pack_flight(vehicle, simulate(vehicle, SimConfig(site=site)), "Calm air"),
            pack_flight(vehicle, simulate(vehicle, SimConfig(site=site, wind=Wind(6.0, 270.0))),
                        "6 m/s wind from the west"),
        ],
        "monte_carlo": pack_monte_carlo(),
    }
    blob = json.dumps(data, separators=(",", ":"))
    page = (VIEWER / "template.html").read_text(encoding="utf-8").replace("/*__FLIGHT_DATA__*/null", blob)
    # flight_viewer.html opens straight from disk (double-click); the
    # _publish copy has no document wrapper, which the web host adds itself.
    out = VIEWER / "flight_viewer.html"
    out.write_text('<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n' + page + "\n</html>\n",
                   encoding="utf-8")
    (VIEWER / "_publish").mkdir(exist_ok=True)
    (VIEWER / "_publish" / "flight_viewer.html").write_text(page, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
