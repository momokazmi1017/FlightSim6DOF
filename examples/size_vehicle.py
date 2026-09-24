"""Size the liquid sounding rocket for a 30,000 ft apogee above the Spaceport America site.

Chooses the burn time (propellant load) and fin size, prints the result and
saves the design to data/sized_vehicle.json for the 6-DOF simulation.

Run from the FlightSim6DOF folder:
    python examples/size_vehicle.py
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flightsim.sizing import FT, LaunchSite, min_static_margin, size_vehicle
from flightsim.vehicle import VehicleSpec

if __name__ == "__main__":
    site = LaunchSite()
    spec, v, r = size_vehicle(VehicleSpec(burn_time=8.0), site)
    f = v.airframe.fins
    rows = {
        "burn time (s)": spec.burn_time,
        "fin scale": spec.fin_scale,
        "length (m)": v.airframe.length,
        "diameter (m)": v.airframe.diameter,
        "dry mass (kg)": v.dry_mass,
        "propellant mass (kg)": v.propellant_mass,
        "liftoff mass (kg)": v.dry_mass + v.propellant_mass,
        "total impulse, vacuum (N s)": v.engine.data.thrust_vac * spec.burn_time,
        "fin root / tip / span (m)": f"{f.root_chord:.3f} / {f.tip_chord:.3f} / {f.span:.3f}",
        "rail exit speed (ft/s)": r.rail_exit_speed / FT,
        "apogee above site (ft)": r.apogee_agl / FT,
        "time to apogee (s)": r.t_apogee,
        "max Mach": r.max_mach,
        "max dynamic pressure (kPa)": r.max_q / 1e3,
        "min static margin (cal)": min_static_margin(v, r),
    }
    for k, val in rows.items():
        print(f"{k:30s} {val:.3f}" if isinstance(val, float) else f"{k:30s} {val}")
    for k, val in v.notes.items():
        print(f"{k:30s} {val:.3f}")

    out = ROOT / "data" / "sized_vehicle.json"
    out.write_text(json.dumps({"spec": asdict(spec), "site": asdict(site)}, indent=2) + "\n")
    print(f"\nsaved {out}")
