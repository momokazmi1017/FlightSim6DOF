"""Import an engine design from the Liquid Rocket Engine Design Tool (project 1).

Reads the engine's design summary and writes the few numbers the flight
simulator needs to data/<name>_engine.json, with a note of where they came from.

Thrust at any ambient pressure follows from the design point. At the design
ambient pressure pa_design the nozzle exit pressure equals pa_design, so
    F(pa) = F_vac - pa * A_exit,   with   F_vac = F_design + pa_design * A_exit.

Run from the FlightSim6DOF folder:
    python tools/import_engine.py ../LiquidRocketEngine/outputs/E5-75/design_summary.json
"""
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
G0 = 9.80665


def main(summary_path: Path):
    s = json.loads(summary_path.read_text())
    pa_design = s["exit_pressure_bar"] * 1e5          # optimum expansion: pe = pa at design
    a_exit = math.pi * (s["exit_diameter_mm"] / 1e3) ** 2 / 4
    f_vac = s["thrust_N"] + pa_design * a_exit
    mdot = s["mdot_total_kg_s"]

    try:
        commit = subprocess.run(["git", "-C", str(summary_path.parent), "log", "-1", "--format=%h"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"

    engine = {
        "name": s["name"],
        "source": f"Liquid Rocket Engine Design Tool, {summary_path.name} (commit {commit})",
        "propellants": s["propellants"],
        "thrust_vac_N": f_vac,
        "thrust_sea_level_N": s["thrust_N"],
        "exit_area_m2": a_exit,
        "mdot_ox_kg_s": s["mdot_ox_kg_s"],
        "mdot_fuel_kg_s": s["mdot_fuel_kg_s"],
        "isp_vac_s": f_vac / (mdot * G0),
        "isp_sea_level_s": s["thrust_N"] / (mdot * G0),
        "chamber_pressure_bar": s["chamber_pressure_bar"],
        "exit_diameter_m": s["exit_diameter_mm"] / 1e3,
        "length_m": s["overall_length_mm"] / 1e3,
    }
    out = ROOT / "data" / f"{s['name'].lower().replace('-', '_')}_engine.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(engine, indent=2) + "\n")
    print(f"wrote {out}")
    for k, v in engine.items():
        print(f"  {k:22s} {v:.4g}" if isinstance(v, float) else f"  {k:22s} {v}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
