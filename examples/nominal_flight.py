"""Fly the sized rocket in calm air and in a 6 m/s crosswind, and plot the missions.

Run from the FlightSim6DOF folder:
    python examples/nominal_flight.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from flightsim.design import load_sized
from flightsim.plots import AXIS, INK_2, SERIES, limit_line, plt
from flightsim.sim import SimConfig, Wind, simulate
from flightsim.sizing import FT
from flightsim.structures import flutter_speed

OUT = ROOT / "outputs"


def flutter_margin(vehicle, site, res):
    m = res.phase <= 2
    V = np.linalg.norm(res.v[m], axis=1)
    Vf = np.array([flutter_speed(vehicle.airframe.fins, site.altitude + z) for z in res.r[m, 2]])
    return res.t[m], Vf / np.maximum(V, 1.0)


def plot_trajectory(runs, path):
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for k, (label, res) in enumerate(runs):
        dist = np.hypot(res.r[:, 0], res.r[:, 1]) / FT
        ax.plot(dist, res.r[:, 2] / FT, color=SERIES[k], label=label)
        i_ap = int(np.argmax(res.r[:, 2]))
        ax.plot(dist[i_ap], res.r[i_ap, 2] / FT, "o", color=SERIES[k], ms=7, mec="white", mew=1.5)
    ax.annotate("apogee, drogue out", (dist[i_ap], res.r[i_ap, 2] / FT), xytext=(10, 4),
                textcoords="offset points", fontsize=9, color=INK_2)
    ax.axhline(30_000, color=AXIS, lw=1.0, ls="--")
    ax.annotate("30,000 ft target", (0, 30_000), xytext=(4, 4), textcoords="offset points",
                fontsize=9, color=INK_2)
    ax.set_xlabel("Ground distance from the pad (ft)")
    ax.set_ylabel("Altitude above the pad (ft)")
    ax.set_title("Launch to landing")
    ax.legend(loc="center right")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_history(vehicle, site, runs, path):
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True, constrained_layout=True)
    t_ap = max(res.events["apogee"] for _, res in runs)
    for k, (label, res) in enumerate(runs):
        m = res.phase <= 2
        t = res.t[m]
        axes[0].plot(t, res.r[m, 2] / FT, color=SERIES[k], label=label)
        axes[1].plot(t, res.mach[m], color=SERIES[k])
        axes[2].plot(t, np.degrees(res.alpha[m]), color=SERIES[k])
        axes[3].plot(t, res.margin[m], color=SERIES[k])
        tf, fm = flutter_margin(vehicle, site, res)
        axes[4].plot(tf, fm, color=SERIES[k])
    axes[0].set_title("Altitude above the pad")
    axes[0].set_ylabel("ft")
    axes[0].legend(loc="lower right")
    axes[1].set_title("Mach number")
    axes[1].set_ylabel("M")
    axes[2].set_title("Angle of attack")
    axes[2].set_ylabel("deg")
    axes[3].set_title("Static margin")
    axes[3].set_ylabel("calibers")
    axes[3].set_ylim(0, None)
    limit_line(axes[3], 1.5, "1.5 cal minimum", above=False)
    axes[4].set_title("Fin flutter margin (flutter speed / airspeed)")
    axes[4].set_ylabel("ratio")
    axes[4].set_ylim(0, 8)
    limit_line(axes[4], 1.5, "1.5 minimum", above=False)
    axes[4].set_xlabel("Time from ignition (s)")
    for ax in axes:
        ax.axvline(vehicle.engine.t_end, color=AXIS, lw=0.8)
        ax.set_xlim(0, t_ap)
    axes[0].annotate("burnout", (vehicle.engine.t_end, 0), xytext=(4, 4), textcoords="offset points",
                     fontsize=9, color=INK_2)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_ground_track(runs, path):
    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    for k, (label, res) in enumerate(runs):
        ax.plot(res.r[:, 0] / FT, res.r[:, 1] / FT, color=SERIES[k], label=label)
        ax.plot(res.r[-1, 0] / FT, res.r[-1, 1] / FT, "X", color=SERIES[k], ms=10, mec="white", mew=1.5)
    ax.plot(0, 0, "^", color=INK_2, ms=9)
    ax.annotate("pad", (0, 0), xytext=(8, -4), textcoords="offset points", fontsize=9, color=INK_2)
    ax.set_aspect("equal")
    ax.set_xlabel("East (ft)")
    ax.set_ylabel("North (ft)")
    ax.set_title("Ground track (X = landing)")
    ax.legend(loc="lower right")
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    spec, vehicle, site = load_sized()
    runs = [
        ("Calm", simulate(vehicle, SimConfig(site=site))),
        ("6 m/s wind from the west", simulate(vehicle, SimConfig(site=site, wind=Wind(6.0, 270.0)))),
    ]
    OUT.mkdir(exist_ok=True)
    summaries = {}
    for label, res in runs:
        s = res.summary()
        s["min fin flutter margin"] = float(flutter_margin(vehicle, site, res)[1].min())
        summaries[label] = s
        print(label)
        for k, v in s.items():
            print(f"   {k:40s} {v:.2f}" if isinstance(v, float) else f"   {k:40s} {v}")
    (OUT / "nominal_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    plot_trajectory(runs, OUT / "trajectory.png")
    plot_history(vehicle, site, runs, OUT / "flight_history.png")
    plot_ground_track(runs, OUT / "ground_track.png")
    print(f"saved plots to {OUT}")
