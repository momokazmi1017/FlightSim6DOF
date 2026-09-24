"""Monte Carlo dispersion analysis of the sized rocket.

Run from the FlightSim6DOF folder (takes a few minutes; uses all CPU cores):
    python examples/monte_carlo.py            # 500 runs
    python examples/monte_carlo.py 100        # quicker
"""
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from flightsim.design import load_sized
from flightsim.montecarlo import ellipse, run_one, sample_inputs, sensitivity
from flightsim.plots import AXIS, CRITICAL, INK_2, SERIES, SURFACE, limit_line, plt
from flightsim.sizing import FT

OUT = ROOT / "outputs"
SEED = 2026

LABELS = {
    "wind_speed": "Wind speed",
    "wind_from_deg": "Wind direction",
    "wind_exponent": "Wind profile shape",
    "thrust_scale": "Thrust / Isp",
    "cd_scale": "Drag coefficient",
    "cn_scale": "Normal-force slope",
    "dry_mass_scale": "Dry mass",
    "thrust_misalignment_deg": "Thrust misalignment",
    "fin_cant_deg": "Fin cant",
    "rail_elevation_deg": "Rail elevation",
    "rail_azimuth_deg": "Rail azimuth",
}


def plot_apogee(ap, path):
    fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
    ax.hist(ap, bins=30, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5)
    p5, p50, p95 = np.percentile(ap, [5, 50, 95])
    for v, txt in [(p5, "5th percentile"), (p50, "median"), (p95, "95th percentile")]:
        ax.axvline(v, color=INK_2, lw=1.0, ls=":")
        ax.annotate(f"{txt}\n{v:,.0f} ft", (v, 1.0), xycoords=("data", "axes fraction"), xytext=(7, -4),
                    textcoords="offset points", va="top", fontsize=9, color=INK_2)
    ax.axvline(30_000, color=CRITICAL, lw=1.2, ls="--")
    ax.annotate("30,000 ft target (dashed red)", (0.99, 0.62), xycoords="axes fraction", ha="right",
                fontsize=9, color=INK_2)
    ax.set_xlabel("Apogee above the pad (ft)")
    ax.set_ylabel("Number of flights")
    ax.set_title(f"Apogee over {len(ap)} dispersed flights")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_landing(runs, path):
    pts = np.array([[r["landing_east_m"], r["landing_north_m"]] for r in runs]) / FT
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    for r in runs[:60]:
        tr = np.array(r["track"]) / FT
        ax.plot(tr[:, 0], tr[:, 1], color=AXIS, lw=0.6, zorder=1)
    ax.scatter(pts[:, 0], pts[:, 1], s=14, color=SERIES[0], edgecolor=SURFACE, linewidth=0.6, zorder=3,
               label="Landing points")
    for p, k, name in [(0.99, 1, "99 % ellipse"), (0.90, 2, "90 % ellipse")]:
        e = ellipse(pts, p)
        ax.plot(e[:, 0], e[:, 1], color=SERIES[k], lw=2.0, zorder=4, label=name)
    ax.plot(0, 0, "^", color=INK_2, ms=10, zorder=5)
    ax.annotate("pad", (0, 0), xytext=(8, -4), textcoords="offset points", fontsize=9, color=INK_2)
    ax.set_aspect("equal")
    ax.set_xlabel("East (ft)")
    ax.set_ylabel("North (ft)")
    ax.set_title("Landing footprint (grey: ground tracks of 60 flights)")
    ax.legend(loc="lower left")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_sensitivity(sens, path, title):
    names = [LABELS[n] for n, _ in sens][::-1]
    vals = [c for _, c in sens][::-1]
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    colors = [SERIES[0] if v >= 0 else SERIES[1] for v in vals]
    ax.barh(names, vals, color=colors, height=0.6)
    ax.axvline(0, color=AXIS, lw=0.8)
    for y, v in enumerate(vals):
        ax.annotate(f"{v:+.2f}", (v, y), xytext=(4 if v >= 0 else -4, 0), textcoords="offset points",
                    ha="left" if v >= 0 else "right", va="center", fontsize=9, color=INK_2)
    ax.set_xlabel("Standardised regression coefficient (blue: raises it, orange: lowers it)")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    lim = max(abs(v) for v in vals) * 1.25
    ax.set_xlim(-lim, lim)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(n_runs: int):
    spec, vehicle, site = load_sized()
    rng = np.random.default_rng(SEED)
    inputs = sample_inputs(rng, n_runs, site)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        results = list(pool.map(run_one, [(spec, site, x) for x in inputs], chunksize=4))
    print(f"{n_runs} flights in {time.time() - t0:.0f} s on {os.cpu_count()} cores")

    runs = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    ap = np.array([r["apogee_ft"] for r in runs])
    pts = np.array([[r["landing_east_m"], r["landing_north_m"]] for r in runs])
    dist = np.hypot(pts[:, 0], pts[:, 1]) / FT

    names = list(LABELS)
    X = np.array([[r[k] for k in names] for r in runs])
    # Wind direction is circular; use the crosswind/headwind component of the wind
    # relative to the rail (which leans north) instead of the raw angle.
    X[:, names.index("wind_from_deg")] = np.array(
        [r["wind_speed"] * math.cos(math.radians(r["wind_from_deg"])) for r in runs])
    LABELS["wind_from_deg"] = "Headwind component"
    sens_ap = sensitivity(X, ap, names)
    sens_land = sensitivity(X, dist, names)

    worst = {
        "min static margin (cal)": min(r["min_margin_cal"] for r in runs),
        "min flutter margin": min(r["min_flutter_margin"] for r in runs),
        "min rail exit speed (ft/s)": min(r["rail_exit_fps"] for r in runs),
        "max angle of attack (deg)": max(r["max_alpha_deg"] for r in runs),
        "max roll rate (deg/s)": max(r["max_roll_rate_dps"] for r in runs),
        "max landing descent rate (ft/s)": max(r["landing_descent_fps"] for r in runs),
        "max landing distance (ft)": float(dist.max()),
    }
    e99 = ellipse(pts / FT, 0.99)
    summary = {
        "runs": n_runs, "failed": len(failed), "seed": SEED,
        "apogee_ft": {"mean": float(ap.mean()), "std": float(ap.std()),
                      "p5": float(np.percentile(ap, 5)), "p50": float(np.percentile(ap, 50)),
                      "p95": float(np.percentile(ap, 95)), "min": float(ap.min()), "max": float(ap.max())},
        "within_10pct_of_target": float(np.mean(np.abs(ap - 30_000) <= 3000)),
        "landing_ellipse_99_semi_axes_ft": sorted(
            float(v) for v in np.sqrt(-2 * math.log(0.01) * np.linalg.eigvalsh(np.cov((pts / FT).T)))),
        "landing_ellipse_99_max_distance_from_pad_ft": float(np.hypot(e99[:, 0], e99[:, 1]).max()),
        "worst_case": worst,
        "sensitivity_apogee": [(LABELS[n], float(c)) for n, c in sens_ap],
        "sensitivity_landing_distance": [(LABELS[n], float(c)) for n, c in sens_land],
    }
    print(json.dumps(summary, indent=2))

    OUT.mkdir(exist_ok=True)
    (OUT / "monte_carlo_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "monte_carlo_runs.json").write_text(json.dumps(results) + "\n")
    plot_apogee(ap, OUT / "mc_apogee.png")
    plot_landing(runs, OUT / "mc_landing.png")
    plot_sensitivity(sens_ap, OUT / "mc_sensitivity_apogee.png", "What drives the apogee spread")
    plot_sensitivity(sens_land, OUT / "mc_sensitivity_landing.png", "What drives the landing distance")
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
