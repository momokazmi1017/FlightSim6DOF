"""Monte Carlo dispersion analysis.

Each run draws the uncertain inputs from the distributions below, flies the
full 6-DOF mission, and records apogee, landing point and safety margins.
Inputs are drawn from a seeded generator, so a study is reproducible.

Uncertainties (1-sigma unless noted):
- thrust / specific impulse       +/- 3 %
- zero-lift drag                  +/- 10 %  (component buildup, worst near Mach 1)
- normal-force slope              +/- 5 %
- dry mass                        +/- 2 %
- thrust misalignment             0.1 deg (half-normal), random direction
- net fin cant                    0.1 deg
- rail elevation                  84 deg +/- 0.5 deg; rail azimuth +/- 2 deg
- surface wind speed              uniform 0-8 m/s (the range's ~20 mph limit is 9 m/s)
- wind direction                  from 240 deg +/- 40 deg (prevailing south-westerly;
                                  an assumption for Spaceport America in June)
- wind profile exponent           uniform 0.10-0.20
- parachute drag areas            +/- 10 % each
"""
import math
from dataclasses import asdict, dataclass, replace

import numpy as np

from .sim import Disturbances, Recovery, SimConfig, Wind, simulate
from .sizing import LaunchSite
from .structures import flutter_speed
from .vehicle import VehicleSpec, build_vehicle


@dataclass(frozen=True)
class RunInputs:
    thrust_scale: float
    cd_scale: float
    cn_scale: float
    dry_mass_scale: float
    thrust_misalignment_deg: float
    thrust_misalignment_azimuth_deg: float
    fin_cant_deg: float
    rail_elevation_deg: float
    rail_azimuth_deg: float
    wind_speed: float
    wind_from_deg: float
    wind_exponent: float
    drogue_scale: float
    main_scale: float


def sample_inputs(rng: np.random.Generator, n: int, site: LaunchSite) -> list[RunInputs]:
    def normal(mean, sd, lo=-np.inf, hi=np.inf):
        return np.clip(rng.normal(mean, sd, n), lo, hi)

    cols = dict(
        thrust_scale=normal(1.0, 0.03),
        cd_scale=normal(1.0, 0.10, 0.6, 1.4),
        cn_scale=normal(1.0, 0.05),
        dry_mass_scale=normal(1.0, 0.02),
        thrust_misalignment_deg=np.abs(rng.normal(0.0, 0.1, n)),
        thrust_misalignment_azimuth_deg=rng.uniform(0.0, 360.0, n),
        fin_cant_deg=rng.normal(0.0, 0.1, n),
        rail_elevation_deg=normal(site.rail_elevation_deg, 0.5, 80.0, 89.0),
        rail_azimuth_deg=rng.normal(0.0, 2.0, n),
        wind_speed=rng.uniform(0.0, 8.0, n),
        wind_from_deg=rng.normal(240.0, 40.0, n) % 360.0,
        wind_exponent=rng.uniform(0.10, 0.20, n),
        drogue_scale=normal(1.0, 0.10, 0.6, 1.4),
        main_scale=normal(1.0, 0.10, 0.6, 1.4),
    )
    return [RunInputs(**{k: float(v[i]) for k, v in cols.items()}) for i in range(n)]


def run_one(args) -> dict:
    """Fly one dispersed mission. args = (base VehicleSpec, LaunchSite, RunInputs)."""
    spec, site, x = args
    vehicle = build_vehicle(replace(spec, dry_mass_scale=x.dry_mass_scale))
    base = Recovery()
    cfg = SimConfig(
        site=replace(site, rail_elevation_deg=x.rail_elevation_deg),
        rail_azimuth_deg=x.rail_azimuth_deg,
        wind=Wind(x.wind_speed, x.wind_from_deg, x.wind_exponent),
        disturbances=Disturbances(x.thrust_scale, x.thrust_misalignment_deg,
                                  x.thrust_misalignment_azimuth_deg, x.fin_cant_deg,
                                  x.cd_scale, x.cn_scale),
        recovery=replace(base, drogue_cda=base.drogue_cda * x.drogue_scale,
                         main_cda=base.main_cda * x.main_scale),
        dt_out=0.05,
    )
    out = asdict(x)
    try:
        res = simulate(vehicle, cfg)
    except RuntimeError as e:
        out.update(ok=False, error=str(e))
        return out
    flight = res.phase <= 2
    V = np.linalg.norm(res.v[flight], axis=1)
    Vf = np.array([flutter_speed(vehicle.airframe.fins, site.altitude + z) for z in res.r[flight, 2]])
    s = res.summary()
    out.update(
        ok=True,
        apogee_ft=s["apogee above site (ft)"],
        t_apogee=s["time to apogee (s)"],
        max_mach=s["max Mach"],
        max_q_kpa=s["max dynamic pressure (kPa)"],
        rail_exit_fps=s["rail exit speed (ft/s)"],
        max_alpha_deg=s["max angle of attack after rail (deg)"],
        min_margin_cal=s["min static margin, M > 0.3 (cal)"],
        min_flutter_margin=float(np.min(Vf / np.maximum(V, 1.0))),
        landing_east_m=float(res.r[-1, 0]),
        landing_north_m=float(res.r[-1, 1]),
        landing_descent_fps=s["landing descent rate (ft/s)"],
        landing_drift_fps=s["landing ground speed (ft/s)"],
        max_roll_rate_dps=float(np.degrees(np.nanmax(np.abs(res.omega[flight, 0])))),
        # ground track, thinned, for plotting and for the 3D viewer
        track=res.r[::20].round(1).tolist(),
    )
    return out


def ellipse(points: np.ndarray, probability: float = 0.99, n: int = 200) -> np.ndarray:
    """Boundary of the confidence ellipse containing `probability` of a 2-D
    Gaussian fitted to the points (chi-square quantile with 2 degrees of freedom)."""
    k2 = -2 * math.log(1 - probability)
    mean = points.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov(points.T))
    t = np.linspace(0, 2 * math.pi, n)
    circle = np.stack([np.cos(t), np.sin(t)])
    return (mean[:, None] + vecs @ (np.sqrt(k2 * vals)[:, None] * circle)).T


def sensitivity(inputs: np.ndarray, output: np.ndarray, names: list[str]) -> list[tuple[str, float]]:
    """Standardised regression coefficients: fit output ~ inputs on standardised
    variables. Each squared coefficient is roughly that input's share of the
    output variance (when the response is close to linear)."""
    X = (inputs - inputs.mean(axis=0)) / inputs.std(axis=0)
    y = (output - output.mean()) / output.std()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return sorted(zip(names, coef), key=lambda kv: -abs(kv[1]))
