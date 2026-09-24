"""Vehicle sizing with a fast point-mass (3-DOF, planar) trajectory.

The rocket slides up a launch rail, then flies with its axis along the
velocity vector (zero angle of attack, a gravity turn). Forces: thrust,
zero-lift drag and gravity. This is used to choose the propellant load and
the fin size; the 6-DOF simulation then verifies the design.
"""
import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .aero import drag
from .atmosphere import atmosphere, gravity
from .vehicle import Vehicle, VehicleSpec, build_vehicle

FT = 0.3048


@dataclass(frozen=True)
class LaunchSite:
    name: str = "Spaceport America"
    altitude: float = 1401.0          # m above mean sea level (4,596 ft)
    rail_elevation_deg: float = 84.0  # rail angle from horizontal


@dataclass
class PointMassResult:
    t: np.ndarray
    x: np.ndarray             # downrange, m
    z: np.ndarray             # altitude above the site, m
    v: np.ndarray             # speed, m/s
    mach: np.ndarray
    q: np.ndarray             # dynamic pressure, Pa
    rail_exit_speed: float
    apogee_agl: float
    t_apogee: float

    @property
    def max_mach(self) -> float:
        return float(self.mach.max())

    @property
    def max_q(self) -> float:
        return float(self.q.max())


def _drag_force(vehicle: Vehicle, t: float, z_msl: float, V: float) -> tuple[float, float, float]:
    s = atmosphere(z_msl)
    M = V / s.a
    cd = drag(vehicle.airframe, M, s.rho, V, s.mu, burning=vehicle.engine.throttle(t) > 0,
              exit_area=vehicle.engine.data.exit_area).total
    q = 0.5 * s.rho * V * V
    return q * vehicle.airframe.ref_area * cd, M, q


def point_mass(vehicle: Vehicle, site: LaunchSite = LaunchSite(), dt_out: float = 0.05) -> PointMassResult:
    eng = vehicle.engine
    el = math.radians(site.rail_elevation_deg)

    # Phase 1: on the rail (1-D along the rail; held until thrust exceeds weight)
    def rail(t, y):
        s, v = y
        z = site.altitude + s * math.sin(el)
        m = vehicle.mass_properties(t).mass
        T = eng.thrust(t, atmosphere(z).p)
        D, _, _ = _drag_force(vehicle, t, z, v)
        a = (T - D) / m - gravity(z) * math.sin(el)
        return [v, a if (s > 0 or a > 0) else 0.0]

    def off_rail(t, y):
        return y[0] - vehicle.rail_length
    off_rail.terminal, off_rail.direction = True, 1

    r1 = solve_ivp(rail, (0, 30), [0.0, 0.0], events=off_rail, max_step=0.01, rtol=1e-8, atol=1e-9)
    t0 = r1.t[-1]
    v0 = r1.y[1, -1]
    y0 = [vehicle.rail_length * math.cos(el), vehicle.rail_length * math.sin(el),
          v0 * math.cos(el), v0 * math.sin(el)]

    # Phase 2: free flight along the velocity vector, to apogee
    def flight(t, y):
        x, z, vx, vz = y
        V = math.hypot(vx, vz)
        zm = site.altitude + z
        m = vehicle.mass_properties(t).mass
        T = eng.thrust(t, atmosphere(zm).p)
        D, _, _ = _drag_force(vehicle, t, zm, V)
        ux, uz = vx / V, vz / V
        return [vx, vz, (T - D) * ux / m, (T - D) * uz / m - gravity(zm)]

    def apogee(t, y):
        return y[3]
    apogee.terminal, apogee.direction = True, -1

    r2 = solve_ivp(flight, (t0, 300), y0, events=apogee, max_step=0.1, rtol=1e-8, atol=1e-6,
                   dense_output=True)
    t_ap = r2.t[-1]
    t = np.arange(t0, t_ap, dt_out)
    x, z, vx, vz = r2.sol(t)
    V = np.hypot(vx, vz)
    M = np.empty_like(t)
    q = np.empty_like(t)
    for i, (ti, zi, Vi) in enumerate(zip(t, z, V)):
        _, M[i], q[i] = _drag_force(vehicle, ti, site.altitude + zi, Vi)
    return PointMassResult(t, x, z, V, M, q, float(v0), float(r2.y[1, -1]), float(t_ap))


def min_static_margin(vehicle: Vehicle, result: PointMassResult) -> float:
    """Smallest static margin (calibers) over the flight, from rail exit to apogee,
    ignoring the slow final coast (Mach < 0.3), where it does not matter."""
    margins = [vehicle.stability_margin(t, M) for t, M in zip(result.t, result.mach) if M > 0.3]
    return min(margins)


def size_vehicle(spec: VehicleSpec, site: LaunchSite = LaunchSite(), target_apogee_agl: float = 30_000 * FT,
                 target_margin: float = 2.0, iterations: int = 3):
    """Choose the burn time (propellant load) for the target apogee and the fin
    scale for the target minimum static margin. The two interact (fins add
    drag and mass, propellant load moves the CG), so they are iterated."""
    for _ in range(iterations):
        def apogee_error(burn):
            v = build_vehicle(replace(spec, burn_time=burn))
            return point_mass(v, site).apogee_agl - target_apogee_agl
        spec = replace(spec, burn_time=brentq(apogee_error, 4.0, 40.0, xtol=0.01))

        def margin_error(k):
            v = build_vehicle(replace(spec, fin_scale=k))
            return min_static_margin(v, point_mass(v, site)) - target_margin
        spec = replace(spec, fin_scale=brentq(margin_error, 0.3, 2.0, xtol=0.005))
    v = build_vehicle(spec)
    return spec, v, point_mass(v, site)
