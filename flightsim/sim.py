"""Six-degree-of-freedom flight simulation, launch to landing.

Frames: inertial East-North-Up (ENU) at the launch site (flat, non-rotating
Earth; gravity falls off with altitude). Body frame: x along the rocket axis
toward the nose, y and z lateral. Stations are measured from the nose tip
toward the tail.

State in free flight (13): position r, velocity v (inertial), attitude
quaternion q (body -> inertial), body angular rate omega.

Loads, in the body frame:
- Thrust along the (possibly misaligned) engine axis at the nozzle exit.
- Axial drag: -q A CD0(M), from the zero-lift drag buildup.
- Normal force on each lifting component (nose, fins), from the airflow
  velocity it sees locally, v_b + omega x r_i. Evaluating it component by
  component gives both the restoring (weathercock) moment and the aerodynamic
  pitch/yaw damping from one expression. CN = CN_alpha sin(alpha).
- Jet damping from the exhaust carrying away angular momentum:
  M = -mdot (x_exit - x_cg)^2 omega_lateral.
- Roll: forcing from fin cant (misalignment), damping from rotating fins.

Rigid-body dynamics with time-varying mass properties:
    m dv/dt = F,    I domega/dt = M - omega x (I omega).

Phases: launch rail (1-D along the rail), free flight to apogee, drogue
descent, main descent to landing (3-DOF under parachute).
"""
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from . import aero
from .atmosphere import atmosphere, gravity
from .rotations import attitude_from_axis, quat_multiply, quat_to_dcm
from .sizing import FT, LaunchSite
from .vehicle import Vehicle


@dataclass(frozen=True)
class Wind:
    """Horizontal wind with a power-law boundary-layer profile:
    speed = speed_10m * (z_agl / 10 m) ** exponent."""
    speed_10m: float = 0.0            # m/s at 10 m above ground
    from_deg: float = 270.0           # direction the wind blows FROM, degrees from North
    exponent: float = 1 / 7

    def at(self, z_agl: float) -> np.ndarray:
        if self.speed_10m == 0.0:
            return np.zeros(3)
        s = self.speed_10m * (max(z_agl, 1.0) / 10.0) ** self.exponent
        to = math.radians(self.from_deg + 180.0)
        return np.array([s * math.sin(to), s * math.cos(to), 0.0])


@dataclass(frozen=True)
class Disturbances:
    """Deviations from the nominal vehicle (used by the Monte Carlo analysis)."""
    thrust_scale: float = 1.0
    thrust_misalignment_deg: float = 0.0
    thrust_misalignment_azimuth_deg: float = 0.0   # around the body axis
    fin_cant_deg: float = 0.0                       # net fin misalignment
    cd_scale: float = 1.0
    cn_scale: float = 1.0


@dataclass(frozen=True)
class Recovery:
    drogue_cda: float = 0.80          # m^2: about 30 m/s (100 ft/s) descent
    main_cda: float = 20.0            # m^2: about 6 m/s (20 ft/s) landing
    main_altitude_agl: float = 1500 * FT


@dataclass(frozen=True)
class SimConfig:
    site: LaunchSite = LaunchSite()
    rail_azimuth_deg: float = 0.0     # direction the rail leans toward, degrees from North
    wind: Wind = Wind()
    disturbances: Disturbances = Disturbances()
    recovery: Recovery = Recovery()
    vacuum: bool = False              # no atmosphere: for validation
    dt_out: float = 0.02
    rtol: float = 1e-7


@dataclass
class Loads:
    force_b: np.ndarray
    moment_b: np.ndarray
    thrust: float
    mach: float
    alpha: float             # total angle of attack, rad
    qbar: float
    x_cp: float
    x_cg: float


def rail_direction(cfg: SimConfig) -> np.ndarray:
    el = math.radians(cfg.site.rail_elevation_deg)
    az = math.radians(cfg.rail_azimuth_deg)
    return np.array([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)])


def body_loads(vehicle: Vehicle, cfg: SimConfig, t: float, z_msl: float,
               v_air_b: np.ndarray, omega: np.ndarray) -> Loads:
    """Aerodynamic and propulsive force and moment about the CG, in the body frame.
    v_air_b is the velocity of the rocket relative to the air, in body axes."""
    dist = cfg.disturbances
    af = vehicle.airframe
    mp = vehicle.mass_properties(t)
    atm = atmosphere(z_msl)
    p_amb = 0.0 if cfg.vacuum else atm.p

    F = np.zeros(3)
    M = np.zeros(3)

    # Thrust
    T = dist.thrust_scale * vehicle.engine.thrust(t, p_amb)
    if T > 0:
        e = math.radians(dist.thrust_misalignment_deg)
        phi = math.radians(dist.thrust_misalignment_azimuth_deg)
        F_T = T * np.array([math.cos(e), math.sin(e) * math.cos(phi), math.sin(e) * math.sin(phi)])
        r_T = np.array([mp.x_cg - vehicle.nozzle_exit_station, 0.0, 0.0])
        F += F_T
        M += np.cross(r_T, F_T)
        # Jet damping
        mdot = dist.thrust_scale * (vehicle.engine.mdot_ox(t) + vehicle.engine.mdot_fuel(t))
        arm2 = (vehicle.nozzle_exit_station - mp.x_cg) ** 2
        M += -mdot * arm2 * np.array([0.0, omega[1], omega[2]])

    V = float(np.linalg.norm(v_air_b))
    if cfg.vacuum or V < 1.0:
        return Loads(F, M, T, 0.0, 0.0, 0.0, float("nan"), mp.x_cg)

    mach = V / atm.a
    qbar = 0.5 * atm.rho * V * V
    qA = qbar * af.ref_area

    # Axial drag
    cd = dist.cd_scale * aero.drag(af, mach, atm.rho, V, atm.mu, burning=T > 0,
                                   exit_area=vehicle.engine.data.exit_area).total
    F[0] -= qA * cd * v_air_b[0] / V

    # Normal force, component by component, from the local airflow (includes damping)
    cn_total = x_cp_moment = 0.0
    for c in aero.components(af, mach):
        cn = dist.cn_scale * c.cn_alpha
        r = mp.x_cg - c.x_cp                        # position along body x, from the CG
        v_lat = np.array([v_air_b[1] + omega[2] * r, v_air_b[2] - omega[1] * r])
        F_lat = -qA * cn * v_lat / V                 # CN = CN_alpha sin(alpha)
        F[1] += F_lat[0]
        F[2] += F_lat[1]
        M[1] += -r * F_lat[1]
        M[2] += r * F_lat[0]
        cn_total += cn
        x_cp_moment += cn * c.x_cp

    # Roll: fin cant forcing and roll damping
    fins = af.fins
    cna_panel = dist.cn_scale * aero.fin_cn_alpha_panel(af, mach)
    y_t = af.radius + fins.span * (fins.root_chord + 2 * fins.tip_chord) / (
        3 * (fins.root_chord + fins.tip_chord))
    M[0] += fins.n * qA * cna_panel * y_t * (math.radians(dist.fin_cant_deg) - omega[0] * y_t / V)

    alpha = math.atan2(math.hypot(v_air_b[1], v_air_b[2]), v_air_b[0])
    return Loads(F, M, T, mach, alpha, qbar, x_cp_moment / cn_total, mp.x_cg)


@dataclass
class FlightResult:
    t: np.ndarray
    r: np.ndarray            # (N, 3) position ENU, m
    v: np.ndarray            # (N, 3) velocity ENU, m/s
    q: np.ndarray            # (N, 4) attitude quaternion (NaN under parachute)
    omega: np.ndarray        # (N, 3) body rates, rad/s
    mach: np.ndarray
    alpha: np.ndarray        # rad
    qbar: np.ndarray
    thrust: np.ndarray
    margin: np.ndarray       # static margin, calibers
    phase: np.ndarray        # 0 rail, 1 powered, 2 coast, 3 drogue, 4 main
    events: dict = field(default_factory=dict)

    @property
    def apogee_agl(self) -> float:
        return float(self.r[:, 2].max())

    def summary(self) -> dict:
        e = self.events
        land = self.r[-1]
        flight = self.phase <= 2
        return {
            "apogee above site (ft)": self.apogee_agl / FT,
            "time to apogee (s)": e["apogee"],
            "rail exit speed (ft/s)": e["rail_exit_speed"] / FT,
            "burnout altitude (ft)": e["burnout_altitude"] / FT,
            "max Mach": float(np.nanmax(self.mach)),
            "max dynamic pressure (kPa)": float(np.nanmax(self.qbar) / 1e3),
            "max angle of attack after rail (deg)": float(np.degrees(self.alpha[self.phase == 1].max(initial=0))),
            "min static margin, M > 0.3 (cal)": float(np.nanmin(np.where(flight & (self.mach > 0.3),
                                                                          self.margin, np.nan))),
            "drogue descent rate (ft/s)": e.get("drogue_rate", float("nan")) / FT,
            "landing descent rate (ft/s)": float(-self.v[-1, 2]) / FT,
            "landing ground speed (ft/s)": float(np.hypot(self.v[-1, 0], self.v[-1, 1])) / FT,
            "flight time (s)": float(self.t[-1]),
            "landing distance (m)": float(math.hypot(land[0], land[1])),
            "landing east / north (m)": (float(land[0]), float(land[1])),
        }


def _state_derivative(vehicle: Vehicle, cfg: SimConfig, t: float, y: np.ndarray) -> np.ndarray:
    r, v, q, w = y[0:3], y[3:6], y[6:10], y[10:13]
    qn = q / np.linalg.norm(q)
    R = quat_to_dcm(qn)
    z_msl = cfg.site.altitude + r[2]
    wind = np.zeros(3) if cfg.vacuum else cfg.wind.at(r[2])
    v_air_b = R.T @ (v - wind)
    L = body_loads(vehicle, cfg, t, z_msl, v_air_b, w)
    mp = vehicle.mass_properties(t)

    a = R @ L.force_b / mp.mass - np.array([0.0, 0.0, gravity(z_msl)])
    I = np.array([mp.I_axial, mp.I_transverse, mp.I_transverse])
    w_dot = (L.moment_b - np.cross(w, I * w)) / I
    # Quaternion kinematics, with a gentle pull back toward unit norm
    q_dot = 0.5 * quat_multiply(q, [0.0, *w]) + 1.0 * (1 - q @ q) * q
    return np.concatenate([v, a, q_dot, w_dot])


def simulate(vehicle: Vehicle, cfg: SimConfig = SimConfig()) -> FlightResult:
    eng = vehicle.engine
    site = cfg.site
    d = rail_direction(cfg)
    el = math.radians(site.rail_elevation_deg)

    # --- Phase 0: launch rail --------------------------------------------------
    def rail(t, y):
        s, u = y
        z = site.altitude + s * d[2]
        m = vehicle.mass_properties(t).mass
        v_air = u * d - (np.zeros(3) if cfg.vacuum else cfg.wind.at(s * d[2]))
        u_air = float(v_air @ d)
        L = body_loads(vehicle, cfg, t, z, np.array([u_air, 0.0, 0.0]), np.zeros(3))
        acc = L.force_b[0] / m - gravity(z) * math.sin(el)
        return [u, acc if (s > 0 or acc > 0) else 0.0]

    def off_rail(t, y):
        return y[0] - vehicle.rail_length
    off_rail.terminal, off_rail.direction = True, 1

    s_rail = solve_ivp(rail, (0.0, 60.0), [0.0, 0.0], events=off_rail, max_step=0.005,
                       rtol=1e-9, atol=1e-9, dense_output=True)
    if s_rail.status != 1:
        raise RuntimeError("the rocket never left the rail")
    t_rail = float(s_rail.t[-1])
    u_rail = float(s_rail.y[1, -1])

    # --- Phase 1-2: 6-DOF free flight to apogee -----------------------------------
    y0 = np.concatenate([vehicle.rail_length * d, u_rail * d, attitude_from_axis(d), np.zeros(3)])

    def apogee(t, y):
        return y[5] if t > eng.t_end else 1.0
    apogee.terminal, apogee.direction = True, -1

    def crashed(t, y):
        return y[2] + 1.0
    crashed.terminal = True

    breaks = [tb for tb in (eng.t_startup, eng.t_cutoff, eng.t_end) if tb > t_rail] + [t_rail + 600.0]
    segments = []
    t0, y = t_rail, y0
    for tb in breaks:
        sol = solve_ivp(lambda t, yy: _state_derivative(vehicle, cfg, t, yy), (t0, tb), y,
                        method="DOP853", events=[apogee, crashed], rtol=cfg.rtol, atol=1e-8,
                        dense_output=True, max_step=0.05)
        segments.append(sol)
        t0, y = sol.t[-1], sol.y[:, -1]
        if sol.status == 1:
            break
    if segments[-1].t_events[1].size:
        raise RuntimeError("the rocket hit the ground before apogee")
    t_apogee, y_apogee = t0, y

    # --- Phase 3-4: descent under parachutes (3-DOF) ------------------------------
    m_final = vehicle.mass_properties(t_apogee).mass
    rec = cfg.recovery

    def descent(cda):
        def f(t, yy):
            r, v = yy[:3], yy[3:]
            atm = atmosphere(site.altitude + r[2])
            v_air = v - cfg.wind.at(r[2])
            F = -0.5 * atm.rho * cda * np.linalg.norm(v_air) * v_air
            return np.concatenate([v, F / m_final - np.array([0, 0, gravity(site.altitude + r[2])])])
        return f

    def main_deploy(t, yy):
        return yy[2] - rec.main_altitude_agl
    main_deploy.terminal, main_deploy.direction = True, -1

    def landed(t, yy):
        return yy[2]
    landed.terminal, landed.direction = True, -1

    y_d = np.concatenate([y_apogee[:3], y_apogee[3:6]])
    s_drogue = solve_ivp(descent(rec.drogue_cda), (t_apogee, t_apogee + 3000), y_d, events=main_deploy,
                         rtol=1e-7, atol=1e-6, dense_output=True, max_step=1.0)
    s_main = solve_ivp(descent(rec.main_cda), (s_drogue.t[-1], s_drogue.t[-1] + 3000), s_drogue.y[:, -1],
                       events=landed, rtol=1e-7, atol=1e-6, dense_output=True, max_step=1.0)

    # --- Sample everything on a common time grid --------------------------------------
    rows = []

    def add(times, phase, sol, six_dof):
        for t in times:
            yy = sol.sol(t)
            if six_dof:
                q = yy[6:10] / np.linalg.norm(yy[6:10])
                R = quat_to_dcm(q)
                z_msl = site.altitude + yy[2]
                v_air_b = R.T @ (yy[3:6] - (np.zeros(3) if cfg.vacuum else cfg.wind.at(yy[2])))
                L = body_loads(vehicle, cfg, t, z_msl, v_air_b, yy[10:13])
                margin = (L.x_cp - L.x_cg) / vehicle.airframe.diameter
                ph = 1 if L.thrust > 0 else 2
                rows.append((t, *yy[0:6], *q, *yy[10:13], L.mach, L.alpha, L.qbar, L.thrust, margin, ph))
            else:
                rows.append((t, *yy[0:6], *[np.nan] * 7, np.nan, np.nan, np.nan, 0.0, np.nan, phase))

    for t in np.arange(0.0, t_rail, cfg.dt_out):
        s, u = s_rail.sol(t)
        z = site.altitude + s * d[2]
        rows.append((t, *(s * d), *(u * d), *attitude_from_axis(d), 0, 0, 0,
                     u / atmosphere(z).a, 0.0, 0.5 * atmosphere(z).rho * u * u,
                     eng.thrust(t, atmosphere(z).p), float("nan"), 0))
    for sol in segments:
        add(np.arange(sol.t[0], sol.t[-1], cfg.dt_out), None, sol, True)
    add(np.arange(s_drogue.t[0], s_drogue.t[-1], 0.5), 3, s_drogue, False)
    add(np.append(np.arange(s_main.t[0], s_main.t[-1], 0.5), s_main.t[-1]), 4, s_main, False)

    a = np.array(rows)
    z_burnout = next((float(sol.sol(eng.t_end)[2]) for sol in segments
                      if sol.t[0] <= eng.t_end <= sol.t[-1]), 0.0)
    events = {
        "rail_exit": t_rail, "rail_exit_speed": u_rail, "burnout": eng.t_end,
        "burnout_altitude": z_burnout, "apogee": t_apogee,
        "main_deploy": float(s_drogue.t[-1]), "landing": float(s_main.t[-1]),
        "drogue_rate": float(-s_drogue.y[5, -1]),
    }
    return FlightResult(t=a[:, 0], r=a[:, 1:4], v=a[:, 4:7], q=a[:, 7:11], omega=a[:, 11:14],
                        mach=a[:, 14], alpha=a[:, 15], qbar=a[:, 16], thrust=a[:, 17], margin=a[:, 18],
                        phase=a[:, 19].astype(int), events=events)
