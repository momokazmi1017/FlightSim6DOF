"""Vehicle layout and time-varying mass properties.

The rocket is a stack of sections from the nose tip down. Each section holds
components, each modelled as a uniform cylinder (a thin shell or a solid) so
its centre of gravity and inertia follow from its mass, station and length.

Propellant tanks are integral (the tank wall is the airframe). Wall thickness
comes from hoop stress at the tank's maximum operating pressure (MEOP):
    t = MEOP * r * SF / sigma_yield.
The pressurant (nitrogen) bottle is sized to push all the propellant out at
tank pressure. During flight the liquid in each tank sits at the aft end
(thrust pushes it there), so its centre of gravity moves aft as it drains.
"""
import math
from dataclasses import dataclass, field, replace

import numpy as np

from .engine import Engine, EngineData
from .geometry import Airframe, FinSet, Nose

# Materials and propellants
RHO_AL = 2700.0            # 6061-T6
SIGMA_Y_AL = 276e6
RHO_LOX = 1141.0
RHO_FUEL = 856.0           # 75 % ethanol / 25 % water by mass, 20 C (measured, includes mixing contraction)
R_N2 = 296.8
RHO_FIN = 1850.0           # G10 fibreglass


@dataclass(frozen=True)
class Part:
    """A uniform cylinder: station of its forward end, length, mass, radius, shell or solid."""
    name: str
    mass: float
    station: float
    length: float
    radius: float
    shell: bool = False

    @property
    def x_cg(self) -> float:
        return self.station + self.length / 2

    def inertia_about_own_cg(self) -> tuple[float, float]:
        """(axial, transverse) moments of inertia of a thin-walled or solid cylinder."""
        m, r2, L2 = self.mass, self.radius ** 2, self.length ** 2
        if self.shell:
            return m * r2, m * r2 / 2 + m * L2 / 12
        return m * r2 / 2, m * r2 / 4 + m * L2 / 12


@dataclass(frozen=True)
class Tank:
    name: str
    station: float          # forward end
    length: float
    inner_radius: float
    propellant_density: float
    propellant_mass: float  # at liftoff, including residuals

    def liquid(self, mass: float) -> Part:
        """The remaining liquid as a solid cylinder at the aft end of the tank."""
        area = math.pi * self.inner_radius ** 2
        h = min(mass / (self.propellant_density * area), self.length)
        return Part(f"{self.name} liquid", mass, self.station + self.length - h, max(h, 1e-6),
                    self.inner_radius)


@dataclass
class MassProperties:
    mass: float
    x_cg: float             # station from the nose tip
    I_axial: float          # roll
    I_transverse: float     # pitch / yaw, about the CG


@dataclass
class Vehicle:
    name: str
    airframe: Airframe
    engine: Engine
    parts: list[Part]               # dry structure and fixed masses
    ox_tank: Tank
    fuel_tank: Tank
    residual_fraction: float
    nozzle_exit_station: float
    rail_length: float = 12.2       # m (40 ft tower)
    notes: dict = field(default_factory=dict)

    def __post_init__(self):
        # Mass properties change only through propellant use, so tabulate them
        # once over the burn and interpolate during integration.
        t = np.linspace(0.0, self.engine.t_end, 200)
        props = [self._mass_properties_exact(ti) for ti in t]
        self._t_table = t
        self._props_table = np.array([[p.mass, p.x_cg, p.I_axial, p.I_transverse] for p in props])

    @property
    def dry_mass(self) -> float:
        return sum(p.mass for p in self.parts)

    @property
    def propellant_mass(self) -> float:
        return self.ox_tank.propellant_mass + self.fuel_tank.propellant_mass

    def mass_properties(self, t: float) -> MassProperties:
        tt = min(max(t, 0.0), self._t_table[-1])
        row = [float(np.interp(tt, self._t_table, self._props_table[:, k])) for k in range(4)]
        return MassProperties(*row)

    def _mass_properties_exact(self, t: float) -> MassProperties:
        ox_used, fuel_used = self.engine.consumed(t)
        parts = list(self.parts) + [
            self.ox_tank.liquid(self.ox_tank.propellant_mass - ox_used),
            self.fuel_tank.liquid(self.fuel_tank.propellant_mass - fuel_used),
        ]
        m = sum(p.mass for p in parts)
        x_cg = sum(p.mass * p.x_cg for p in parts) / m
        I_ax = I_tr = 0.0
        for p in parts:
            a, tr = p.inertia_about_own_cg()
            I_ax += a
            I_tr += tr + p.mass * (p.x_cg - x_cg) ** 2
        return MassProperties(m, x_cg, I_ax, I_tr)

    def stability_margin(self, t: float, mach: float) -> float:
        """(x_cp - x_cg) in calibers (body diameters). Positive is stable."""
        from .aero import normal_force
        _, x_cp = normal_force(self.airframe, mach)
        return (x_cp - self.mass_properties(t).x_cg) / self.airframe.diameter


# --- Parametric vehicle ----------------------------------------------------------

@dataclass(frozen=True)
class VehicleSpec:
    """Top-level choices for the liquid sounding rocket."""
    burn_time: float                      # s, sets the propellant load
    diameter: float = 0.1524              # m (6.0 in)
    nose_fineness: float = 5.0
    payload_mass: float = 4.0             # kg (Spaceport America Cup minimum, 8.8 lb)
    ullage_fraction: float = 0.08
    residual_fraction: float = 0.03       # unusable propellant left at shutdown
    meop_ox: float = 33e5                 # Pa: chamber + injector drop + feed losses
    meop_fuel: float = 37e5               # Pa: also the regen-jacket pressure drop
    tank_safety_factor: float = 2.0       # on yield
    min_wall: float = 2.0e-3              # m, handling / manufacturing minimum
    copv_pressure: float = 300e5          # Pa
    copv_mass_per_litre: float = 0.65     # kg/L, carbon-overwrapped bottle
    tube_mass_per_m: float = 1.3          # kg/m, carbon-fibre airframe (non-tank sections)
    engine_mass: float = 7.0              # kg, regen chamber + injector + mount
    # n, root, tip, span, sweep, thickness. 1/4 in G10: 3/16 in fins failed the flutter check (margin 1.32).
    fins: tuple = (4, 0.30, 0.10, 0.15, 0.20, 6.35e-3)
    fin_scale: float = 1.0                # scales root, tip, span and sweep together
    dry_mass_scale: float = 1.0           # as-built / predicted dry mass (Monte Carlo)
    rail_length: float = 12.2             # m


def build_vehicle(spec: VehicleSpec, engine_data: EngineData | None = None) -> Vehicle:
    ed = engine_data or EngineData.load()
    eng = Engine(ed, spec.burn_time)
    d, r = spec.diameter, spec.diameter / 2
    r_in = r - spec.min_wall

    # Propellant load: usable flow over the burn, plus unusable residuals
    m_ox = ed.mdot_ox * spec.burn_time / (1 - spec.residual_fraction)
    m_fuel = ed.mdot_fuel * spec.burn_time / (1 - spec.residual_fraction)

    def tank_length(m, rho):
        return m / rho * (1 + spec.ullage_fraction) / (math.pi * r_in ** 2)

    def wall(meop):
        return max(meop * r * spec.tank_safety_factor / SIGMA_Y_AL, spec.min_wall)

    # Pressurant: N2 filling both tanks at MEOP at the end of the burn (isothermal
    # estimate at 280 K), plus what stays in the bottle at tank pressure.
    v_tanks = (m_ox / RHO_LOX + m_fuel / RHO_FUEL) * (1 + spec.ullage_fraction)
    p_tank = max(spec.meop_ox, spec.meop_fuel)
    m_n2_expelled = p_tank * v_tanks / (R_N2 * 280.0)
    copv_volume = m_n2_expelled * R_N2 * 280.0 / (spec.copv_pressure - p_tank)
    m_n2 = spec.copv_pressure * copv_volume / (R_N2 * 280.0)
    m_copv = spec.copv_mass_per_litre * copv_volume * 1e3

    parts: list[Part] = []
    x = 0.0

    def section(name, length, contents):
        """Add a tube section of given length holding (name, mass) items spread along it."""
        nonlocal x
        parts.append(Part(f"{name} tube", spec.tube_mass_per_m * length, x, length, r, shell=True))
        for item, mass in contents:
            parts.append(Part(item, mass, x, length, 0.7 * r))
        x += length

    # Nose cone (fibreglass shell) with the payload in its base
    Ln = spec.nose_fineness * d
    parts.append(Part("nose cone", 2.0, 0.0, Ln, 0.6 * r, shell=True))
    parts.append(Part("payload", spec.payload_mass, 0.6 * Ln, 0.4 * Ln, 0.6 * r))
    x = Ln
    section("recovery bay", 0.70, [("parachutes and harness", 4.0), ("recovery electronics", 1.0)])
    section("avionics bay", 0.30, [("flight computer and GPS", 1.5), ("batteries", 1.0)])
    section("pressurant bay", max(0.35, copv_volume / (math.pi * (0.8 * r) ** 2) + 0.10),
            [("COPV", m_copv), ("nitrogen", m_n2), ("regulator and valves", 2.5)])

    tanks = {}
    for name, m, rho, meop in [("LOX", m_ox, RHO_LOX, spec.meop_ox),
                               ("fuel", m_fuel, RHO_FUEL, spec.meop_fuel)]:
        L = tank_length(m, rho)
        t_w = wall(meop)
        shell = RHO_AL * math.pi * d * t_w * L
        bulkheads = 2 * RHO_AL * math.pi * r ** 2 * 0.012 * 0.5     # two domed/machined ends
        parts.append(Part(f"{name} tank wall", shell, x, L, r, shell=True))
        parts.append(Part(f"{name} tank bulkheads", bulkheads, x, L, 0.8 * r))
        tanks[name] = Tank(name, x, L, r_in, rho, m)
        x += L
        if name == "LOX":
            section("intertank", 0.25, [("feed line and valve", 2.0)])

    section("engine bay", 0.45, [("main valves and plumbing", 4.5)])
    engine_station = x
    parts.append(Part("engine", spec.engine_mass, engine_station, ed.length, 0.35 * r))
    x += ed.length

    n, cr, ct, span, sweep, t_fin = spec.fins
    k = spec.fin_scale
    cr, ct, span, sweep = k * cr, k * ct, k * span, k * sweep
    fins = FinSet(n, cr, ct, span, sweep, t_fin, station=x - cr)
    fin_mass = RHO_FIN * n * 0.5 * span * (cr + ct) * t_fin + 0.5
    parts.append(Part("fins and fin can", fin_mass, x - cr, cr, r + span / 3))
    parts.append(Part("aft skirt", spec.tube_mass_per_m * ed.length, engine_station, ed.length, r,
                      shell=True))

    if spec.dry_mass_scale != 1.0:
        parts = [replace(p, mass=p.mass * spec.dry_mass_scale) for p in parts]
    airframe = Airframe(d, x, Nose(Ln, r), fins)
    notes = {"tank wall LOX (mm)": wall(spec.meop_ox) * 1e3, "tank wall fuel (mm)": wall(spec.meop_fuel) * 1e3,
             "COPV volume (L)": copv_volume * 1e3, "nitrogen (kg)": m_n2}
    return Vehicle("Liquid sounding rocket", airframe, eng, parts, tanks["LOX"], tanks["fuel"],
                   spec.residual_fraction, nozzle_exit_station=x, rail_length=spec.rail_length, notes=notes)
