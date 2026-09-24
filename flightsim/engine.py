"""Liquid engine model: thrust and propellant flow versus time and ambient pressure.

The engine runs at a fixed operating point (pressure-fed, fixed chamber
pressure), with a linear start-up ramp and tail-off. Thrust depends on the
ambient pressure through the nozzle pressure term:
    F = s(t) F_vac - p_a A_exit,
where s(t) is the throttle profile (0 -> 1 -> 0).
"""
import json
from dataclasses import dataclass
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass(frozen=True)
class EngineData:
    name: str
    thrust_vac: float       # N
    exit_area: float        # m^2
    mdot_ox: float          # kg/s at full thrust
    mdot_fuel: float        # kg/s at full thrust
    exit_diameter: float    # m
    length: float           # m

    @property
    def mdot(self) -> float:
        return self.mdot_ox + self.mdot_fuel

    @classmethod
    def load(cls, name: str = "e5_75") -> "EngineData":
        d = json.loads((ENGINE_DIR / f"{name}_engine.json").read_text())
        return cls(d["name"], d["thrust_vac_N"], d["exit_area_m2"], d["mdot_ox_kg_s"],
                   d["mdot_fuel_kg_s"], d["exit_diameter_m"], d["length_m"])


@dataclass(frozen=True)
class Engine:
    data: EngineData
    burn_time: float           # equivalent full-thrust burn time (s): usable propellant / mdot
    t_startup: float = 0.3     # s, linear ramp to full thrust
    t_tailoff: float = 0.2     # s, linear ramp to zero

    @property
    def t_cutoff(self) -> float:
        """Start of the tail-off, chosen so the total flow equals burn_time x mdot."""
        return self.burn_time + 0.5 * self.t_startup - 0.5 * self.t_tailoff

    @property
    def t_end(self) -> float:
        return self.t_cutoff + self.t_tailoff

    def throttle(self, t: float) -> float:
        if t <= 0 or t >= self.t_end:
            return 0.0
        if t < self.t_startup:
            return t / self.t_startup
        if t <= self.t_cutoff:
            return 1.0
        return (self.t_end - t) / self.t_tailoff

    def thrust(self, t: float, p_ambient: float) -> float:
        s = self.throttle(t)
        if s == 0.0:
            return 0.0
        return max(s * self.data.thrust_vac - p_ambient * self.data.exit_area, 0.0)

    def mdot_ox(self, t: float) -> float:
        return self.throttle(t) * self.data.mdot_ox

    def mdot_fuel(self, t: float) -> float:
        return self.throttle(t) * self.data.mdot_fuel

    def consumed(self, t: float) -> tuple[float, float]:
        """Oxidiser and fuel mass consumed by time t (kg): integral of the throttle profile."""
        tu, tc, te = self.t_startup, self.t_cutoff, self.t_end
        t = min(max(t, 0.0), te)
        if t <= tu:
            area = 0.5 * t * t / tu
        elif t <= tc:
            area = 0.5 * tu + (t - tu)
        else:
            area = 0.5 * tu + (tc - tu) + (te - tc) * 0.5 * (1 - ((te - t) / (te - tc)) ** 2)
        return area * self.data.mdot_ox, area * self.data.mdot_fuel
