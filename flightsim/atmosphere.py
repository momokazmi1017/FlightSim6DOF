"""U.S. Standard Atmosphere 1976, 0-86 km geometric altitude.

Seven layers of constant temperature lapse rate in geopotential altitude H.
Within a layer with base (Hb, Tb, pb) and lapse rate L:
    T = Tb + L (H - Hb)
    p = pb (T / Tb) ^ (-g0 / (R L))           if L != 0
    p = pb exp(-g0 (H - Hb) / (R Tb))         if L == 0
Density from the ideal gas law, speed of sound a = sqrt(gamma R T), and
dynamic viscosity from Sutherland's law.
"""
import math
from dataclasses import dataclass

G0 = 9.80665               # m/s^2
R_AIR = 287.05287          # J/kg-K  (R* / M0 of the 1976 standard)
GAMMA = 1.4
R_EARTH = 6_356_766.0      # m, effective Earth radius for geopotential altitude

# Layer base geopotential altitude (m), base temperature (K), lapse rate (K/m)
_LAYERS = [
    (0.0, 288.15, -0.0065),
    (11_000.0, 216.65, 0.0),
    (20_000.0, 216.65, 0.0010),
    (32_000.0, 228.65, 0.0028),
    (47_000.0, 270.65, 0.0),
    (51_000.0, 270.65, -0.0028),
    (71_000.0, 214.65, -0.0020),
]
_H_TOP = 84_852.0          # geopotential altitude of 86 km geometric


def _base_pressures():
    p = [101_325.0]
    for (hb, tb, lapse), (h_next, _, _) in zip(_LAYERS, _LAYERS[1:]):
        p.append(_layer_pressure(p[-1], tb, lapse, h_next - hb))
    return p


def _layer_pressure(pb, tb, lapse, dh):
    if lapse == 0.0:
        return pb * math.exp(-G0 * dh / (R_AIR * tb))
    return pb * ((tb + lapse * dh) / tb) ** (-G0 / (R_AIR * lapse))


_P_BASE = _base_pressures()


@dataclass(frozen=True)
class AtmosphereState:
    T: float        # K
    p: float        # Pa
    rho: float      # kg/m^3
    a: float        # m/s
    mu: float       # Pa-s


def atmosphere(z: float) -> AtmosphereState:
    """Standard atmosphere at geometric altitude z (m above mean sea level).
    Clamped to 0-86 km."""
    z = min(max(z, 0.0), 86_000.0)
    h = R_EARTH * z / (R_EARTH + z)
    i = len(_LAYERS) - 1
    while i > 0 and h < _LAYERS[i][0]:
        i -= 1
    hb, tb, lapse = _LAYERS[i]
    T = tb + lapse * (h - hb)
    p = _layer_pressure(_P_BASE[i], tb, lapse, h - hb)
    rho = p / (R_AIR * T)
    return AtmosphereState(T, p, rho, math.sqrt(GAMMA * R_AIR * T),
                           1.458e-6 * T ** 1.5 / (T + 110.4))


def gravity(z: float) -> float:
    """Gravitational acceleration (m/s^2) at geometric altitude z, inverse-square law."""
    return G0 * (R_EARTH / (R_EARTH + z)) ** 2
