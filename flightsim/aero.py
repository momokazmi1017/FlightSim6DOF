"""Rocket aerodynamics: normal-force slope, centre of pressure and zero-lift drag.

Normal force (per radian, referenced to the body cross-section):
- Nose: CN_alpha = 2 at all Mach numbers (slender-body theory), acting at
  0.466 L for a tangent ogive (Barrowman).
- Fins: Diederich's lifting-surface formula (Barrowman's method with
  compressibility) below Mach 0.8,
      CN_alpha,panel = 2 pi AR (S/A_ref) / (2 + sqrt(4 + (beta AR / cos Gamma_c)^2)),
  Ackeret supersonic thin-wing theory with the finite-span tip-loss
  correction above Mach 1.2 (or higher for low-aspect-ratio fins, so AR beta >= 1),
      CN_alpha,panel = 4 / beta (1 - 1 / (2 AR beta)) (S/A_ref),
  and a linear blend between. n fins act like n/2 panels in the plane of the
  angle of attack. The body-fin interference factor is K = 1 + R / (s + R).
  The fin centre of pressure moves from the quarter to the half mean
  aerodynamic chord between Mach 0.5 and 2.

Zero-lift drag is a component buildup, following the forms used by OpenRocket
(Niskanen 2009):
- skin friction, with a roughness limit and compressibility correction,
  over the body and fins (with fineness and thickness form factors);
- base drag (reduced to the annulus around the nozzle while the engine burns);
- fin leading-edge pressure drag and blunt trailing-edge base drag;
- nose wave drag above Mach 0.8, from Stoney's cone correlation
  (0.083 + 0.096 / M^2)(theta / 10)^1.69 at the equivalent cone half-angle,
  scaled by 0.6 for an ogive (an approximation; see README).
"""
import math
from dataclasses import dataclass

from .geometry import Airframe

M_SUB, M_SUP = 0.8, 1.2         # transonic blending range for fin lift
OGIVE_CP = 0.466                 # tangent-ogive nose centre of pressure / length
OGIVE_WAVE_FACTOR = 0.6          # ogive wave drag relative to the equivalent cone
PROTUBERANCE_FACTOR = 1.10       # rail buttons, plumbing fairings, fasteners


def _beta(M):
    return math.sqrt(abs(1 - M * M))


def _blend(M, lo, hi):
    """0 at M <= lo, 1 at M >= hi, linear between."""
    return min(max((M - lo) / (hi - lo), 0.0), 1.0)


@dataclass(frozen=True)
class Component:
    name: str
    cn_alpha: float     # per radian
    x_cp: float         # station from the nose tip, m


def supersonic_onset(af: Airframe) -> float:
    """Mach number above which the supersonic fin theory is used: 1.2, or higher
    for low-aspect-ratio fins, so that AR * beta >= 1 (tip Mach cones do not
    overlap the whole panel)."""
    ar = af.fins.aspect_ratio
    return max(M_SUP, math.sqrt(1 + 1 / ar ** 2))


def fin_cn_alpha_panel(af: Airframe, M: float) -> float:
    f = af.fins
    s_ratio = f.area / af.ref_area
    ar = f.aspect_ratio
    m_sup = supersonic_onset(af)

    def subsonic(m):
        b = _beta(min(m, M_SUB))
        return 2 * math.pi * ar * s_ratio / (
            2 + math.sqrt(4 + (b * ar / math.cos(f.midchord_sweep)) ** 2))

    def supersonic(m):
        # Ackeret thin-wing lift with the tip-loss correction for finite span:
        # inside the Mach cones from the tips the lift is reduced.
        b = _beta(max(m, m_sup))
        return 4 / b * (1 - 1 / (2 * ar * b)) * s_ratio

    w = _blend(M, M_SUB, m_sup)
    return (1 - w) * subsonic(M) + w * supersonic(M)


def components(af: Airframe, M: float) -> list[Component]:
    f = af.fins
    k_interference = 1 + af.radius / (f.span + af.radius)
    fins_cn = f.n / 2 * fin_cn_alpha_panel(af, M) * k_interference
    frac = 0.25 + 0.25 * _blend(M, 0.5, 2.0)
    fins_cp = f.mac_le_station + frac * f.mac
    return [
        Component("nose", 2.0, OGIVE_CP * af.nose.length),
        Component("fins", fins_cn, fins_cp),
    ]


def normal_force(af: Airframe, M: float) -> tuple[float, float]:
    """(CN_alpha per radian, centre-of-pressure station in m)."""
    comps = components(af, M)
    cn = sum(c.cn_alpha for c in comps)
    return cn, sum(c.cn_alpha * c.x_cp for c in comps) / cn


# --- Drag ---------------------------------------------------------------------

def skin_friction(Re: float, M: float, length: float, roughness: float) -> float:
    """Turbulent flat-plate skin friction coefficient with a roughness limit and
    compressibility correction."""
    Re = max(Re, 1e4)
    cf = 1 / (1.50 * math.log(Re) - 5.6) ** 2
    cf_rough = 0.032 * (roughness / length) ** 0.2
    cf = max(cf, cf_rough)
    if M < 1:
        return cf * (1 - 0.1 * M * M)
    return cf / (1 + 0.15 * M * M) ** 0.58


def base_drag(M: float) -> float:
    return 0.12 + 0.13 * M * M if M < 1 else 0.25 / M


def fin_leading_edge_drag(M: float) -> float:
    """Pressure drag coefficient of a rounded leading edge (per unit frontal area)."""
    if M < 0.9:
        return (1 - M * M) ** -0.417 - 1
    if M < 1:
        return 1 - 1.785 * (M - 0.9)
    return 1.214 - 0.502 / M ** 2 + 0.1095 / M ** 4


def nose_wave_drag(af: Airframe, M: float) -> float:
    if M <= 0.8:
        return 0.0
    theta = af.nose.half_angle_deg
    cone = (0.083 + 0.096 / max(M, 1.0) ** 2) * (theta / 10) ** 1.69
    return OGIVE_WAVE_FACTOR * cone * _blend(M, 0.8, 1.0)


@dataclass(frozen=True)
class DragBreakdown:
    friction: float
    base: float
    fins_pressure: float
    nose_wave: float

    @property
    def total(self) -> float:
        return self.friction + self.base + self.fins_pressure + self.nose_wave


def drag(af: Airframe, M: float, rho: float, V: float, mu: float,
         burning: bool = False, exit_area: float = 0.0) -> DragBreakdown:
    """Zero-lift axial drag coefficient, referenced to the body cross-section."""
    A = af.ref_area
    f = af.fins
    Re = rho * max(V, 1e-3) * af.length / mu
    cf_body = skin_friction(Re, M, af.length, af.roughness)
    cf_fin = skin_friction(Re * f.mac / af.length, M, f.mac, af.roughness)
    fineness = af.length / af.diameter
    friction = PROTUBERANCE_FACTOR * (
        cf_body * (1 + 1 / (2 * fineness)) * af.body_wetted_area
        + cf_fin * (1 + 2 * f.thickness / f.mac) * af.fin_wetted_area) / A

    base_area = max(A - exit_area, 0.0) if burning else A
    base = base_drag(M) * base_area / A

    fin_frontal = f.n * f.thickness * f.span
    fins_pressure = (fin_leading_edge_drag(M) * math.cos(f.le_sweep) ** 2 + base_drag(M)) * fin_frontal / A

    return DragBreakdown(friction, base, fins_pressure, nose_wave_drag(af, M))
