"""External geometry of a single-stage rocket: tangent-ogive nose, cylindrical body, trapezoidal fins.

Stations (x) are measured from the nose tip toward the tail, in metres.
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Nose:
    length: float
    radius: float

    def profile(self, x):
        """Radius of a tangent ogive at distance x from the tip."""
        R, L = self.radius, self.length
        rho = (R * R + L * L) / (2 * R)
        return np.sqrt(rho * rho - (L - np.asarray(x)) ** 2) + R - rho

    def wetted_area(self) -> float:
        x = np.linspace(0.0, self.length, 400)
        r = self.profile(x)
        ds = np.hypot(np.diff(x), np.diff(r))
        return float(np.sum(2 * np.pi * 0.5 * (r[1:] + r[:-1]) * ds))

    @property
    def half_angle_deg(self) -> float:
        """Half-angle of the cone with the same length and base."""
        return math.degrees(math.atan(self.radius / self.length))


@dataclass(frozen=True)
class FinSet:
    n: int                # number of fins (3 or 4)
    root_chord: float
    tip_chord: float
    span: float           # root to tip (semi-span measured from the body)
    sweep: float          # axial distance from root leading edge to tip leading edge
    thickness: float
    station: float        # root leading edge, from the nose tip

    @property
    def area(self) -> float:
        """Planform area of one fin."""
        return 0.5 * self.span * (self.root_chord + self.tip_chord)

    @property
    def aspect_ratio(self) -> float:
        return 2 * self.span ** 2 / self.area

    @property
    def mac(self) -> float:
        cr, ct = self.root_chord, self.tip_chord
        return 2 / 3 * (cr + ct - cr * ct / (cr + ct))

    @property
    def mac_le_station(self) -> float:
        """Station of the mean aerodynamic chord's leading edge."""
        cr, ct = self.root_chord, self.tip_chord
        y_mac = self.span * (cr + 2 * ct) / (3 * (cr + ct))
        return self.station + self.sweep * y_mac / self.span

    @property
    def midchord_sweep(self) -> float:
        return math.atan((self.sweep + self.tip_chord / 2 - self.root_chord / 2) / self.span)

    @property
    def le_sweep(self) -> float:
        return math.atan(self.sweep / self.span)


@dataclass(frozen=True)
class Airframe:
    diameter: float
    length: float          # overall, nose tip to tail
    nose: Nose
    fins: FinSet
    roughness: float = 20e-6   # m, painted / filament-wound finish

    @property
    def ref_area(self) -> float:
        return math.pi * self.diameter ** 2 / 4

    @property
    def radius(self) -> float:
        return self.diameter / 2

    @property
    def body_wetted_area(self) -> float:
        return self.nose.wetted_area() + math.pi * self.diameter * (self.length - self.nose.length)

    @property
    def fin_wetted_area(self) -> float:
        return 2 * self.fins.n * self.fins.area
