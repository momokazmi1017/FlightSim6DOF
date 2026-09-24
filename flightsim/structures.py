"""Fin flutter check (NACA TN 4197, Martin 1958, in the form used in amateur
rocketry, e.g. Apogee Peak of Flight #291):

    V_f = a * sqrt( G / [ 1.337 AR^3 p (lambda + 1) / (2 (AR + 2) (t/c)^3) ] )

a: local speed of sound, p: local static pressure, G: fin material shear
modulus, AR = span^2 / panel area, lambda = tip chord / root chord, t/c =
thickness / root chord. A design should keep flutter margin V_f / V >= 1.5
throughout the flight: the formula is an estimate for flat plates, and
bonding, layup and edge shaping all move the real boundary.
"""
import math

from .atmosphere import atmosphere
from .geometry import FinSet

G_G10 = 2.93e9      # Pa (425,000 psi), fibreglass laminate shear modulus


def flutter_speed(fins: FinSet, z_msl: float, shear_modulus: float = G_G10) -> float:
    atm = atmosphere(z_msl)
    ar = fins.span ** 2 / fins.area
    lam = fins.tip_chord / fins.root_chord
    tc = fins.thickness / fins.root_chord
    denom = 1.337 * ar ** 3 * atm.p * (lam + 1) / (2 * (ar + 2) * tc ** 3)
    return atm.a * math.sqrt(shear_modulus / denom)
