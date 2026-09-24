import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flightsim import aero
from flightsim.engine import Engine, EngineData
from flightsim.vehicle import VehicleSpec, build_vehicle

ED = EngineData.load()


def test_engine_thrust_matches_design_points():
    eng = Engine(ED, burn_time=6.0)
    assert eng.thrust(3.0, 101_325.0) == pytest.approx(5000.0, rel=1e-6)   # sea-level design point
    assert eng.thrust(3.0, 0.0) == pytest.approx(ED.thrust_vac, rel=1e-12)
    assert eng.thrust(eng.t_end + 0.1, 0.0) == 0.0


def test_engine_consumes_exactly_the_usable_propellant():
    eng = Engine(ED, burn_time=6.0)
    ox, fuel = eng.consumed(eng.t_end)
    assert ox == pytest.approx(6.0 * ED.mdot_ox, rel=1e-12)
    assert fuel == pytest.approx(6.0 * ED.mdot_fuel, rel=1e-12)
    # consumed() is the integral of the flow rate
    ts = [i * eng.t_end / 2000 for i in range(2001)]
    integral = sum(0.5 * (eng.mdot_ox(a) + eng.mdot_ox(b)) * (b - a) for a, b in zip(ts, ts[1:]))
    assert integral == pytest.approx(ox, rel=1e-4)


def test_mass_properties_over_the_burn():
    v = build_vehicle(VehicleSpec(burn_time=6.0))
    m0 = v.mass_properties(0.0)
    m1 = v.mass_properties(v.engine.t_end)
    assert m0.mass == pytest.approx(v.dry_mass + v.propellant_mass, rel=1e-9)
    assert m1.mass == pytest.approx(v.dry_mass + v.residual_fraction * v.propellant_mass, rel=1e-6)
    assert m1.I_transverse < m0.I_transverse
    # All the parts lie inside the airframe
    assert all(0 <= p.station and p.station + p.length <= v.airframe.length + 1e-9 for p in v.parts)


def test_fin_lift_reduces_to_barrowman_at_low_mach():
    v = build_vehicle(VehicleSpec(burn_time=6.0))
    af, f = v.airframe, v.airframe.fins
    # Barrowman's closed form (incompressible)
    lm = f.span / math.cos(f.midchord_sweep)
    barrowman = 4 * f.n * (f.span / af.diameter) ** 2 / (
        1 + math.sqrt(1 + (2 * lm / (f.root_chord + f.tip_chord)) ** 2))
    k = 1 + af.radius / (f.span + af.radius)
    fins = [c for c in aero.components(af, 1e-4) if c.name == "fins"][0]
    assert fins.cn_alpha == pytest.approx(barrowman * k, rel=1e-6)
    # Barrowman's fin centre of pressure: quarter chord of the mean aerodynamic chord
    xr, cr, ct = f.sweep, f.root_chord, f.tip_chord
    x_barrowman = f.station + xr * (cr + 2 * ct) / (3 * (cr + ct)) + (cr + ct - cr * ct / (cr + ct)) / 6
    assert fins.x_cp == pytest.approx(x_barrowman, rel=1e-9)


def test_skin_friction_matches_flat_plate_value():
    # Turbulent flat plate at Re = 1e7: Cf ~ 0.0029
    assert aero.skin_friction(1e7, 0.0, 1.0, 1e-9) == pytest.approx(0.0029, rel=0.03)
