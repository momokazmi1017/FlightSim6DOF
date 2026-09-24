"""Validation of the 6-DOF simulation against analytic results."""
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flightsim import aero
from flightsim.atmosphere import G0, R_EARTH, atmosphere
from flightsim.design import load_sized
from flightsim.rotations import quat_to_dcm
from flightsim.sim import SimConfig, Wind, _state_derivative, body_loads, simulate
from flightsim.sizing import point_mass

SPEC, VEHICLE, SITE = load_sized()


@pytest.fixture(scope="module")
def calm():
    return simulate(VEHICLE, SimConfig(site=SITE))


def test_torque_free_rigid_body():
    # After burnout in vacuum there are no moments: the angular momentum vector
    # must stay fixed in inertial space and rotational energy is conserved. For
    # an axisymmetric body the roll rate is constant and the lateral rate
    # rotates in the body frame at (I_x - I_t) / I_t * p.
    cfg = SimConfig(site=SITE, vacuum=True)
    t0 = VEHICLE.engine.t_end + 1.0
    mp = VEHICLE.mass_properties(t0)
    I = np.array([mp.I_axial, mp.I_transverse, mp.I_transverse])
    w0 = np.array([6.0, 0.8, 0.0])
    y0 = np.concatenate([[0, 0, 5000.0], [0, 0, 100.0], [1.0, 0, 0, 0], w0])
    sol = solve_ivp(lambda t, y: _state_derivative(VEHICLE, cfg, t, y), (t0, t0 + 5), y0,
                    method="DOP853", rtol=1e-10, atol=1e-12, dense_output=True)
    H0 = quat_to_dcm(y0[6:10]) @ (I * w0)
    for t in np.linspace(t0, t0 + 5, 11):
        y = sol.sol(t)
        q, w = y[6:10] / np.linalg.norm(y[6:10]), y[10:13]
        assert quat_to_dcm(q) @ (I * w) == pytest.approx(H0, abs=1e-6 * np.linalg.norm(H0))
        assert 0.5 * w @ (I * w) == pytest.approx(0.5 * w0 @ (I * w0), rel=1e-8)
        assert w[0] == pytest.approx(w0[0], rel=1e-10)
        lam = (mp.I_axial - mp.I_transverse) / mp.I_transverse * w0[0]
        assert math.atan2(w[2], w[1]) == pytest.approx(math.remainder(lam * (t - t0), 2 * math.pi), abs=1e-6)


def test_vacuum_launch_matches_rocket_equation_and_energy():
    # Vertical launch in vacuum: burnout speed from the rocket equation minus the
    # gravity loss, and apogee from conservation of energy in inverse-square gravity.
    site = replace(SITE, rail_elevation_deg=90.0)
    res = simulate(VEHICLE, SimConfig(site=site, vacuum=True, dt_out=0.001))
    eng = VEHICLE.engine
    ve = eng.data.thrust_vac / eng.data.mdot
    moving = res.v[:, 2] > 0
    t_lift = res.t[np.argmax(moving)] - 0.001
    burn = (res.t >= t_lift) & (res.t <= eng.t_end)
    g = G0 * (R_EARTH / (R_EARTH + SITE.altitude + res.r[burn, 2])) ** 2
    dv = ve * math.log(VEHICLE.mass_properties(t_lift).mass / VEHICLE.mass_properties(eng.t_end).mass) \
        - np.trapezoid(g, res.t[burn])
    v_bo = np.interp(eng.t_end, res.t, res.v[:, 2])
    assert v_bo == pytest.approx(dv, rel=2e-3)

    mu = G0 * R_EARTH ** 2
    r_bo = R_EARTH + SITE.altitude + np.interp(eng.t_end, res.t, res.r[:, 2])
    r_ap = 1 / (1 / r_bo - v_bo ** 2 / (2 * mu))
    assert SITE.altitude + res.apogee_agl + R_EARTH == pytest.approx(r_ap, rel=1e-5)


def test_six_dof_agrees_with_point_mass_in_calm_air(calm):
    # With no wind the rocket flies at near-zero angle of attack, so the 6-DOF
    # and the point-mass sizing model should agree closely.
    assert calm.apogee_agl == pytest.approx(point_mass(VEHICLE, SITE).apogee_agl, rel=0.01)
    assert np.degrees(np.nanmax(calm.alpha[calm.phase == 1])) < 1.0


def test_aerodynamic_stiffness_and_damping_match_linear_theory():
    # Pitching moment per radian of angle of attack, and per unit pitch rate,
    # from the component-by-component loads, against the closed-form derivatives.
    cfg = SimConfig(site=SITE)
    t, z, V = 20.0, SITE.altitude + 5000.0, 250.0
    atm = atmosphere(z)
    mp = VEHICLE.mass_properties(t)
    M = V / atm.a
    qA = 0.5 * atm.rho * V * V * VEHICLE.airframe.ref_area
    comps = aero.components(VEHICLE.airframe, M)
    stiffness = sum(c.cn_alpha * (c.x_cp - mp.x_cg) for c in comps) * qA          # per rad
    damping = sum(c.cn_alpha * (c.x_cp - mp.x_cg) ** 2 for c in comps) * qA / V   # per rad/s

    a = 1e-3
    Lp = body_loads(VEHICLE, cfg, t, z, V * np.array([math.cos(a), math.sin(a), 0]), np.zeros(3))
    assert Lp.moment_b[2] / math.sin(a) == pytest.approx(stiffness, rel=1e-6)
    Lr = body_loads(VEHICLE, cfg, t, z, np.array([V, 0, 0]), np.array([0, 0, 0.01]))
    assert Lr.moment_b[2] / 0.01 == pytest.approx(-damping, rel=1e-6)


def test_rocket_weathercocks_into_the_wind(calm):
    # Wind from the west: the rocket turns into it, so its apogee moves west (upwind).
    windy = simulate(VEHICLE, SimConfig(site=SITE, wind=Wind(speed_10m=6.0, from_deg=270.0)))
    x_apogee = lambda res: res.r[np.argmax(res.r[:, 2]), 0]
    assert x_apogee(windy) < x_apogee(calm) - 100.0
    # ... and, drifting under the parachutes, lands downwind (east)
    assert windy.r[-1, 0] > 500.0


def test_parachute_descent_reaches_terminal_velocity(calm):
    m = VEHICLE.mass_properties(calm.events["apogee"]).mass
    rho = atmosphere(SITE.altitude).rho
    v_terminal = math.sqrt(2 * m * G0 / (rho * SimConfig().recovery.main_cda))
    assert abs(calm.v[-1, 2]) == pytest.approx(v_terminal, rel=0.01)
