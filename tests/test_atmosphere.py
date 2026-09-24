import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flightsim.atmosphere import atmosphere

# U.S. Standard Atmosphere 1976 tabulated values (geometric altitude)
TABLE = [
    # z [m], T [K], p [Pa], rho [kg/m^3]
    (0, 288.150, 101_325.0, 1.2250),
    (5_000, 255.676, 54_048.0, 0.73643),
    (10_000, 223.252, 26_500.0, 0.41351),
    (20_000, 216.650, 5_529.3, 0.088910),
    (30_000, 226.509, 1_197.0, 0.018410),
    (50_000, 270.650, 79.779, 0.0010269),
]


@pytest.mark.parametrize("z, T, p, rho", TABLE)
def test_matches_standard_table(z, T, p, rho):
    s = atmosphere(z)
    assert s.T == pytest.approx(T, abs=0.01)
    assert s.p == pytest.approx(p, rel=1e-3)
    assert s.rho == pytest.approx(rho, rel=1e-3)


def test_sea_level_speed_of_sound_and_viscosity():
    s = atmosphere(0)
    assert s.a == pytest.approx(340.294, rel=1e-4)
    assert s.mu == pytest.approx(1.7894e-5, rel=1e-3)
