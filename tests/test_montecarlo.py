import sys
from pathlib import Path

import numpy as np
import pytest
from matplotlib.path import Path as Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flightsim.design import load_sized
from flightsim.montecarlo import ellipse, run_one, sample_inputs, sensitivity


def test_ellipse_contains_the_stated_fraction_of_gaussian_points():
    rng = np.random.default_rng(1)
    cov = np.array([[9.0, 4.0], [4.0, 4.0]])
    pts = rng.multivariate_normal([100.0, -50.0], cov, 50_000)
    for p in (0.90, 0.99):
        inside = Polygon(ellipse(pts, p, n=720)).contains_points(pts).mean()
        assert inside == pytest.approx(p, abs=0.005)


def test_sensitivity_recovers_known_linear_coefficients():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(5000, 3)) * [1.0, 2.0, 0.5]
    y = 3.0 * X[:, 0] - 1.0 * X[:, 1] + 0.0 * X[:, 2]
    # Standardised coefficients: b_i * sd(x_i) / sd(y)
    sd_y = np.sqrt(9 * 1 + 1 * 4)
    got = dict(sensitivity(X, y, ["a", "b", "c"]))
    assert got["a"] == pytest.approx(3 * 1.0 / sd_y, abs=0.02)
    assert got["b"] == pytest.approx(-1 * 2.0 / sd_y, abs=0.02)
    assert abs(got["c"]) < 0.02


def test_dispersed_run_is_reproducible_and_sane():
    spec, _, site = load_sized()
    x = sample_inputs(np.random.default_rng(7), 1, site)[0]
    a, b = run_one((spec, site, x)), run_one((spec, site, x))
    assert a["ok"] and a["apogee_ft"] == b["apogee_ft"]
    assert 20_000 < a["apogee_ft"] < 40_000
    assert a["min_margin_cal"] > 1.5 and a["min_flutter_margin"] > 1.5
