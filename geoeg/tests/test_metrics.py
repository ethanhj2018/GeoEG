import math

import pytest

from geoeg.eval.metrics import THRESHOLDS_KM, accuracy_at, haversine_km, summarize


def test_haversine_zero():
    assert haversine_km(48.8566, 2.3522, 48.8566, 2.3522) == 0.0


def test_haversine_known_pairs():
    # Paris – London ≈ 343.5 km ; Beijing – Shanghai ≈ 1067 km
    assert haversine_km(48.8566, 2.3522, 51.5074, -0.1278) == pytest.approx(343.5, abs=2)
    assert haversine_km(39.9042, 116.4074, 31.2304, 121.4737) == pytest.approx(1067, abs=5)


def test_haversine_antipodal_and_symmetry():
    assert haversine_km(0, 0, 0, 180) == pytest.approx(math.pi * 6371.0088, rel=1e-6)
    assert haversine_km(10, 20, -30, 40) == pytest.approx(haversine_km(-30, 40, 10, 20))


def test_accuracy_thresholds():
    acc = accuracy_at([0.5, 10, 150, 700, 2000, 5000, None])
    assert set(acc) == {f"acc@{t}km" for t in THRESHOLDS_KM}
    assert acc["acc@1km"] == pytest.approx(1 / 7)
    assert acc["acc@25km"] == pytest.approx(2 / 7)
    assert acc["acc@200km"] == pytest.approx(3 / 7)
    assert acc["acc@750km"] == pytest.approx(4 / 7)
    assert acc["acc@2500km"] == pytest.approx(5 / 7)


def test_accuracy_empty_and_summary():
    assert accuracy_at([])["acc@1km"] == 0.0
    s = summarize([1, 3, 2])
    assert s["n"] == 3 and s["median_km"] == 2
