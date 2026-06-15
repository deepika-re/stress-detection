from app.services.stress_engine import calculate_stress


def test_stress_engine_low_without_face_score():
    result = calculate_stress(72, 0.4)

    assert result["stress_level"] == "LOW"
    assert 0 <= result["stress_score"] <= 100
    assert "face_score_unavailable" in result["reasons"]


def test_stress_engine_high_with_elevated_signals():
    result = calculate_stress(126, 7.2, 0.8)

    assert result["stress_level"] == "HIGH"
    assert result["stress_score"] >= 65
    assert "heart_rate_very_high" in result["reasons"]
    assert "gsr_very_high" in result["reasons"]


def test_hr_gsr_dominate_even_when_face_score_is_low():
    result = calculate_stress(105, 3.5, 0.0)

    assert result["stress_level"] == "HIGH"
    assert result["components"]["heart_rate"] > 0.8
    assert result["components"]["gsr"] > 0.8


def test_face_score_alone_does_not_force_high_stress():
    result = calculate_stress(72, 0.4, 1.0)

    assert result["stress_level"] == "LOW"
    assert result["stress_score"] < 25
