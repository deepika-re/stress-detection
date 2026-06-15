from app.services.emotion_stress import calculate_emotion_stress


def test_fear_distribution_maps_to_high_face_stress():
    result = calculate_emotion_stress([
        {"label": "fear", "score": 0.82},
        {"label": "neutral", "score": 0.12},
        {"label": "happy", "score": 0.06},
    ])

    assert result["dominant_emotion"] == "fear"
    assert result["stress_score"] > 0.75
    assert result["confidence"] == 0.82


def test_happy_distribution_maps_to_low_face_stress():
    result = calculate_emotion_stress([
        {"label": "happy", "score": 0.88},
        {"label": "neutral", "score": 0.08},
        {"label": "surprise", "score": 0.04},
    ])

    assert result["dominant_emotion"] == "happy"
    assert result["stress_score"] < 0.2
