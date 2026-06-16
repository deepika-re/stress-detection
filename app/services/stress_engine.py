import math
import statistics
from typing import Any

MIN_HISTORY_FOR_PERSONALIZATION = 6

HR_REST_CENTER = 78.0
HR_REST_SCALE = 12.0
HR_MIN_PERSONAL_SCALE = 5.5

GSR_REFERENCE = 1.8
GSR_LOG_SCALE = 0.42
GSR_MIN_PERSONAL_LOG_SCALE = 0.10


def clamp(value: float, minimum: float = 0, maximum: float = 100) -> float:
    return max(minimum, min(maximum, value))


def clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _read_attr(reading: Any, field: str) -> Any:
    if isinstance(reading, dict):
        return reading.get(field)
    return getattr(reading, field, None)


def _series(recent_readings: Any, field: str) -> list[float]:
    if not recent_readings:
        return []
    values = []
    for reading in recent_readings:
        v = _safe_float(_read_attr(reading, field))
        if v is not None:
            values.append(v)
    return values


def _personal_scale(values: list[float], fallback_scale: float, min_scale: float) -> float:
    """MAD-based robust scale, only used once enough history exists."""
    if len(values) < MIN_HISTORY_FOR_PERSONALIZATION:
        return fallback_scale
    center = statistics.median(values)
    mad = statistics.median(abs(v - center) for v in values)
    return max(1.4826 * mad, min_scale) if mad > 0 else max(fallback_scale, min_scale)


def _blended_z(
    value: float,
    values: list[float],
    fallback_center: float,
    fallback_scale: float,
    min_scale: float,
) -> float:
    """
    Z-score against personal baseline, smoothly blended with the population
    baseline as history grows (no hard cliff at MIN_HISTORY_FOR_PERSONALIZATION).
    """
    n = len(values)
    weight = clamp_unit(n / MIN_HISTORY_FOR_PERSONALIZATION)  # 0 -> 1
    if n > 0:
        personal_center = statistics.median(values)
        personal_scale = _personal_scale(values, fallback_scale, min_scale)
    else:
        personal_center, personal_scale = fallback_center, fallback_scale

    center = weight * personal_center + (1 - weight) * fallback_center
    scale = weight * personal_scale + (1 - weight) * fallback_scale
    return (value - center) / max(scale, 0.001)


def _heart_rate_reasons(heart_rate: float) -> str:
    if heart_rate >= 120:
        return "heart_rate_very_high"
    if heart_rate >= 100:
        return "heart_rate_high"
    if heart_rate >= 86:
        return "heart_rate_elevated"
    if heart_rate <= 55:
        return "heart_rate_low"
    return "heart_rate_normal"


def _gsr_reasons(gsr: float) -> str:
    if gsr >= 6:
        return "gsr_very_high"
    if gsr >= 3:
        return "gsr_high"
    if gsr >= 1.4:
        return "gsr_elevated"
    return "gsr_low_or_normal"


def _face_reasons(face_score: float | None) -> str:
    if face_score is None:
        return "face_score_unavailable"
    if face_score >= 0.75:
        return "face_score_very_high"
    if face_score >= 0.5:
        return "face_score_high"
    if face_score >= 0.3:
        return "face_score_elevated"
    return "face_score_low"


def calculate_stress(
    heart_rate: Any,
    gsr: Any,
    face_score: float | None = None,
    recent_readings: Any = None,
) -> dict:
    """
    Fuse physiological signals into a 0-100 stress score.

    Each signal contributes a 0-1 "z-to-unit" component (how far above personal/
    population baseline it sits), and the three components are combined with a
    straight weighted average. No compounding interaction/volatility/disagreement
    layers, so high inputs actually reach a high score instead of being damped
    out by several sigmoids stacked on top of each other.
    """
    heart_rate = _safe_float(heart_rate, HR_REST_CENTER)
    gsr = max(0.0, _safe_float(gsr, 0.0))

    hr_values = _series(recent_readings, "heart_rate")
    gsr_values = _series(recent_readings, "gsr")

    log_gsr = math.log1p(gsr)
    log_gsr_ref = math.log1p(GSR_REFERENCE)
    log_gsr_values = [math.log1p(max(0.0, v)) for v in gsr_values]

    # z >= 0 roughly means "at or above baseline arousal"; scale so that
    # z=0 -> 0.5, z=+2 (about 2 SDs above baseline) -> ~1.0, z<<0 -> 0.
    hr_z = _blended_z(heart_rate, hr_values, HR_REST_CENTER, HR_REST_SCALE, HR_MIN_PERSONAL_SCALE)
    gsr_z = _blended_z(log_gsr, log_gsr_values, log_gsr_ref, GSR_LOG_SCALE, GSR_MIN_PERSONAL_LOG_SCALE)

    hr_component = clamp_unit(0.5 + hr_z / 4.0)
    gsr_component = clamp_unit(0.5 + gsr_z / 4.0)
    face_component = clamp_unit(float(face_score)) if face_score is not None else 0.18

    fused = clamp_unit(0.40 * hr_component + 0.45 * gsr_component + 0.15 * face_component)
    score = int(round(clamp(100.0 * fused)))

    if score >= 64:
        level = "HIGH"
    elif score >= 34:
        level = "MEDIUM"
    else:
        level = "LOW"

    reasons: list[str] = [
        _heart_rate_reasons(heart_rate),
        _gsr_reasons(gsr),
        _face_reasons(face_score),
    ]
    if hr_z >= 0.85:
        reasons.append("heart_rate_above_personal_baseline")
    if gsr_z >= 0.85:
        reasons.append("gsr_above_personal_baseline")
    if hr_component >= 0.68:
        reasons.append("heart_rate_dominant_response")
    if gsr_component >= 0.68:
        reasons.append("gsr_dominant_response")

    return {
        "stress_score": score,
        "stress_level": level,
        "reasons": reasons,
        "components": {
            "heart_rate": round(hr_component, 4),
            "gsr": round(gsr_component, 4),
            "face": round(face_component, 4) if face_score is not None else None,
            "hr_z": round(hr_z, 4),
            "gsr_z": round(gsr_z, 4),
        },
    }
