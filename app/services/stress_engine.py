import bisect
import math
import statistics
from typing import Any

MIN_HISTORY_FOR_PERSONALIZATION = 6

# ── Population fallbacks ──────────────────────────────────────────────────────
# Not clinical thresholds; demo-friendly physiological arousal priors.
HR_REST_CENTER = 78.0
HR_REST_SCALE = 12.0
HR_MIN_PERSONAL_SCALE = 5.5

# GSR/EDA varies widely by sensor.  log1p() compresses large values so small
# changes near zero still register.
GSR_REFERENCE = 1.8
GSR_LOG_SCALE = 0.42
GSR_MIN_PERSONAL_LOG_SCALE = 0.10


def clamp(value: float, minimum: float = 0, maximum: float = 100) -> float:
    return max(minimum, min(maximum, value))


def clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def sigmoid(value: float) -> float:
    # Numerically stable implementation: always works with a positive exponent.
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    z = math.exp(value)
    return z / (1.0 + z)


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
    """Extract a finite-float series from a list of readings or dicts."""
    if not recent_readings:
        return []
    values: list[float] = []
    for reading in recent_readings:
        v = _safe_float(_read_attr(reading, field))
        if v is not None:
            values.append(v)
    return values


def _median(values: list[float], fallback: float) -> float:
    if not values:
        return fallback
    return statistics.median(values)


def _robust_scale(values: list[float], fallback_scale: float, min_scale: float) -> float:
    """
    MAD/IQR-based robust scale estimator.

    Falls back to *fallback_scale* when fewer than MIN_HISTORY_FOR_PERSONALIZATION
    samples are available.  Always returns at least *min_scale*.
    """
    if len(values) < MIN_HISTORY_FOR_PERSONALIZATION:
        return fallback_scale

    center = statistics.median(values)
    deviations = [abs(s - center) for s in values]
    mad = statistics.median(deviations)
    mad_scale = 1.4826 * mad if mad > 0 else 0.0

    iqr_scale = 0.0
    if len(values) >= 8:
        q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
        if q3 > q1:
            iqr_scale = (q3 - q1) / 1.349

    return max(mad_scale, iqr_scale, min_scale)


def _robust_z(
    value: float,
    values: list[float],
    fallback_center: float,
    fallback_scale: float,
    min_scale: float,
) -> float:
    """Z-score using median + robust scale, personalised when history is present."""
    n = len(values)
    if n >= MIN_HISTORY_FOR_PERSONALIZATION:
        center = statistics.median(values)
        scale = _robust_scale(values, fallback_scale, min_scale)
    else:
        center = fallback_center
        scale = fallback_scale
    return (value - center) / max(scale, 0.001)


def _percentile_rank(value: float, values: list[float]) -> float:
    """
    Percentile rank of *value* within *values* using bisect — O(log n).

    Returns 0.5 when history is too short to be meaningful.
    """
    if len(values) < MIN_HISTORY_FOR_PERSONALIZATION:
        return 0.5

    sorted_vals = sorted(values)
    below = bisect.bisect_left(sorted_vals, value)
    equal = bisect.bisect_right(sorted_vals, value) - below
    return clamp_unit((below + 0.5 * equal) / len(sorted_vals))


def _trend_activation(
    value: float,
    values: list[float],
    fallback_scale: float,
    min_scale: float,
) -> float:
    """
    Measures how much *value* has risen relative to the recent local baseline.

    Returns a fixed prior (0.18) when fewer than 3 samples are available,
    producing negligible volatility contribution for new streams.
    """
    if len(values) < 3:
        return 0.18

    tail = values[-5:]
    local_baseline = statistics.median(tail)
    scale = _robust_scale(values, fallback_scale, min_scale)

    delta = value - local_baseline
    return sigmoid((delta - 0.18 * scale) / max(0.28 * scale, 0.001))


# ── Reason helpers ────────────────────────────────────────────────────────────

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


# ── Main entry point ──────────────────────────────────────────────────────────

def calculate_stress(
    heart_rate: Any,
    gsr: Any,
    face_score: float | None = None,
    recent_readings: Any = None,
) -> dict:
    """
    Fuse physiological signals into a 0–100 stress score.

    Parameters
    ----------
    heart_rate:
        Current heart rate in BPM.
    gsr:
        Galvanic skin response (0–20 µS).
    face_score:
        Optional 0–1 facial-expression stress score from the emotion model.
        Pass *None* when the face pipeline is unavailable.
    recent_readings:
        Iterable of recent readings (dicts or ORM objects) with ``heart_rate``
        and ``gsr`` fields.  Used for personalisation and trend detection.

    Returns
    -------
    dict with keys: stress_score (int 0–100), stress_level, reasons, components.
    """
    heart_rate = _safe_float(heart_rate, HR_REST_CENTER)
    gsr = max(0.0, _safe_float(gsr, 0.0))  # type: ignore[arg-type]

    hr_values = _series(recent_readings, "heart_rate")
    gsr_values = _series(recent_readings, "gsr")

    log_gsr = math.log1p(gsr)
    log_gsr_ref = math.log1p(GSR_REFERENCE)
    log_gsr_values = [math.log1p(max(0.0, v)) for v in gsr_values]

    # ── 1. Heart-rate component ───────────────────────────────────────────────
    # Blend of: absolute level, personal deviation, population percentile, trend.
    hr_absolute = sigmoid((heart_rate - 82.0) / 11.0)
    hr_personal_z = _robust_z(
        heart_rate, hr_values, HR_REST_CENTER, HR_REST_SCALE, HR_MIN_PERSONAL_SCALE
    )
    hr_personal = sigmoid((hr_personal_z - 0.15) * 1.05)
    hr_percentile = _percentile_rank(heart_rate, hr_values)
    hr_trend = _trend_activation(heart_rate, hr_values, HR_REST_SCALE, HR_MIN_PERSONAL_SCALE)

    hr_component = clamp_unit(
        0.34 * hr_absolute
        + 0.40 * hr_personal
        + 0.16 * hr_percentile
        + 0.10 * hr_trend
    )

    # ── 2. GSR component ──────────────────────────────────────────────────────
    # GSR gets a slightly higher fusion weight because it reflects sympathetic
    # arousal more directly than heart rate alone.
    gsr_absolute = sigmoid((log_gsr - log_gsr_ref) / GSR_LOG_SCALE)
    gsr_personal_z = _robust_z(
        log_gsr, log_gsr_values, log_gsr_ref, GSR_LOG_SCALE, GSR_MIN_PERSONAL_LOG_SCALE
    )
    gsr_personal = sigmoid((gsr_personal_z - 0.10) * 1.15)
    gsr_percentile = _percentile_rank(log_gsr, log_gsr_values)
    gsr_trend = _trend_activation(
        log_gsr, log_gsr_values, GSR_LOG_SCALE, GSR_MIN_PERSONAL_LOG_SCALE
    )

    gsr_component = clamp_unit(
        0.28 * gsr_absolute
        + 0.44 * gsr_personal
        + 0.18 * gsr_percentile
        + 0.10 * gsr_trend
    )

    # ── 3. Face component ─────────────────────────────────────────────────────
    # Low-impact because facial emotion models are noisy in live demos.
    # A fixed 0.18 prior is used when the face pipeline is unavailable so the
    # overall score stays stable (not zeroed-out) on inference failures.
    face_component = (
        clamp_unit(float(face_score))  # type: ignore[arg-type]
        if face_score is not None
        else 0.18
    )

    # ── 4. Fusion ─────────────────────────────────────────────────────────────
    # Interaction term rises when both HR and GSR rise together.
    # Volatility captures rapid physiological state changes.
    cardio_dermal_interaction = math.sqrt(max(hr_component * gsr_component, 0.0))
    volatility = clamp_unit(0.45 * hr_trend + 0.55 * gsr_trend)
    physio_load = clamp_unit(1.0 - (1.0 - hr_component) * (1.0 - gsr_component))

    fused = (
        0.28 * hr_component
        + 0.43 * gsr_component
        + 0.12 * cardio_dermal_interaction
        + 0.10 * volatility
        + 0.07 * face_component
    )

    # Penalise HR/GSR disagreement: fused * (1 - 0.15 * |hr - gsr|).
    # Algebraically equivalent to the original two-term expression:
    #   (fused * sa) + (0.5 * fused * (1 - sa))  where sa = 1 - 0.30 * |hr - gsr|
    disagreement = abs(hr_component - gsr_component)
    fused = clamp_unit(fused * (1.0 - 0.15 * disagreement))

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

    if hr_personal_z >= 0.85:
        reasons.append("heart_rate_above_personal_baseline")
    if gsr_personal_z >= 0.85:
        reasons.append("gsr_above_personal_baseline")
    if hr_percentile >= 0.80:
        reasons.append("heart_rate_high_for_recent_window")
    if gsr_percentile >= 0.80:
        reasons.append("gsr_high_for_recent_window")
    if volatility >= 0.68:
        reasons.append("rapid_signal_rise")
    if cardio_dermal_interaction >= 0.62:
        reasons.append("heart_gsr_coupled_response")
    if gsr_component >= 0.68:
        reasons.append("gsr_dominant_response")
    if hr_component >= 0.68:
        reasons.append("heart_rate_dominant_response")

    return {
        "stress_score": score,
        "stress_level": level,
        "reasons": reasons,
        "components": {
            "heart_rate": round(hr_component, 4),
            "gsr": round(gsr_component, 4),
            "face": round(face_component, 4) if face_score is not None else None,
            "physio_load": round(physio_load, 4),
            "interaction": round(cardio_dermal_interaction, 4),
            "volatility": round(volatility, 4),
            "hr_personal_z": round(hr_personal_z, 4),
            "gsr_personal_z": round(gsr_personal_z, 4),
            "hr_percentile": round(hr_percentile, 4),
            "gsr_percentile": round(gsr_percentile, 4),
        },
    }