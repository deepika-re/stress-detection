import math
from typing import Any

# ── Heart rate bands (BPM) ────────────────────────────────────────────────────
HR_LOW_MAX = 60.0
HR_NORMAL_MAX = 72.0
HR_MEDIUM_MAX = 95.0
# above HR_MEDIUM_MAX -> high

# ── GSR bands (raw sensor units) ──────────────────────────────────────────────
GSR_NORMAL_MAX = 350.0
GSR_MEDIUM_MAX = 800.0
GSR_FORCE_HIGH_THRESHOLD = 800.0
# above GSR_MEDIUM_MAX -> high; above GSR_FORCE_HIGH_THRESHOLD -> hard override

# ── Face score bands (0-1, from emotion model) ───────────────────────────────
FACE_NORMAL_MAX = 0.30
FACE_MEDIUM_MAX = 0.60
# above FACE_MEDIUM_MAX -> high

# ── Fusion weights: GSR weighted highest, then HR, then face ────────────────
WEIGHT_HR = 0.35
WEIGHT_GSR = 0.45
WEIGHT_FACE = 0.20

# Fallback used when face_score is unavailable, kept low/neutral so the
# fused score doesn't swing just because the face pipeline dropped out.
FACE_UNAVAILABLE_COMPONENT = 0.18


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


def _band_component(value: float, normal_max: float, medium_max: float) -> float:
    """
    Map a raw value to a 0-1 component using three linear bands:
      value <= normal_max              -> 0.0 .. 0.33  (low)
      normal_max < value <= medium_max -> 0.33 .. 0.66  (medium)
      value > medium_max               -> 0.66 .. 1.0   (high, ramps over a
                                           span equal to the medium band width
                                           then saturates at 1.0)
    """
    if value <= normal_max:
        if normal_max <= 0:
            return 0.0
        return clamp_unit(0.33 * (value / normal_max)) if value > 0 else 0.0

    if value <= medium_max:
        span = medium_max - normal_max
        if span <= 0:
            return 0.66
        return 0.33 + 0.33 * clamp_unit((value - normal_max) / span)

    span = max(medium_max - normal_max, 1e-6)
    return 0.66 + 0.34 * clamp_unit((value - medium_max) / span)


def _heart_rate_component(heart_rate: float) -> float:
    return _band_component(heart_rate, HR_NORMAL_MAX, HR_MEDIUM_MAX)


def _gsr_component(gsr: float) -> float:
    return _band_component(gsr, GSR_NORMAL_MAX, GSR_MEDIUM_MAX)


def _face_component(face_score: float) -> float:
    return _band_component(face_score, FACE_NORMAL_MAX, FACE_MEDIUM_MAX)


def _heart_rate_reasons(heart_rate: float) -> str:
    if heart_rate > HR_MEDIUM_MAX:
        return "heart_rate_high"
    if heart_rate > HR_NORMAL_MAX:
        return "heart_rate_medium"
    if heart_rate < HR_LOW_MAX:
        return "heart_rate_low"
    return "heart_rate_normal"


def _gsr_reasons(gsr: float) -> str:
    if gsr > GSR_MEDIUM_MAX:
        return "gsr_high"
    if gsr > GSR_NORMAL_MAX:
        return "gsr_medium"
    return "gsr_normal"


def _face_reasons(face_score: float | None) -> str:
    if face_score is None:
        return "face_score_unavailable"
    if face_score > FACE_MEDIUM_MAX:
        return "face_score_high"
    if face_score > FACE_NORMAL_MAX:
        return "face_score_medium"
    return "face_score_low"


def calculate_stress(
    heart_rate: Any,
    gsr: Any,
    face_score: float | None = None,
    recent_readings: Any = None,
) -> dict:
    """
    Fuse physiological signals into a 0-100 stress score using fixed bands.

    Bands
    -----
    heart_rate (BPM): <60 trending low, 60-72 normal, 72-95 medium, >95 high.
    gsr (raw sensor units): 0-350 normal, 350-800 medium, >800 high.
    face_score (0-1): 0-0.30 low, 0.30-0.60 medium, >0.60 high.

    GSR above GSR_FORCE_HIGH_THRESHOLD (800) is a hard override: the result
    is forced to stress_score=100 / HIGH regardless of HR or face.

    Fusion weights: GSR highest (0.45), then HR (0.35), then face (0.20).
    *recent_readings* is accepted for backward-compatible call signatures
    but is not used by this band-based version (no personalisation).

    Returns
    -------
    dict with keys: stress_score (int 0-100), stress_level, reasons, components.
    """
    heart_rate = _safe_float(heart_rate, (HR_NORMAL_MAX + HR_LOW_MAX) / 2)
    gsr = max(0.0, _safe_float(gsr, 0.0))

    if gsr > GSR_FORCE_HIGH_THRESHOLD:
        reasons: list[str] = [
            _heart_rate_reasons(heart_rate),
            "gsr_forced_high_threshold_exceeded",
            _face_reasons(face_score),
        ]
        return {
            "stress_score": 100,
            "stress_level": "HIGH",
            "reasons": reasons,
            "components": {
                "heart_rate": round(_heart_rate_component(heart_rate), 4),
                "gsr": 1.0,
                "face": round(_face_component(float(face_score)), 4) if face_score is not None else None,
            },
        }

    hr_component = _heart_rate_component(heart_rate)
    gsr_component = _gsr_component(gsr)
    face_component = (
        _face_component(clamp_unit(float(face_score)))
        if face_score is not None
        else FACE_UNAVAILABLE_COMPONENT
    )

    fused = clamp_unit(
        WEIGHT_HR * hr_component
        + WEIGHT_GSR * gsr_component
        + WEIGHT_FACE * face_component
    )
    score = int(round(clamp(100.0 * fused)))

    if score >= 64:
        level = "HIGH"
    elif score >= 34:
        level = "MEDIUM"
    else:
        level = "LOW"

    reasons = [
        _heart_rate_reasons(heart_rate),
        _gsr_reasons(gsr),
        _face_reasons(face_score),
    ]
    if hr_component >= 0.66:
        reasons.append("heart_rate_dominant_response")
    if gsr_component >= 0.66:
        reasons.append("gsr_dominant_response")
    if face_score is not None and face_component >= 0.66:
        reasons.append("face_score_dominant_response")

    return {
        "stress_score": score,
        "stress_level": level,
        "reasons": reasons,
        "components": {
            "heart_rate": round(hr_component, 4),
            "gsr": round(gsr_component, 4),
            "face": round(face_component, 4) if face_score is not None else None,
        },
    }
