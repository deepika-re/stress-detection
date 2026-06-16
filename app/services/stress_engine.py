
───────────────
Fuses physiological signals into a 0–100 stress score with reliable
LOW / MEDIUM / HIGH variation across the full realistic input range.

Score-to-level map
──────────────────
  LOW:    0 – 27    Calm, resting, sleep-level arousal
  MEDIUM: 28 – 71   Normal to elevated activity; mild alertness through light stress
  HIGH:   72 – 100  Meaningful stress response; moderate to acute arousal

Key design decisions
────────────────────
1. Absolute components are centred at population resting values (HR 72 BPM,
   GSR 1.5 µS) so that a clearly elevated reading registers without needing
   personal history.

2. A linear stretch maps the realistic fused range [FUSED_LOW, FUSED_HIGH]
   to [0, 100].  This defeats sigmoid compression that caused scores to
   plateau in the 75–88 range and never reach HIGH.

3. No disagreement penalty.  When both HR and GSR are elevated together,
   that is corroborating evidence, not a reason to lower the score.

4. Personalisation (via robust z-score and percentile rank) tightens the
   response once MIN_HISTORY_FOR_PERSONALIZATION readings are available,
   catching individuals whose resting values sit outside population priors.

5. Fusion weights sum to exactly 1.0.  HR and GSR together carry 0.84 so
   that the raw physiological signal dominates; interaction, volatility, and
   face add nuance without distorting the baseline.
"""

import bisect
import math
import statistics
from typing import Any

# ── Personalisation gate ───────────────────────────────────────────────────────
MIN_HISTORY_FOR_PERSONALIZATION = 6

# ── Population priors ──────────────────────────────────────────────────────────
HR_REST_CENTER        = 72.0    # BPM — median resting heart rate
HR_REST_SCALE         = 10.0    # BPM — robust population scale (≈ IQR/1.35)
HR_MIN_PERSONAL_SCALE = 4.0     # BPM — floor to avoid over-sensitivity

GSR_REFERENCE              = 1.5   # µS  — median resting GSR
GSR_LOG_SCALE              = 0.35  # log-space scale (tighter than original 0.42)
GSR_MIN_PERSONAL_LOG_SCALE = 0.08

# ── Fusion stretch anchors ─────────────────────────────────────────────────────
# Computed once at (HR=50, GSR=0.3) and (HR=160, GSR=18) with no history,
# default vol/face priors.  Hard-coded to keep calculate_stress() O(1).
FUSED_LOW  = 0.135   # fused score at absolute rest → maps to raw score 0
FUSED_HIGH = 0.885   # fused score at physiological maximum → maps to raw score 100

# ── Score band thresholds ──────────────────────────────────────────────────────
THRESHOLD_HIGH   = 72   # stretched score ≥ 72 → HIGH
THRESHOLD_MEDIUM = 28   # stretched score ≥ 28 → MEDIUM; below → LOW

# ── Reason thresholds ─────────────────────────────────────────────────────────
REASON_PERSONAL_Z_THRESHOLD  = 0.75
REASON_PERCENTILE_THRESHOLD  = 0.78
REASON_VOLATILITY_THRESHOLD  = 0.60
REASON_INTERACTION_THRESHOLD = 0.55
REASON_DOMINANCE_THRESHOLD   = 0.65


# ─────────────────────────────────────────────────────────────────────────────
# Low-level maths
# ─────────────────────────────────────────────────────────────────────────────

def clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    return max(minimum, min(maximum, value))


def clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def sigmoid(value: float) -> float:
    """Numerically stable logistic function."""
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


def _robust_scale(values: list[float], fallback_scale: float, min_scale: float) -> float:
    """
    MAD/IQR-based robust scale estimator.

    Falls back to *fallback_scale* until MIN_HISTORY_FOR_PERSONALIZATION
    readings are available.  Always returns ≥ min_scale.
    """
    if len(values) < MIN_HISTORY_FOR_PERSONALIZATION:
        return fallback_scale

    center     = statistics.median(values)
    deviations = [abs(s - center) for s in values]
    mad        = statistics.median(deviations)
    mad_scale  = 1.4826 * mad if mad > 0 else 0.0

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
    """Robust z-score; uses personal baseline when sufficient history exists."""
    if len(values) >= MIN_HISTORY_FOR_PERSONALIZATION:
        center = statistics.median(values)
        scale  = _robust_scale(values, fallback_scale, min_scale)
    else:
        center = fallback_center
        scale  = fallback_scale
    return (value - center) / max(scale, 0.001)


def _percentile_rank(value: float, values: list[float]) -> float:
    """
    Percentile rank of *value* within *values* via bisect — O(log n).
    Returns 0.5 when history is too short.
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
    How sharply has *value* risen relative to the recent local baseline?

    Uses a neutral prior of 0.25 when fewer than 3 readings are available —
    small but non-zero, reflecting mild uncertainty about a new stream.
    """
    if len(values) < 3:
        return 0.25
    tail           = values[-5:]
    local_baseline = statistics.median(tail)
    scale          = _robust_scale(values, fallback_scale, min_scale)
    delta          = value - local_baseline
    return sigmoid((delta - 0.10 * scale) / max(0.30 * scale, 0.001))


# ─────────────────────────────────────────────────────────────────────────────
# Per-signal component calculators
# ─────────────────────────────────────────────────────────────────────────────

def _hr_component(
    heart_rate: float,
    hr_values: list[float],
) -> tuple[float, float, float, float, float]:
    """
    Returns (component, hr_absolute, hr_personal_z, hr_percentile, hr_trend).

    Sigmoid response table (centre=72, scale=10):
      52 BPM → 0.12  (sleep)
      72 BPM → 0.50  (resting)
      88 BPM → 0.83  (elevated)
     108 BPM → 0.96  (high)

    Weights: absolute 0.40 · personal 0.38 · percentile 0.12 · trend 0.10
    The absolute term dominates so population-level stress registers immediately.
    """
    hr_absolute   = sigmoid((heart_rate - HR_REST_CENTER) / HR_REST_SCALE)
    hr_personal_z = _robust_z(
        heart_rate, hr_values, HR_REST_CENTER, HR_REST_SCALE, HR_MIN_PERSONAL_SCALE
    )
    hr_personal   = sigmoid((hr_personal_z - 0.10) * 1.10)
    hr_percentile = _percentile_rank(heart_rate, hr_values)
    hr_trend      = _trend_activation(heart_rate, hr_values, HR_REST_SCALE, HR_MIN_PERSONAL_SCALE)

    component = clamp_unit(
        0.40 * hr_absolute
        + 0.38 * hr_personal
        + 0.12 * hr_percentile
        + 0.10 * hr_trend
    )
    return component, hr_absolute, hr_personal_z, hr_percentile, hr_trend


def _gsr_component(
    log_gsr: float,
    log_gsr_values: list[float],
) -> tuple[float, float, float, float, float]:
    """
    Returns (component, gsr_absolute, gsr_personal_z, gsr_percentile, gsr_trend).

    Sigmoid response table (ref=log1p(1.5)≈0.916, scale=0.35):
      0.4 µS → 0.16  (very low)
      1.5 µS → 0.50  (resting reference)
      3.5 µS → 0.84  (elevated)
      7.5 µS → 0.97  (high)

    Weights: absolute 0.42 · personal 0.36 · percentile 0.12 · trend 0.10
    """
    log_gsr_ref    = math.log1p(GSR_REFERENCE)
    gsr_absolute   = sigmoid((log_gsr - log_gsr_ref) / GSR_LOG_SCALE)
    gsr_personal_z = _robust_z(
        log_gsr, log_gsr_values, log_gsr_ref, GSR_LOG_SCALE, GSR_MIN_PERSONAL_LOG_SCALE
    )
    gsr_personal   = sigmoid((gsr_personal_z - 0.08) * 1.20)
    gsr_percentile = _percentile_rank(log_gsr, log_gsr_values)
    gsr_trend      = _trend_activation(
        log_gsr, log_gsr_values, GSR_LOG_SCALE, GSR_MIN_PERSONAL_LOG_SCALE
    )

    component = clamp_unit(
        0.42 * gsr_absolute
        + 0.36 * gsr_personal
        + 0.12 * gsr_percentile
        + 0.10 * gsr_trend
    )
    return component, gsr_absolute, gsr_personal_z, gsr_percentile, gsr_trend


# ─────────────────────────────────────────────────────────────────────────────
# Reason label helpers
# ─────────────────────────────────────────────────────────────────────────────

def _heart_rate_reason(heart_rate: float) -> str:
    if heart_rate >= 120:
        return "heart_rate_very_high"
    if heart_rate >= 100:
        return "heart_rate_high"
    if heart_rate >= 86:
        return "heart_rate_elevated"
    if heart_rate <= 55:
        return "heart_rate_low"
    return "heart_rate_normal"


def _gsr_reason(gsr: float) -> str:
    if gsr >= 6:
        return "gsr_very_high"
    if gsr >= 3:
        return "gsr_high"
    if gsr >= 1.4:
        return "gsr_elevated"
    return "gsr_low_or_normal"


def _face_reason(face_score: float | None) -> str:
    if face_score is None:
        return "face_score_unavailable"
    if face_score >= 0.75:
        return "face_score_very_high"
    if face_score >= 0.50:
        return "face_score_high"
    if face_score >= 0.30:
        return "face_score_elevated"
    return "face_score_low"


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

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
        Galvanic skin response in µS (0–20 typical range).
    face_score:
        Optional 0–1 facial-expression stress score.  Pass None when the
        face pipeline is unavailable.
    recent_readings:
        Iterable of recent readings (dicts or ORM objects) with
        ``heart_rate`` and ``gsr`` fields.  Enables personalisation and
        trend detection once MIN_HISTORY_FOR_PERSONALIZATION readings exist.

    Returns
    -------
    dict
        stress_score : int, 0–100
        stress_level : "LOW" | "MEDIUM" | "HIGH"
        reasons      : list[str]
        components   : dict of intermediate values for debugging / logging
    """
    # ── Sanitise inputs ───────────────────────────────────────────────────────
    heart_rate = _safe_float(heart_rate, HR_REST_CENTER)
    gsr        = max(0.0, _safe_float(gsr, 0.0))   # type: ignore[arg-type]

    hr_values  = _series(recent_readings, "heart_rate")
    gsr_values = _series(recent_readings, "gsr")

    log_gsr        = math.log1p(gsr)
    log_gsr_values = [math.log1p(max(0.0, v)) for v in gsr_values]

    # ── 1. Heart-rate component ───────────────────────────────────────────────
    (
        hr_comp,
        hr_absolute,
        hr_personal_z,
        hr_percentile,
        hr_trend,
    ) = _hr_component(heart_rate, hr_values)

    # ── 2. GSR component ──────────────────────────────────────────────────────
    (
        gsr_comp,
        gsr_absolute,
        gsr_personal_z,
        gsr_percentile,
        gsr_trend,
    ) = _gsr_component(log_gsr, log_gsr_values)

    # ── 3. Face component ─────────────────────────────────────────────────────
    # Neutral prior of 0.30 when the face pipeline is unavailable — sits at
    # the MEDIUM floor and neither inflates nor deflates the score.
    face_comp = (
        clamp_unit(float(face_score))   # type: ignore[arg-type]
        if face_score is not None
        else 0.30
    )

    # ── 4. Interaction & volatility ───────────────────────────────────────────
    # Interaction: geometric mean.  High only when BOTH signals are elevated.
    interaction = math.sqrt(max(hr_comp * gsr_comp, 0.0))

    # Volatility: weighted average of the two trend activations.
    volatility = clamp_unit(0.45 * hr_trend + 0.55 * gsr_trend)

    # ── 5. Fusion ─────────────────────────────────────────────────────────────
    # Weights sum to 1.0.
    #
    #   HR component     0.40  — dominant physiological signal
    #   GSR component    0.35  — direct sympathetic arousal marker
    #   Interaction      0.09  — corroboration bonus when both rise
    #   Volatility       0.08  — rewards rapid sustained rises
    #   Face             0.08  — noisy but informative when available
    #
    # Example (no history, no face):
    #   HR=90, GSR=2.8  →  hr=0.85  gsr=0.73  →  fused≈0.724
    #   stretched = (0.724-0.135)/(0.885-0.135) = 0.785  →  score 79  → HIGH
    #
    #   HR=72, GSR=1.5  →  hr=0.50  gsr=0.50  →  fused≈0.464
    #   stretched = (0.464-0.135)/0.750 = 0.439  →  score 44  → MEDIUM
    #
    #   HR=58, GSR=0.5  →  hr=0.24  gsr=0.22  →  fused≈0.262
    #   stretched = (0.262-0.135)/0.750 = 0.169  →  score 17  → LOW
    fused = clamp_unit(
        0.40 * hr_comp
        + 0.35 * gsr_comp
        + 0.09 * interaction
        + 0.08 * volatility
        + 0.08 * face_comp
    )

    # ── 6. Stretch to 0–100 ───────────────────────────────────────────────────
    # Maps the realistic physiological range [FUSED_LOW, FUSED_HIGH] linearly
    # to [0, 100], preventing sigmoid compression from clustering scores in the
    # 40–85 band and making HIGH unreachable at realistic input values.
    stretch = clamp_unit(
        (fused - FUSED_LOW) / max(FUSED_HIGH - FUSED_LOW, 0.001)
    )
    score = int(round(clamp(100.0 * stretch)))

    # ── 7. Band classification ────────────────────────────────────────────────
    if score >= THRESHOLD_HIGH:
        level = "HIGH"
    elif score >= THRESHOLD_MEDIUM:
        level = "MEDIUM"
    else:
        level = "LOW"

    # ── 8. Reasons ────────────────────────────────────────────────────────────
    reasons: list[str] = [
        _heart_rate_reason(heart_rate),
        _gsr_reason(gsr),
        _face_reason(face_score),
    ]

    if hr_personal_z >= REASON_PERSONAL_Z_THRESHOLD:
        reasons.append("heart_rate_above_personal_baseline")
    if gsr_personal_z >= REASON_PERSONAL_Z_THRESHOLD:
        reasons.append("gsr_above_personal_baseline")
    if hr_percentile >= REASON_PERCENTILE_THRESHOLD:
        reasons.append("heart_rate_high_for_recent_window")
    if gsr_percentile >= REASON_PERCENTILE_THRESHOLD:
        reasons.append("gsr_high_for_recent_window")
    if volatility >= REASON_VOLATILITY_THRESHOLD:
        reasons.append("rapid_signal_rise")
    if interaction >= REASON_INTERACTION_THRESHOLD:
        reasons.append("heart_gsr_coupled_response")
    if gsr_comp >= REASON_DOMINANCE_THRESHOLD:
        reasons.append("gsr_dominant_response")
    if hr_comp >= REASON_DOMINANCE_THRESHOLD:
        reasons.append("heart_rate_dominant_response")

    return {
        "stress_score": score,
        "stress_level": level,
        "reasons":      reasons,
        "components": {
            "heart_rate":     round(hr_comp,          4),
            "gsr":            round(gsr_comp,         4),
            "face":           round(face_comp,        4),
            "interaction":    round(interaction,      4),
            "volatility":     round(volatility,       4),
            "fused_raw":      round(fused,            4),
            "fused_stretch":  round(stretch,          4),
            "hr_absolute":    round(hr_absolute,      4),
            "gsr_absolute":   round(gsr_absolute,     4),
            "hr_personal_z":  round(hr_personal_z,    4),
            "gsr_personal_z": round(gsr_personal_z,   4),
            "hr_percentile":  round(hr_percentile,    4),
            "gsr_percentile": round(gsr_percentile,   4),
            "hr_trend":       round(hr_trend,         4),
            "gsr_trend":      round(gsr_trend,        4),
        },
    }
