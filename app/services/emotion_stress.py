import math
from typing import Any

EMOTION_STRESS_PRIORS: dict[str, float] = {
    "angry": 0.82,
    "anger": 0.82,
    "disgust": 0.74,
    "fear": 0.92,
    "fearful": 0.92,
    "sad": 0.68,
    "sadness": 0.68,
    "surprise": 0.46,
    "surprised": 0.46,
    "neutral": 0.22,
    "happy": 0.04,
    "happiness": 0.04,
}

HIGH_AROUSAL_LABELS: frozenset[str] = frozenset(
    {"angry", "anger", "disgust", "fear", "fearful", "surprise", "surprised"}
)

# Default prior for unknown labels (placed here for easy tuning).
_UNKNOWN_PRIOR = 0.35

# Sigmoid parameters for final stress mapping.
_SIG_SCALE = 4.2
_SIG_SHIFT = 0.35


def _label_of(item: Any) -> str | None:
    if isinstance(item, dict):
        return item.get("label")
    return getattr(item, "label", None)


def _score_of(item: Any) -> float | None:
    if isinstance(item, dict):
        return item.get("score")
    return getattr(item, "score", None)


def normalize_emotion_distribution(predictions: Any) -> list[dict]:
    """
    Converts raw model predictions into a probability distribution summing to 1.

    Filters out items with missing/invalid labels or non-finite scores, then
    normalises so all probabilities sum exactly to 1.  Returns an empty list
    when no valid items are found.
    """
    if not predictions:
        return []

    rows: list[dict] = []
    for item in predictions:
        label = str(_label_of(item) or "").strip().lower()
        if not label:
            continue
        try:
            probability = float(_score_of(item))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if not math.isfinite(probability) or probability < 0:
            continue
        rows.append({"label": label, "score": probability})

    if not rows:
        return []

    total = sum(row["score"] for row in rows)
    if total <= 0:
        return []

    inv_total = 1.0 / total
    normalised = [
        {"label": row["label"], "score": row["score"] * inv_total}
        for row in rows
    ]
    normalised.sort(key=lambda row: row["score"], reverse=True)
    return normalised


def calculate_emotion_stress(predictions: Any) -> dict:
    """
    Derives a 0–1 stress score from a face-expression probability distribution.

    The formula blends:
      - A weighted prior for each emotion label (e.g. fear = 0.92).
      - Arousal mass (sum of probabilities for high-arousal labels).
      - Confidence (top-label probability) scaling the prior weight.
      - Penalties for happiness, neutrality, and distributional entropy.

    Returns a dict with keys: stress_score, dominant_emotion, confidence,
    entropy, distribution.
    """
    distribution = normalize_emotion_distribution(predictions)
    if not distribution:
        return {
            "stress_score": 0.0,
            "dominant_emotion": "unknown",
            "confidence": 0.0,
            "entropy": 1.0,
            "distribution": [],
        }

    confidence = distribution[0]["score"]

    # Weighted evidence signal: prior ∝ confidence so uncertain predictions
    # contribute less.
    weighted_prior = sum(
        row["score"] * EMOTION_STRESS_PRIORS.get(row["label"], _UNKNOWN_PRIOR)
        for row in distribution
    )

    # Arousal mass directly boosts the signal.
    arousal = sum(
        row["score"]
        for row in distribution
        if row["label"] in HIGH_AROUSAL_LABELS
    )

    # Only scan once for both happy & neutral to avoid two linear passes.
    happy = 0.0
    neutral = 0.0
    for row in distribution:
        lbl = row["label"]
        if lbl == "happy" or lbl == "happiness":
            happy += row["score"]
        elif lbl == "neutral":
            neutral += row["score"]

    # Normalised entropy: 0 = fully peaked, 1 = uniform over all labels.
    n = len(distribution)
    raw_entropy = -sum(row["score"] * math.log(row["score"] + 1e-12) for row in distribution)
    entropy = raw_entropy / max(math.log(n), 1e-12)

    evidence = (
        weighted_prior * (0.72 + 0.28 * confidence)
        + 0.18 * arousal
        - 0.24 * happy
        - 0.10 * neutral
        - 0.08 * entropy
    )

    # Logistic mapping: evidence=0.35 → stress_score=0.50.
    raw = 1.0 / (1.0 + math.exp(-_SIG_SCALE * (evidence - _SIG_SHIFT)))
    stress_score = max(0.0, min(1.0, raw))

    return {
        "stress_score": round(stress_score, 4),
        "dominant_emotion": distribution[0]["label"],
        "confidence": round(confidence, 4),
        "entropy": round(entropy, 4),
        "distribution": [
            {"label": row["label"], "score": round(row["score"], 4)}
            for row in distribution
        ],
    }
