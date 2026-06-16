from datetime import datetime, timedelta, timezone
from typing import Any


class ValidationError(ValueError):
    pass


# ── Constants ─────────────────────────────────────────────────────────────────

# Extend this set to accept additional sources (e.g. "mobile_ble") without
# touching the validation logic.
ALLOWED_SOURCES: frozenset[str] = frozenset({"browser_serial"})

STREAM_ID_MAX_LEN = 80

# Clock-skew windows for client timestamps.
_FUTURE_TOLERANCE = timedelta(minutes=2)
_PAST_TOLERANCE = timedelta(days=1)

# Physiological bounds for sensor readings.
HR_MIN = 35.0
HR_MAX = 220.0
GSR_MIN = 0.0
# Raised from the old 0-20 µS range to match the engine's new band
# thresholds (normal <=350, medium <=800, hard-override >800). Adjust if
# your sensor's real ceiling differs.
GSR_MAX = 1000.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_client_timestamp(value: Any) -> datetime:
    """
    Parse an ISO-8601 client timestamp and return it as a naive UTC datetime.

    Rejects timestamps that are too far in the future (clock skew) or too old
    (stale replay).  The returned datetime is *naive* to match the project-wide
    convention of storing naive UTC with a manual 'Z' suffix.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValidationError("captured_at is required")

    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValidationError("captured_at must be an ISO-8601 timestamp") from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    parsed_utc = parsed.astimezone(timezone.utc).replace(tzinfo=None)

    # Compute now once to keep the comparison window consistent.
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    if parsed_utc > now + _FUTURE_TOLERANCE:
        raise ValidationError("captured_at is too far in the future")
    if parsed_utc < now - _PAST_TOLERANCE:
        raise ValidationError("captured_at is too old")

    return parsed_utc


def validate_stream_id(stream_id: Any) -> str:
    if not isinstance(stream_id, str):
        raise ValidationError("stream_id must be a string")
    stream_id = stream_id.strip()
    if not stream_id or len(stream_id) > STREAM_ID_MAX_LEN:
        raise ValidationError(
            f"stream_id must be 1 to {STREAM_ID_MAX_LEN} characters"
        )
    return stream_id


def validate_reading_payload(reading: Any) -> dict:
    """
    Validate a single sensor reading dict and return a cleaned copy.

    All numeric fields are coerced to their target types; out-of-range values
    raise ValidationError so the caller can surface them as per-reading errors
    without aborting the entire batch.
    """
    if not isinstance(reading, dict):
        raise ValidationError("reading must be an object")

    stream_id = validate_stream_id(reading.get("stream_id"))

    try:
        seq = int(reading.get("seq"))  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValidationError("seq must be an integer") from exc
    if seq < 0:
        raise ValidationError("seq must be non-negative")

    try:
        heart_rate = float(reading.get("heart_rate"))  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValidationError("heart_rate must be numeric") from exc
    if not (HR_MIN <= heart_rate <= HR_MAX):
        raise ValidationError(
            f"heart_rate must be between {HR_MIN:.0f} and {HR_MAX:.0f} bpm"
        )

    try:
        gsr = float(reading.get("gsr"))  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValidationError("gsr must be numeric") from exc
    if not (GSR_MIN <= gsr <= GSR_MAX):
        raise ValidationError(
            f"gsr must be between {GSR_MIN:.0f} and {GSR_MAX:.0f}"
        )

    source = reading.get("source", "browser_serial")
    if source not in ALLOWED_SOURCES:
        raise ValidationError(
            f"source must be one of: {', '.join(sorted(ALLOWED_SOURCES))}"
        )

    captured_at = parse_client_timestamp(reading.get("captured_at"))

    return {
        "stream_id": stream_id,
        "seq": seq,
        "heart_rate": heart_rate,
        "gsr": gsr,
        "captured_at": captured_at,
        "source": source,
    }


def validate_face_score_payload(payload: Any) -> dict:
    """
    Validate a face-stress score submission and return a cleaned copy.
    """
    if not isinstance(payload, dict):
        raise ValidationError("payload must be an object")

    try:
        score = float(payload.get("score"))  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValidationError("score must be numeric") from exc

    if not (0.0 <= score <= 1.0):
        raise ValidationError("score must be between 0 and 1")

    raw_captured_at = payload.get("captured_at")
    if raw_captured_at:
        captured_at = parse_client_timestamp(raw_captured_at)
    else:
        captured_at = datetime.now(timezone.utc).replace(tzinfo=None)

    source = str(payload.get("source", "face_model"))[:40]
    return {
        "score": score,
        "captured_at": captured_at,
        "source": source,
    }
