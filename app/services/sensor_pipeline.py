import json
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from flask import current_app

from ..extensions import db
from ..models import BrowserStream, FaceScore, SensorReading
from .alert_service import create_high_stress_event, maybe_send_alert
from .stress_engine import calculate_stress
from .validation_service import ValidationError, validate_reading_payload, validate_stream_id


RECENT_CONTEXT_LIMIT = 40
MAX_ERRORS_IN_ACK = 10


@dataclass
class SensorBatchResult:
    stream_id: str
    last_accepted_seq: int
    accepted_count: int
    duplicate_count: int
    rejected_count: int
    dropped_count: int
    errors: list
    latest_update: dict | None
    # Total error count before MAX_ERRORS_IN_ACK truncation, useful for
    # debugging batches where many readings fail validation simultaneously.
    total_errors: int = field(default=0)

    def to_ack(self) -> dict:
        return {
            "stream_id": self.stream_id,
            "last_accepted_seq": self.last_accepted_seq,
            "accepted_count": self.accepted_count,
            "duplicate_count": self.duplicate_count,
            "rejected_count": self.rejected_count,
            "dropped_count": self.dropped_count,
            "errors": self.errors[:MAX_ERRORS_IN_ACK],
            "total_errors": self.total_errors,
            "server_time": utc_now_naive().isoformat() + "Z",
        }


def utc_now_naive() -> datetime:
    """
    Return the current UTC time as a *naive* datetime.

    The project uses naive UTC timestamps with a manual 'Z' suffix throughout,
    so this helper keeps that convention consistent in one place.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def user_room(user_id: int) -> str:
    return f"user:{user_id}"


def get_or_create_stream(user, stream_id: str):
    """
    Fetch the BrowserStream for this user, creating it if it does not exist.

    A single flush is required only on creation so that SensorReading can
    reference stream.id via a foreign key.

    NOTE: Concurrent requests for the same new stream_id may race here.
    The current mitigation relies on a UNIQUE constraint on stream_id at
    the DB level; the second writer will receive an IntegrityError which
    Flask-SQLAlchemy surfaces as a 500 that the client retries.  For
    higher-traffic deployments, replace with an ON CONFLICT / upsert.
    """
    stream = BrowserStream.query.filter_by(stream_id=stream_id).first()

    if stream and stream.user_id != user.id:
        raise ValidationError("stream_id belongs to another user")

    if not stream:
        stream = BrowserStream(
            stream_id=stream_id,
            user_id=user.id,
            status="started",
        )
        db.session.add(stream)
        db.session.flush()

    return stream


def latest_recent_face_score(user_id: int) -> float | None:
    """
    Return the most recent face stress score within the configured age window.

    Selects only the score column (not a full ORM object) for minimal overhead.
    Returns *None* when no valid face score is available; the stress engine
    treats None as "use the neutral prior" rather than "low stress", which
    prevents the overall stress score from jumping when the face pipeline
    flickers on and off.
    """
    cutoff = utc_now_naive() - timedelta(
        seconds=current_app.config["FACE_SCORE_MAX_AGE_SECONDS"]
    )
    return (
        db.session.query(FaceScore.score)
        .filter(
            FaceScore.user_id == user_id,
            FaceScore.captured_at >= cutoff,
        )
        .order_by(FaceScore.captured_at.desc())
        .limit(1)
        .scalar()
    )


def recent_sensor_context(user_id: int, limit: int = RECENT_CONTEXT_LIMIT) -> deque:
    """
    Load the most recent sensor readings for personalisation and trend detection.

    Returns a ``deque(maxlen=limit)`` of ``{"heart_rate": …, "gsr": …}`` dicts
    (oldest-first) so that the caller can append new readings without extra
    slicing overhead.  Only the two columns needed by calculate_stress() are
    fetched.
    """
    rows = (
        db.session.query(SensorReading.heart_rate, SensorReading.gsr)
        .filter(SensorReading.user_id == user_id)
        .order_by(SensorReading.captured_at.desc(), SensorReading.id.desc())
        .limit(limit)
        .all()
    )

    ctx: deque = deque(maxlen=limit)
    for heart_rate, gsr in reversed(rows):
        ctx.append({"heart_rate": heart_rate, "gsr": gsr})
    return ctx


def serialize_live_update(reading, stress_result: dict, face_score_available: bool) -> dict:
    captured_at = reading.captured_at
    received_at = reading.received_at or utc_now_naive()

    return {
        "stream_id": reading.stream_id,
        "seq": reading.seq,
        "heart_rate": reading.heart_rate,
        "gsr": reading.gsr,
        "source": reading.source,
        "stress_score": stress_result["stress_score"],
        "stress_level": stress_result["stress_level"],
        "reasons": stress_result["reasons"],
        "components": stress_result.get("components", {}),
        # Lets the UI distinguish "face score is genuinely low" (False) from
        # "face inference is unavailable right now" (also False score but
        # face_score_available=False) so it can show a placeholder instead
        # of a misleading low reading.
        "face_score_available": face_score_available,
        "captured_at": captured_at.isoformat() + "Z",
        "received_at": received_at.isoformat() + "Z",
    }


def normalize_sensor_batch(
    stream_id: str,
    raw_readings,
) -> tuple[list, int, list]:
    """
    Validate raw readings and remove within-batch duplicates.

    Returns
    -------
    normalized:
        Valid, de-duplicated readings.
    duplicate_count:
        Number of seqs that appeared more than once in this request.
    errors:
        Validation error dicts with ``index`` and ``message`` fields.
    """
    normalized: list = []
    errors: list = []
    seen_in_batch: set = set()
    duplicate_count = 0

    for index, raw in enumerate(raw_readings or []):
        try:
            if isinstance(raw, dict):
                payload = {**raw, "stream_id": raw.get("stream_id", stream_id)}
            else:
                payload = raw

            reading = validate_reading_payload(payload)

            if reading["stream_id"] != stream_id:
                raise ValidationError("reading stream_id does not match batch stream_id")

            key = (reading["stream_id"], reading["seq"])
            if key in seen_in_batch:
                duplicate_count += 1
                continue

            seen_in_batch.add(key)
            normalized.append(reading)

        except ValidationError as exc:
            errors.append({"index": index, "message": str(exc)})

    return normalized, duplicate_count, errors


def existing_sequences_for_stream(stream_id: str, seqs: list[int]) -> set[int]:
    """
    Return the subset of *seqs* that are already stored for *stream_id*.

    Prevents duplicate inserts and skips redundant stress calculations.
    The query uses an IN clause which is efficient for typical batch sizes
    (< 1 000 seqs); for very large batches the caller is expected to chunk.
    """
    if not seqs:
        return set()

    return {
        seq
        for (seq,) in (
            db.session.query(SensorReading.seq)
            .filter(
                SensorReading.stream_id == stream_id,
                SensorReading.seq.in_(seqs),
            )
            .all()
        )
    }


def build_sensor_reading(
    user,
    stream,
    stream_id: str,
    reading_data: dict,
    stress_result: dict,
    received_at: datetime,
):
    return SensorReading(
        user_id=user.id,
        browser_stream_id=stream.id,
        stream_id=stream_id,
        seq=reading_data["seq"],
        heart_rate=reading_data["heart_rate"],
        gsr=reading_data["gsr"],
        source=reading_data["source"],
        stress_score=stress_result["stress_score"],
        stress_level=stress_result["stress_level"],
        reasons_json=json.dumps(stress_result["reasons"]),
        captured_at=reading_data["captured_at"],
        received_at=received_at,
    )


def _fire_high_stress_alerts(user, high_stress_candidates: list) -> None:
    """
    Background worker: process alert for the latest HIGH reading in the batch.

    Limiting to the *last* HIGH reading prevents a batch of 20 HIGH readings
    from firing 20 alert flows in one request.
    """
    if not high_stress_candidates:
        return
    latest_reading, latest_result = high_stress_candidates[-1]
    stress_event = create_high_stress_event(user, latest_reading, latest_result)
    maybe_send_alert(user, stress_event, latest_reading)


def process_high_stress_alerts(user, high_stress_candidates: list) -> None:
    """
    Dispatch alert processing to a daemon thread so it does not block the
    HTTP response.

    The thread is a fire-and-forget daemon; it will not prevent the process
    from shutting down.  For guaranteed delivery move this to a task queue
    (Celery, RQ, etc.) instead.
    """
    if not high_stress_candidates:
        return
    t = threading.Thread(
        target=_fire_high_stress_alerts,
        args=(user, high_stress_candidates),
        daemon=True,
        name="high-stress-alert",
    )
    t.start()


def process_sensor_batch(
    user,
    stream_id: str,
    raw_readings,
    dropped_count: int = 0,
) -> SensorBatchResult:
    stream_id = validate_stream_id(stream_id)
    stream = get_or_create_stream(user, stream_id)

    normalized, duplicate_count, errors = normalize_sensor_batch(stream_id, raw_readings)
    total_errors = len(errors)

    if not normalized:
        stream.dropped_count += int(dropped_count or 0)
        db.session.commit()
        return SensorBatchResult(
            stream_id=stream_id,
            last_accepted_seq=stream.last_seq_acked,
            accepted_count=0,
            duplicate_count=duplicate_count,
            rejected_count=total_errors,
            dropped_count=stream.dropped_count,
            errors=errors,
            latest_update=None,
            total_errors=total_errors,
        )

    seqs = list({r["seq"] for r in normalized})
    existing_seqs = existing_sequences_for_stream(stream_id, seqs)

    accepted_count = 0
    last_acked_candidate = stream.last_seq_acked
    latest_update: dict | None = None

    received_at = utc_now_naive()
    face_score = latest_recent_face_score(user.id)
    face_score_available = face_score is not None

    # deque(maxlen=…) gives O(1) append + automatic eviction of the oldest
    # element — no list re-allocation on every iteration.
    recent_context: deque = recent_sensor_context(user.id)

    new_readings: list = []
    high_stress_candidates: list = []

    with db.session.no_autoflush:
        for reading_data in normalized:
            seq = reading_data["seq"]
            last_acked_candidate = max(last_acked_candidate, seq)

            if seq in existing_seqs:
                duplicate_count += 1
                continue

            stress_result = calculate_stress(
                reading_data["heart_rate"],
                reading_data["gsr"],
                face_score,
                recent_readings=recent_context,
            )

            reading = build_sensor_reading(
                user=user,
                stream=stream,
                stream_id=stream_id,
                reading_data=reading_data,
                stress_result=stress_result,
                received_at=received_at,
            )
            new_readings.append(reading)

            # Append to the deque — maxlen automatically drops the oldest entry.
            recent_context.append(
                {"heart_rate": reading_data["heart_rate"], "gsr": reading_data["gsr"]}
            )

            accepted_count += 1
            latest_update = serialize_live_update(reading, stress_result, face_score_available)

            if stress_result["stress_level"] == "HIGH":
                high_stress_candidates.append((reading, stress_result))

    if new_readings:
        db.session.add_all(new_readings)

    stream.last_seq_acked = max(stream.last_seq_acked, last_acked_candidate)
    stream.dropped_count += int(dropped_count or 0)

    # Single flush for the whole batch instead of one flush per reading.
    db.session.flush()

    # Alert processing runs in a daemon thread so it does not delay the HTTP
    # response.  Move to a proper task queue for guaranteed delivery.
    process_high_stress_alerts(user, high_stress_candidates)

    db.session.commit()

    return SensorBatchResult(
        stream_id=stream_id,
        last_accepted_seq=stream.last_seq_acked,
        accepted_count=accepted_count,
        duplicate_count=duplicate_count,
        rejected_count=total_errors,
        dropped_count=stream.dropped_count,
        errors=errors,
        latest_update=latest_update,
        total_errors=total_errors,
    )
