from flask import Blueprint, current_app, jsonify, request
from flask_login import current_user, login_required

from ..extensions import db
from ..models import FaceScore, SensorReading, utcnow
from ..services.face_inference import FaceInferenceError, classify_face_expression, decode_image_payload
from ..services.validation_service import ValidationError, parse_client_timestamp, validate_face_score_payload

api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.get("/health")
def health():
    return jsonify({"status": "ok", "service": "stress-monitor"})


@api_bp.get("/readings/latest")
@login_required
def latest_reading():
    reading = (
        SensorReading.query.filter_by(user_id=current_user.id)
        .order_by(SensorReading.captured_at.desc(), SensorReading.id.desc())
        .first()
    )
    if not reading:
        return jsonify({})

    return jsonify(_reading_to_dict(reading))


@api_bp.get("/readings/history")
@login_required
def readings_history():
    try:
        limit = min(max(int(request.args.get("limit", 100)), 1), 500)
    except ValueError:
        limit = 100

    rows = (
        SensorReading.query.filter_by(user_id=current_user.id)
        .order_by(SensorReading.captured_at.desc(), SensorReading.id.desc())
        .limit(limit)
        .all()
    )
    return jsonify([_reading_to_dict(row) for row in reversed(rows)])


@api_bp.post("/face-score")
@login_required
def face_score():
    try:
        data = validate_face_score_payload(request.get_json(silent=True) or {})
    except ValidationError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400

    score = FaceScore(
        user_id=current_user.id,
        score=data["score"],
        captured_at=data["captured_at"],
        source=data["source"],
    )
    db.session.add(score)
    db.session.flush()
    _prune_old_face_scores(current_user.id, score.id)
    db.session.commit()
    return jsonify({"status": "ok"})


@api_bp.post("/face-inference")
@login_required
def face_inference():
    payload = request.get_json(silent=True) or {}
    try:
        image_bytes = decode_image_payload(payload.get("image"))
        captured_at = payload.get("captured_at")
        captured_at = parse_client_timestamp(captured_at) if captured_at else None
    except FaceInferenceError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    except ValidationError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400

    try:
        inference = classify_face_expression(image_bytes)
    except FaceInferenceError as exc:
        return jsonify({
            "status": "error",
            "message": str(exc),
            "retry_after_seconds": exc.retry_after_seconds,
        }), 503

    previous = (
        FaceScore.query.filter_by(user_id=current_user.id)
        .order_by(FaceScore.captured_at.desc(), FaceScore.id.desc())
        .first()
    )
    raw_score = inference["stress_score"]
    alpha = max(0.0, min(1.0, current_app.config["FACE_EWMA_ALPHA"]))
    if previous:
        smoothed_score = (alpha * raw_score) + ((1 - alpha) * previous.score)
    else:
        smoothed_score = raw_score

    score = FaceScore(
        user_id=current_user.id,
        score=smoothed_score,
        captured_at=captured_at or utcnow(),
        source="hf_face_expression",
    )
    db.session.add(score)
    db.session.flush()
    _prune_old_face_scores(current_user.id, score.id)
    db.session.commit()

    return jsonify({
        "status": "ok",
        "stress_score": round(smoothed_score, 4),
        "raw_stress_score": raw_score,
        "dominant_emotion": inference["dominant_emotion"],
        "confidence": inference["confidence"],
        "entropy": inference["entropy"],
        "distribution": inference["distribution"],
        "model": inference["model"],
        "provider": inference["provider"],
        "latency_ms": inference["latency_ms"],
    })


def _prune_old_face_scores(user_id, latest_score_id):
    FaceScore.query.filter(
        FaceScore.user_id == user_id,
        FaceScore.id != latest_score_id,
    ).delete(synchronize_session=False)


def _reading_to_dict(reading):
    return {
        "stream_id": reading.stream_id,
        "seq": reading.seq,
        "heart_rate": reading.heart_rate,
        "gsr": reading.gsr,
        "source": reading.source,
        "stress_score": reading.stress_score,
        "stress_level": reading.stress_level,
        "captured_at": reading.captured_at.isoformat() + "Z",
        "received_at": reading.received_at.isoformat() + "Z",
    }
