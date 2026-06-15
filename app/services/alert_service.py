import json
from datetime import datetime, timedelta, timezone

from flask import current_app

from ..extensions import db
from ..models import AlertLog, StressEvent
from .email_service import send_stress_alert


def create_high_stress_event(user, reading, stress_result):
    event = StressEvent(
        user_id=user.id,
        sensor_reading_id=reading.id,
        stress_score=stress_result["stress_score"],
        stress_level=stress_result["stress_level"],
        reasons_json=json.dumps(stress_result["reasons"]),
    )
    db.session.add(event)
    db.session.flush()
    return event


def log_alert(user_id, stress_event_id, status, recipient_email=None, message=None):
    entry = AlertLog(
        user_id=user_id,
        stress_event_id=stress_event_id,
        status=status,
        recipient_email=recipient_email,
        message=message,
    )
    db.session.add(entry)
    return entry


def maybe_send_alert(user, stress_event, reading):
    cooldown_seconds = current_app.config["ALERT_COOLDOWN_SECONDS"]
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=cooldown_seconds)
    recent_alert = (
        AlertLog.query.filter(
            AlertLog.user_id == user.id,
            AlertLog.status == "sent",
            AlertLog.created_at >= cutoff,
        )
        .order_by(AlertLog.created_at.desc())
        .first()
    )

    if recent_alert:
        return log_alert(
            user.id,
            stress_event.id,
            "cooldown",
            recent_alert.recipient_email,
            "Alert suppressed by cooldown",
        )

    result = send_stress_alert(user, stress_event, reading)
    return log_alert(
        user.id,
        stress_event.id,
        result["status"],
        result.get("recipient_email"),
        result.get("message"),
    )
