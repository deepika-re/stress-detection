from datetime import datetime, timezone

from flask import request
from flask_login import current_user
from flask_socketio import emit, join_room, leave_room

from .extensions import db
from .models import BrowserStream
from .services.sensor_pipeline import process_sensor_batch, user_room
from .services.validation_service import ValidationError, validate_stream_id


def register_socket_events(socketio):
    @socketio.on("connect")
    def handle_connect():
        if not current_user.is_authenticated:
            return False
        join_room(user_room(current_user.id))

    @socketio.on("disconnect")
    def handle_disconnect():
        if current_user.is_authenticated:
            leave_room(user_room(current_user.id))

    @socketio.on("start_stream")
    def handle_start_stream(payload):
        if not current_user.is_authenticated:
            emit("stream_error", {"message": "Authentication required"})
            return

        payload = payload or {}
        try:
            stream_id = validate_stream_id(payload.get("stream_id"))
            stream = BrowserStream.query.filter_by(stream_id=stream_id).first()
            if stream and stream.user_id != current_user.id:
                raise ValidationError("stream_id belongs to another user")
            if not stream:
                stream = BrowserStream(stream_id=stream_id, user_id=current_user.id)
                db.session.add(stream)
            stream.status = "started"
            stream.started_at = datetime.now(timezone.utc).replace(tzinfo=None)
            stream.stopped_at = None
            db.session.commit()
            emit("sensor_ack", {
                "stream_id": stream_id,
                "last_accepted_seq": stream.last_seq_acked,
                "accepted_count": 0,
                "duplicate_count": 0,
                "rejected_count": 0,
                "dropped_count": stream.dropped_count,
                "errors": [],
            })
        except ValidationError as exc:
            emit("stream_error", {"message": str(exc)})

    @socketio.on("stop_stream")
    def handle_stop_stream(payload):
        if not current_user.is_authenticated:
            emit("stream_error", {"message": "Authentication required"})
            return

        payload = payload or {}
        try:
            stream_id = validate_stream_id(payload.get("stream_id"))
            stream = BrowserStream.query.filter_by(stream_id=stream_id, user_id=current_user.id).first()
            if stream:
                stream.status = "stopped"
                stream.stopped_at = datetime.now(timezone.utc).replace(tzinfo=None)
                db.session.commit()
            emit("sensor_ack", {
                "stream_id": stream_id,
                "last_accepted_seq": stream.last_seq_acked if stream else -1,
                "accepted_count": 0,
                "duplicate_count": 0,
                "rejected_count": 0,
                "dropped_count": stream.dropped_count if stream else 0,
                "errors": [],
            })
        except ValidationError as exc:
            emit("stream_error", {"message": str(exc)})

    @socketio.on("sensor_batch")
    def handle_sensor_batch(payload):
        if not current_user.is_authenticated:
            emit("stream_error", {"message": "Authentication required"})
            return

        payload = payload or {}
        readings = payload.get("readings") or []
        if not isinstance(readings, list):
            emit("stream_error", {"message": "readings must be a list"})
            return

        try:
            result = process_sensor_batch(
                current_user,
                payload.get("stream_id"),
                readings[:50],
                dropped_count=payload.get("dropped_count", 0),
            )
            emit("sensor_ack", result.to_ack(), to=request.sid)
            if result.latest_update:
                socketio.emit("live_update", result.latest_update, room=user_room(current_user.id))
        except ValidationError as exc:
            emit("stream_error", {"message": str(exc)}, to=request.sid)
        except Exception as exc:
            db.session.rollback()
            emit("stream_error", {"message": "Failed to process sensor batch"}, to=request.sid)
            raise exc
