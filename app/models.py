from datetime import datetime, timezone

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from .extensions import db


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(255), nullable=False, unique=True, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    phone = db.Column(db.String(40))
    caregiver_email = db.Column(db.String(255))
    caregiver_phone = db.Column(db.String(40))
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False)

    streams = db.relationship("BrowserStream", back_populates="user", cascade="all, delete-orphan")
    readings = db.relationship("SensorReading", back_populates="user", cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class BrowserStream(db.Model):
    __tablename__ = "browser_streams"

    id = db.Column(db.Integer, primary_key=True)
    stream_id = db.Column(db.String(80), nullable=False, unique=True, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    status = db.Column(db.String(30), nullable=False, default="started")
    last_seq_acked = db.Column(db.Integer, default=-1, nullable=False)
    dropped_count = db.Column(db.Integer, default=0, nullable=False)
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False)
    started_at = db.Column(db.DateTime, default=utcnow, nullable=False)
    stopped_at = db.Column(db.DateTime)

    user = db.relationship("User", back_populates="streams")
    readings = db.relationship("SensorReading", back_populates="stream")


class SensorReading(db.Model):
    __tablename__ = "sensor_readings"
    __table_args__ = (
        db.UniqueConstraint("stream_id", "seq", name="uq_sensor_readings_stream_seq"),
        db.Index("ix_sensor_readings_user_captured", "user_id", "captured_at"),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    browser_stream_id = db.Column(db.Integer, db.ForeignKey("browser_streams.id", ondelete="SET NULL"))
    stream_id = db.Column(db.String(80), nullable=False)
    seq = db.Column(db.Integer, nullable=False)
    heart_rate = db.Column(db.Float, nullable=False)
    gsr = db.Column(db.Float, nullable=False)
    source = db.Column(db.String(40), nullable=False, default="browser_serial")
    stress_score = db.Column(db.Integer, nullable=False)
    stress_level = db.Column(db.String(20), nullable=False)
    reasons_json = db.Column(db.Text, nullable=False, default="[]")
    captured_at = db.Column(db.DateTime, nullable=False)
    received_at = db.Column(db.DateTime, default=utcnow, nullable=False)

    user = db.relationship("User", back_populates="readings")
    stream = db.relationship("BrowserStream", back_populates="readings")
    stress_events = db.relationship("StressEvent", back_populates="reading")


class FaceScore(db.Model):
    __tablename__ = "face_scores"
    __table_args__ = (db.Index("ix_face_scores_user_captured", "user_id", "captured_at"),)

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    score = db.Column(db.Float, nullable=False)
    source = db.Column(db.String(40), nullable=False, default="face_model")
    captured_at = db.Column(db.DateTime, nullable=False)
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False)


class StressEvent(db.Model):
    __tablename__ = "stress_events"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    sensor_reading_id = db.Column(db.Integer, db.ForeignKey("sensor_readings.id", ondelete="SET NULL"))
    stress_score = db.Column(db.Integer, nullable=False)
    stress_level = db.Column(db.String(20), nullable=False)
    reasons_json = db.Column(db.Text, nullable=False, default="[]")
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False)

    reading = db.relationship("SensorReading", back_populates="stress_events")
    alert_logs = db.relationship("AlertLog", back_populates="stress_event")


class AlertLog(db.Model):
    __tablename__ = "alert_logs"
    __table_args__ = (db.Index("ix_alert_logs_user_created", "user_id", "created_at"),)

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    stress_event_id = db.Column(db.Integer, db.ForeignKey("stress_events.id", ondelete="SET NULL"))
    status = db.Column(db.String(30), nullable=False)
    recipient_email = db.Column(db.String(255))
    message = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=utcnow, nullable=False)

    stress_event = db.relationship("StressEvent", back_populates="alert_logs")
