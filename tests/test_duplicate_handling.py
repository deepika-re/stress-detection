from datetime import datetime, timezone

import pytest

from app import create_app
from app.extensions import db
from app.models import FaceScore, SensorReading, User
from app.services.sensor_pipeline import process_sensor_batch


@pytest.fixture()
def flask_app(tmp_path):
    app = create_app(
        {
            "TESTING": True,
            "WTF_CSRF_ENABLED": False,
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'test.sqlite'}",
            "MAIL_USERNAME": None,
            "MAIL_PASSWORD": None,
            "MAIL_DEFAULT_SENDER": None,
        }
    )
    return app


def make_user():
    user = User(name="Test User", email="test@example.com")
    user.set_password("password")
    db.session.add(user)
    db.session.commit()
    return user


def make_reading(seq):
    return {
        "seq": seq,
        "heart_rate": 82,
        "gsr": 0.63,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source": "browser_serial",
    }


def test_duplicate_stream_seq_is_not_inserted_twice(flask_app):
    with flask_app.app_context():
        user = make_user()

        first = process_sensor_batch(
            user,
            "stream-duplicates",
            [make_reading(0), make_reading(0)],
        )
        second = process_sensor_batch(
            user,
            "stream-duplicates",
            [make_reading(0)],
        )

        assert first.accepted_count == 1
        assert first.duplicate_count == 1
        assert second.accepted_count == 0
        assert second.duplicate_count == 1
        assert SensorReading.query.count() == 1


def test_face_score_endpoint_keeps_only_latest_score(flask_app):
    with flask_app.app_context():
        user = make_user()
        user_id = user.id
        user_email = user.email

    client = flask_app.test_client()
    with client:
        response = client.post(
            "/auth/login",
            data={"email": user_email, "password": "password"},
            follow_redirects=False,
        )
        assert response.status_code == 302

        for score in (0.2, 0.7, 0.4):
            response = client.post("/api/face-score", json={"score": score})
            assert response.status_code == 200

    with flask_app.app_context():
        rows = FaceScore.query.filter_by(user_id=user_id).all()
        assert len(rows) == 1
        assert rows[0].score == 0.4
