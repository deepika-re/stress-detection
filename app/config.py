import os


def normalize_database_url(url):
    if not url:
        return url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+pg8000://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+pg8000://", 1)
    return url


class Config:
    SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "dev-change-me")
    SQLALCHEMY_DATABASE_URI = normalize_database_url(os.environ.get("DATABASE_URL"))
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = os.environ.get("SESSION_COOKIE_SAMESITE", "Lax")
    SESSION_COOKIE_SECURE = os.environ.get("SESSION_COOKIE_SECURE", "false").lower() == "true"
    WTF_CSRF_TIME_LIMIT = None

    SOCKETIO_ASYNC_MODE = os.environ.get("SOCKETIO_ASYNC_MODE", "threading")
    SOCKETIO_CORS_ALLOWED_ORIGINS = os.environ.get("SOCKETIO_CORS_ALLOWED_ORIGINS") or None

    ALERT_COOLDOWN_SECONDS = int(os.environ.get("ALERT_COOLDOWN_SECONDS", "300"))
    FACE_SCORE_MAX_AGE_SECONDS = int(os.environ.get("FACE_SCORE_MAX_AGE_SECONDS", "60"))
    FACE_INFERENCE_MIN_INTERVAL_SECONDS = float(os.environ.get("FACE_INFERENCE_MIN_INTERVAL_SECONDS", "1.4"))

    HF_TOKEN = os.environ.get("HF_TOKEN")
    HF_INFERENCE_PROVIDER = os.environ.get("HF_INFERENCE_PROVIDER", "hf-inference")
    HF_FACE_EXPRESSION_MODEL = os.environ.get("HF_FACE_EXPRESSION_MODEL", "trpakov/vit-face-expression")
    HF_INFERENCE_TIMEOUT_SECONDS = float(os.environ.get("HF_INFERENCE_TIMEOUT_SECONDS", "30"))
    HF_DIRECT_API_FALLBACK = os.environ.get("HF_DIRECT_API_FALLBACK", "true").lower() == "true"
    FACE_EWMA_ALPHA = float(os.environ.get("FACE_EWMA_ALPHA", "0.55"))

    MAIL_SERVER = os.environ.get("MAIL_SERVER", "smtp.gmail.com")
    MAIL_PORT = int(os.environ.get("MAIL_PORT", "587"))
    MAIL_USE_TLS = os.environ.get("MAIL_USE_TLS", "true").lower() == "true"
    MAIL_USERNAME = os.environ.get("MAIL_USERNAME") or os.environ.get("EMAIL_USER")
    MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD") or os.environ.get("EMAIL_PASS")
    MAIL_DEFAULT_SENDER = os.environ.get("MAIL_DEFAULT_SENDER") or MAIL_USERNAME
