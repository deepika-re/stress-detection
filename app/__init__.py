from pathlib import Path

from flask import Flask, redirect, url_for

from .db_init import init_database, prepare_sqlite_database, register_sqlite_pragmas
from .extensions import csrf, db, login_manager, socketio

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):
        return False


def create_app(config_overrides=None):
    load_dotenv()
    register_sqlite_pragmas()

    from .config import Config
    from .models import User
    from .routes.api_routes import api_bp
    from .routes.auth_routes import auth_bp
    from .routes.dashboard_routes import dashboard_bp
    from .socket_events import register_socket_events

    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(Config)
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)

    if not app.config.get("SQLALCHEMY_DATABASE_URI"):
        db_path = Path(app.instance_path) / "stress_app.sqlite"
        app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"

    if config_overrides:
        app.config.update(config_overrides)

    prepare_sqlite_database(app)

    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)
    socketio.init_app(
        app,
        cors_allowed_origins=app.config["SOCKETIO_CORS_ALLOWED_ORIGINS"],
        async_mode=app.config["SOCKETIO_ASYNC_MODE"],
        manage_session=False,
    )

    login_manager.login_view = "auth.login"
    login_manager.login_message_category = "info"

    @login_manager.user_loader
    def load_user(user_id):
        try:
            return db.session.get(User, int(user_id))
        except (TypeError, ValueError):
            return None

    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(api_bp)
    csrf.exempt(api_bp)

    @app.get("/")
    def index():
        return redirect(url_for("dashboard.dashboard"))

    init_database(app)
    register_socket_events(socketio)

    return app
