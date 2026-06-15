import sqlite3
from pathlib import Path

from sqlalchemy import event, text
from sqlalchemy.engine import Engine

from .extensions import db

_PRAGMAS_REGISTERED = False


def prepare_sqlite_database(app):
    uri = app.config.get("SQLALCHEMY_DATABASE_URI", "")
    if not uri.startswith("sqlite:///") or uri == "sqlite:///:memory:":
        return

    raw_path = uri.replace("sqlite:///", "", 1)
    db_path = Path(raw_path)
    if not db_path.is_absolute():
        project_root = Path(app.root_path).parent
        db_path = project_root / db_path

    db_path.parent.mkdir(parents=True, exist_ok=True)
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"


def register_sqlite_pragmas():
    global _PRAGMAS_REGISTERED
    if _PRAGMAS_REGISTERED:
        return

    @event.listens_for(Engine, "connect")
    def set_sqlite_pragmas(dbapi_connection, _connection_record):
        if not isinstance(dbapi_connection, sqlite3.Connection):
            return
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

    _PRAGMAS_REGISTERED = True


def init_database(app):
    with app.app_context():
        db.create_all()
        uri = app.config.get("SQLALCHEMY_DATABASE_URI", "")
        if uri.startswith("sqlite"):
            db.session.execute(text("PRAGMA journal_mode=WAL"))
            db.session.execute(text("PRAGMA foreign_keys=ON"))
            db.session.execute(text("PRAGMA busy_timeout=5000"))
            db.session.commit()
