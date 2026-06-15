import os

from app import create_app
from app.extensions import socketio

app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "0.0.0.0")
    socketio.run(
        app,
        host=host,
        port=port,
        debug=app.config["DEBUG"],
        allow_unsafe_werkzeug=True,
    )
