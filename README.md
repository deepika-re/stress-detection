# Browser Serial Stress Monitor

A local-first Flask + JavaScript stress monitoring app. A logged-in user opens the dashboard, connects an Arduino/ESP32 from the browser with the Web Serial API, streams heart-rate/GSR readings over Socket.IO, optionally adds browser-camera facial emotion inference through Hugging Face, stores readings, calculates stress, updates the dashboard live, and logs email alert attempts for high stress.

## Data Flow

```text
Browser Web Serial
  -> parse HR/GSR lines
  -> in-memory buffer with seq numbers
  -> Socket.IO sensor_batch every 75 ms or 10 readings
  -> Flask validates current logged-in user session
  -> SQLite batch insert with UNIQUE(stream_id, seq)
  -> stress engine fuses HR/GSR with the latest face score
  -> alert service logs/sends high-stress alerts with cooldown
  -> Socket.IO sensor_ack + live_update
  -> dashboard reconciles UI and chart
```

The browser never sends `user_id`. The backend uses the authenticated session for all routes and socket events.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python run.py
```

Open:

```text
http://localhost:5000/dashboard
```

Register a user, log in, then click **Connect Device**.

## Arduino / ESP32 Serial Format

Send newline-delimited lines at `115200` baud:

```text
HR:82,GSR:0.63
HR: 94, GSR: 1.25
```

Malformed lines are ignored in the browser before they reach the backend.

## Browser Serial Notes

Web Serial works in supported Chromium-based browsers on `localhost` or HTTPS origins. If unsupported, the dashboard shows a Local Bridge fallback message instead of crashing.

## Socket.IO Events

Client sends:

- `start_stream`
- `stop_stream`
- `sensor_batch`

Server emits:

- `sensor_ack`
- `live_update`
- `stream_error`

Each reading has:

```json
{
  "stream_id": "...",
  "seq": 1,
  "heart_rate": 82,
  "gsr": 0.63,
  "captured_at": "2026-06-14T15:30:00.000Z",
  "source": "browser_serial"
}
```

## Facial Emotion Inference

Set `HF_TOKEN` to enable server-side Hugging Face inference with `trpakov/vit-face-expression`:

```text
HF_TOKEN=<your-hugging-face-token>
HF_INFERENCE_PROVIDER=hf-inference
HF_FACE_EXPRESSION_MODEL=trpakov/vit-face-expression
HF_INFERENCE_TIMEOUT_SECONDS=30
HF_DIRECT_API_FALLBACK=true
FACE_INFERENCE_MIN_INTERVAL_SECONDS=1.4
FACE_EWMA_ALPHA=0.55
```

The dashboard captures a small centered camera frame in the browser, downsizes it to `224x224`, posts it to `/api/face-inference`, and stores the smoothed `0..1` face stress score in `face_scores`. The backend maps all returned emotion probabilities into a stress score; it does not rely only on the top label.

Manual or external clients can still post a direct face score:

Authenticated clients can post a recent face score:

```http
POST /api/face-score
Content-Type: application/json

{"score": 0.42, "captured_at": "2026-06-14T15:30:00.000Z"}
```

Scores must be between `0` and `1`. Sensor scoring uses the latest face score from the last 60 seconds.

## SQLite

Local default:

```text
instance/stress_app.sqlite
```

The app enables:

- WAL mode
- foreign keys
- `busy_timeout`
- transactional batch inserts
- duplicate protection with `UNIQUE(stream_id, seq)`

## Railway Deployment

Set environment variables:

```text
FLASK_SECRET_KEY=<long-random-secret>
DATABASE_URL=<Railway PostgreSQL DATABASE_URL>
SESSION_COOKIE_SECURE=true
HF_TOKEN=<your-hugging-face-token>
MAIL_USERNAME=<optional>
MAIL_PASSWORD=<optional>
MAIL_DEFAULT_SENDER=<optional>
```

The included `Procfile` runs:

```text
gunicorn -w 1 --threads 100 --timeout 120 run:app --bind 0.0.0.0:$PORT
```

For production, use Railway PostgreSQL. The app normalizes Railway's `postgresql://` URL to SQLAlchemy's `postgresql+pg8000://` dialect, so the pure-Python `pg8000` driver is enough. Disable Railway Serverless/app sleeping for the live dashboard to avoid cold starts during sensor sessions. SQLite remains useful for local demos and simple single-instance experiments.

## Tests

Python tests:

```bash
pytest
```

JavaScript parser test:

```bash
node --test tests/serial_line_parser.test.mjs
```
