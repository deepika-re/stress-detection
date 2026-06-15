import base64
import binascii
import json
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from flask import current_app

from .emotion_stress import calculate_emotion_stress

MAX_FACE_IMAGE_BYTES = 900_000
DATA_URL_MARKER = ";base64,"

# Thread-safe client cache: double-checked locking so only one InferenceClient
# is constructed per (provider, token, timeout) tuple even under concurrent
# requests.  A bare dict without a lock caused silent score drops under
# multi-threaded gunicorn workers when two threads raced on the first request.
_CLIENT_CACHE: dict = {}
_CLIENT_CACHE_LOCK = threading.Lock()


class FaceInferenceError(RuntimeError):
    def __init__(self, message: str, retry_after_seconds: int = 15) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


def decode_image_payload(value: object) -> bytes:
    if not isinstance(value, str) or not value.strip():
        raise FaceInferenceError("image is required")

    raw = value.strip()
    if DATA_URL_MARKER in raw:
        header, raw = raw.split(DATA_URL_MARKER, 1)
        if not header.lower().startswith("data:image/"):
            raise FaceInferenceError("image must be a data:image payload")

    try:
        image_bytes = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise FaceInferenceError("image must be base64 encoded") from exc

    if not image_bytes:
        raise FaceInferenceError("image is empty")
    if len(image_bytes) > MAX_FACE_IMAGE_BYTES:
        raise FaceInferenceError("image is too large")
    return image_bytes


def get_inference_client():
    """
    Return a cached InferenceClient for the current app config.

    Thread-safe: uses double-checked locking so the client is constructed
    exactly once per (provider, token, timeout) key even under concurrent
    first-requests.
    """
    token = current_app.config.get("HF_TOKEN")
    if not token:
        raise FaceInferenceError("HF_TOKEN is not configured")

    provider = current_app.config.get("HF_INFERENCE_PROVIDER", "hf-inference")
    timeout = current_app.config.get("HF_INFERENCE_TIMEOUT_SECONDS", 12)
    cache_key = (provider, token, timeout)

    # Fast path: already cached (no lock needed for reads after initial write).
    client = _CLIENT_CACHE.get(cache_key)
    if client is not None:
        return client

    with _CLIENT_CACHE_LOCK:
        # Re-check inside the lock: another thread may have populated the cache
        # while we were waiting.
        client = _CLIENT_CACHE.get(cache_key)
        if client is None:
            try:
                from huggingface_hub import InferenceClient
            except ImportError as exc:
                raise FaceInferenceError("huggingface_hub is not installed") from exc

            client = InferenceClient(
                provider=provider,
                api_key=token,
                timeout=timeout,
            )
            _CLIENT_CACHE[cache_key] = client

    return client


def sanitize_error(exc: BaseException) -> str:
    message = str(exc) or exc.__class__.__name__
    token = current_app.config.get("HF_TOKEN")
    if token:
        message = message.replace(token, "[redacted]")
    return " ".join(message.split())[:420]


def direct_api_classification(
    image_bytes: bytes,
    model: str,
    token: str,
    timeout: float,
) -> list:
    url = f"https://api-inference.huggingface.co/models/{model}"
    request = Request(
        url,
        data=image_bytes,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "image/jpeg",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except HTTPError as exc:
        # Parse retry-after for 429 so callers can back off correctly.
        retry_after = 15
        try:
            retry_after = int(exc.headers.get("Retry-After", 15))
        except (TypeError, ValueError):
            pass

        detail = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(detail)
            message = body.get("error") or detail
        except (TypeError, ValueError, json.JSONDecodeError):
            message = detail or exc.reason
        raise FaceInferenceError(
            f"Hugging Face HTTP {exc.code}: {message}",
            retry_after_seconds=retry_after,
        ) from exc
    except URLError as exc:
        raise FaceInferenceError(f"Hugging Face network error: {exc.reason}") from exc

    try:
        predictions = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise FaceInferenceError("Hugging Face returned non-JSON inference output") from exc

    if isinstance(predictions, dict) and predictions.get("error"):
        raise FaceInferenceError(f"Hugging Face error: {predictions['error']}")
    if not isinstance(predictions, list):
        raise FaceInferenceError("Hugging Face returned an unexpected inference shape")
    return predictions


def _is_network_or_api_error(exc: BaseException) -> bool:
    """
    Return True only for errors that indicate a network/API problem and
    therefore warrant a fallback attempt.

    Programming errors (AttributeError, TypeError, etc.) are *not* retriable
    and should propagate immediately so bugs surface during development.
    """
    # Always fall back on standard network/HTTP errors.
    if isinstance(exc, (URLError, HTTPError, ConnectionError, TimeoutError, OSError)):
        return True

    # huggingface_hub-specific errors (imported lazily to avoid hard dependency
    # at module level).
    try:
        from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
        if isinstance(exc, (HfHubHTTPError, RepositoryNotFoundError)):
            return True
    except ImportError:
        pass

    try:
        import requests
        if isinstance(exc, requests.exceptions.RequestException):
            return True
    except ImportError:
        pass

    return False


def classify_face_expression(image_bytes: bytes) -> dict:
    """
    Run face-expression image classification and return an emotion-stress result.

    Attempts the huggingface_hub InferenceClient first; falls back to a direct
    HTTP call to the Inference API only on genuine network/API failures.
    Programming errors are not silenced and will propagate as-is.
    """
    token = current_app.config.get("HF_TOKEN")
    if not token:
        raise FaceInferenceError("HF_TOKEN is not configured")

    model = current_app.config.get(
        "HF_FACE_EXPRESSION_MODEL", "trpakov/vit-face-expression"
    )
    provider = current_app.config.get("HF_INFERENCE_PROVIDER", "hf-inference")
    timeout = current_app.config.get("HF_INFERENCE_TIMEOUT_SECONDS", 30)
    started = time.perf_counter()
    provider_used = provider

    try:
        client = get_inference_client()
        try:
            predictions = client.image_classification(image_bytes, model=model, top_k=7)
        except TypeError:
            # Older SDK versions do not accept top_k.
            predictions = client.image_classification(image_bytes, model=model)

    except Exception as exc:
        # Only retry via the direct API for transient network / HTTP errors.
        # Let programming errors (AttributeError, KeyError, etc.) surface.
        if not _is_network_or_api_error(exc):
            raise

        client_error = sanitize_error(exc)
        if not current_app.config.get("HF_DIRECT_API_FALLBACK", True):
            raise FaceInferenceError(
                f"Hugging Face inference failed: {client_error}"
            ) from exc

        try:
            predictions = direct_api_classification(image_bytes, model, token, timeout)
            provider_used = "hf-inference-api"
        except FaceInferenceError as fallback_exc:
            fallback_error = sanitize_error(fallback_exc)
            raise FaceInferenceError(
                f"Hugging Face inference failed: {client_error}; "
                f"fallback failed: {fallback_error}"
            ) from exc

    result = calculate_emotion_stress(predictions)
    result["model"] = model
    result["provider"] = provider_used
    result["latency_ms"] = int((time.perf_counter() - started) * 1000)
    return result