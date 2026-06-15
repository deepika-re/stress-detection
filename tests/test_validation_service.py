from datetime import datetime, timezone

import pytest

from app.services.validation_service import ValidationError, validate_reading_payload


def valid_reading(**overrides):
    payload = {
        "stream_id": "stream-test",
        "seq": 1,
        "heart_rate": 82,
        "gsr": 0.63,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source": "browser_serial",
    }
    payload.update(overrides)
    return payload


def test_validate_reading_accepts_expected_payload():
    result = validate_reading_payload(valid_reading())

    assert result["heart_rate"] == 82
    assert result["gsr"] == 0.63
    assert result["source"] == "browser_serial"


@pytest.mark.parametrize(
    "field,value",
    [
        ("heart_rate", 34),
        ("heart_rate", 221),
        ("gsr", -0.1),
        ("gsr", 21),
        ("source", "manual"),
    ],
)
def test_validate_reading_rejects_invalid_ranges(field, value):
    with pytest.raises(ValidationError):
        validate_reading_payload(valid_reading(**{field: value}))
