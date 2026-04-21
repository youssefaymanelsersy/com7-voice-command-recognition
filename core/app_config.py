"""Shared runtime settings, thresholds, and user-facing messages."""

import os


def _env_int(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_optional_int(name, default=None):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


AUDIO_SAMPLE_RATE = _env_int("VOICE_AUDIO_SAMPLE_RATE", 8000)
RECORD_DURATION_SEC = _env_float("VOICE_RECORD_DURATION_SEC", 1.0)
DEVICE_ID = _env_optional_int("VOICE_DEVICE_ID")
USB_DEVICE_KEYWORDS = tuple(
    item.strip().lower()
    for item in os.getenv("VOICE_USB_DEVICE_KEYWORDS", "usb,codec,audio").split(",")
    if item.strip()
)

SERIAL_PORT = os.getenv("VOICE_SERIAL_PORT", "COM7")
SERIAL_BAUDRATE = _env_int("VOICE_SERIAL_BAUDRATE", 115200 * 2)
SERIAL_SAMPLE_WIDTH_BYTES = _env_int("VOICE_SERIAL_SAMPLE_WIDTH_BYTES", 1)
SERIAL_START_POLL_SEC = _env_float("VOICE_SERIAL_START_POLL_SEC", 0.01)
SERIAL_READ_CHUNK_SIZE = _env_int("VOICE_SERIAL_READ_CHUNK_SIZE", 512)
SERIAL_AFTER_START_TIMEOUT_SEC = _env_float("VOICE_SERIAL_AFTER_START_TIMEOUT_SEC", 8.0)
MODEL_FILE = os.getenv("VOICE_MODEL_FILE", "models/model.json")

QUIET_THRESHOLD = 0.08
CLIPPING_THRESHOLD = 0.95
MIN_COMMAND_LENGTH = 4000

MSG_SERIAL_MODE = "Serial mode active on COM input. Waiting for board audio automatically..."
MSG_TOO_QUIET = "Too quiet, speak louder"
MSG_CLIPPING = "Clipping detected, speak a bit softer"
MSG_INVALID_CHOICE = "Invalid choice"
