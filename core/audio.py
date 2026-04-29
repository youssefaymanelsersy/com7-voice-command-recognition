import numpy as np
import time

from core.app_config import (
    AUDIO_SAMPLE_RATE,
    RECORD_DURATION_SEC,
    DEVICE_ID,
    USB_DEVICE_KEYWORDS,
    SERIAL_PORT,
    SERIAL_BAUDRATE,
    SERIAL_SAMPLE_WIDTH_BYTES,
    SERIAL_START_POLL_SEC,
    SERIAL_READ_CHUNK_SIZE,
    SERIAL_AFTER_START_TIMEOUT_SEC,
)

try:
    import sounddevice as sd
except ImportError:
    sd = None

FS = AUDIO_SAMPLE_RATE
DURATION = RECORD_DURATION_SEC


def _is_serial_selector(value):
    return isinstance(value, str) and value.upper().startswith("COM")


def _record_audio_from_serial(port):
    """Record mono audio bytes from a serial-connected board."""
    try:
        import serial
    except ImportError as exc:
        import sys
        raise RuntimeError(
            "pyserial is required for COM port recording. Install with: "
            f"{sys.executable} -m pip install pyserial"
        ) from exc

    total_samples = int(DURATION * FS)
    expected_bytes = total_samples * SERIAL_SAMPLE_WIDTH_BYTES
    raw = bytearray()

    with serial.Serial(port, SERIAL_BAUDRATE, timeout=0.1) as ser:
        ser.reset_input_buffer()

        # Wait continuously for the board to start streaming bytes.
        print(f"Waiting for serial audio on {port}...")
        while ser.in_waiting == 0:
            time.sleep(SERIAL_START_POLL_SEC)

        deadline = time.time() + SERIAL_AFTER_START_TIMEOUT_SEC
        while len(raw) < expected_bytes:
            to_read = min(SERIAL_READ_CHUNK_SIZE, expected_bytes - len(raw))
            chunk = ser.read(to_read)
            if chunk:
                raw.extend(chunk)
                deadline = time.time() + SERIAL_AFTER_START_TIMEOUT_SEC
                continue

            if time.time() >= deadline:
                break

    received = len(raw)
    if received == 0:
        raise RuntimeError(
            f"No serial audio data received from {port}. Check board firmware, baud rate, and COM port."
        )

    if received < expected_bytes:
        missing = expected_bytes - received
        print(
            f"Warning: short serial frame from {port}: {received}/{expected_bytes} bytes. "
            f"Padding {missing} bytes with silence."
        )
        raw.extend(b"\x80" * missing if SERIAL_SAMPLE_WIDTH_BYTES == 1 else b"\x00" * missing)

    serial_bytes = bytes(raw[:expected_bytes])
    if SERIAL_SAMPLE_WIDTH_BYTES == 1:
        pcm_u8 = np.frombuffer(serial_bytes, dtype=np.uint8).astype(np.int32)
        pcm_i16 = (pcm_u8 - 128) << 8
        return pcm_i16.astype(np.int16)

    pcm = np.frombuffer(serial_bytes, dtype="<i2")
    return pcm.astype(np.int16)


def using_serial_input(device=None):
    """Return True when recording is configured to read from serial COM input."""
    if _is_serial_selector(device):
        return True
    if device is None and _is_serial_selector(SERIAL_PORT):
        return True
    return False

def list_input_devices():
    """Return all input-capable devices with useful metadata."""
    if sd is None:
        raise RuntimeError(
            "sounddevice is required for microphone input. Install with: "
            "python3 -m pip install sounddevice"
        )

    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    default_input, _ = sd.default.device

    inputs = []
    for idx, dev in enumerate(devices):
        max_in = int(dev.get("max_input_channels", 0))
        if max_in <= 0:
            continue

        hostapi_idx = int(dev.get("hostapi", -1))
        hostapi_name = (
            hostapis[hostapi_idx]["name"]
            if 0 <= hostapi_idx < len(hostapis)
            else "Unknown"
        )

        inputs.append(
            {
                "index": idx,
                "name": dev.get("name", "Unknown"),
                "hostapi": hostapi_name,
                "max_input_channels": max_in,
                "is_default": idx == default_input,
            }
        )

    return inputs


def print_input_devices():
    """Print all microphones to help pick the USB input device."""
    inputs = list_input_devices()
    if not inputs:
        print("No microphone/input devices found.")
        return

    print("Available microphones:")
    for dev in inputs:
        default_tag = " (default)" if dev["is_default"] else ""
        print(
            f"[{dev['index']}] {dev['name']} | "
            f"Host API: {dev['hostapi']} | "
            f"Input channels: {dev['max_input_channels']}{default_tag}"
        )


def _find_usb_input_device():
    """Find a likely USB microphone device by name."""
    for dev in list_input_devices():
        name = str(dev["name"]).lower()
        if any(keyword in name for keyword in USB_DEVICE_KEYWORDS):
            return dev["index"]
    return None


def _resolve_recording_device(device):
    """Resolve input device with priority: explicit -> configured -> USB board."""
    if device is not None:
        return device
    if DEVICE_ID is not None:
        return DEVICE_ID

    usb_device = _find_usb_input_device()
    if usb_device is not None:
        return usb_device

    raise RuntimeError(
        "No USB microphone board detected. Connect the board and use main.py (option 9 -> option 2) to see available input devices."
    )


def record_audio(device=None):
    serial_port = None
    if _is_serial_selector(device):
        serial_port = device
    elif device is None and _is_serial_selector(SERIAL_PORT):
        serial_port = SERIAL_PORT

    print("Recording now...")

    if serial_port is not None:
        audio = _record_audio_from_serial(serial_port)
    else:
        selected_device = _resolve_recording_device(device)

        # Fail early with clear error if selected input device is invalid.
        if selected_device is not None:
            try:
                sd.query_devices(selected_device, "input")
            except Exception as exc:
                raise ValueError(
                    f"Invalid input device '{selected_device}'. Use main.py (option 9 -> option 2) or print_input_devices() to see valid microphones."
                ) from exc

        audio = sd.rec(
            int(DURATION * FS),
            samplerate=FS,
            channels=1,
            device=selected_device,
            dtype='int16'
        )

        sd.wait()
        audio = audio.flatten()

    raw_max = int(np.max(np.abs(audio)))
    print("Max amplitude:", raw_max)

    print("Done recording")

    return audio, audio, raw_max
