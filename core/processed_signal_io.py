"""Helpers for buffered processed-signal serial transmission."""

import numpy as np


def _to_int16_samples(samples):
    arr = np.asarray(samples)
    if arr.size == 0:
        return np.array([], dtype=np.int16)
    if np.issubdtype(arr.dtype, np.integer):
        return np.clip(arr.astype(np.int32), -32768, 32767).astype(np.int16)

    return np.clip(arr.astype(np.int32), -32768, 32767).astype(np.int16)


def build_processed_signal_frame(samples):
    """Build START/END framed payload only after full sample buffering."""
    pcm = _to_int16_samples(samples)

    lines = ["START"]
    lines.extend(str(int(value)) for value in pcm)
    lines.append("END")
    return "\n".join(lines) + "\n"


def transmit_processed_signal(samples, port, baudrate=115200):
    """Transmit an already-processed signal frame over serial in one pass."""
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError(
            "pyserial is required for processed-signal transmission. Install with: python -m pip install pyserial"
        ) from exc

    payload = build_processed_signal_frame(samples).encode("ascii")
    with serial.Serial(port, int(baudrate), timeout=1) as ser:
        ser.write(payload)
        ser.flush()
