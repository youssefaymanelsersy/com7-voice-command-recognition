"""Play processed samples framed as START/END integers over serial."""

import argparse

import numpy as np

from core.app_config import AUDIO_SAMPLE_RATE, PROCESSED_SIGNAL_TX_BAUDRATE, PROCESSED_SIGNAL_TX_PORT


def _read_processed_frame(port, baudrate):
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError(
            "pyserial is required. Install with: python -m pip install pyserial"
        ) from exc

    with serial.Serial(port, int(baudrate), timeout=1) as ser:
        samples = []
        in_frame = False

        while True:
            raw = ser.readline()
            if not raw:
                continue

            line = raw.decode("ascii", errors="ignore").strip()
            if not line:
                continue

            if line == "START":
                samples = []
                in_frame = True
                continue

            if line == "END":
                if in_frame and samples:
                    return np.array(samples, dtype=np.int16)
                samples = []
                in_frame = False
                continue

            if not in_frame:
                continue

            try:
                value = int(line)
            except ValueError:
                continue
            samples.append(max(-32768, min(32767, value)))


def main():
    parser = argparse.ArgumentParser(
        description="Listen on a serial port for processed START/END framed samples and play them back."
    )
    parser.add_argument(
        "--port",
        default=PROCESSED_SIGNAL_TX_PORT or None,
        help="Serial port carrying START/END framed processed samples.",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=PROCESSED_SIGNAL_TX_BAUDRATE,
        help=f"Serial baudrate (default: {PROCESSED_SIGNAL_TX_BAUDRATE})",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=AUDIO_SAMPLE_RATE,
        help=f"Playback sample rate in Hz (default: {AUDIO_SAMPLE_RATE})",
    )
    args = parser.parse_args()

    if not args.port:
        raise ValueError(
            "A serial port is required. Set VOICE_PROCESSED_SIGNAL_TX_PORT or pass --port COMx."
        )

    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "sounddevice is required. Install with: python -m pip install sounddevice"
        ) from exc

    print("Waiting for processed serial frames. Press Ctrl+C to stop.")
    try:
        while True:
            print(f"Listening on {args.port} for START/END framed samples...")
            audio = _read_processed_frame(args.port, args.baudrate)

            if audio.size == 0:
                print("Received an empty frame; waiting for the next one.")
                continue

            pcm = audio.astype(np.float32)
            peak = float(np.max(np.abs(pcm)))
            if peak > 0:
                pcm = pcm / peak

            print(f"Received {len(audio)} processed samples. Playing...")
            sd.play(pcm, samplerate=args.sample_rate, blocking=True)
            print("Playback complete.")
    except KeyboardInterrupt:
        print("\nInterrupted by user; exiting playback loop.")


if __name__ == "__main__":
    main()
