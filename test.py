"""Unified test menu for audio input and recognition checks."""

import json

from audio import print_input_devices, record_audio, using_serial_input
from app_config import (
    QUIET_THRESHOLD,
    CLIPPING_THRESHOLD,
    MSG_SERIAL_MODE,
    MSG_TOO_QUIET,
    MSG_CLIPPING,
    MSG_INVALID_CHOICE,
)
from features import extract_features
from recognize import recognize

COMMANDS = ["on", "off", "start", "stop", "left", "right", "up", "down"]
TRIALS_PER_COMMAND = 10
MODEL_FILE = "model.json"


def _load_model():
    try:
        with open(MODEL_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Model file not found: {MODEL_FILE}")
        return None


def _prompt_if_needed(message):
    if not using_serial_input():
        input(message)


def test_input_level():
    print("\nTesting input level...")
    if using_serial_input():
        print(MSG_SERIAL_MODE)
    _prompt_if_needed("Press Enter, then say something loud: ")

    _, _, max_amp = record_audio()
    if max_amp < QUIET_THRESHOLD:
        print(MSG_TOO_QUIET)
        return False

    print(f"Input working (level: {max_amp:.4f})")
    return True


def test_features():
    print("\nTesting feature extraction...")
    _prompt_if_needed("Press Enter, then say a command: ")

    _, audio, max_amp = record_audio()
    if max_amp < QUIET_THRESHOLD:
        print(MSG_TOO_QUIET)
        return False

    zcr, energy, length, centroid = extract_features(audio)
    print(f"Features: ZCR={zcr:.4f}, Energy={energy:.4f}, Length={length:.0f}, Centroid={centroid:.0f}")
    print("Features extracted")
    return True


def test_model():
    print("\nTesting model loading...")
    model = _load_model()
    if model is None:
        return False
    print(f"Model loaded: {list(model.keys())}")
    return True


def test_recognition_once():
    print("\nTesting one-shot recognition...")
    model = _load_model()
    if model is None:
        return False

    _prompt_if_needed("Press Enter, then say a command: ")
    _, audio, max_amp = record_audio()

    if max_amp < QUIET_THRESHOLD:
        print(MSG_TOO_QUIET)
        return False

    result = recognize(audio, model)
    print(f"Recognition result: {result}")
    return True


def run_quick_suite():
    print("\n" + "=" * 50)
    print("QUICK SYSTEM TEST")
    print("=" * 50)

    results = [
        ("Input", test_input_level()),
        ("Features", test_features()),
        ("Model", test_model()),
        ("Recognition", test_recognition_once()),
    ]

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}")
    print("=" * 50)


def run_one_shot_capture():
    print("\n" + "=" * 50)
    print("INPUT CAPTURE TEST")
    print("=" * 50)
    print_input_devices()
    print()

    device = input("Optional device (index or COM port, Enter for default): ").strip()
    if not device:
        device = None
    elif device.isdigit():
        device = int(device)

    _, _, max_amp = record_audio(device=device)
    print(f"Captured max amplitude: {max_amp:.6f}")


def _choose_command():
    print("\nAvailable commands:")
    for index, command in enumerate(COMMANDS, start=1):
        print(f"  {index}. {command}")

    while True:
        choice = input("\nEnter the word you want to test: ").strip().lower()
        if choice in COMMANDS:
            return choice
        if choice.isdigit() and 1 <= int(choice) <= len(COMMANDS):
            return COMMANDS[int(choice) - 1]
        print("Please enter one of the listed commands.")


def run_command_trials():
    model = _load_model()
    if model is None:
        return

    target_word = _choose_command()
    serial_mode = using_serial_input()

    print(f"\nTesting '{target_word}' {TRIALS_PER_COMMAND} times")
    if serial_mode:
        print(MSG_SERIAL_MODE)
    else:
        print("Press Enter before each trial, then speak the target command.")
    print("Press Ctrl+C to stop early.\n")

    results = []
    try:
        for trial in range(1, TRIALS_PER_COMMAND + 1):
            if not serial_mode:
                input(f"Trial {trial}/{TRIALS_PER_COMMAND} - press Enter: ")

            _, audio, max_amp = record_audio()

            if max_amp < QUIET_THRESHOLD:
                prediction = "too_quiet"
            elif (not serial_mode) and max_amp > CLIPPING_THRESHOLD:
                prediction = "clipped"
            else:
                prediction = recognize(audio, model)

            results.append(prediction)
            print(f"Trial {trial}: {prediction}")
    except KeyboardInterrupt:
        print("\nStopped early.")

    if not results:
        print("No trials recorded.")
        return

    total = len(results)
    correct = sum(1 for item in results if item == target_word)
    unknown = sum(1 for item in results if item == "unknown")
    bad_audio = sum(1 for item in results if item in {"too_quiet", "clipped"})
    accuracy = (correct / total) * 100.0

    print("\n" + "=" * 60)
    print(f"SUMMARY FOR: {target_word}")
    print("=" * 60)
    print(f"Trials: {total}")
    print(f"Correct: {correct}")
    print(f"Unknown: {unknown}")
    print(f"Bad audio: {bad_audio}")
    print(f"Accuracy: {accuracy:.1f}%")
    print("Predictions:")
    print(", ".join(results))
    print("=" * 60)


def main():
    while True:
        print("\n" + "=" * 55)
        print("VOICE RECOGNITION TEST MENU")
        print("=" * 55)
        print("1. Quick system test")
        print("2. List input devices")
        print("3. One-shot input capture")
        print("4. Command evaluation trials")
        print("5. Exit")
        print("=" * 55)

        choice = input("Choose (1-5): ").strip()
        if choice == "1":
            run_quick_suite()
        elif choice == "2":
            print_input_devices()
        elif choice == "3":
            run_one_shot_capture()
        elif choice == "4":
            run_command_trials()
        elif choice == "5":
            break
        else:
            print(MSG_INVALID_CHOICE)


if __name__ == "__main__":
    main()