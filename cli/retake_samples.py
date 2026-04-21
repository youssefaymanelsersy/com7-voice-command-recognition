import argparse
import json
import os
import time
from pathlib import Path

from core.audio import record_audio, using_serial_input
from core.features import extract_features


COMMANDS = ["on", "off", "start", "stop", "left", "right", "up", "down"]
DATASET_FILE = "data/test_samples.json"
SAMPLES_PER_COMMAND = 10


def load_dataset():
    if not os.path.exists(DATASET_FILE):
        return {command: [] for command in COMMANDS}

    with open(DATASET_FILE, "r") as f:
        data = json.load(f)

    for command in COMMANDS:
        data.setdefault(command, [])

    return data


def save_dataset(dataset):
    Path(DATASET_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_FILE, "w") as f:
        json.dump(dataset, f, indent=2)


def parse_commands(value):
    if not value:
        return []

    selected = []
    for item in value.split(","):
        name = item.strip().lower()
        if not name:
            continue
        if name.isdigit() and 1 <= int(name) <= len(COMMANDS):
            name = COMMANDS[int(name) - 1]
        if name not in COMMANDS:
            raise ValueError(f"Unknown command: {name}")
        if name not in selected:
            selected.append(name)
    return selected


def choose_commands_interactively():
    print("Available commands:")
    for index, command in enumerate(COMMANDS, start=1):
        print(f"  {index}. {command}")
    print("\nEnter commands to re-record as a comma-separated list.")
    print("Examples: off,left,right or 2,5,6")

    while True:
        try:
            selected = parse_commands(input("Commands to re-record: "))
        except ValueError as exc:
            print(exc)
            continue

        if selected:
            return selected
        print("Please choose at least one command.")


def record_for_command(command, sample_count):
    samples = []
    serial_mode = using_serial_input()
    print("\n" + "=" * 55)
    print(f"Re-recording '{command}' ({sample_count} samples)")
    print("=" * 55)
    if serial_mode:
        print("Serial mode active on COM input. Waiting for board audio automatically...")

    while len(samples) < sample_count:
        if not serial_mode:
            input(f"Say '{command}' ({len(samples) + 1}/{sample_count}) and press Enter...")
        time.sleep(0.3)

        _, audio, max_amp = record_audio()
        if max_amp < 0.03:
            print("Too quiet. Retry.")
            continue
        if (not serial_mode) and max_amp > 0.95:
            print("Clipping. Retry.")
            continue

        zcr, energy, length, centroid = extract_features(audio)
        if length < 4000:
            print("Too short/noisy. Retry.")
            continue

        samples.append(
            {
                "zcr": float(zcr),
                "energy": float(energy),
                "length": float(length),
                "spectral_centroid": float(centroid),
            }
        )
        print("Accepted")

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Re-record only selected commands into TEST dataset data/test_samples.json without touching training data/samples.json."
    )
    parser.add_argument(
        "commands",
        nargs="?",
        help="Comma-separated command list to re-record, for example: off,left,right",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=SAMPLES_PER_COMMAND,
        help="Number of samples to record per selected command (default: 10)",
    )
    args = parser.parse_args()

    if args.commands:
        selected = parse_commands(args.commands)
        if not selected:
            print("No valid commands provided.")
            return
    else:
        selected = choose_commands_interactively()

    dataset = load_dataset()

    print(f"\nSaving TEST updates into {DATASET_FILE}")
    print(f"Commands being replaced: {', '.join(selected)}")

    for command in selected:
        dataset[command] = record_for_command(command, args.count)

    save_dataset(dataset)
    print(f"\nUpdated TEST dataset {DATASET_FILE} successfully.")
    print("Use main.py option 4 (Autotune model) to rebuild/search parameters.")


if __name__ == "__main__":
    main()