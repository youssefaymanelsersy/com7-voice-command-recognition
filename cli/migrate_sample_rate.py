"""Rescale saved spectral-centroid features after changing sample rate."""

import argparse
import json

from core.app_config import AUDIO_SAMPLE_RATE


DEFAULT_OLD_SAMPLE_RATE = 44100


def migrate_file(path, old_sample_rate=DEFAULT_OLD_SAMPLE_RATE, new_sample_rate=AUDIO_SAMPLE_RATE):
    with open(path, "r") as f:
        dataset = json.load(f)

    scale = float(new_sample_rate) / float(old_sample_rate)
    changed = 0
    for samples in dataset.values():
        for sample in samples:
            if "spectral_centroid" in sample:
                sample["spectral_centroid"] = float(sample["spectral_centroid"]) * scale
                changed += 1

    with open(path, "w") as f:
        json.dump(dataset, f, indent=2)
        f.write("\n")

    return changed


def main():
    parser = argparse.ArgumentParser(
        description="Rescale saved spectral_centroid values between sample rates."
    )
    parser.add_argument("files", nargs="+", help="Dataset JSON files to migrate.")
    parser.add_argument("--old-rate", type=float, default=DEFAULT_OLD_SAMPLE_RATE)
    parser.add_argument("--new-rate", type=float, default=AUDIO_SAMPLE_RATE)
    args = parser.parse_args()

    for path in args.files:
        changed = migrate_file(path, args.old_rate, args.new_rate)
        print(
            f"{path}: rescaled {changed} spectral_centroid values "
            f"from {args.old_rate:g} Hz to {args.new_rate:g} Hz"
        )


if __name__ == "__main__":
    main()
