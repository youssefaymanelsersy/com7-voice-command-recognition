import json
import time

import numpy as np

from audio import record_audio, using_serial_input
from app_config import (
    QUIET_THRESHOLD,
    CLIPPING_THRESHOLD,
    MIN_COMMAND_LENGTH,
    MSG_SERIAL_MODE,
    MSG_TOO_QUIET,
    MSG_CLIPPING,
)
from export_samples_header import export_samples_zte_and_zcr_header
from features import extract_features

# ============================================================================
# WORKFLOW:
# 1. train() - Record 20 samples per command once, save to samples.json
# 2. Modify features.py or build_model_from_samples() as needed
# 3. retrain_from_saved_samples() - Rebuild model from saved samples.json
#    (NO re-recording needed, just rebuilds with new logic)
# ============================================================================

COMMANDS = ["on", "off", "start", "stop", "left", "right", "up", "down"]
SAMPLES_PER_COMMAND = 20
DATASET_FILE = "samples.json"
MODEL_FILE = "model.json"

FEATURE_ORDER = ["zcr", "energy", "length", "spectral_centroid"]

# Keep zero crossing rate as a strong signal.
FEATURE_WEIGHTS = {
    "zcr": 2.0,
    "energy": 0.4,
    "length": 1.8,
    "spectral_centroid": 1.8,
}


def _filter_command_outliers(vectors, z_limit=2.5):
    """Remove feature outliers inside a single command cluster."""
    if len(vectors) < 8:
        return vectors

    arr = np.array(vectors, dtype=np.float64)
    means = np.mean(arr, axis=0)
    stds = np.std(arr, axis=0) + 1e-6

    keep = []
    for row in arr:
        z = np.abs((row - means) / stds)
        if np.all(z <= z_limit):
            keep.append(row.tolist())

    # Avoid over-pruning small datasets.
    if len(keep) < max(10, len(vectors) // 2):
        return vectors
    return keep


def build_model_from_samples(sample_database):
    """Build a robust model from training samples using all recorded examples."""
    model = {
        "meta": {
            "feature_order": FEATURE_ORDER,
            "feature_weights": FEATURE_WEIGHTS,
            "k_neighbors": 3,
            "unknown_threshold": 7.0,
            "min_margin": 0.05,
        },
        "feature_stats": {},
        "command_stats": {},
        "commands": {},
    }

    all_vectors = []

    for cmd in COMMANDS:
        samples = sample_database.get(cmd, [])
        if not samples:
            continue

        vectors = []
        for sample in samples:
            vector = [float(sample.get(feature, 0.0)) for feature in FEATURE_ORDER]
            vectors.append(vector)

        vectors = _filter_command_outliers(vectors)

        for vector in vectors:
            all_vectors.append(vector)

        if not vectors:
            continue

        centroid = np.mean(vectors, axis=0).tolist()
        cmd_arr = np.array(vectors, dtype=np.float64)

        stats = {}
        for i, feature in enumerate(FEATURE_ORDER):
            stats[feature] = {
                "mean": float(np.mean(cmd_arr[:, i])),
                "std": float(np.std(cmd_arr[:, i]) + 1e-6),
            }

        model["command_stats"][cmd] = stats

        model["commands"][cmd] = {
            "samples": vectors,
            "centroid": [float(v) for v in centroid],
        }

    if all_vectors:
        all_arr = np.array(all_vectors, dtype=np.float64)

        # Use within-command spread so separability between commands is preserved.
        per_feature_within_stds = {f: [] for f in FEATURE_ORDER}
        for cmd, info in model["commands"].items():
            cmd_arr = np.array(info["samples"], dtype=np.float64)
            if len(cmd_arr) < 2:
                continue
            for i, feature in enumerate(FEATURE_ORDER):
                per_feature_within_stds[feature].append(float(np.std(cmd_arr[:, i])))

        for i, feature in enumerate(FEATURE_ORDER):
            within = per_feature_within_stds[feature]
            if within:
                std = float(np.mean(within))
            else:
                std = float(np.std(all_arr[:, i]))
            model["feature_stats"][feature] = {
                "mean": float(np.mean(all_arr[:, i])),
                "std": float(std + 1e-6),
            }

    return model


def save_model_from_samples(sample_database):
    model = build_model_from_samples(sample_database)

    with open(MODEL_FILE, "w") as f:
        json.dump(model, f, indent=2)

    return model


def retrain_from_saved_samples():
    try:
        with open(DATASET_FILE, "r") as f:
            sample_database = json.load(f)
    except FileNotFoundError:
        print(f"No saved sample file found at {DATASET_FILE}")
        return None

    model = save_model_from_samples(sample_database)
    export_samples_zte_and_zcr_header(DATASET_FILE)
    print(f"Rebuilt {MODEL_FILE} from saved samples in {DATASET_FILE}")
    return model


def train():
    sample_database = {}
    serial_mode = using_serial_input()

    if serial_mode:
        print(f"\n{MSG_SERIAL_MODE}\n")

    for cmd in COMMANDS:
        print(f"\n{'='*50}")
        print(f"Training '{cmd.upper()}' ({SAMPLES_PER_COMMAND} samples)")
        print('='*50)
        samples = []

        i = 0
        while i < SAMPLES_PER_COMMAND:
            if not serial_mode:
                input(f"Say '{cmd}' ({i + 1}/{SAMPLES_PER_COMMAND}): ")
            time.sleep(0.3)

            _, audio, max_amp = record_audio()

            if max_amp < QUIET_THRESHOLD:
                print(f"{MSG_TOO_QUIET} - try again")
                continue

            if (not serial_mode) and max_amp > CLIPPING_THRESHOLD:
                print(f"{MSG_CLIPPING} - retry")
                continue

            zcr, energy, length, centroid = extract_features(audio)

            if length < MIN_COMMAND_LENGTH:
                print("Too short - speak clearly")
                continue

            print(f"✓ Accepted")

            samples.append({
                "zcr": float(zcr),
                "energy": float(energy),
                "length": float(length),
                "spectral_centroid": float(centroid),
            })
            i += 1

        sample_database[cmd] = samples

    with open(DATASET_FILE, "w") as f:
        json.dump(sample_database, f, indent=2)

    export_samples_zte_and_zcr_header(DATASET_FILE)

    save_model_from_samples(sample_database)

    print(f"\n{'='*50}")
    print("✅ Training complete")
    print(f"Saved: {DATASET_FILE}, {MODEL_FILE}")
    print('='*50)
