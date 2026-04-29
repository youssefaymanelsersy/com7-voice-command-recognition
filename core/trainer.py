import json
import time
from pathlib import Path

import numpy as np

from core.app_config import (
    QUIET_THRESHOLD,
    CLIPPING_THRESHOLD,
    MIN_COMMAND_LENGTH,
    MSG_SERIAL_MODE,
    MSG_TOO_QUIET,
    MSG_CLIPPING,
)
from core.export_model_header import export_voice_model_header
from core.export_samples_header import export_samples_zte_and_zcr_header
from core.features import (
    FRAME_FEATURE_ORDER,
    average_recording_features,
    extract_features,
    extract_frame_features,
    flatten_frame_features,
)

# ============================================================================
# WORKFLOW:
# 1. train() - Record 20 samples per command once, save to samples.json
# 2. Modify trainer/scoring logic as needed
# 3. retrain_from_saved_samples() - Rebuild model from saved extracted features
#    (NO re-recording needed, but this does not recreate raw-waveform features)
# ============================================================================

COMMANDS = ["on", "off", "start", "stop", "left", "right", "up", "down"]
SAMPLES_PER_COMMAND = 20
DATASET_FILE = "data/samples.json"
FRAME_AVERAGE_FILE = "data/frame_feature_averages.json"
MODEL_FILE = "models/model.json"
ZCR_ENERGY_MODEL_FILE = "models/model_zcr_energy.json"

FEATURE_ORDER = FRAME_FEATURE_ORDER + ["length", "spectral_centroid"]
ZCR_ENERGY_FEATURE_ORDER = FRAME_FEATURE_ORDER

# Keep zero crossing rate as a strong signal. STE needs sufficient weight (1.0, not 0.4)
# to contribute meaningfully after normalization by its large standard deviations (~8 billion).
FEATURE_WEIGHTS = {
    **{feature: 2.0 for feature in FRAME_FEATURE_ORDER if feature.startswith("zcr_")},
    **{feature: 1.0 for feature in FRAME_FEATURE_ORDER if feature.startswith("ste_")},
    "length": 1.8,
    "spectral_centroid": 1.8,
}

ZCR_ENERGY_FEATURE_WEIGHTS = {
    **{feature: 2.0 for feature in FRAME_FEATURE_ORDER if feature.startswith("zcr_")},
    **{feature: 1.0 for feature in FRAME_FEATURE_ORDER if feature.startswith("ste_")},
}


def _rebuild_frame_averages(sample_database):
    """Build zcr_avg/ste_avg per command from stored recording frame features."""
    out = {}
    has_frame_data = False
    for cmd in COMMANDS:
        samples = sample_database.get(cmd, [])
        frame_ready = [
            sample
            for sample in samples
            if isinstance(sample, dict)
            and "zcr_features" in sample
            and "ste_features" in sample
        ]
        has_frame_data = has_frame_data or bool(frame_ready)
        out[cmd] = average_recording_features(frame_ready)
    return out if has_frame_data else None


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


def _sample_feature_value(sample, feature):
    flattened = None

    if "zcr_features" in sample or "ste_features" in sample:
        flattened = flatten_frame_features(sample)
        if feature in flattened:
            return float(flattened[feature])

    if feature in sample:
        return float(sample.get(feature, 0.0))

    return 0.0


def sample_to_feature_vector(sample, feature_order):
    return [_sample_feature_value(sample, feature) for feature in feature_order]


def build_model_from_samples(
    sample_database,
    feature_order=None,
    feature_weights=None,
    feature_floors=None,
    k_neighbors=3,
    unknown_threshold=60.0,
    min_margin=2.0,
):
    """Build a robust model from training samples using all recorded examples."""
    feature_order = list(feature_order or FEATURE_ORDER)
    feature_weights = dict(feature_weights or FEATURE_WEIGHTS)
    model = {
        "meta": {
            "feature_order": feature_order,
            "feature_weights": feature_weights,
            "feature_floors": dict(feature_floors or {}),
            "k_neighbors": k_neighbors,
            "unknown_threshold": unknown_threshold,
            "min_margin": min_margin,
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
            vector = sample_to_feature_vector(sample, feature_order)
            vectors.append(vector)

        vectors = _filter_command_outliers(vectors)

        for vector in vectors:
            all_vectors.append(vector)

        if not vectors:
            continue

        centroid = np.mean(vectors, axis=0).tolist()
        cmd_arr = np.array(vectors, dtype=np.float64)

        stats = {}
        for i, feature in enumerate(feature_order):
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
        per_feature_within_stds = {f: [] for f in feature_order}
        for cmd, info in model["commands"].items():
            cmd_arr = np.array(info["samples"], dtype=np.float64)
            if len(cmd_arr) < 2:
                continue
            for i, feature in enumerate(feature_order):
                per_feature_within_stds[feature].append(float(np.std(cmd_arr[:, i])))

        for i, feature in enumerate(feature_order):
            within = per_feature_within_stds[feature]
            if within:
                std = float(np.mean(within))
            else:
                std = float(np.std(all_arr[:, i]))
            # Use relative epsilon for large values, absolute for small ones
            mean_val = float(np.mean(all_arr[:, i]))
            if abs(mean_val) > 1e6:
                epsilon = max(std * 1e-6, 1e-9)  # Relative epsilon for large values
            else:
                epsilon = 1e-6  # Absolute epsilon for small values
            model["feature_stats"][feature] = {
                "mean": mean_val,
                "std": float(std + epsilon),
            }

    return model


def save_model_from_samples(
    sample_database,
    model_file=MODEL_FILE,
    feature_order=None,
    feature_weights=None,
    feature_floors=None,
    k_neighbors=3,
    unknown_threshold=60.0,
    min_margin=2.0,
):
    model = build_model_from_samples(
        sample_database,
        feature_order=feature_order,
        feature_weights=feature_weights,
        feature_floors=feature_floors,
        k_neighbors=k_neighbors,
        unknown_threshold=unknown_threshold,
        min_margin=min_margin,
    )

    Path(model_file).parent.mkdir(parents=True, exist_ok=True)
    with open(model_file, "w") as f:
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

    frame_averages = _rebuild_frame_averages(sample_database)
    if frame_averages is not None:
        Path(FRAME_AVERAGE_FILE).parent.mkdir(parents=True, exist_ok=True)
        with open(FRAME_AVERAGE_FILE, "w") as f:
            json.dump(frame_averages, f, indent=2)

    export_samples_zte_and_zcr_header(DATASET_FILE)
    export_voice_model_header(MODEL_FILE)
    print(f"Rebuilt {MODEL_FILE} from saved extracted features in {DATASET_FILE}")
    return model


def train():
    from audio import record_audio, using_serial_input

    sample_database = {}
    word_averages = {}
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
            frame_features = extract_frame_features(audio)

            if length < MIN_COMMAND_LENGTH:
                print("Too short - speak clearly")
                continue

            print(f"✓ Accepted")

            samples.append({
                "zcr": int(zcr),
                "energy": int(energy),
                "length": int(length),
                "spectral_centroid": int(centroid),
                # Store intermediate frame-wise features per recording.
                "zcr_features": [int(v) for v in frame_features["zcr_features"]],
                "ste_features": [int(v) for v in frame_features["ste_features"]],
            })
            i += 1

        sample_database[cmd] = samples
        word_averages[cmd] = average_recording_features(samples)

    Path(DATASET_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_FILE, "w") as f:
        json.dump(sample_database, f, indent=2)

    if any(samples and "zcr_features" in samples[0] for samples in sample_database.values()):
        with open(FRAME_AVERAGE_FILE, "w") as f:
            json.dump(word_averages, f, indent=2)

    export_samples_zte_and_zcr_header(DATASET_FILE)

    save_model_from_samples(sample_database)
    export_voice_model_header(MODEL_FILE)

    print(f"\n{'='*50}")
    print("✅ Training complete")
    print(f"Saved: {DATASET_FILE}, {FRAME_AVERAGE_FILE}, {MODEL_FILE}")
    print('='*50)
