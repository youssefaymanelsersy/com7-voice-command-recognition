import json
import time
from pathlib import Path

import numpy as np

from core.export_model_header import export_voice_model_header
from core.features import extract_features
from core.recognize import recognize_features


COMMANDS = ["on", "off", "start", "stop", "left", "right", "up", "down"]
FEATURE_ORDER = ["zcr", "energy", "length", "spectral_centroid"]

TEST_DATASET_FILE = "data/test_samples.json"
MAIN_DATASET_FILE = "data/samples.json"
MODEL_FILE = "models/model.json"
SAMPLES_PER_COMMAND = 10


def _filter_command_outliers(vectors, z_limit=2.5):
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

    if len(keep) < max(5, len(vectors) // 2):
        return vectors
    return keep


def build_model_from_samples(sample_database, params):
    model = {
        "meta": {
            "feature_order": FEATURE_ORDER,
            "feature_weights": params["feature_weights"],
            "feature_floors": params.get("feature_floors", {}),
            "k_neighbors": params["k_neighbors"],
            "unknown_threshold": params["unknown_threshold"],
            "min_margin": params["min_margin"],
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
            vectors.append([float(sample.get(f, 0.0)) for f in FEATURE_ORDER])

        vectors = _filter_command_outliers(vectors)
        if not vectors:
            continue

        all_vectors.extend(vectors)
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

    if not all_vectors:
        return model

    all_arr = np.array(all_vectors, dtype=np.float64)

    per_feature_within_stds = {f: [] for f in FEATURE_ORDER}
    for cmd, info in model["commands"].items():
        cmd_arr = np.array(info["samples"], dtype=np.float64)
        if len(cmd_arr) < 2:
            continue
        for i, feature in enumerate(FEATURE_ORDER):
            per_feature_within_stds[feature].append(float(np.std(cmd_arr[:, i])))

    for i, feature in enumerate(FEATURE_ORDER):
        within = per_feature_within_stds[feature]
        std = float(np.mean(within)) if within else float(np.std(all_arr[:, i]))
        model["feature_stats"][feature] = {
            "mean": float(np.mean(all_arr[:, i])),
            "std": float(std + 1e-6),
        }

    return model


def _normalize_vector(vector, feature_stats):
    floors = {
        "zcr": 0.01,
        "energy": 0.01,
        "length": 1500.0,
        "spectral_centroid": 100.0,
    }

    out = []
    for i, feature in enumerate(FEATURE_ORDER):
        stats = feature_stats.get(feature, {})
        mean = float(stats.get("mean", 0.0))
        std = max(float(stats.get("std", 0.0)), floors[feature])
        out.append((float(vector[i]) - mean) / std)
    return np.array(out, dtype=np.float64)


def predict_feature_vector(feature_vector, model):
    return recognize_features(feature_vector, model)


def loocv_score(sample_database, params):
    total = 0
    correct = 0
    unknown = 0

    for cmd in COMMANDS:
        samples = sample_database.get(cmd, [])
        for i in range(len(samples)):
            train_db = {}
            for c2 in COMMANDS:
                vals = sample_database.get(c2, [])
                if c2 == cmd:
                    train_db[c2] = vals[:i] + vals[i + 1 :]
                else:
                    train_db[c2] = vals[:]

            model = build_model_from_samples(train_db, params)
            feature_vec = [float(samples[i].get(f, 0.0)) for f in FEATURE_ORDER]
            pred = predict_feature_vector(feature_vec, model)

            total += 1
            if pred == "unknown":
                unknown += 1
            if pred == cmd:
                correct += 1

    acc = (correct / total) if total else 0.0
    unk = (unknown / total) if total else 0.0
    # Penalize unknown a bit to avoid trivial over-reject models.
    objective = acc - (0.35 * unk)
    return {
        "accuracy": acc,
        "unknown_rate": unk,
        "objective": objective,
    }


def search_best_params(sample_database):
    grid = []
    for zcr_w in [2.0, 2.4, 2.8, 3.2]:
        for len_w in [1.0, 1.4, 1.8]:
            for cent_w in [1.2, 1.8, 2.4]:
                for k in [3, 5, 7]:
                    for thr in [5.5, 6.0, 6.5, 7.0]:
                        for margin in [0.05, 0.08, 0.12]:
                            grid.append(
                                {
                                    "feature_weights": {
                                        "zcr": zcr_w,
                                        "energy": 0.4,
                                        "length": len_w,
                                        "spectral_centroid": cent_w,
                                    },
                                    "k_neighbors": k,
                                    "unknown_threshold": thr,
                                    "min_margin": margin,
                                }
                            )

    best = None
    best_metrics = None
    for idx, params in enumerate(grid, start=1):
        metrics = loocv_score(sample_database, params)
        if best is None or metrics["objective"] > best_metrics["objective"]:
            best = params
            best_metrics = metrics

        if idx % 60 == 0:
            print(f"Checked {idx}/{len(grid)} configs...")

    return best, best_metrics


def validation_score(train_database, validation_database, params):
    model = build_model_from_samples(train_database, params)
    total = 0
    correct = 0
    unknown = 0

    for cmd in COMMANDS:
        for sample in validation_database.get(cmd, []):
            feature_vec = [float(sample.get(f, 0.0)) for f in FEATURE_ORDER]
            pred = predict_feature_vector(feature_vec, model)
            total += 1
            if pred == cmd:
                correct += 1
            if pred == "unknown":
                unknown += 1

    acc = (correct / total) if total else 0.0
    unk = (unknown / total) if total else 0.0
    unknown_bonus = 0.05 * min(unk, 0.15)
    excessive_unknown_penalty = 0.25 * max(0.0, unk - 0.15)
    objective = acc + unknown_bonus - excessive_unknown_penalty
    return {
        "accuracy": acc,
        "unknown_rate": unk,
        "objective": objective,
    }


def search_best_validation_params(train_database, validation_database):
    grid = []
    for zcr_w in [1.5, 2.0, 2.5, 3.0]:
        for energy_w in [0.0, 0.4]:
            for len_w in [0.8, 1.4, 2.0]:
                for cent_w in [0.8, 1.4, 2.0, 2.8]:
                    for k in [1, 3, 5]:
                        for thr in [3.5, 5.0, 6.5, 8.0, 10.0]:
                            for margin in [0.02, 0.05, 0.1]:
                                grid.append(
                                    {
                                        "feature_weights": {
                                            "zcr": zcr_w,
                                            "energy": energy_w,
                                            "length": len_w,
                                            "spectral_centroid": cent_w,
                                        },
                                        "k_neighbors": k,
                                        "unknown_threshold": thr,
                                        "min_margin": margin,
                                    }
                                )

    best = None
    best_metrics = None
    for idx, params in enumerate(grid, start=1):
        metrics = validation_score(train_database, validation_database, params)
        if best is None or metrics["objective"] > best_metrics["objective"]:
            best = params
            best_metrics = metrics

        if idx % 500 == 0:
            print(f"Checked {idx}/{len(grid)} configs...")

    return best, best_metrics


def record_80_samples(output_file=TEST_DATASET_FILE):
    from audio import record_audio, using_serial_input

    serial_mode = using_serial_input()
    print("Recording 10 samples per command (80 total).")
    print("Speak clearly and consistently.\n")
    if serial_mode:
        print("Serial mode active on COM input. Waiting for board audio automatically...\n")
    db = {}

    for cmd in COMMANDS:
        db[cmd] = []
        print("=" * 55)
        print(f"Command: {cmd} ({SAMPLES_PER_COMMAND} samples)")
        print("=" * 55)

        i = 0
        while i < SAMPLES_PER_COMMAND:
            if not serial_mode:
                input(f"Say '{cmd}' ({i + 1}/{SAMPLES_PER_COMMAND}) and press Enter...")
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

            db[cmd].append(
                {
                    "zcr": float(zcr),
                    "energy": float(energy),
                    "length": float(length),
                    "spectral_centroid": float(centroid),
                }
            )
            print("Accepted")
            i += 1

    with open(output_file, "w") as f:
        json.dump(db, f, indent=2)

    print(f"\nSaved dataset to {output_file}")
    return db


def load_dataset(dataset_file=TEST_DATASET_FILE):
    with open(dataset_file, "r") as f:
        return json.load(f)


def save_model(model):
    Path(MODEL_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_FILE, "w") as f:
        json.dump(model, f, indent=2)
    export_voice_model_header(MODEL_FILE)


def main():
    print("\nAUTO TUNE WORKFLOW")
    print(f"Main dataset is preserved in: {MAIN_DATASET_FILE}")
    print(f"Autotune dataset file is: {TEST_DATASET_FILE}")
    print("1. Record 80 test samples (10 per command) and tune model")
    print("2. Tune model from existing data/test_samples.json")
    print("3. Tune model from main data/samples.json (read-only)")
    print("4. Tune model on data/samples.json and validate with data/test_samples.json")
    choice = input("Choose (1/2/3/4): ").strip()

    if choice == "1":
        db = record_80_samples(TEST_DATASET_FILE)
    elif choice == "2":
        db = load_dataset(TEST_DATASET_FILE)
    elif choice == "4":
        train_db = load_dataset(MAIN_DATASET_FILE)
        validation_db = load_dataset(TEST_DATASET_FILE)
        print("\nRunning validation parameter search...")
        best_params, metrics = search_best_validation_params(train_db, validation_db)
        print("\nBEST CONFIG")
        print(json.dumps(best_params, indent=2))
        print(
            "Validation estimate: "
            f"accuracy={metrics['accuracy']*100:.1f}% | "
            f"unknown={metrics['unknown_rate']*100:.1f}% | "
            f"objective={metrics['objective']:.4f}"
        )
        model = build_model_from_samples(train_db, best_params)
        save_model(model)
        print(f"Saved tuned model to {MODEL_FILE}")
        return
    else:
        db = load_dataset(MAIN_DATASET_FILE)

    print("\nRunning parameter search (this may take a few minutes)...")
    best_params, metrics = search_best_params(db)

    print("\nBEST CONFIG")
    print(json.dumps(best_params, indent=2))
    print(
        "Estimated LOOCV: "
        f"accuracy={metrics['accuracy']*100:.1f}% | "
        f"unknown={metrics['unknown_rate']*100:.1f}% | "
        f"objective={metrics['objective']:.4f}"
    )

    model = build_model_from_samples(db, best_params)
    save_model(model)
    print(f"Saved tuned model to {MODEL_FILE}")


if __name__ == "__main__":
    main()
