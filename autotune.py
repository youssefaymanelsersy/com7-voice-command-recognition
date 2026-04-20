import json
import time

import numpy as np

from audio import record_audio, using_serial_input
from features import extract_features


COMMANDS = ["on", "off", "start", "stop", "left", "right", "up", "down"]
FEATURE_ORDER = ["zcr", "energy", "length", "spectral_centroid"]

TEST_DATASET_FILE = "test_samples.json"
MAIN_DATASET_FILE = "samples.json"
MODEL_FILE = "model.json"
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
            "k_neighbors": params["k_neighbors"],
            "unknown_threshold": params["unknown_threshold"],
            "min_margin": params["min_margin"],
        },
        "feature_stats": {},
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
    meta = model.get("meta", {})
    commands = model.get("commands", {})
    feature_stats = model.get("feature_stats", {})

    if not commands:
        return "unknown"

    weights = meta.get("feature_weights", {})
    weight_arr = np.array([float(weights.get(f, 1.0)) for f in FEATURE_ORDER], dtype=np.float64)

    k = int(meta.get("k_neighbors", 5))
    unknown_threshold = float(meta.get("unknown_threshold", 6.5))
    min_margin = float(meta.get("min_margin", 0.08))

    query = _normalize_vector(feature_vector, feature_stats)

    centroid_scores = []
    for cmd, info in commands.items():
        centroid = info.get("centroid")
        if not centroid:
            continue
        centroid_vec = _normalize_vector(centroid, feature_stats)
        centroid_dist = float(np.sum(np.abs(query - centroid_vec) * weight_arr))
        centroid_scores.append((centroid_dist, cmd))

    if not centroid_scores:
        return "unknown"

    centroid_scores.sort(key=lambda item: item[0])
    centroid_best_dist, centroid_best_cmd = centroid_scores[0]
    centroid_second_dist = centroid_scores[1][0] if len(centroid_scores) > 1 else float("inf")
    centroid_confident = (
        centroid_best_dist <= unknown_threshold
        and (centroid_second_dist - centroid_best_dist) >= 0.05
    )

    neighbors = []
    for cmd, info in commands.items():
        for sample in info.get("samples", []):
            sample_norm = _normalize_vector(sample, feature_stats)
            dist = float(np.sum(np.abs(query - sample_norm) * weight_arr))
            neighbors.append((dist, cmd))

    if not neighbors:
        return centroid_best_cmd if centroid_confident else "unknown"

    neighbors.sort(key=lambda x: x[0])
    k = max(1, min(k, len(neighbors)))
    top_k = neighbors[:k]

    cmd_scores = {}
    cmd_distances = {}
    cmd_counts = {}
    for dist, cmd in top_k:
        score = 1.0 / (dist + 1e-6)
        cmd_scores[cmd] = cmd_scores.get(cmd, 0.0) + score
        cmd_distances.setdefault(cmd, []).append(dist)
        cmd_counts[cmd] = cmd_counts.get(cmd, 0) + 1

    ranked = sorted(cmd_scores.items(), key=lambda item: item[1], reverse=True)
    best_cmd, best_vote = ranked[0]
    second_vote = ranked[1][1] if len(ranked) > 1 else 0.0
    best_count = cmd_counts.get(best_cmd, 0)

    ranked_by_dist = sorted(
        ((cmd, float(np.mean(dists))) for cmd, dists in cmd_distances.items()),
        key=lambda item: item[1],
    )
    best_mean_dist = ranked_by_dist[0][1]
    second_mean_dist = ranked_by_dist[1][1] if len(ranked_by_dist) > 1 else float("inf")
    best_dist = top_k[0][0]

    if best_dist > unknown_threshold:
        return centroid_best_cmd if centroid_confident else "unknown"

    if best_count >= 3:
        return centroid_best_cmd if centroid_confident else best_cmd

    if (best_vote - second_vote) < min_margin:
        return centroid_best_cmd if centroid_confident else "unknown"

    if second_mean_dist < float("inf") and (second_mean_dist - best_mean_dist) < 0.06:
        return centroid_best_cmd if centroid_confident else "unknown"

    directional = {"left", "right"}
    if best_cmd in directional and len(ranked_by_dist) > 1:
        second_cmd = ranked_by_dist[1][0]
        if second_cmd in directional and (ranked_by_dist[1][1] - ranked_by_dist[0][1]) < 0.2:
            return centroid_best_cmd if centroid_confident else "unknown"

    return centroid_best_cmd if centroid_confident else best_cmd


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


def record_80_samples(output_file=TEST_DATASET_FILE):
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
    with open(MODEL_FILE, "w") as f:
        json.dump(model, f, indent=2)


def main():
    print("\nAUTO TUNE WORKFLOW")
    print(f"Main dataset is preserved in: {MAIN_DATASET_FILE}")
    print(f"Autotune dataset file is: {TEST_DATASET_FILE}")
    print("1. Record 80 test samples (10 per command) and tune model")
    print("2. Tune model from existing test_samples.json")
    print("3. Tune model from main samples.json (read-only)")
    choice = input("Choose (1/2/3): ").strip()

    if choice == "1":
        db = record_80_samples(TEST_DATASET_FILE)
    elif choice == "2":
        db = load_dataset(TEST_DATASET_FILE)
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