"""Build and tune a model that uses only ZCR and energy."""

import argparse
import json

import numpy as np

from core.export_model_header import export_voice_model_header
from core.recognize import recognize_features
from core.trainer import (
    COMMANDS,
    DATASET_FILE,
    ZCR_ENERGY_FEATURE_ORDER,
    ZCR_ENERGY_MODEL_FILE,
    build_model_from_samples,
    sample_to_feature_vector,
)


TEST_DATASET_FILE = "data/test_samples.json"
ZCR_ENERGY_HEADER_FILE = "include/voice_model_zcr_energy.h"


def _load_dataset(path):
    with open(path, "r") as f:
        return json.load(f)


def _feature_vector(sample):
    return sample_to_feature_vector(sample, ZCR_ENERGY_FEATURE_ORDER)


def _frame_weight_map(zcr_weight, ste_weight):
    return {
        **{feature: zcr_weight for feature in ZCR_ENERGY_FEATURE_ORDER if feature.startswith("zcr_")},
        **{feature: ste_weight for feature in ZCR_ENERGY_FEATURE_ORDER if feature.startswith("ste_")},
    }


def _frame_floor_map(zcr_floor, ste_floor):
    return {
        **{feature: zcr_floor for feature in ZCR_ENERGY_FEATURE_ORDER if feature.startswith("zcr_")},
        **{feature: ste_floor for feature in ZCR_ENERGY_FEATURE_ORDER if feature.startswith("ste_")},
    }


def _build_search_context(model, validation_db):
    commands = list(model["commands"].keys())
    command_index = {command: index for index, command in enumerate(commands)}

    feature_stats = np.array(
        [
            [
                model["feature_stats"][feature]["mean"],
                model["feature_stats"][feature]["std"],
            ]
            for feature in ZCR_ENERGY_FEATURE_ORDER
        ],
        dtype=np.float64,
    )

    command_means = []
    command_stds = []
    centroids = []
    sample_vectors = []
    sample_command_indices = []
    for command in commands:
        stats = model["command_stats"][command]
        command_means.append([stats[feature]["mean"] for feature in ZCR_ENERGY_FEATURE_ORDER])
        command_stds.append([stats[feature]["std"] for feature in ZCR_ENERGY_FEATURE_ORDER])
        centroids.append(model["commands"][command]["centroid"])
        for sample in model["commands"][command]["samples"]:
            sample_vectors.append(sample)
            sample_command_indices.append(command_index[command])

    validation_vectors = []
    validation_expected_indices = []
    for command in commands:
        for sample in validation_db.get(command, []):
            validation_vectors.append(_feature_vector(sample))
            validation_expected_indices.append(command_index[command])

    return {
        "commands": commands,
        "feature_stats": feature_stats,
        "command_means": np.array(command_means, dtype=np.float64),
        "command_stds": np.array(command_stds, dtype=np.float64),
        "centroids": np.array(centroids, dtype=np.float64),
        "sample_vectors": np.array(sample_vectors, dtype=np.float64),
        "sample_command_indices": np.array(sample_command_indices, dtype=np.int64),
        "validation_vectors": np.array(validation_vectors, dtype=np.float64),
        "validation_expected_indices": np.array(validation_expected_indices, dtype=np.int64),
    }


def _predict_index(vector, context, params):
    weights = np.array(
        [params["feature_weights"][feature] for feature in ZCR_ENERGY_FEATURE_ORDER],
        dtype=np.float64,
    )
    floors = np.array(
        [params.get("feature_floors", {}).get(feature, 0.0) for feature in ZCR_ENERGY_FEATURE_ORDER],
        dtype=np.float64,
    )
    floors = np.maximum(floors, 1e-9)
    threshold = float(params["unknown_threshold"])
    margin = float(params["min_margin"])

    command_stds = np.maximum(context["command_stds"], floors)
    command_scores = np.sum(
        weights * np.abs(vector - context["command_means"]) / command_stds,
        axis=1,
    )
    command_order = np.argsort(command_scores)
    best_command = int(command_order[0])
    second_command_score = (
        float(command_scores[command_order[1]]) if len(command_order) > 1 else float("inf")
    )
    best_command_score = float(command_scores[best_command])
    if best_command_score <= threshold and (second_command_score - best_command_score) >= margin:
        return best_command

    feature_means = context["feature_stats"][:, 0]
    feature_stds = np.maximum(context["feature_stats"][:, 1], floors)
    query = (vector - feature_means) / feature_stds
    centroid_norm = (context["centroids"] - feature_means) / feature_stds
    centroid_scores = np.sum(np.abs(query - centroid_norm) * weights, axis=1)
    centroid_order = np.argsort(centroid_scores)
    centroid_best = int(centroid_order[0])
    centroid_second_score = (
        float(centroid_scores[centroid_order[1]]) if len(centroid_order) > 1 else float("inf")
    )
    centroid_best_score = float(centroid_scores[centroid_best])
    centroid_confident = (
        centroid_best_score <= threshold
        and (centroid_second_score - centroid_best_score) >= 0.05
    )

    sample_norm = (context["sample_vectors"] - feature_means) / feature_stds
    distances = np.sum(np.abs(query - sample_norm) * weights, axis=1)
    k = max(1, min(int(params["k_neighbors"]), len(distances)))
    top_indices = np.argpartition(distances, k - 1)[:k]
    top_indices = top_indices[np.argsort(distances[top_indices])]
    top_distances = distances[top_indices]
    top_commands = context["sample_command_indices"][top_indices]

    if float(top_distances[0]) > threshold:
        return centroid_best if centroid_confident else -1

    votes = {}
    distance_groups = {}
    counts = {}
    for distance, command_index in zip(top_distances, top_commands):
        command_index = int(command_index)
        votes[command_index] = votes.get(command_index, 0.0) + (1.0 / (float(distance) + 1e-6))
        distance_groups.setdefault(command_index, []).append(float(distance))
        counts[command_index] = counts.get(command_index, 0) + 1

    ranked_votes = sorted(votes.items(), key=lambda item: item[1], reverse=True)
    best_vote_command, best_vote = ranked_votes[0]
    second_vote = ranked_votes[1][1] if len(ranked_votes) > 1 else 0.0
    if counts.get(best_vote_command, 0) >= 3:
        return centroid_best if centroid_confident else best_vote_command

    if (best_vote - second_vote) < margin:
        return centroid_best if centroid_confident else -1

    ranked_distances = sorted(
        ((command_index, float(np.mean(distances))) for command_index, distances in distance_groups.items()),
        key=lambda item: item[1],
    )
    if len(ranked_distances) > 1 and (ranked_distances[1][1] - ranked_distances[0][1]) < 0.06:
        return centroid_best if centroid_confident else -1

    directional = {"left", "right"}
    best_name = context["commands"][best_vote_command]
    if best_name in directional and len(ranked_distances) > 1:
        second_name = context["commands"][ranked_distances[1][0]]
        if second_name in directional and (ranked_distances[1][1] - ranked_distances[0][1]) < 0.2:
            return centroid_best if centroid_confident else -1

    return centroid_best if centroid_confident else best_vote_command


def _apply_params(model, params):
    model["meta"]["feature_weights"] = params["feature_weights"]
    model["meta"]["feature_floors"] = params.get("feature_floors", {})
    model["meta"]["k_neighbors"] = params["k_neighbors"]
    model["meta"]["unknown_threshold"] = params["unknown_threshold"]
    model["meta"]["min_margin"] = params["min_margin"]


def _score(context, params):
    total = len(context["validation_vectors"])
    if total == 0:
        return {
            "accuracy": 0.0,
            "unknown_rate": 0.0,
            "objective": 0.0,
        }

    correct = 0
    unknown = 0
    for vector, expected in zip(context["validation_vectors"], context["validation_expected_indices"]):
        prediction = _predict_index(vector, context, params)
        if prediction == int(expected):
            correct += 1
        if prediction == -1:
            unknown += 1

    accuracy = correct / total
    unknown_rate = unknown / total
    unknown_bonus = 0.04 * min(unknown_rate, 0.15)
    excessive_unknown_penalty = 0.25 * max(0.0, unknown_rate - 0.15)
    return {
        "accuracy": accuracy,
        "unknown_rate": unknown_rate,
        "objective": accuracy + unknown_bonus - excessive_unknown_penalty,
    }


def _score_model(model, validation_db, params):
    _apply_params(model, params)

    total = 0
    correct = 0
    unknown = 0
    for command in COMMANDS:
        for sample in validation_db.get(command, []):
            prediction = recognize_features(_feature_vector(sample), model)
            total += 1
            if prediction == command:
                correct += 1
            if prediction == "unknown":
                unknown += 1

    accuracy = (correct / total) if total else 0.0
    unknown_rate = (unknown / total) if total else 0.0
    unknown_bonus = 0.04 * min(unknown_rate, 0.15)
    excessive_unknown_penalty = 0.25 * max(0.0, unknown_rate - 0.15)
    return {
        "accuracy": accuracy,
        "unknown_rate": unknown_rate,
        "objective": accuracy + unknown_bonus - excessive_unknown_penalty,
    }


def search_best_params(model, validation_db):
    context = _build_search_context(model, validation_db)
    grid = []
    for zcr_w in [3.0, 3.5, 4.0, 4.5, 5.0]:
        for energy_w in [1.2, 1.5, 1.8, 2.1, 2.5]:
            for zcr_floor in [0.005, 0.01, 0.015]:
                for energy_floor in [0.005, 0.01, 0.02]:
                    for k in [3, 5, 7]:
                        for threshold in [30.0, 45.0, 60.0, 80.0, 100.0]:
                            for margin in [0.0, 1.0, 2.0, 5.0]:
                                grid.append(
                                    {
                                        "feature_weights": {
                                            **_frame_weight_map(zcr_w, energy_w),
                                        },
                                        "feature_floors": {
                                            **_frame_floor_map(zcr_floor, energy_floor),
                                        },
                                        "k_neighbors": k,
                                        "unknown_threshold": threshold,
                                        "min_margin": margin,
                                    }
                                )

    best_params = None
    best_metrics = None
    top = []
    for index, params in enumerate(grid, start=1):
        metrics = _score(context, params)
        candidate = (metrics["objective"], metrics["accuracy"], -metrics["unknown_rate"], params, metrics)
        top.append(candidate)
        top.sort(key=lambda item: item[:3], reverse=True)
        del top[10:]

        if best_metrics is None or metrics["objective"] > best_metrics["objective"]:
            best_params = params
            best_metrics = metrics
        if index % 1000 == 0:
            print(f"Checked {index}/{len(grid)} configs...", flush=True)

    return best_params, best_metrics, top


def search_fast_params(model, validation_db):
    grid = []
    for zcr_w in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        for energy_w in [0.0, 0.2, 0.4, 0.8, 1.2, 1.8, 2.5, 3.5]:
            for k in [1, 3, 5, 7]:
                for threshold in [30.0, 45.0, 60.0, 80.0, 100.0, 120.0]:
                    for margin in [0.0, 1.0, 2.0, 5.0]:
                        grid.append(
                            {
                                "feature_weights": {
                                    **_frame_weight_map(zcr_w, energy_w),
                                },
                                "k_neighbors": k,
                                "unknown_threshold": threshold,
                                "min_margin": margin,
                            }
                        )

    best_params = None
    best_metrics = None
    for index, params in enumerate(grid, start=1):
        metrics = _score_model(model, validation_db, params)
        if best_metrics is None or metrics["objective"] > best_metrics["objective"]:
            best_params = params
            best_metrics = metrics
        if index % 500 == 0:
            print(f"Checked {index}/{len(grid)} configs...")

    return best_params, best_metrics, []


def save_model(model, model_path):
    with open(model_path, "w") as f:
        json.dump(model, f, indent=2)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Build a ZCR+energy-only model.")
    parser.add_argument("--train", default=DATASET_FILE, help="Training dataset JSON.")
    parser.add_argument("--validation", default=TEST_DATASET_FILE, help="Validation dataset JSON.")
    parser.add_argument("--model", default=ZCR_ENERGY_MODEL_FILE, help="Output model JSON.")
    parser.add_argument("--fast", action="store_true", help="Use the smaller legacy search grid.")
    parser.add_argument(
        "--header",
        default=ZCR_ENERGY_HEADER_FILE,
        help="Output C header for the ZCR+energy model.",
    )
    args = parser.parse_args()

    train_db = _load_dataset(args.train)
    validation_db = _load_dataset(args.validation)
    base_model = build_model_from_samples(
        train_db,
        feature_order=ZCR_ENERGY_FEATURE_ORDER,
        feature_weights=_frame_weight_map(1.0, 1.0),
        k_neighbors=3,
        unknown_threshold=60.0,
        min_margin=2.0,
    )
    if args.fast:
        best_params, metrics, top = search_fast_params(base_model, validation_db)
    else:
        best_params, metrics, top = search_best_params(base_model, validation_db)

    print("\nBEST ZCR+ENERGY CONFIG")
    print(json.dumps(best_params, indent=2))
    print(
        "Validation estimate: "
        f"accuracy={metrics['accuracy'] * 100:.1f}% | "
        f"unknown={metrics['unknown_rate'] * 100:.1f}% | "
        f"objective={metrics['objective']:.4f}"
    )
    if top:
        print("\nTop configs:")
        for rank, (_, _, _, params, top_metrics) in enumerate(top[:5], start=1):
            print(
                f"{rank}. accuracy={top_metrics['accuracy'] * 100:.1f}% | "
                f"unknown={top_metrics['unknown_rate'] * 100:.1f}% | "
                f"objective={top_metrics['objective']:.4f} | "
                f"params={json.dumps(params, sort_keys=True)}"
            )

    model = build_model_from_samples(
        train_db,
        feature_order=ZCR_ENERGY_FEATURE_ORDER,
        feature_weights=best_params["feature_weights"],
        feature_floors=best_params.get("feature_floors", {}),
        k_neighbors=best_params["k_neighbors"],
        unknown_threshold=best_params["unknown_threshold"],
        min_margin=best_params["min_margin"],
    )
    save_model(model, args.model)
    export_voice_model_header(args.model, args.header)
    print(f"Saved {args.model} and {args.header}")


if __name__ == "__main__":
    main()
