import numpy as np
from core.features import extract_features

FEATURE_ORDER = ["zcr", "energy", "length", "spectral_centroid"]

FALLBACK_WEIGHTS = {
    "zcr": 2.0,
    "energy": 0.4,
    "length": 1.8,
    "spectral_centroid": 1.8,
}

FEATURE_FLOORS = {
    "zcr": 0.01,
    "energy": 0.01,
    "length": 1500.0,
    "spectral_centroid": 100.0,
}


def _model_feature_order(model):
    meta = model.get("meta", {}) if isinstance(model, dict) else {}
    feature_order = meta.get("feature_order", FEATURE_ORDER)
    return list(feature_order) if feature_order else FEATURE_ORDER


def _coerce_feature_vector(feature_vector, feature_order):
    if isinstance(feature_vector, dict):
        return [float(feature_vector.get(feature, 0.0)) for feature in feature_order]

    values = list(feature_vector)
    if len(values) == len(feature_order):
        return [float(value) for value in values]

    if len(values) == len(FEATURE_ORDER):
        by_feature = dict(zip(FEATURE_ORDER, values))
        return [float(by_feature.get(feature, 0.0)) for feature in feature_order]

    return [float(value) for value in values]


def _feature_floor(feature, floors):
    return float(floors.get(feature, FEATURE_FLOORS[feature]))


def _command_normalized_distance(feature_vector, cmd_stats, weights, feature_order, floors):
    total = 0.0
    for i, feature in enumerate(feature_order):
        stats = cmd_stats.get(feature, {})
        mean = float(stats.get("mean", 0.0))
        std = max(float(stats.get("std", 0.0)), _feature_floor(feature, floors))
        total += float(weights.get(feature, 1.0)) * abs(float(feature_vector[i]) - mean) / std
    return float(total)


def _normalize_vector(vector, feature_stats, feature_order, floors):
    normalized = []
    for i, feature in enumerate(feature_order):
        stats = feature_stats.get(feature, {})
        mean = float(stats.get("mean", 0.0))
        std = max(float(stats.get("std", 0.0)), _feature_floor(feature, floors))
        normalized.append((float(vector[i]) - mean) / std)
    return np.array(normalized, dtype=np.float64)


def _legacy_recognize(feature_vector, model):
    """Fallback for older mean/std-only model files."""
    weights = FALLBACK_WEIGHTS

    def score_feature(value, stored_value, feature_name):
        floor = FEATURE_FLOORS[feature_name]
        if isinstance(stored_value, dict):
            mean = stored_value["mean"]
            std = max(stored_value["std"], floor)
        else:
            mean = stored_value
            std = floor
        return weights[feature_name] * abs(value - mean) / std

    zcr, energy, length, centroid = feature_vector

    best_cmd = None
    best_score = float("inf")
    second_score = float("inf")

    for cmd, values in model.items():
        if not isinstance(values, dict):
            continue
        if "zcr" not in values:
            continue

        score = (
            score_feature(zcr, values.get("zcr", 0), "zcr")
            + score_feature(energy, values.get("energy", 0), "energy")
            + score_feature(length, values.get("length", 0), "length")
            + score_feature(centroid, values.get("spectral_centroid", 0), "spectral_centroid")
        )

        if score < best_score:
            second_score = best_score
            best_score = score
            best_cmd = cmd
        elif score < second_score:
            second_score = score

    if best_score > 5.0:
        return "unknown"

    if second_score < float("inf") and best_score > 0:
        if (second_score - best_score) < 0.35:
            return "unknown"

    return best_cmd or "unknown"


def recognize_features(feature_vector, model):
    """Recognize command from a precomputed feature vector."""
    if not isinstance(model, dict) or "commands" not in model:
        return _legacy_recognize(feature_vector, model)

    meta = model.get("meta", {})
    commands = model.get("commands", {})
    command_stats = model.get("command_stats", {})
    feature_stats = model.get("feature_stats", {})
    feature_order = _model_feature_order(model)
    feature_vector = _coerce_feature_vector(feature_vector, feature_order)

    if not commands:
        return "unknown"

    weights = meta.get("feature_weights", FALLBACK_WEIGHTS)
    floors = meta.get("feature_floors", {})
    k = int(meta.get("k_neighbors", 3))
    unknown_threshold = float(meta.get("unknown_threshold", 7.0))
    min_margin = float(meta.get("min_margin", 0.05))

    weight_arr = np.array([float(weights.get(f, 1.0)) for f in feature_order], dtype=np.float64)
    query = _normalize_vector(feature_vector, feature_stats, feature_order, floors)

    # Prefer command-specific z-normalized distances when available.
    if command_stats:
        command_scores = []
        for cmd in commands.keys():
            stats = command_stats.get(cmd)
            if not stats:
                continue
            score = _command_normalized_distance(feature_vector, stats, weights, feature_order, floors)
            command_scores.append((score, cmd))

        if command_scores:
            command_scores.sort(key=lambda item: item[0])
            d1, c1 = command_scores[0]
            d2 = command_scores[1][0] if len(command_scores) > 1 else float("inf")
            if d1 <= unknown_threshold and (d2 - d1) >= min_margin:
                return c1

    centroid_scores = []
    for cmd, info in commands.items():
        centroid = info.get("centroid")
        if not centroid:
            continue
        centroid_vec = _normalize_vector(centroid, feature_stats, feature_order, floors)
        centroid_dist = float(np.sum(np.abs(query - centroid_vec) * weight_arr))
        centroid_scores.append((centroid_dist, cmd))

    if not centroid_scores:
        return _legacy_recognize(feature_vector, model)

    centroid_scores.sort(key=lambda item: item[0])
    centroid_best_dist, centroid_best_cmd = centroid_scores[0]
    centroid_second_dist = centroid_scores[1][0] if len(centroid_scores) > 1 else float("inf")

    # If the centroid match is clearly separated, use it directly.
    if centroid_best_dist <= unknown_threshold and (centroid_second_dist - centroid_best_dist) >= 0.05:
        centroid_confident = True
    else:
        centroid_confident = False

    neighbors = []
    for cmd, info in commands.items():
        samples = info.get("samples", [])
        for sample in samples:
            sample_norm = _normalize_vector(sample, feature_stats, feature_order, floors)
            # Weighted L1 distance keeps interpretation simple and robust.
            dist = float(np.sum(np.abs(query - sample_norm) * weight_arr))
            neighbors.append((dist, cmd))

    if not neighbors:
        return centroid_best_cmd if centroid_confident else "unknown"

    neighbors.sort(key=lambda x: x[0])
    k = max(1, min(k, len(neighbors)))
    top_k = neighbors[:k]

    cmd_scores = {}
    cmd_distances = {}
    for dist, cmd in top_k:
        score = 1.0 / (dist + 1e-6)
        cmd_scores[cmd] = cmd_scores.get(cmd, 0.0) + score
        cmd_distances.setdefault(cmd, []).append(dist)

    ranked = sorted(cmd_scores.items(), key=lambda item: item[1], reverse=True)
    best_cmd, best_vote = ranked[0]
    second_vote = ranked[1][1] if len(ranked) > 1 else 0.0

    cmd_counts = {}
    for _, cmd in top_k:
        cmd_counts[cmd] = cmd_counts.get(cmd, 0) + 1
    best_count = cmd_counts.get(best_cmd, 0)

    ranked_by_dist = sorted(
        ((cmd, float(np.mean(dists))) for cmd, dists in cmd_distances.items()),
        key=lambda item: item[1],
    )
    best_dist_cmd, best_mean_dist = ranked_by_dist[0]
    second_mean_dist = ranked_by_dist[1][1] if len(ranked_by_dist) > 1 else float("inf")
    best_dist = top_k[0][0]

    if best_dist > unknown_threshold:
        return centroid_best_cmd if centroid_confident else "unknown"

    # If nearest neighbors strongly agree, accept instead of over-rejecting.
    if best_count >= 3:
        return centroid_best_cmd if centroid_confident else best_cmd

    if (best_vote - second_vote) < min_margin:
        return centroid_best_cmd if centroid_confident else "unknown"

    # Ambiguity guard for very close command clusters.
    if second_mean_dist < float("inf") and (second_mean_dist - best_mean_dist) < 0.06:
        return centroid_best_cmd if centroid_confident else "unknown"

    # Directional safety: avoid wrong left/right decisions on close distances.
    directional = {"left", "right"}
    if best_cmd in directional and len(ranked_by_dist) > 1:
        second_cmd = ranked_by_dist[1][0]
        if second_cmd in directional and (ranked_by_dist[1][1] - ranked_by_dist[0][1]) < 0.2:
            return centroid_best_cmd if centroid_confident else "unknown"

    return centroid_best_cmd if centroid_confident else best_cmd


def recognize(audio, model):
    """Recognize command using centroid-first scoring over saved samples."""
    return recognize_features(extract_features(audio), model)
