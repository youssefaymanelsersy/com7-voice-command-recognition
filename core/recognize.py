from core.features import (
    FRAME_FEATURE_ORDER,
    extract_features,
    extract_frame_features,
    flatten_frame_features,
)

FEATURE_ORDER = ["zcr", "energy", "length", "spectral_centroid"]

FALLBACK_WEIGHTS = {
    "zcr": 2,
    "energy": 1,
    "length": 2,
    "spectral_centroid": 2,
}


def _model_feature_order(model):
    meta = model.get("meta", {}) if isinstance(model, dict) else {}
    feature_order = meta.get("feature_order", FEATURE_ORDER)
    return list(feature_order) if feature_order else FEATURE_ORDER


def _to_int(value, default=0):
    try:
        # JSON numeric values may be float; this converts to fixed integer storage.
        return int(round(value))
    except Exception:
        return int(default)


def _coerce_feature_vector(feature_vector, feature_order):
    if isinstance(feature_vector, dict) and (
        "zcr_features" in feature_vector or "ste_features" in feature_vector
    ):
        processed_signal = feature_vector.get("processed_signal")
        if processed_signal is not None:
            feature_vector = extract_frame_features(processed_signal)

        flattened = flatten_frame_features(feature_vector)
        if all(feature in flattened for feature in feature_order):
            return [_to_int(flattened.get(feature, 0)) for feature in feature_order]

        if all(feature in flattened for feature in FEATURE_ORDER):
            return [_to_int(flattened.get(feature, 0)) for feature in FEATURE_ORDER]

        if all(feature in feature_order for feature in FEATURE_ORDER):
            scalar_vector = extract_features(processed_signal if processed_signal is not None else feature_vector.get("processed_signal", []))
            by_feature = dict(zip(FEATURE_ORDER, scalar_vector))
            return [_to_int(by_feature.get(feature, 0)) for feature in feature_order]

    if isinstance(feature_vector, dict):
        return [_to_int(feature_vector.get(feature, 0)) for feature in feature_order]

    values = list(feature_vector)
    if len(values) == len(feature_order):
        return [_to_int(value) for value in values]

    if len(values) == len(FEATURE_ORDER):
        by_feature = dict(zip(FEATURE_ORDER, values))
        return [_to_int(by_feature.get(feature, 0)) for feature in feature_order]

    return [_to_int(value) for value in values]


def _weight_map(meta, feature_order):
    weights = meta.get("feature_weights", {}) if isinstance(meta, dict) else {}
    mapped = {}
    for feature in feature_order:
        if feature.startswith("zcr_"):
            fallback = 2.0
        elif feature.startswith("ste_"):
            fallback = 0.4
        else:
            fallback = FALLBACK_WEIGHTS.get(feature, 1)

        raw = weights.get(feature, fallback)
        try:
            value = float(raw)
        except Exception:
            value = float(fallback)
        mapped[feature] = max(0.0, value)
    return mapped


def _weighted_l1_distance(a, b, feature_order, weights):
    total = 0.0
    for idx, feature in enumerate(feature_order):
        total += float(weights.get(feature, 1.0)) * abs(float(a[idx]) - float(b[idx]))
    return float(total)


def _normalized_weighted_l1_distance(a, b, feature_order, weights, feature_stats):
    total = 0.0
    for idx, feature in enumerate(feature_order):
        stats = feature_stats.get(feature, {}) if isinstance(feature_stats, dict) else {}
        std = float(stats.get("std", 1.0) or 1.0)
        if std <= 0:
            std = 1.0
        diff = abs(float(a[idx]) - float(b[idx])) / std
        total += float(weights.get(feature, 1.0)) * diff
    return float(total)


def _legacy_recognize(feature_vector, model):
    """Fallback for older mean/std-only model files using integer-safe scoring."""
    best_cmd = None
    best_score = None
    second_score = None

    for cmd, values in model.items():
        if not isinstance(values, dict) or "zcr" not in values:
            continue

        score = 0
        for feature in FEATURE_ORDER:
            stored = values.get(feature, 0)
            if isinstance(stored, dict):
                mean = _to_int(stored.get("mean", 0))
                std = max(1, _to_int(stored.get("std", 1), 1))
            else:
                mean = _to_int(stored)
                std = 1

            value = _to_int(feature_vector[FEATURE_ORDER.index(feature)])
            score += FALLBACK_WEIGHTS.get(feature, 1) * (abs(value - mean) // std)

        if best_score is None or score < best_score:
            second_score = best_score
            best_score = score
            best_cmd = cmd
        elif second_score is None or score < second_score:
            second_score = score

    if best_cmd is None:
        return "unknown"

    if second_score is not None and (second_score - best_score) <= 1:
        return "unknown"

    return best_cmd


def recognize_features(feature_vector, model):
    """Recognize command from a feature vector using integer-only distance math."""
    if not isinstance(model, dict) or "commands" not in model:
        return _legacy_recognize(feature_vector, model)

    commands = model.get("commands", {})
    if not commands:
        return "unknown"

    meta = model.get("meta", {})
    feature_stats = model.get("feature_stats", {}) if isinstance(model, dict) else {}
    feature_order = _model_feature_order(model)
    query = _coerce_feature_vector(feature_vector, feature_order)
    weights = _weight_map(meta, feature_order)

    k = max(1, _to_int(meta.get("k_neighbors", 3), 3))
    try:
        min_margin = max(0.0, float(meta.get("min_margin", 1.0)))
    except Exception:
        min_margin = 1.0
    unknown_threshold = float(meta.get("unknown_threshold", 0.0) or 0.0)

    neighbors = []
    centroid_scores = []

    for cmd, info in commands.items():
        centroid = info.get("centroid")
        if centroid:
            centroid_vec = _coerce_feature_vector(centroid, feature_order)
            centroid_scores.append((
                _normalized_weighted_l1_distance(
                    query,
                    centroid_vec,
                    feature_order,
                    weights,
                    feature_stats,
                ),
                cmd,
            ))

        samples = info.get("samples", [])
        for sample in samples:
            sample_vec = _coerce_feature_vector(sample, feature_order)
            neighbors.append((
                _normalized_weighted_l1_distance(
                    query,
                    sample_vec,
                    feature_order,
                    weights,
                    feature_stats,
                ),
                cmd,
            ))

    if not neighbors and not centroid_scores:
        return "unknown"

    if centroid_scores:
        centroid_scores.sort(key=lambda item: item[0])
        centroid_best_dist, centroid_best_cmd = centroid_scores[0]
        centroid_second_dist = centroid_scores[1][0] if len(centroid_scores) > 1 else None
        if centroid_second_dist is None or (centroid_second_dist - centroid_best_dist) >= min_margin:
            centroid_confident = True
        else:
            centroid_confident = False
    else:
        centroid_best_cmd = "unknown"
        centroid_confident = False

    if not neighbors:
        return centroid_best_cmd if centroid_confident else "unknown"

    neighbors.sort(key=lambda item: item[0])
    top_k = neighbors[: min(k, len(neighbors))]

    best_neighbor_dist = float(top_k[0][0]) if top_k else float("inf")
    if unknown_threshold > 0 and best_neighbor_dist > unknown_threshold:
        return "unknown"

    cmd_counts = {}
    cmd_dist_total = {}
    for dist, cmd in top_k:
        cmd_counts[cmd] = cmd_counts.get(cmd, 0) + 1
        cmd_dist_total[cmd] = cmd_dist_total.get(cmd, 0) + dist

    ranked = sorted(
        cmd_counts.items(),
        key=lambda item: (-item[1], cmd_dist_total.get(item[0], 0)),
    )

    best_cmd = ranked[0][0]
    best_count = ranked[0][1]
    best_dist = cmd_dist_total.get(best_cmd, 0)

    if len(ranked) > 1:
        second_cmd = ranked[1][0]
        second_count = ranked[1][1]
        second_dist = cmd_dist_total.get(second_cmd, 0)

        if best_count == second_count and abs(second_dist - best_dist) < min_margin:
            return centroid_best_cmd if centroid_confident else "unknown"

    if best_count <= 1 and not centroid_confident:
        return "unknown"

    return centroid_best_cmd if centroid_confident else best_cmd


def recognize(audio, model):
    """Recognize command from audio using integer-based feature extraction/scoring."""
    if isinstance(audio, dict):
        return recognize_features(audio, model)
    return recognize_features(extract_frame_features(audio), model)
