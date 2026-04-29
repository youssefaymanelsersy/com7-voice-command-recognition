import numpy as np

from core.app_config import AUDIO_SAMPLE_RATE

FS = AUDIO_SAMPLE_RATE
FRAME_COUNT = 32
FRAME_SIZE = 250
RECORDING_SAMPLES = FRAME_COUNT * FRAME_SIZE
FRAME_ZCR_FEATURE_NAMES = [f"zcr_{index}" for index in range(FRAME_COUNT)]
FRAME_STE_FEATURE_NAMES = [f"ste_{index}" for index in range(FRAME_COUNT)]
FRAME_FEATURE_ORDER = FRAME_ZCR_FEATURE_NAMES + FRAME_STE_FEATURE_NAMES

# Integer thresholds keep feature extraction MCU-friendly.
INT16_SILENCE_THRESHOLD = 600
MIN_ACTIVE_SAMPLES = 10


def _to_int_signal(signal):
    """Convert incoming samples to signed 16-bit style integers."""
    arr = np.asarray(signal)
    if arr.size == 0:
        return np.array([], dtype=np.int32)

    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.int32, copy=False)

    return np.clip(arr.astype(np.int32), -32768, 32767)


def remove_silence(signal, threshold=INT16_SILENCE_THRESHOLD):
    """Remove near-silent samples using an integer amplitude threshold."""
    int_signal = _to_int_signal(signal)
    if int_signal.size == 0:
        return int_signal

    mask = np.abs(int_signal) > int(threshold)
    if int(np.sum(mask)) < MIN_ACTIVE_SAMPLES:
        return int_signal
    return int_signal[mask]


def _trim_to_active_window(signal, threshold=INT16_SILENCE_THRESHOLD):
    """Trim leading/trailing silence while preserving the spoken region."""
    int_signal = _to_int_signal(signal)
    if int_signal.size == 0:
        return int_signal

    active = np.flatnonzero(np.abs(int_signal) > int(threshold))
    if active.size < MIN_ACTIVE_SAMPLES:
        return int_signal

    start = int(active[0])
    end = int(active[-1]) + 1
    return int_signal[start:end]


def _count_zero_crossings(signal):
    if signal.size < 2:
        return 0

    shifted = signal[1:] * signal[:-1]
    return int(np.sum(shifted < 0))


def _compute_energy(signal):
    if signal.size == 0:
        return 0

    # Use int64 accumulator and integer division for mean square energy.
    squared = signal.astype(np.int64) * signal.astype(np.int64)
    return int(np.sum(squared) // signal.size)


def _compute_time_centroid(signal):
    """Integer surrogate for spectral centroid using time-index weighted magnitude."""
    if signal.size < 2:
        return 0

    mags = np.abs(signal.astype(np.int64))
    total = int(np.sum(mags))
    if total <= 0:
        return 0

    idx = np.arange(signal.size, dtype=np.int64)
    weighted = int(np.sum(idx * mags))
    centroid_index = weighted // total
    return int((centroid_index * FS) // max(signal.size, 1))


def _normalize_recording_length(signal):
    """Return exactly 8000 samples using truncation or zero-padding."""
    int_signal = _trim_to_active_window(signal)
    if int_signal.size >= RECORDING_SAMPLES:
        return int_signal[:RECORDING_SAMPLES]

    out = np.zeros(RECORDING_SAMPLES, dtype=np.int32)
    out[: int_signal.size] = int_signal
    return out


def _compute_ste(frame):
    """Compute short-time energy as sum of squares inside one frame."""
    squared = frame.astype(np.int64) * frame.astype(np.int64)
    return int(np.sum(squared))


def extract_frame_features(signal):
    """Extract frame-based integer ZCR/STE arrays with fixed 32x250 indexing."""
    normalized = _normalize_recording_length(signal)

    zcr_features = np.zeros(FRAME_COUNT, dtype=np.int64)
    ste_features = np.zeros(FRAME_COUNT, dtype=np.int64)

    for frame_index in range(FRAME_COUNT):
        start = frame_index * FRAME_SIZE
        end = start + FRAME_SIZE
        frame = normalized[start:end]

        zcr_features[frame_index] = _count_zero_crossings(frame)
        ste_features[frame_index] = _compute_ste(frame)

    return {
        "zcr_features": zcr_features.astype(np.int32).tolist(),
        "ste_features": ste_features.astype(np.int64).tolist(),
        "length": int(RECORDING_SAMPLES),
        "spectral_centroid": int(_compute_time_centroid(normalized)),
        "processed_signal": normalized,
    }


def flatten_frame_features(feature_vector):
    """Flatten frame-based features into a stable model vector ordering."""
    zcr_values = [int(value) for value in feature_vector.get("zcr_features", [])]
    ste_values = [int(value) for value in feature_vector.get("ste_features", [])]

    if len(zcr_values) < FRAME_COUNT:
        zcr_values.extend([0] * (FRAME_COUNT - len(zcr_values)))
    if len(ste_values) < FRAME_COUNT:
        ste_values.extend([0] * (FRAME_COUNT - len(ste_values)))

    zcr_values = zcr_values[:FRAME_COUNT]
    ste_values = ste_values[:FRAME_COUNT]

    return {
        **{name: int(value) for name, value in zip(FRAME_ZCR_FEATURE_NAMES, zcr_values)},
        **{name: int(value) for name, value in zip(FRAME_STE_FEATURE_NAMES, ste_values)},
        "length": int(feature_vector.get("length", RECORDING_SAMPLES)),
        "spectral_centroid": int(feature_vector.get("spectral_centroid", 0)),
    }


def average_recording_features(recordings):
    """Average per-recording frame features into zcr_avg/ste_avg arrays."""
    if not recordings:
        return {
            "zcr_avg": [0] * FRAME_COUNT,
            "ste_avg": [0] * FRAME_COUNT,
        }

    zcr_stack = np.array([r["zcr_features"] for r in recordings], dtype=np.int64)
    ste_stack = np.array([r["ste_features"] for r in recordings], dtype=np.int64)
    count = len(recordings)

    # Integer rounding: (sum + count/2) // count.
    zcr_sum = np.sum(zcr_stack, axis=0)
    ste_sum = np.sum(ste_stack, axis=0)
    zcr_avg = ((zcr_sum + (count // 2)) // count).astype(np.int32).tolist()
    ste_avg = ((ste_sum + (count // 2)) // count).astype(np.int64).tolist()

    return {
        "zcr_avg": zcr_avg,
        "ste_avg": ste_avg,
    }


def extract_features(signal, return_processed=False):
    """Compatibility wrapper around frame features for existing call sites."""
    frame_features = extract_frame_features(signal)
    processed = frame_features["processed_signal"]
    flattened = flatten_frame_features(frame_features)
    zcr = int(sum(flattened[name] for name in FRAME_ZCR_FEATURE_NAMES))
    # Normalize energy: average per-frame STE instead of dividing by total recording samples
    total_ste = sum(flattened[name] for name in FRAME_STE_FEATURE_NAMES)
    energy = int(total_ste // FRAME_COUNT) if FRAME_COUNT > 0 else int(total_ste)
    length = int(flattened["length"])
    spectral_centroid = int(flattened["spectral_centroid"])

    features = (zcr, energy, length, spectral_centroid)
    return (features, processed) if return_processed else features
