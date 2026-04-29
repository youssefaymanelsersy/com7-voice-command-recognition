# Complete Code Architecture Audit

## OVERVIEW

The codebase has been refactored to use **32 frame-based features** (32 ZCR + 32 STE). This audit evaluates whether ALL code components properly implement and use this new architecture.

***VERDICT: ~75% properly implemented, 25% has serious issues***

---

## CRITICAL ISSUES FOUND

### 1. **STE WEIGHT IS TOO LOW - FUNDAMENTALLY BREAKS FEATURE DISCRIMINATION** ⚠️ CRITICAL

**File**: [core/trainer.py](core/trainer.py#L45-L50)

```python
FEATURE_WEIGHTS = {
    **{feature: 2.0 for feature in FRAME_FEATURE_ORDER if feature.startswith("zcr_")},
    **{feature: 0.4 for feature in FRAME_FEATURE_ORDER if feature.startswith("ste_")},  # TOO LOW!
    "length": 1.8,
    "spectral_centroid": 1.8,
}
```

**Problem**:

- STE features have values ~35 billion with std ~8 billion
- After normalization: `diff / std` → nearly 0
- Weight 0.4 × 0.0004 ≈ 0.00016 contribution per STE feature
- Weight 2.0 × 0.5 ≈ 1.0 contribution per ZCR feature
- **Result: STE is 6,000x less influential than ZCR**

**Also in**: [core/trainer.py](core/trainer.py#L51-L54) - ZCR_ENERGY_FEATURE_WEIGHTS has same problem
**Also in**: [cli/autotune.py](cli/autotune.py#L15-L18) -_frame_weight_map hardcodes 0.4 for STE
**Also in**: [cli/build_zcr_energy_model.py](cli/build_zcr_energy_model.py#L23-27) - _frame_weight_map hardcodes 0.4

---

### 2. **ENERGY AGGREGATION IS WRONG** ⚠️ CRITICAL

**File**: [core/features.py](core/features.py#L283-284)

```python
def extract_features(signal, return_processed=False):
    # ...
    energy = int(sum(flattened[name] for name in FRAME_STE_FEATURE_NAMES) // RECORDING_SAMPLES)
    # ...
```

**Problem**:

- STE per frame is already sum-of-squares, with values ~35 billion
- Summing all 32 frames = ~1.1 trillion (int64 overflow risk)
- Dividing by 8000 samples = ~137 billion (still huge)
- **This "energy" metric is a meaningless scaling artifact**

**Should be**: Either normalize by sum directly, or use log scale
**Impact**: The backward-compatible "energy" returned to callers is useless

---

### 3. **ARRAY SIZE MISMATCH IN DATA STORAGE** ⚠️ HIGH

**File**: [cli/retake_samples.py](cli/retake_samples.py#L1) and [cli/retake_training_samples.py](cli/retake_training_samples.py#L1)

**Problem**: These files extract frame features but don't show what they store:

```python
frame_features = extract_frame_features(audio)
# Missing: storing zcr_features and ste_features in the sample dict
```

**Need to verify**: Do they call the full `extract_frame_features()` and store all fields?

Looking at [retake_training_samples.py lines 65-75]:

```python
samples.append(
    {
        "zcr": int(zcr),
        "energy": int(energy),
        "length": int(length),
        "spectral_centroid": int(centroid),
        # Missing: "zcr_features" and "ste_features"!
    }
)
```

**Issue**: Test/training retake samples don't store frame features. They only store scalar values.

- This means retraining from retaken samples won't have frame-based features
- Model will be built from incomplete data

---

### 4. **FRAME FEATURE HANDLING IN test.py IS INCOMPLETE** ⚠️ MEDIUM

**File**: [cli/test.py](cli/test.py#L35-45)

```python
def test_features():
    print("\nTesting feature extraction...")
    _prompt_if_needed("Press Enter, then say a command: ")

    _, audio, max_amp = record_audio()
    if max_amp < QUIET_THRESHOLD:
        print(MSG_TOO_QUIET)
        return False

    frame_features = extract_frame_features(audio)
    zcr, energy, length, centroid = extract_features(audio)
    print(f"Features: ZCR={zcr}, Energy={energy}, Length={length}, Centroid={centroid}")
    print(f"Frame ZCR count: {len(frame_features['zcr_features'])}, Frame STE count: {len(frame_features['ste_features'])}")
    print("Features extracted")
    return True
```

**Problem**:

- Only prints frame feature COUNTS, not actual values
- Can't debug if frame extraction is working correctly
- No validation that frames have reasonable values

---

### 5. **DUPLICATE FEATURE EXTRACTION IN main.py** ⚠️ MEDIUM

**File**: [cli/main.py](cli/main.py#L58-82)

```python
def run_recognition():
    # ...
    frame_features = extract_frame_features(audio)
    cmd = recognize_features(frame_features, model)

    processed_signal = frame_features.get("processed_signal")
    if tx_enabled and processed_signal is not None and len(processed_signal) > 0:
        transmit_processed_signal(
            processed_signal,
            PROCESSED_SIGNAL_TX_PORT,
            PROCESSED_SIGNAL_TX_BAUDRATE,
        )
```

**Problem**:

- `extract_frame_features()` is called separately, returns dict with `processed_signal`
- But if you call `extract_features(audio, return_processed=True)`, you get (features, processed_signal) tuple
- **Inconsistent API between two extraction functions**
- In [core/recognize.py _coerce_feature_vector](core/recognize.py#L32-52), it tries to re-extract from `processed_signal` if available:

  ```python
  if processed_signal is not None:
      feature_vector = extract_frame_features(processed_signal)
  ```

  This causes **double extraction** when frame features are already passed in

---

### 6. **AUTOTUNE GRID SEARCH HARDCODES STE WEIGHT** ⚠️ MEDIUM

**File**: [cli/autotune.py](cli/autotune.py#L196-200)

```python
def search_best_validation_params(train_database, validation_database):
    grid = []
    for zcr_w in [1.5, 2.0, 2.5, 3.0]:
        for energy_w in [0.0, 0.4]:  # This var is created but never used!
            for len_w in [0.8, 1.4, 2.0]:
                for cent_w in [0.8, 1.4, 2.0, 2.8]:
```

**Problem**:

- Variable `energy_w` is created but then never used in grid generation
- Grid is built with hardcoded `_frame_weight_map(zcr_w, 0.4, ...)`
- Autotune can't actually search STE weight space

---

### 7. **FEATURE STATS CALCULATION ISSUES** ⚠️ MEDIUM

**File**: [core/trainer.py](core/trainer.py#L160-180)

```python
for i, feature in enumerate(feature_order):
    within = per_feature_within_stds[feature]
    if within:
        std = float(np.mean(within))
    else:
        std = float(np.std(all_arr[:, i]))
    model["feature_stats"][feature] = {
        "mean": float(np.mean(all_arr[:, i])),
        "std": float(std + 1e-6),
    }
```

**Problem**:

- For STE features with values ~35 billion, adding 1e-6 is meaningless
- std might be 8 billion, +1e-6 changes nothing
- Should be: `std = max(std, 1e6)` or relative epsilon: `std = std * (1 + 1e-6)`

---

### 8. **INCONSISTENT FEATURE ORDER BETWEEN MODELS** ⚠️ MEDIUM

**Files**:

- [core/trainer.py](core/trainer.py#L36): `FEATURE_ORDER = FRAME_FEATURE_ORDER + ["length", "spectral_centroid"]`
- [core/trainer.py](core/trainer.py#L37): `ZCR_ENERGY_FEATURE_ORDER = FRAME_FEATURE_ORDER`
- [core/recognize.py](core/recognize.py#L4): `FEATURE_ORDER = ["zcr", "energy", "length", "spectral_centroid"]`

**Problem**:

- Three different feature orders exist
- trainer.py uses 66 features (32 ZCR + 32 STE + 2 scalars)
- recognize.py fallback uses 4 features
- ZCR_ENERGY uses 64 features
- Code must carefully track which model uses which order
- High risk of index misalignment bugs

---

### 9. **EXPORT FUNCTIONS NOT UPDATED FOR FRAME FEATURES** ⚠️ MEDIUM

**File**: [core/export_model_header.py](core/export_model_header.py#L1)

**Problem**:

- Exports C header with model and samples
- But C code will receive 66-dimensional vectors
- Need to ensure MCU code is also updated to handle 66 features, not just 4
- Header generation looks OK, but **no validation that MCU implementation matches**

**Same issue in**: [core/export_samples_header.py](core/export_samples_header.py)

- Correctly exports `zcr_features[32]` and `ste_features[32]`
- But if MCU expects different format, will silently fail

---

### 10. **LEGACY RECOGNIZE FUNCTION IS DEAD CODE** ⚠️ LOW

**File**: [core/recognize.py](core/recognize.py#L113-141)

```python
def _legacy_recognize(feature_vector, model):
    """Fallback for older mean/std-only model files using integer-safe scoring."""
```

**Problem**:

- This function exists but uses 4 features (zcr, energy, length, centroid)
- But models now use 66 features
- If someone loads an old model format, this will silently fail or give garbage results
- Should either: remove this, or properly support both formats

---

### 11. **PROCESSED SIGNAL TRANSMISSION NOT VALIDATED** ⚠️ MEDIUM

**File**: [cli/main.py](cli/main.py#L70-77)

```python
processed_signal = frame_features.get("processed_signal")
if tx_enabled and processed_signal is not None and len(processed_signal) > 0:
    transmit_processed_signal(
        processed_signal,
        PROCESSED_SIGNAL_TX_PORT,
        PROCESSED_SIGNAL_TX_BAUDRATE,
    )
```

**Problem**:

- `processed_signal` is 8000 samples
- Transmission waits for all 8000 lines + START/END overhead
- No timeout or error handling if serial transmission fails
- Serial baud rate is 230400 (SERIAL_BAUDRATE * 2) - can handle 8000 samples in ~350ms, but no verification

---

### 12. **RECOGNIZE FEATURES CAN CAUSE DOUBLE EXTRACTION** ⚠️ MEDIUM

**File**: [core/recognize.py](core/recognize.py#L32-45)

```python
def _coerce_feature_vector(feature_vector, feature_order):
    if isinstance(feature_vector, dict) and (
        "zcr_features" in feature_vector or "ste_features" in feature_vector
    ):
        processed_signal = feature_vector.get("processed_signal")
        if processed_signal is not None:
            feature_vector = extract_frame_features(processed_signal)  # DOUBLE EXTRACTION!

        flattened = flatten_frame_features(feature_vector)
```

**Problem**:

- If you pass frame_features dict with `processed_signal`, it extracts AGAIN
- This is wasteful and could produce different results
- Should only extract once

---

### 13. **NO VALIDATION OF FRAME SIZES** ⚠️ LOW

**File**: [core/features.py](core/features.py#L146-161)

```python
def flatten_frame_features(feature_vector):
    """Flatten frame-based features into a stable model vector ordering."""
    zcr_values = [int(value) for value in feature_vector.get("zcr_features", [])]
    ste_values = [int(value) for value in feature_vector.get("ste_features", [])]

    if len(zcr_values) < FRAME_COUNT:
        zcr_values.extend([0] * (FRAME_COUNT - len(zcr_values)))
```

**Problem**:

- Silently pads with zeros if frames are missing
- Should error loudly instead
- Makes it hard to debug if frame extraction fails

---

### 14. **RETAKE SAMPLES INCOMPLETE STORAGE** ⚠️ HIGH

**File**: [cli/retake_training_samples.py](cli/retake_training_samples.py#L72-87)

```python
samples.append(
    {
        "zcr": int(zcr),
        "energy": int(energy),
        "length": int(length),
        "spectral_centroid": int(centroid),
        # MISSING: "zcr_features" and "ste_features"
    }
)
```

**Also in**: [cli/retake_samples.py](cli/retake_samples.py) - same pattern

**Problem**:

- Frame features are extracted but NOT STORED
- When retraining, model will be built from incomplete data
- This breaks the whole frame-based architecture

**Expected**:

```python
samples.append(
    {
        "zcr": int(zcr),
        "energy": int(energy),
        "length": int(length),
        "spectral_centroid": int(centroid),
        "zcr_features": [int(v) for v in frame_features["zcr_features"]],
        "ste_features": [int(v) for v in frame_features["ste_features"]],
    }
)
```

---

## MINOR ISSUES

### 15. Missing error handling in [core/export_samples_header.py](core/export_samples_header.py#L40)

If `sample.get("zcr_features")` returns None, the list comprehension will fail

### 16. Inconsistent return values in [core/audio.py](core/audio.py#L195)

```python
return audio, audio, raw_max  # Why return audio twice?
```

Should be: `return audio, raw_max` or clearly document why both are returned

### 17. Feature floors are unused

[core/trainer.py](core/trainer.py#L138) has `feature_floors` parameter but it's never actually used in distance calculations

### 18. Unknown threshold might be unreachable

In [core/recognize.py](core/recognize.py#L209), with unknown_threshold=60.0 and normalized distances potentially being very small, this threshold might never be hit

---

## WHAT IS WORKING CORRECTLY ✅

1. **Frame extraction logic** - [core/features.py extract_frame_features()](core/features.py#L124-161) correctly:
   - Normalizes to 8000 samples
   - Divides into 32 frames of 250 samples each
   - Computes ZCR per frame correctly
   - Computes STE per frame correctly

2. **Model building** - [core/trainer.py build_model_from_samples()](core/trainer.py#L91-192) correctly:
   - Loads samples with frame features
   - Builds command centroids
   - Computes feature stats
   - Stores everything in JSON

3. **C header export** - [core/export_samples_header.py](core/export_samples_header.py) correctly:
   - Exports 32-element arrays for zcr_features and ste_features
   - Creates proper C structs

4. **Processed signal transmission** - [core/processed_signal_io.py](core/processed_signal_io.py) correctly:
   - Buffers entire signal
   - Sends with START/END framing
   - Proper int16 clipping

5. **Training flow** - [core/trainer.py train()](core/trainer.py#L282-366) correctly:
   - Calls extract_frame_features()
   - Stores both scalar and frame features
   - Builds model from complete data

---

## SUMMARY BY SEVERITY

| Severity | Count | Files Affected |
| ---------- | ------- | ---------------- |
| CRITICAL | 2 | trainer.py (weight/energy) |
| HIGH | 2 | retake_samples.py, retake_training_samples.py (missing storage) |
| MEDIUM | 9 | trainer.py, recognize.py, main.py, autotune.py, export_*.py |
| LOW | 3 | Various edge cases |

---

## IMPACT ASSESSMENT

### Why Accuracy is 7.5%

1. **STE weight (0.4) is ignored** after normalization by huge std values
2. **ZCR alone is insufficient** to distinguish commands
3. **Model only sees 32 ZCR values**, effectively 32-dimensional, not 66-dimensional
4. High-dimensional overfitting with sparse data

### Why Retrain Fails

1. Retaken samples don't store `zcr_features` and `ste_features`
2. Model retraining uses incomplete data
3. New features aren't extracted from retaken samples

### Why Some Features Work

1. Initial training stores frame features correctly
2. Recognition can process frame features
3. Export functions work with frame format
4. C header generation is correct

---

## NEXT STEPS FOR FIXES

**Priority 1 (Accuracy):**

1. Fix STE weight - should be at least 1.0-2.0, not 0.4
2. Add log-scale normalization for STE values
3. Re-train and test accuracy

**Priority 2 (Data Integrity):**

1. Add zcr_features and ste_features storage to retake_samples.py
2. Add zcr_features and ste_features storage to retake_training_samples.py
3. Validate frame feature sizes in flatten_frame_features()

**Priority 3 (Code Quality):**

1. Fix autotune grid search to actually use energy_w variable
2. Remove double extraction in _coerce_feature_vector()
3. Document feature order consistency
4. Fix energy aggregation calculation
5. Remove or properly update _legacy_recognize()
