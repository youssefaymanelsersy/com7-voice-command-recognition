# Why Accuracy is Only 7.5% (not 12.5%)

## Key Findings

### 1. **Fundamental Feature Scaling Problem**

The most critical issue is that **ZCR features are dominating the distance metric** while STE features are being mathematically neutralized:

- **ZCR range**: 0-25 per frame (small integers)
- **STE range**: ~35 billion per frame (massive integers)
- **ZCR weight**: 2.0
- **STE weight**: 0.4

When normalized by their standard deviations:

- ZCR std: ~1-5, so normalized distance is still significant (value 2.0)
- STE std: ~8 billion, so normalized distance becomes nearly 0 (value → 0.0004 or less)

**Result**: Despite having 32 STE features with lower weight, they contribute almost NOTHING to the final distance metric after normalization. The model relies almost entirely on 32 ZCR features.

### 2. **ZCR Alone is Not Discriminative Enough**

Looking at the test data, ZCR values are nearly identical across different commands:

- All commands have similar zero-crossing patterns
- ZCR values in frames typically range from 0-20
- There's massive overlap between "on", "off", "left", "right", etc.

**Confusion matrix shows**:

- "start" gets 50% accuracy (the most common prediction)
- "on" and "start" are dominant predictions, often confused with each other
- Most other commands get 0% accuracy
- 27.5% of predictions are "unknown"

### 3. **Curse of Dimensionality**

- **Feature space**: 66 dimensions (32 ZCR + 32 STE + length + spectral_centroid)
- **Training samples**: ~160 total (20 per command × 8 commands)
- **Test samples**: 80 (10 per command)

With only 20 samples per command in a 66-dimensional space, the model has **extreme data sparsity**. The k-NN algorithm (k=3) cannot find meaningful nearest neighbors.

### 4. **STE Feature Implementation is Correct But Misused**

The STE values ARE calculated correctly:

```python
squared = frame.astype(np.int64) * frame.astype(np.int64)
return int(np.sum(squared))
```

- 250 samples per frame × 16-bit samples (up to 32767²) = billions
- This is mathematically correct

**But the problem**: These huge values are being treated as absolute differences in the distance metric, which makes them meaningless relative to ZCR after normalization.

---

## Why This Happened

When you refactored to frame-based features, you:

1. ✅ Correctly extracted 32 ZCR frames
2. ✅ Correctly extracted 32 STE frames
3. ❌ **Did NOT account for the scale mismatch** between integer ZCR (0-25) and integer STE (billions)
4. ❌ **Used too many features** relative to training data size
5. ❌ **Weighted STE too low** (0.4 vs 2.0 for ZCR)

The feature extraction is **mathematically correct**, but the **model architecture is flawed** for this problem.

---

## What's Wrong in Code

### In `core/trainer.py` - FEATURE_WEIGHTS

```python
FEATURE_WEIGHTS = {
    **{feature: 2.0 for feature in FRAME_FEATURE_ORDER if feature.startswith("zcr_")},  # Weight 2.0
    **{feature: 0.4 for feature in FRAME_FEATURE_ORDER if feature.startswith("ste_")},  # Weight 0.4 TOO LOW!
    "length": 1.8,
    "spectral_centroid": 1.8,
}
```

With weight 0.4 and huge standard deviations, STE features contribute almost zero to the distance.

### In `core/recognize.py` - Normalization Math

```python
def _normalized_weighted_l1_distance(a, b, feature_order, weights, feature_stats):
    for idx, feature in enumerate(feature_order):
        std = float(stats.get("std", 1.0) or 1.0)
        diff = abs(float(a[idx]) - float(b[idx])) / std  # STE std is ~8 billion!
        total += float(weights.get(feature, 1.0)) * diff
```

When `std = 8,000,000,000` and `diff = 10,000,000`, the normalized value is only 0.00000125!

---

## The Real Solution (Not Quick Fixes)

You need ONE of these approaches:

### Option A: Normalize STE Features to Reasonable Scale

```python
# Instead of raw sum of squares, use:
energy_normalized = sum_of_squares / (250 * 32767²)  # Normalize to 0-1 range
```

### Option B: Use log-scale for STE

```python
energy_log = math.log(1 + sum_of_squares)  # Compress the scale
```

### Option C: Reduce dimensionality

Instead of 32 STE features, use aggregate energy over blocks:

```python
# 8 blocks instead of 32 frames = fewer features, less overfitting
```

### Option D: Different classifier

Instead of k-NN, use algorithms that handle high-dimensional sparse data:

- SVM with RBF kernel
- Random Forest
- Neural network

---

## Current Test Results

```text
Overall: 6/80 correct (7.5%), unknown=22 (27.5%)

Per command:
  on      0/10 accuracy= 0.0%
  off     0/10 accuracy= 0.0%
  start   5/10 accuracy=50.0%  ← Dominant prediction
  stop    1/10 accuracy=10.0%
  left    0/10 accuracy= 0.0%
  right   0/10 accuracy= 0.0%
  up      0/10 accuracy= 0.0%
  down    0/10 accuracy= 0.0%
```

The model barely works because the feature representation doesn't discriminate well between commands.
