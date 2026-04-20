import numpy as np
from scipy.fft import fft
from scipy.fftpack import fftfreq

FS = 44100


def remove_silence(signal, threshold=0.02):
    """Remove silence from signal."""
    mask = np.abs(signal) > threshold
    if np.sum(mask) < 10:
        return signal
    return signal[mask]


def compute_spectral_centroid(signal):
    """Compute spectral centroid (center of mass of frequency spectrum)."""
    if len(signal) < 2:
        return 0.0
    
    spectrum = np.abs(fft(signal))
    freqs = np.abs(fftfreq(len(signal), 1/FS))
    
    if np.sum(spectrum) == 0:
        return 0.0
    
    centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
    return float(centroid)


def extract_features(signal):
    """Extract features: ZCR, energy, length, spectral centroid."""
    if len(signal) == 0:
        return 0.0, 0.0, 0.0, 0.0

    signal = remove_silence(signal)

    if len(signal) < 10:
        return 0.0, 0.0, 0.0, 0.0

    zcr = np.mean(np.abs(np.diff(np.sign(signal)))) / 2
    energy = np.sum(signal**2) / len(signal)
    length = len(signal)
    spectral_centroid = compute_spectral_centroid(signal)

    return float(zcr), float(energy), float(length), float(spectral_centroid)