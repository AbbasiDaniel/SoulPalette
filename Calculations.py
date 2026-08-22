import numpy as np
from scipy.signal import find_peaks

EMOTION_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
_EMOTION_MAP = {emotion: idx for idx, emotion in enumerate(EMOTION_CLASSES)}

def _to_numeric(arr):
    """تبدیل آرایه رشته‌ای احساسات به مقادیر عددی در صورت نیاز"""
    if arr.dtype.kind in ['U', 'S', 'O']:
        return np.array([_EMOTION_MAP.get(str(x).lower(), 0) for x in arr], dtype=float)
    return np.asarray(arr, dtype=float)


def calculate_intensity(signal: np.ndarray) -> float:
    """۱. شدت (Intensity) - میانگین مقادیر سیگنال"""
    data = _to_numeric(signal)
    return float(np.mean(data))


def calculate_stability(signal: np.ndarray) -> float:
    """۲. پایداری (Stability) - معکوس انحراف معیار (1 - std)"""
    data = _to_numeric(signal)
    return float(1.0 - np.std(data))


def calculate_wavelength(signal: np.ndarray, fps: float = 1.0) -> float:
    """۳. طول موج (Wavelength) - معکوس فرکانس غالب حاصل از FFT"""
    data = _to_numeric(signal)
    centered = data - np.mean(data)
    
    fft_vals = np.abs(np.fft.rfft(centered))
    fft_freqs = np.fft.rfftfreq(len(centered), d=1.0/fps)
    
    if np.max(fft_vals) == 0:
        return float('inf')
    
    dominant_freq = fft_freqs[np.argmax(fft_vals)]
    return float(1.0 / dominant_freq) if dominant_freq != 0 else float('inf')


def calculate_burstiness(signal: np.ndarray, threshold: float = None) -> int:
    data = _to_numeric(signal)
    if threshold is None:
        threshold = np.mean(data) + np.std(data)
        
    peaks, _ = find_peaks(data, height=threshold)
    return float(len(peaks))