"""Five low-frequency acoustic features supplied to the augmented branch."""

import numpy as np
from scipy.signal import butter, sosfilt


FEATURE_NAMES = (
    "peak_to_mean_10_300_hz",
    "coefficient_of_variation_10_300_hz",
    "harmonic_structure_10_37_over_37_173_hz",
    "rumble_to_total_energy_10_173_hz",
    "sustained_sub_50_hz_seconds",
)


def _band_energy(frame: np.ndarray, sample_rate: int, low: float, high: float) -> float:
    nyquist = sample_rate / 2.0
    low_normalised = max(low / nyquist, 0.001)
    high_normalised = min(high / nyquist, 0.999)
    if low_normalised >= high_normalised or len(frame) < 20:
        return 0.0
    band_filter = butter(
        4, [low_normalised, high_normalised], btype="band", output="sos"
    )
    return float(np.mean(sosfilt(band_filter, frame) ** 2))


def _low_frequency_energy(frame: np.ndarray, sample_rate: int) -> float:
    low_filter = butter(4, 50 / (sample_rate / 2.0), btype="low", output="sos")
    return float(np.mean(sosfilt(low_filter, frame) ** 2))


def extract_acoustic_features(
    waveform: np.ndarray, sample_rate: int, window_seconds: float = 0.5
) -> np.ndarray:
    """Return the five features in ``FEATURE_NAMES`` order."""
    waveform = np.asarray(waveform, dtype=np.float64)
    frame_length = int(sample_rate * window_seconds)
    frame_count = max(1, len(waveform) // frame_length)

    intermediate = np.zeros(frame_count)
    rumble = np.zeros(frame_count)
    fundamental = np.zeros(frame_count)
    upper_harmonics = np.zeros(frame_count)
    below_50 = np.zeros(frame_count)
    total = np.zeros(frame_count)

    for index in range(frame_count):
        frame = waveform[index * frame_length : (index + 1) * frame_length]
        intermediate[index] = _band_energy(frame, sample_rate, 10, 300)
        rumble[index] = _band_energy(frame, sample_rate, 10, 173)
        fundamental[index] = _band_energy(frame, sample_rate, 10, 37)
        upper_harmonics[index] = _band_energy(frame, sample_rate, 37, 173)
        below_50[index] = _low_frequency_energy(frame, sample_rate)
        total[index] = float(np.mean(frame ** 2))

    epsilon = 1e-12
    peak_to_mean = float(np.max(intermediate) / (np.mean(intermediate) + epsilon))
    coefficient_of_variation = float(
        np.std(intermediate) / (np.mean(intermediate) + epsilon)
    )
    harmonic_structure = float(
        (np.mean(fundamental) + epsilon) / (np.mean(upper_harmonics) + epsilon)
    )
    rumble_to_total = float(
        (np.mean(rumble) + epsilon) / (np.mean(total) + epsilon)
    )

    longest_run = current_run = 0
    if np.max(below_50) > epsilon:
        for is_above_median in below_50 > np.median(below_50):
            current_run = current_run + 1 if is_above_median else 0
            longest_run = max(longest_run, current_run)
    sustained_seconds = float(longest_run * window_seconds)

    return np.asarray(
        [peak_to_mean, coefficient_of_variation, harmonic_structure,
         rumble_to_total, sustained_seconds],
        dtype=np.float32,
    )
