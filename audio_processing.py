"""Locate raw recordings, make ten-second clips, and apply two-band denoising

 Birdnet uses raw recordings and we therefore need to sometimes find the clip from parent recording.
 ."""

from functools import lru_cache
import os
from pathlib import Path
import re
from dataclasses import dataclass

import numpy as np

from .config import (
    CLIP_SECONDS,
    FIELD_RECORDING_NAME,
    NOISE_PROFILE_DIR,
    PROJECT_DIR,
    RAW_SAMPLE_RATE,
)


# These clip numbers were cut from longer recordings. The mapping was established
# by matching their audio content to the source recordings.
CLIP_PARENT_RECORDING = {
    36: "Conkp_23_20231229_134938", 41: "Conkp_23_20231229_134938",
    64: "Conkp_23_20231229_134938", 99: "Conkp_23_20231229_134938",
    148: "Conkp_23_20231229_134938", 203: "Conkp_23_20231229_134938",
    206: "Conkp_23_20231229_134938", 214: "Conkp_23_20231229_134938",
    215: "Conkp_23_20231229_134938", 222: "Conkp_23_20231229_134938",
    42: "Conkp_23_20240103_160006_day time",
    76: "Conkp_23_20240103_160006_day time",
    79: "Conkp_23_20240103_160006_day time",
    86: "Conkp_23_20240103_160006_day time",
    89: "Conkp_23_20240103_160006_day time",
    100: "Conkp_23_20240103_160006_day time",
    118: "Conkp_23_20240103_160006_day time",
    123: "Conkp_23_20240103_160006_day time",
    143: "Conkp_23_20240103_160006_day time",
    152: "Conkp_23_20240103_160006_day time",
    156: "Conkp_23_20240103_160006_day time",
    171: "Conkp_23_20240103_160006_day time",
    183: "Conkp_23_20240103_160006_day time",
}

# No raw source exists for these three curated files. They are the only files
# skipped when the 462 labelled candidates become the 459-clip dataset.
KNOWN_UNRESOLVED_RAW_FILES = {
    "elephant__elephant_231226_125119.wav",
    "rumble_231226_125050.wav",
    "Copy of clip_0084_elephant recording_boost025.wav",
}


@dataclass(frozen=True)
class RawSource:
    """A source WAV and, for long recordings, the ten-second starting offset."""

    path: Path
    kind: str
    start_seconds: int | None = None


def load_audio(path: Path, sample_rate: int = RAW_SAMPLE_RATE) -> np.ndarray:
    import librosa

    waveform, _ = librosa.load(path, sr=sample_rate, mono=True)
    return np.asarray(waveform, dtype=np.float32)


def centre_crop_or_pad(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return exactly ten seconds, cropping or zero-padding symmetrically."""
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    target = CLIP_SECONDS * sample_rate
    if len(waveform) > target:
        start = (len(waveform) - target) // 2
        return waveform[start : start + target].copy()
    if len(waveform) < target:
        missing = target - len(waveform)
        return np.pad(waveform, (missing // 2, missing - missing // 2))
    return waveform.copy()


@lru_cache(maxsize=None)
def find_long_recording(name: str) -> Path:
    """Find a named parent recording in the project or its Downloads folder."""
    search_folders = (PROJECT_DIR / "Fable context", PROJECT_DIR, PROJECT_DIR.parent)
    candidates = []
    search_names = {name, name.removesuffix("_day time")}
    for folder in search_folders:
        if folder.is_dir():
            for search_name in search_names:
                candidates.extend(folder.glob(f"*{search_name}*.wav"))
    large = sorted(path for path in candidates if path.stat().st_size > 10_000_000)
    if not large:
        raise FileNotFoundError(f"Could not find parent recording containing {name!r}")
    return large[0]


@lru_cache(maxsize=1)
def _other_raw_files() -> dict[str, Path]:
    """Index raw WAV files outside directories whose names indicate denoising."""
    result: dict[str, Path] = {}
    for folder, directories, file_names in os.walk(PROJECT_DIR):
        directories.sort()
        file_names.sort()
        lower = folder.casefold()
        if any(
            word in lower
            for word in ("cleaned", "denoised", ".git", "tfhub", "raw_clips_459")
        ):
            directories[:] = []
            continue
        for file_name in file_names:
            if file_name.casefold().endswith(".wav"):
                result.setdefault(file_name, Path(folder) / file_name)
    return result


def raw_audio_for_curated_clip(cleaned_path: Path) -> tuple[np.ndarray | None, str]:
    """Resolve raw audio without silently falling back to a denoised waveform."""
    source = resolve_raw_source(cleaned_path)
    if source is None:
        return None, "unresolved"
    import librosa

    waveform, _ = librosa.load(
        source.path,
        sr=RAW_SAMPLE_RATE,
        mono=True,
        offset=source.start_seconds or 0,
        duration=CLIP_SECONDS if source.start_seconds is not None else None,
    )
    return np.asarray(waveform, dtype=np.float32), source.kind


def resolve_raw_source(cleaned_path: Path) -> RawSource | None:
    """Return the exact raw file and optional offset for one cleaned clip."""
    file_name = cleaned_path.name
    raw_name = file_name.replace("_cleaned", "")
    if raw_name in KNOWN_UNRESOLVED_RAW_FILES:
        return None

    direct = sorted((PROJECT_DIR / "data_multi_class").glob(f"*/{raw_name}"))
    if direct:
        if len(direct) != 1:
            raise RuntimeError(f"Ambiguous primary raw match for {raw_name}: {direct}")
        return RawSource(direct[0], "data_multi_class")

    match = re.fullmatch(r"clip_(\d{4})\.wav", raw_name)
    if match and int(match.group(1)) in CLIP_PARENT_RECORDING:
        clip_number = int(match.group(1))
        parent = find_long_recording(CLIP_PARENT_RECORDING[clip_number])
        return RawSource(
            parent,
            f"recut:{parent.name}",
            clip_number * CLIP_SECONDS,
        )

    loose = _other_raw_files().get(raw_name)
    if loose is not None:
        return RawSource(loose, f"raw_file:{loose.name}")
    return None


@lru_cache(maxsize=1)
def field_recording() -> np.ndarray:
    return load_audio(find_long_recording(FIELD_RECORDING_NAME))


def raw_audio_for_field_clip(clip_number: int) -> tuple[np.ndarray, str]:
    samples_per_clip = CLIP_SECONDS * RAW_SAMPLE_RATE
    start = clip_number * samples_per_clip
    waveform = field_recording()[start : start + samples_per_clip]
    if len(waveform) != samples_per_clip:
        raise ValueError(f"Field clip {clip_number} extends beyond its parent recording")
    return waveform.copy(), f"{FIELD_RECORDING_NAME}@{clip_number * CLIP_SECONDS}s"


@lru_cache(maxsize=1)
def noise_reference() -> np.ndarray:
    """Join the first 30 seconds of each fixed noise-reference recording."""
    import librosa

    paths = sorted(NOISE_PROFILE_DIR.glob("*.wav"))
    if not paths:
        raise FileNotFoundError(f"No noise-profile WAV files found in {NOISE_PROFILE_DIR}")
    pieces = [
        librosa.load(path, sr=RAW_SAMPLE_RATE, mono=True, duration=30)[0]
        for path in paths
    ]
    combined = np.concatenate(pieces).astype(np.float32)
    peak = float(np.max(np.abs(combined)))
    return combined / peak if peak > 0 else combined


def denoise(waveform: np.ndarray) -> np.ndarray:
    """Apply gentler removal below 300 Hz and stronger removal above 300 Hz."""
    import noisereduce as nr
    from scipy import signal

    waveform = np.asarray(waveform, dtype=np.float32)
    reference = noise_reference()
    repeats = int(np.ceil(len(waveform) / len(reference)))
    matched_noise = np.tile(reference, repeats)[: len(waveform)]

    low_filter = signal.butter(6, 300, "low", fs=RAW_SAMPLE_RATE, output="sos")
    high_filter = signal.butter(6, 300, "high", fs=RAW_SAMPLE_RATE, output="sos")
    low_signal = signal.sosfilt(low_filter, waveform)
    high_signal = signal.sosfilt(high_filter, waveform)
    low_noise = signal.sosfilt(low_filter, matched_noise)
    high_noise = signal.sosfilt(high_filter, matched_noise)

    low_clean = nr.reduce_noise(
        y=low_signal, sr=RAW_SAMPLE_RATE, y_noise=low_noise,
        stationary=True, prop_decrease=0.40,
    )
    high_clean = nr.reduce_noise(
        y=high_signal, sr=RAW_SAMPLE_RATE, y_noise=high_noise,
        stationary=True, prop_decrease=0.85,
    )
    cleaned = low_clean + high_clean
    cleaned += 0.25 * signal.sosfilt(low_filter, cleaned)
    high_pass = signal.butter(4, 5, "high", fs=RAW_SAMPLE_RATE, output="sos")
    cleaned = np.nan_to_num(signal.sosfilt(high_pass, cleaned))
    peak = float(np.max(np.abs(cleaned)))
    if peak > 1.0:
        cleaned = cleaned / peak
    return cleaned.astype(np.float32)
