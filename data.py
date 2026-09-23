"""Load the single dataset used by the model."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import joblib
import numpy as np

from .config import (
    ACOUSTIC_FEATURE_COUNT,
    BIRDNET_SEQUENCE_STEPS,
    CACHE_FORMAT_VERSION,
    EMBEDDING_DIMENSION,
    EXPECTED_CLIP_COUNT,
    EXPECTED_CONTROL_COUNT,
    EXPECTED_ELEPHANT_COUNT,
    FEATURE_CACHE,
    YAMNET_MAX_FRAMES,
)


@dataclass(frozen=True)
class Dataset:
    files: tuple[str, ...]
    labels: np.ndarray
    categories: tuple[str, ...]
    yamnet_sequences: tuple[np.ndarray, ...]
    birdnet_sequences: tuple[np.ndarray, ...]
    acoustic_features: np.ndarray
    cache_sha256: str
    cache_signature: dict[str, object]

    def __len__(self) -> int:
        return len(self.files)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sequence(value, expected_frames: int | None, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(f"{name} has invalid shape {array.shape}")
    if expected_frames is not None and array.shape[0] != expected_frames:
        raise ValueError(f"{name} has {array.shape[0]} frames; expected {expected_frames}")
    if name == "YAMNet" and not 1 <= array.shape[0] <= YAMNET_MAX_FRAMES:
        raise ValueError(f"YAMNet has an unsupported frame count: {array.shape[0]}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-finite value")
    return array


def load_dataset(cache_path: Path = FEATURE_CACHE) -> Dataset:
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Feature cache is missing: {cache_path}\n"
            "Generate it with: python -m readable_three_branch_stratified.feature_cache"
        )
    cache = joblib.load(cache_path)
    required = {
        "signature", "files", "labels", "categories", "yamnet_sequences",
        "birdnet_sequences", "acoustic_features",
    }
    missing = required.difference(cache)
    if missing:
        raise KeyError(f"Feature cache is missing fields: {sorted(missing)}")
    if cache["signature"].get("format_version") != CACHE_FORMAT_VERSION:
        raise ValueError("Feature cache format is out of date; regenerate it with --force")

    files = tuple(cache["files"])
    labels = np.asarray(cache["labels"], dtype=np.int64)
    categories = tuple(cache["categories"])
    acoustic = np.asarray(cache["acoustic_features"], dtype=np.float32)
    yamnet = tuple(_sequence(value, None, "YAMNet") for value in cache["yamnet_sequences"])
    birdnet = tuple(
        _sequence(value, BIRDNET_SEQUENCE_STEPS, "BirdNET")
        for value in cache["birdnet_sequences"]
    )

    lengths = {len(files), len(labels), len(categories), len(yamnet), len(birdnet), len(acoustic)}
    if lengths != {EXPECTED_CLIP_COUNT}:
        raise ValueError(f"The default dataset must contain exactly {EXPECTED_CLIP_COUNT} clips")
    if len(set(files)) != len(files):
        raise ValueError("The cache contains duplicate clip identifiers")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Labels must be binary values 0 and 1")
    if int(labels.sum()) != EXPECTED_ELEPHANT_COUNT:
        raise ValueError("Unexpected elephant count in the 459-clip dataset")
    if int((labels == 0).sum()) != EXPECTED_CONTROL_COUNT:
        raise ValueError("Unexpected control count in the 459-clip dataset")
    if acoustic.shape != (EXPECTED_CLIP_COUNT, ACOUSTIC_FEATURE_COUNT):
        raise ValueError(f"Unexpected acoustic-feature shape {acoustic.shape}")
    if not np.isfinite(acoustic).all():
        raise ValueError("Acoustic features contain a non-finite value")

    return Dataset(
        files=files,
        labels=labels,
        categories=categories,
        yamnet_sequences=yamnet,
        birdnet_sequences=birdnet,
        acoustic_features=acoustic,
        cache_sha256=_file_sha256(cache_path),
        cache_signature=dict(cache["signature"]),
    )
