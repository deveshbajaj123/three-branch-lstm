"""Generate the complete 459-clip YAMNet, BirdNET, and acoustic-feature cache.

This is to make iteration faster and not have to generate embeddings over and over again
"""

import argparse
import csv
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
import sys

import joblib
import librosa
import numpy as np
import tensorflow as tf

from .acoustic_features import FEATURE_NAMES, extract_acoustic_features
from .audio_processing import centre_crop_or_pad
from .config import (
    BIRDNET_HOP_SECONDS,
    BIRDNET_SEQUENCE_STEPS,
    BIRDNET_SAMPLE_RATE,
    BIRDNET_SPEED_FACTOR,
    BIRDNET_WINDOW_SECONDS,
    CACHE_FORMAT_VERSION,
    CLEANED_CLIP_DIR,
    EMBEDDING_DIMENSION,
    EXPECTED_CLIP_COUNT,
    FEATURE_CACHE,
    FIELD_CLIP_IDS,
    FIELD_DENOISED_DIR,
    FIELD_ELEPHANT_IDS,
    PINNED_YAMNET_DIR,
    PROJECT_DIR,
    RAW_SAMPLE_RATE,
    RAW_CLIP_DIR,
    RAW_CLIP_MANIFEST,
    YAMNET_SAMPLE_RATE,
)


BIRDNET_INPUT_SAMPLES = BIRDNET_SAMPLE_RATE * BIRDNET_WINDOW_SECONDS


@dataclass(frozen=True)
class Clip:
    identifier: str
    category: str
    label: int
    denoised_path: Path
    raw_path: Path


def labelled_candidates() -> list[Clip]:
    """Return 428 curated clips followed by the 34 selected passive windows."""
    clips: list[Clip] = []
    for category_folder in sorted(CLEANED_CLIP_DIR.iterdir()):
        if not category_folder.is_dir():
            continue
        label = 0 if category_folder.name.startswith("control") else 1
        for path in sorted(category_folder.glob("*.wav")):
            clips.append(
                Clip(
                    identifier=path.relative_to(PROJECT_DIR).as_posix(),
                    category=category_folder.name,
                    label=label,
                    denoised_path=path,
                    raw_path=(
                        RAW_CLIP_DIR
                        / category_folder.name
                        / path.name.replace("_cleaned", "")
                    ),
                )
            )
    for clip_number in FIELD_CLIP_IDS:
        label = int(clip_number in FIELD_ELEPHANT_IDS)
        clips.append(
            Clip(
                identifier=f"field_16_january/clip_{clip_number:04d}.wav",
                category="field_elephant" if label else "field_control",
                label=label,
                denoised_path=FIELD_DENOISED_DIR / f"clip_{clip_number:04d}.wav",
                raw_path=RAW_CLIP_DIR / "field_16_january" / f"clip_{clip_number:04d}.wav",
            )
        )
    if len(clips) != 462:
        raise AssertionError(f"Expected 462 labelled candidates, found {len(clips)}")
    return clips


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_birdnet_model() -> Path:
    """Locate BirdNET GLOBAL 6K V2.4's full-precision TFLite model."""
    relative = Path("models/analyzer/BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite")
    candidates: list[Path] = []
    package = importlib.util.find_spec("birdnetlib")
    if package and package.submodule_search_locations:
        candidates.append(Path(next(iter(package.submodule_search_locations))) / relative)
    candidates.extend(
        [
            Path(sys.prefix) / "Lib/site-packages/birdnetlib" / relative,
            Path("C:/Python312/Lib/site-packages/birdnetlib") / relative,
            PROJECT_DIR / "birdnetlib" / relative,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "BirdNET GLOBAL 6K V2.4 FP32 was not found. Install birdnetlib or place "
        "its model under birdnetlib/models/analyzer in the project."
    )


def load_yamnet():
    """Load the pinned local YAMNet model; no network model is substituted."""
    saved_model = PINNED_YAMNET_DIR / "saved_model.pb"
    if not saved_model.is_file():
        raise FileNotFoundError(f"Pinned YAMNet is missing: {saved_model}")
    return tf.saved_model.load(str(PINNED_YAMNET_DIR))


class BirdNetEmbedder:
    """Small wrapper exposing BirdNET's validated 1024-value embedding tensor."""

    def __init__(self, model_path: Path) -> None:
        self.interpreter = tf.lite.Interpreter(
            model_path=str(model_path),
            experimental_preserve_all_tensors=True,
            num_threads=4,
        )
        input_detail = self.interpreter.get_input_details()[0]
        self.input_index = int(input_detail["index"])
        self.interpreter.resize_tensor_input(
            self.input_index, [1, BIRDNET_INPUT_SAMPLES]
        )
        self.interpreter.allocate_tensors()
        tensors = [
            detail for detail in self.interpreter.get_tensor_details()
            if detail["name"].endswith("GLOBAL_AVG_POOL/Mean")
            and tuple(detail["shape"]) == (1, EMBEDDING_DIMENSION)
        ]
        if len(tensors) != 1:
            raise AssertionError(
                f"Expected one BirdNET embedding tensor, found {len(tensors)}"
            )
        self.embedding_index = int(tensors[0]["index"])
        self.embedding_name = str(tensors[0]["name"])

    @staticmethod
    def speed_up_16_times(waveform: np.ndarray) -> np.ndarray:
        """Transpose 24-kHz audio by 16x, then tile it to BirdNET's 3-s input."""
        shifted = librosa.resample(
            np.asarray(waveform, dtype=np.float32),
            orig_sr=RAW_SAMPLE_RATE,
            target_sr=BIRDNET_SAMPLE_RATE // BIRDNET_SPEED_FACTOR,
        ).astype(np.float32)
        if not len(shifted):
            return np.zeros(BIRDNET_INPUT_SAMPLES, dtype=np.float32)
        repeats = math.ceil(BIRDNET_INPUT_SAMPLES / len(shifted))
        return np.tile(shifted, repeats)[:BIRDNET_INPUT_SAMPLES]

    def embed(self, waveform: np.ndarray) -> np.ndarray:
        model_input = self.speed_up_16_times(waveform)
        self.interpreter.set_tensor(self.input_index, model_input[None, :])
        self.interpreter.invoke()
        embedding = self.interpreter.get_tensor(self.embedding_index)[0].copy()
        if embedding.shape != (EMBEDDING_DIMENSION,):
            raise AssertionError(f"Unexpected BirdNET embedding shape {embedding.shape}")
        return embedding.astype(np.float32)

    def sequence(self, ten_second_waveform: np.ndarray) -> np.ndarray:
        """Embed eight overlapping 3-s windows at a 1-s interval."""
        window = BIRDNET_WINDOW_SECONDS * RAW_SAMPLE_RATE
        hop = BIRDNET_HOP_SECONDS * RAW_SAMPLE_RATE
        embeddings = [
            self.embed(ten_second_waveform[start : start + window])
            for start in range(0, len(ten_second_waveform) - window + 1, hop)
        ]
        sequence = np.stack(embeddings)
        if sequence.shape != (BIRDNET_SEQUENCE_STEPS, EMBEDDING_DIMENSION):
            raise AssertionError(f"Unexpected BirdNET sequence shape {sequence.shape}")
        return sequence


def cache_signature(clips: list[Clip], birdnet_path: Path) -> dict[str, object]:
    identity = [(clip.identifier, clip.label, clip.category) for clip in clips]
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "candidate_identity_sha256": sha256(
            json.dumps(identity, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "yamnet_saved_model_sha256": file_sha256(PINNED_YAMNET_DIR / "saved_model.pb"),
        "birdnet_model_sha256": file_sha256(birdnet_path),
        "birdnet_model_name": birdnet_path.name,
        "pipeline_source_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in ("feature_cache.py", "acoustic_features.py")
        },
        "raw_clip_manifest_sha256": file_sha256(RAW_CLIP_MANIFEST),
        "birdnet_speed_factor": BIRDNET_SPEED_FACTOR,
        "clip_seconds": 10,
        "crop": "centre",
        "short_clip_policy": "symmetric_zero_padding",
        "yamnet_input": "existing_denoised_clip",
        "birdnet_input": "materialised_raw_clip",
        "acoustic_feature_names": list(FEATURE_NAMES),
    }


def load_and_validate_raw_manifest(clips: list[Clip]) -> dict[str, dict[str, str]]:
    """Make the raw/denoised pairing explicit and reject any changed file."""
    with RAW_CLIP_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_identifier = {row["clip_identifier"]: row for row in rows}
    if len(rows) != EXPECTED_CLIP_COUNT or len(by_identifier) != len(rows):
        raise ValueError("Raw manifest must contain 459 unique clip identifiers")
    for clip in clips:
        row = by_identifier.get(clip.identifier)
        if row is None:
            continue  # the three curated examples with no recoverable raw audio
        if Path(row["raw_clip_path"]).resolve() != clip.raw_path.resolve():
            raise ValueError(f"Raw path mismatch for {clip.identifier}")
        if Path(row["denoised_path"]).resolve() != clip.denoised_path.resolve():
            raise ValueError(f"Denoised path mismatch for {clip.identifier}")
        if int(row["label"]) != clip.label or row["category"] != clip.category:
            raise ValueError(f"Label/category mismatch for {clip.identifier}")
        if "cleaned" in str(clip.raw_path).casefold() or "denoised" in str(clip.raw_path).casefold():
            raise ValueError(f"BirdNET path appears processed: {clip.raw_path}")
    return by_identifier


def build_cache(output_path: Path = FEATURE_CACHE, force: bool = False) -> dict:
    """Extract all three feature families and save one aligned cache."""
    candidates = labelled_candidates()
    if not RAW_CLIP_MANIFEST.is_file():
        raise FileNotFoundError(
            f"Raw clip manifest is missing: {RAW_CLIP_MANIFEST}\n"
            "Create it with: python -m readable_three_branch_stratified.prepare_raw_clips"
        )
    manifest = load_and_validate_raw_manifest(candidates)
    birdnet_path = find_birdnet_model()
    signature = cache_signature(candidates, birdnet_path)
    partial_path = output_path.with_suffix(".partial.joblib")

    if force:
        output_path.unlink(missing_ok=True)
        partial_path.unlink(missing_ok=True)
    if output_path.exists():
        existing = joblib.load(output_path)
        if existing.get("signature") != signature:
            raise RuntimeError("Existing cache uses different inputs; rebuild with --force")
        print(f"Cache is already current: {output_path}")
        return existing

    completed: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    if partial_path.exists():
        partial = joblib.load(partial_path)
        if partial.get("signature") != signature:
            raise RuntimeError("Partial cache uses different inputs; rebuild with --force")
        completed = partial["completed"]
        skipped = partial["skipped"]
        print(f"Resuming after {len(completed) + len(skipped)} candidates")

    yamnet = load_yamnet()
    birdnet = BirdNetEmbedder(birdnet_path)
    yamnet(tf.zeros(YAMNET_SAMPLE_RATE, dtype=tf.float32))

    for position, clip in enumerate(candidates, 1):
        if clip.identifier in completed or clip.identifier in skipped:
            continue
        manifest_row = manifest.get(clip.identifier)
        if manifest_row is None or not clip.raw_path.is_file():
            skipped[clip.identifier] = "raw clip unavailable"
        else:
            if not clip.denoised_path.is_file():
                raise FileNotFoundError(f"Missing denoised clip: {clip.denoised_path}")
            raw, _ = librosa.load(clip.raw_path, sr=RAW_SAMPLE_RATE, mono=True)
            raw_hash = file_sha256(clip.raw_path)
            denoised_hash = file_sha256(clip.denoised_path)
            if raw_hash != manifest_row["raw_clip_sha256"]:
                raise ValueError(f"Raw clip changed after manifest creation: {clip.identifier}")
            if denoised_hash != manifest_row["denoised_clip_sha256"]:
                raise ValueError(
                    f"Denoised clip changed after manifest creation: {clip.identifier}"
                )
            raw = np.asarray(raw, dtype=np.float32)
            raw_ten_seconds = centre_crop_or_pad(raw, RAW_SAMPLE_RATE)
            denoised_16k, _ = librosa.load(
                clip.denoised_path, sr=YAMNET_SAMPLE_RATE, mono=True
            )
            denoised_16k = np.asarray(denoised_16k, dtype=np.float32)
            denoised_16k = centre_crop_or_pad(denoised_16k, YAMNET_SAMPLE_RATE)
            _, yamnet_sequence, _ = yamnet(tf.convert_to_tensor(denoised_16k))
            yamnet_sequence = np.asarray(yamnet_sequence.numpy(), dtype=np.float32)
            acoustic = extract_acoustic_features(denoised_16k, YAMNET_SAMPLE_RATE)
            completed[clip.identifier] = {
                "label": clip.label,
                "category": clip.category,
                "raw_clip_path": str(clip.raw_path.resolve()),
                "denoised_clip_path": str(clip.denoised_path.resolve()),
                "raw_clip_sha256": raw_hash,
                "denoised_clip_sha256": denoised_hash,
                "yamnet_sequence": yamnet_sequence,
                "birdnet_sequence": birdnet.sequence(raw_ten_seconds),
                "acoustic_features": acoustic,
            }
        if position % 10 == 0 or position == len(candidates):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {"signature": signature, "completed": completed, "skipped": skipped},
                partial_path,
            )
            print(f"  {position}/{len(candidates)} candidates", flush=True)

    ordered = [clip for clip in candidates if clip.identifier in completed]
    if len(ordered) != EXPECTED_CLIP_COUNT:
        raise AssertionError(
            f"Expected {EXPECTED_CLIP_COUNT} usable clips, found {len(ordered)}; "
            f"unresolved: {sorted(skipped)}"
        )
    cache = {
        "signature": signature,
        "files": [clip.identifier for clip in ordered],
        "labels": np.asarray([completed[clip.identifier]["label"] for clip in ordered]),
        "categories": [completed[clip.identifier]["category"] for clip in ordered],
        "raw_clip_path": [completed[clip.identifier]["raw_clip_path"] for clip in ordered],
        "denoised_clip_path": [
            completed[clip.identifier]["denoised_clip_path"] for clip in ordered
        ],
        "raw_clip_sha256": [
            completed[clip.identifier]["raw_clip_sha256"] for clip in ordered
        ],
        "denoised_clip_sha256": [
            completed[clip.identifier]["denoised_clip_sha256"] for clip in ordered
        ],
        "yamnet_sequences": tuple(
            completed[clip.identifier]["yamnet_sequence"] for clip in ordered
        ),
        "birdnet_sequences": tuple(
            completed[clip.identifier]["birdnet_sequence"] for clip in ordered
        ),
        "acoustic_features": np.stack(
            [completed[clip.identifier]["acoustic_features"] for clip in ordered]
        ).astype(np.float32),
        "skipped_unresolved_raw": skipped,
    }
    joblib.dump(cache, output_path)
    partial_path.unlink(missing_ok=True)
    print(f"Saved {len(ordered)} clips to {output_path}")
    return cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=FEATURE_CACHE)
    parser.add_argument("--force", action="store_true", help="replace an existing cache")
    arguments = parser.parse_args()
    build_cache(arguments.output, arguments.force)


if __name__ == "__main__":
    main()
