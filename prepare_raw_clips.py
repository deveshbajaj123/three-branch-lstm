"""Materialise the 459 raw ten-second clips used only by the BirdNET branch.

The output folder contains WAV files plus ``manifest.csv``. The manifest records
the labelled/denoised clip, original raw file, any parent-recording offset, and
SHA-256 hashes. No denoising is performed here.
"""

import argparse
import csv
from hashlib import sha256
from pathlib import Path
import shutil

import numpy as np
import soundfile as sf

from .audio_processing import RawSource, find_long_recording, resolve_raw_source
from .config import (
    CLEANED_CLIP_DIR,
    CLIP_SECONDS,
    EXPECTED_CLIP_COUNT,
    EXPECTED_ELEPHANT_COUNT,
    FIELD_CLIP_IDS,
    FIELD_DENOISED_DIR,
    FIELD_ELEPHANT_IDS,
    FIELD_RECORDING_NAME,
    PROJECT_DIR,
    RAW_CLIP_DIR,
    RAW_CLIP_MANIFEST,
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_exact_ten_seconds(source: RawSource) -> tuple[np.ndarray, int, float]:
    """Read/crop/pad at the source rate without resampling or denoising."""
    information = sf.info(source.path)
    sample_rate = int(information.samplerate)
    target_samples = CLIP_SECONDS * sample_rate
    if source.start_seconds is not None:
        first_sample = source.start_seconds * sample_rate
    else:
        first_sample = max((int(information.frames) - target_samples) // 2, 0)
    audio, read_rate = sf.read(
        source.path,
        start=first_sample,
        frames=target_samples,
        dtype="int16",
        always_2d=True,
    )
    if int(read_rate) != sample_rate:
        raise AssertionError("SoundFile returned a different sample rate")
    if audio.shape[1] > 1:
        audio = np.rint(audio.astype(np.float64).mean(axis=1)).astype(np.int16)
    else:
        audio = audio[:, 0]
    if len(audio) < target_samples:
        missing = target_samples - len(audio)
        audio = np.pad(audio, (missing // 2, missing - missing // 2))
    if len(audio) != target_samples:
        raise AssertionError("Raw clip is not exactly ten seconds")
    return audio, sample_rate, first_sample / sample_rate


def _write_clip(source: RawSource, destination: Path) -> tuple[int, float]:
    audio, sample_rate, actual_start = _read_exact_ten_seconds(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, audio, sample_rate, subtype="PCM_16")
    written = sf.info(destination)
    if written.frames != CLIP_SECONDS * written.samplerate:
        raise AssertionError(f"Exported clip has wrong duration: {destination}")
    return sample_rate, actual_start


def build_raw_clip_folder(force: bool = False) -> None:
    """Create 459 raw clips and their complete provenance manifest."""
    if RAW_CLIP_DIR.exists():
        if not force:
            raise FileExistsError(
                f"{RAW_CLIP_DIR} already exists. Use --force to rebuild it."
            )
        resolved = RAW_CLIP_DIR.resolve()
        if resolved.parent != Path(__file__).resolve().parent:
            raise RuntimeError(f"Refusing to remove unexpected directory: {resolved}")
        shutil.rmtree(resolved)

    rows: list[dict[str, object]] = []
    source_hashes: dict[Path, str] = {}

    def add_row(
        identifier: str,
        category: str,
        label: int,
        denoised_path: Path,
        raw_source: RawSource,
        destination: Path,
    ) -> None:
        sample_rate, actual_start = _write_clip(raw_source, destination)
        source_hashes.setdefault(raw_source.path, file_sha256(raw_source.path))
        rows.append(
            {
                "clip_identifier": identifier,
                "category": category,
                "label": label,
                "denoised_path": str(denoised_path.resolve()),
                "raw_clip_path": str(destination.resolve()),
                "raw_source_kind": raw_source.kind,
                "raw_source_path": str(raw_source.path.resolve()),
                "raw_source_start_seconds": f"{actual_start:.6f}",
                "raw_source_end_seconds": f"{actual_start + CLIP_SECONDS:.6f}",
                "sample_rate": sample_rate,
                "source_file_sha256": source_hashes[raw_source.path],
                "raw_clip_sha256": file_sha256(destination),
                "denoised_clip_sha256": file_sha256(denoised_path),
            }
        )

    unresolved = []
    for category_folder in sorted(CLEANED_CLIP_DIR.iterdir()):
        if not category_folder.is_dir():
            continue
        category = category_folder.name
        label = 0 if category.startswith("control") else 1
        for denoised_path in sorted(category_folder.glob("*.wav")):
            raw_source = resolve_raw_source(denoised_path)
            if raw_source is None:
                unresolved.append(str(denoised_path.resolve()))
                continue
            raw_name = denoised_path.name.replace("_cleaned", "")
            destination = RAW_CLIP_DIR / category / raw_name
            add_row(
                denoised_path.relative_to(PROJECT_DIR).as_posix(),
                category,
                label,
                denoised_path,
                raw_source,
                destination,
            )

    field_parent = find_long_recording(FIELD_RECORDING_NAME)
    for clip_number in FIELD_CLIP_IDS:
        label = int(clip_number in FIELD_ELEPHANT_IDS)
        category = "field_elephant" if label else "field_control"
        denoised_path = FIELD_DENOISED_DIR / f"clip_{clip_number:04d}.wav"
        if not denoised_path.is_file():
            raise FileNotFoundError(f"Missing field denoised clip: {denoised_path}")
        raw_source = RawSource(
            field_parent,
            f"recut:{field_parent.name}",
            clip_number * CLIP_SECONDS,
        )
        destination = RAW_CLIP_DIR / "field_16_january" / f"clip_{clip_number:04d}.wav"
        add_row(
            f"field_16_january/clip_{clip_number:04d}.wav",
            category,
            label,
            denoised_path,
            raw_source,
            destination,
        )

    if len(rows) != EXPECTED_CLIP_COUNT:
        raise AssertionError(
            f"Expected {EXPECTED_CLIP_COUNT} raw clips, wrote {len(rows)}; "
            f"unresolved={unresolved}"
        )
    if sum(int(row["label"]) for row in rows) != EXPECTED_ELEPHANT_COUNT:
        raise AssertionError("Unexpected elephant count in raw manifest")
    with RAW_CLIP_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Created {len(rows)} raw clips in {RAW_CLIP_DIR}\n"
        f"Skipped {len(unresolved)} curated files with no raw source\n"
        f"Manifest: {RAW_CLIP_MANIFEST}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace the raw folder")
    arguments = parser.parse_args()
    build_raw_clip_folder(arguments.force)


if __name__ == "__main__":
    main()
