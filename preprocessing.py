"""Fold-local scaling, PCA, and sequence padding.

Each cross-validation fold creates a fresh preprocessor, fits it on that fold's
training indices, and then applies the frozen transforms to training and test data.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import BIRDNET_SEQUENCE_STEPS, PCA_COMPONENTS, YAMNET_MAX_FRAMES
from .data import Dataset


@dataclass(frozen=True)
class ModelInputs:
    yamnet_sequence: np.ndarray
    birdnet_sequence: np.ndarray
    acoustic_features: np.ndarray

    def as_tuple(self) -> tuple[np.ndarray, ...]:
        return self.yamnet_sequence, self.birdnet_sequence, self.acoustic_features


class SequenceProjector:
    """Reduce real training frames to 64 dimensions, then standardise them."""

    def __init__(self) -> None:
        self.pca = PCA(n_components=PCA_COMPONENTS, svd_solver="full")
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, training_sequences: tuple[np.ndarray, ...]) -> None:
        training_frames = np.concatenate(training_sequences, axis=0)
        if min(training_frames.shape) < PCA_COMPONENTS:
            raise ValueError("There are not enough training frames for 64-component PCA")
        projected = self.pca.fit_transform(training_frames)
        self.scaler.fit(projected)
        self.is_fitted = True

    def transform_and_pad(
        self, sequences: tuple[np.ndarray, ...], maximum_frames: int
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("SequenceProjector must be fitted before use")
        output = np.zeros(
            (len(sequences), maximum_frames, PCA_COMPONENTS), dtype=np.float32
        )
        for row, sequence in enumerate(sequences):
            if len(sequence) > maximum_frames:
                raise ValueError(
                    f"Sequence has {len(sequence)} frames; limit is {maximum_frames}"
                )
            transformed = self.scaler.transform(self.pca.transform(sequence))
            output[row, : len(sequence)] = transformed.astype(np.float32)
        return output


class FoldPreprocessor:
    """Prepare the three model inputs using training-fold statistics only."""

    def __init__(self) -> None:
        self.acoustic_scaler = StandardScaler()
        self.yamnet_projector = SequenceProjector()
        self.birdnet_projector = SequenceProjector()
        self.is_fitted = False

    @staticmethod
    def _yamnet_plus_acoustic(
        dataset: Dataset, indices: np.ndarray, scaled_acoustic: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        """Append five clip features to every 1024-value YAMNet frame."""
        sequences = []
        for row, dataset_index in enumerate(indices):
            yamnet = dataset.yamnet_sequences[dataset_index]
            repeated_acoustic = np.repeat(
                scaled_acoustic[row][None, :], len(yamnet), axis=0
            )
            combined = np.concatenate([yamnet, repeated_acoustic], axis=1)
            if combined.shape[1] != 1029:
                raise AssertionError("YAMNet plus acoustic frame must have 1029 values")
            sequences.append(combined.astype(np.float32))
        return tuple(sequences)

    def fit(self, dataset: Dataset, training_indices: np.ndarray) -> "FoldPreprocessor":
        training_indices = np.asarray(training_indices, dtype=np.int64)
        training_acoustic = dataset.acoustic_features[training_indices]
        self.acoustic_scaler.fit(training_acoustic)
        scaled_acoustic = self.acoustic_scaler.transform(training_acoustic)
        self.yamnet_projector.fit(
            self._yamnet_plus_acoustic(dataset, training_indices, scaled_acoustic)
        )
        self.birdnet_projector.fit(
            tuple(dataset.birdnet_sequences[index] for index in training_indices)
        )
        self.is_fitted = True
        return self

    def transform(self, dataset: Dataset, indices: np.ndarray) -> ModelInputs:
        if not self.is_fitted:
            raise RuntimeError("FoldPreprocessor must be fitted before use")
        indices = np.asarray(indices, dtype=np.int64)
        scaled_acoustic = self.acoustic_scaler.transform(
            dataset.acoustic_features[indices]
        ).astype(np.float32)
        yamnet_1029 = self._yamnet_plus_acoustic(dataset, indices, scaled_acoustic)
        birdnet = tuple(dataset.birdnet_sequences[index] for index in indices)
        return ModelInputs(
            yamnet_sequence=self.yamnet_projector.transform_and_pad(
                yamnet_1029, YAMNET_MAX_FRAMES
            ),
            birdnet_sequence=self.birdnet_projector.transform_and_pad(
                birdnet, BIRDNET_SEQUENCE_STEPS
            ),
            acoustic_features=scaled_acoustic,
        )
