"""Reproducible Keras training helpers."""

import random

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .config import BATCH_SIZE, FRAME_MASK_FRACTION
from .preprocessing import ModelInputs


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def balanced_class_weights(labels: np.ndarray) -> tuple[float, float]:
    control_count = max(int((labels == 0).sum()), 1)
    elephant_count = max(int((labels == 1).sum()), 1)
    return len(labels) / (2 * control_count), len(labels) / (2 * elephant_count)


class TrainingBatches(keras.utils.Sequence):
    """Shuffle clips and mask short frame ranges only while training."""

    def __init__(
        self,
        inputs: ModelInputs,
        labels: np.ndarray,
        control_weight: float,
        elephant_weight: float,
        seed: int,
    ) -> None:
        super().__init__()
        self.inputs = inputs.as_tuple()
        self.labels = np.asarray(labels, dtype=np.float32)
        self.control_weight = control_weight
        self.elephant_weight = elephant_weight
        self.random = np.random.default_rng(seed)
        self.order = np.arange(len(labels))

    def __len__(self) -> int:
        return int(np.ceil(len(self.labels) / BATCH_SIZE))

    def on_epoch_end(self) -> None:
        self.random.shuffle(self.order)

    def __getitem__(self, batch_number: int):
        chosen = self.order[
            batch_number * BATCH_SIZE : (batch_number + 1) * BATCH_SIZE
        ]
        batch_inputs = [array[chosen].copy() for array in self.inputs]
        batch_labels = self.labels[chosen].copy()

        # The first two inputs are temporal sequences. Masking a short contiguous
        # region reduces reliance on any single embedding frame.
        for sequence in batch_inputs[:2]:
            maximum = max(int(sequence.shape[1] * FRAME_MASK_FRACTION), 1)
            count = int(self.random.integers(0, maximum + 1))
            if count:
                start = int(self.random.integers(0, sequence.shape[1] - count + 1))
                sequence[:, start : start + count] = 0.0

        sample_weights = np.where(
            batch_labels == 1, self.elephant_weight, self.control_weight
        ).astype(np.float32)
        return tuple(batch_inputs), batch_labels, sample_weights
