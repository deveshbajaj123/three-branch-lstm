"""ordinary binary StratifiedKFold."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .config import N_FOLDS, SPLIT_SEED


def make_folds(labels: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = np.asarray(labels, dtype=np.int64)
    splitter = StratifiedKFold(
        n_splits=N_FOLDS, shuffle=True, random_state=SPLIT_SEED
    )
    folds = list(splitter.split(np.zeros(len(labels)), labels))
    validate_folds(folds, labels)
    return folds


def validate_folds(
    folds: list[tuple[np.ndarray, np.ndarray]], labels: np.ndarray
) -> None:
    all_test_indices: list[int] = []
    expected = set(range(len(labels)))
    for fold_number, (training_indices, test_indices) in enumerate(folds, 1):
        training_set = set(training_indices.tolist())
        test_set = set(test_indices.tolist())
        if training_set & test_set:
            raise AssertionError(f"Fold {fold_number} has train/test index overlap")
        if training_set | test_set != expected:
            raise AssertionError(f"Fold {fold_number} does not cover the population")
        if set(np.unique(labels[test_indices])) != {0, 1}:
            raise AssertionError(f"Fold {fold_number} test set is missing a binary class")
        all_test_indices.extend(test_indices.tolist())
    if sorted(all_test_indices) != list(range(len(labels))):
        raise AssertionError("Each clip must appear in exactly one test fold")
