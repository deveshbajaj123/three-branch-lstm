"""Train and evaluate the canonical model with five-fold StratifiedKFold."""

from collections import Counter
from datetime import datetime, timezone
import gc
import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import (
    DECISION_THRESHOLD,
    LEARNING_RATE,
    N_EPOCHS,
    PIPELINE_VERSION,
    RESULTS_DIR,
    TRAINING_SEED,
)
from .data import Dataset, load_dataset
from .model import build_model
from .preprocessing import FoldPreprocessor
from .splitting import make_folds
from .training import TrainingBatches, balanced_class_weights, set_reproducible_seed


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    predictions = (probabilities >= DECISION_THRESHOLD).astype(np.int64)
    true_control, false_alarm, missed_elephant, true_elephant = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()
    return {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "auc": float(roc_auc_score(labels, probabilities)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(true_control / max(true_control + false_alarm, 1)),
        "true_elephant": int(true_elephant),
        "true_control": int(true_control),
        "false_alarm": int(false_alarm),
        "missed_elephant": int(missed_elephant),
    }


def category_metrics(dataset: Dataset, probabilities: np.ndarray) -> list[dict]:
    predictions = probabilities >= DECISION_THRESHOLD
    rows = []
    for category in sorted(set(dataset.categories)):
        selected = np.asarray([value == category for value in dataset.categories])
        rows.append(
            {
                "category": category,
                "n": int(selected.sum()),
                "elephant_fraction": float(dataset.labels[selected].mean()),
                "accuracy": float((predictions[selected] == dataset.labels[selected]).mean()),
            }
        )
    return rows


def run_cross_validation(dataset: Dataset) -> dict:
    probabilities = np.full(len(dataset), np.nan, dtype=np.float32)
    fold_log = []

    for fold_number, (training_indices, test_indices) in enumerate(
        make_folds(dataset.labels), 1
    ):
        print(f"Fold {fold_number}/5", flush=True)
        preprocessor = FoldPreprocessor().fit(dataset, training_indices)
        training_inputs = preprocessor.transform(dataset, training_indices)
        test_inputs = preprocessor.transform(dataset, test_indices)
        control_weight, elephant_weight = balanced_class_weights(
            dataset.labels[training_indices]
        )

        tf.keras.backend.clear_session()
        set_reproducible_seed(TRAINING_SEED)
        model = build_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="binary_crossentropy",
        )
        batches = TrainingBatches(
            training_inputs,
            dataset.labels[training_indices],
            control_weight,
            elephant_weight,
            TRAINING_SEED,
        )
        history = model.fit(
            batches,
            epochs=N_EPOCHS,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="loss",
                    mode="min",
                    patience=15,
                    restore_best_weights=True,
                    start_from_epoch=12,
                )
            ],
        )
        fold_probabilities = np.asarray(
            model.predict(test_inputs.as_tuple(), verbose=0)
        ).reshape(-1)
        probabilities[test_indices] = fold_probabilities
        fold_log.append(
            {
                "fold": fold_number,
                "training_clips": int(len(training_indices)),
                "test_clips": int(len(test_indices)),
                "epochs": len(history.history["loss"]),
                "metrics": metrics(dataset.labels[test_indices], fold_probabilities),
            }
        )
        del model, batches
        tf.keras.backend.clear_session()
        gc.collect()

    if np.isnan(probabilities).any():
        raise AssertionError("Some clips did not receive an out-of-fold prediction")
    return {
        "protocol": {
            "pipeline_version": PIPELINE_VERSION,
            "model": "three branches: YAMNet LSTM, BirdNET LSTM, acoustic dense",
            "clips": len(dataset),
            "folds": 5,
            "split": "ordinary binary StratifiedKFold",
            "training_seed": TRAINING_SEED,
            "yamnet_pca_input_dimensions": 1029,
            "pca_components_per_embedding_branch": 64,
            "decision_threshold": DECISION_THRESHOLD,
        },
        "overall": metrics(dataset.labels, probabilities),
        "per_category": category_metrics(dataset, probabilities),
        "out_of_fold_probabilities": probabilities.astype(float).tolist(),
        "fold_log": fold_log,
    }


def main() -> None:
    dataset = load_dataset()
    print(
        f"Loaded {len(dataset)} clips: {int(dataset.labels.sum())} elephant and "
        f"{int((dataset.labels == 0).sum())} control",
        flush=True,
    )
    result = run_cross_validation(dataset)
    result.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "cache_sha256": dataset.cache_sha256,
            "cache_signature": dataset.cache_signature,
            "category_counts": dict(sorted(Counter(dataset.categories).items())),
            "interpretation": (
                "This stratified result measures performance on a familiar mixed "
                "distribution. It is not an unseen-recording deployment estimate."
            ),
        }
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "stratified_459_results.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    overall = result["overall"]
    print(
        f"Accuracy={overall['accuracy']:.3f}, AUC={overall['auc']:.3f}, "
        f"F1={overall['f1']:.3f}, recall={overall['recall']:.3f}"
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
