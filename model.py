"""The three-branch Keras classifier used in this project."""

from tensorflow import keras

from .config import (
    ACOUSTIC_FEATURE_COUNT,
    BIRDNET_SEQUENCE_STEPS,
    DROPOUT,
    EXPECTED_MODEL_PARAMETERS,
    L2_STRENGTH,
    LSTM_UNITS,
    PCA_COMPONENTS,
    RECURRENT_DROPOUT,
    YAMNET_MAX_FRAMES,
)


def sequence_branch(sequence_input, name: str):
    """Summarise embedding sequence with one 32-unit LSTM to be used by Birdnet, Yamnet and augmented features."""
    masked = keras.layers.Masking(mask_value=0.0, name=f"{name}_mask")(sequence_input)
    return keras.layers.LSTM(
        LSTM_UNITS,
        dropout=DROPOUT,
        recurrent_dropout=RECURRENT_DROPOUT,
        kernel_regularizer=keras.regularizers.l2(L2_STRENGTH),
        name=f"{name}_lstm",
    )(masked)


def build_model() -> keras.Model:
    """Build the YAMNet, BirdNET, and acoustic-feature branches."""
    yamnet_input = keras.Input(
        shape=(YAMNET_MAX_FRAMES, PCA_COMPONENTS), name="yamnet_sequence"
    )
    birdnet_input = keras.Input(
        shape=(BIRDNET_SEQUENCE_STEPS, PCA_COMPONENTS), name="birdnet_sequence"
    )
    acoustic_input = keras.Input(
        shape=(ACOUSTIC_FEATURE_COUNT,), name="acoustic_features"
    )

    yamnet_summary = sequence_branch(yamnet_input, "yamnet")
    birdnet_summary = sequence_branch(birdnet_input, "birdnet")
    acoustic_summary = keras.layers.Dense(
        16,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(L2_STRENGTH),
        name="acoustic_dense_16",
    )(acoustic_input)
    acoustic_summary = keras.layers.Dense(
        8, activation="relu", name="acoustic_dense_8"
    )(acoustic_summary)

    combined = keras.layers.Concatenate(name="combine_three_branches")(
        [yamnet_summary, birdnet_summary, acoustic_summary]
    )
    combined = keras.layers.Dropout(DROPOUT, name="combined_dropout")(combined)
    combined = keras.layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(L2_STRENGTH),
        name="combined_dense",
    )(combined)
    probability = keras.layers.Dense(
        1, activation="sigmoid", name="elephant_probability"
    )(combined)

    model = keras.Model(
        inputs=(yamnet_input, birdnet_input, acoustic_input),
        outputs=probability,
        name="three_branch_elephant_classifier",
    )
    if model.count_params() != EXPECTED_MODEL_PARAMETERS:
        raise AssertionError(
            f"Expected {EXPECTED_MODEL_PARAMETERS:,} parameters, "
            f"found {model.count_params():,}"
        )
    return model
