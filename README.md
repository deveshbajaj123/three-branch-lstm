# three-branch elephant-call classifier



1. denoised audio → YAMNet frame embeddings → 64-component PCA → 32-unit LSTM;
2. raw audio → BirdNET embeddings at 16× speed → 64-component PCA → 32-unit LSTM;
3. five denoised low-frequency acoustic features → two small dense layers.

The three summaries are concatenated and passed through dropout, one 32-unit
dense layer, and a sigmoid output. 

## Files, in pipeline order

- `config.py` contains every path and experimental setting.
- `audio_processing.py` resolves each denoised clip back to its raw recording and
  retains the original denoising procedure only as a rebuilding utility.
- `prepare_raw_clips.py` creates a separate `raw_clips_459` folder for BirdNET
  and records the source path, time offset, and hashes in `manifest.csv`.
- `acoustic_features.py` calculates the five augmented features.
- `feature_cache.py` runs YAMNet and BirdNET and creates the aligned 459-clip
  feature cache.
- `data.py` loads and validates that cache.
- `splitting.py` creates the five binary-stratified folds.
- `preprocessing.py` fits StandardScaler and PCA on each training fold only.
- `model.py` defines the three-branch Keras network.
- `training.py` handles one-seed batching, class weights, and frame masking.
- `run_stratified_cv.py` trains, predicts, and calculates the final metrics.

Run `python -m readable_three_branch_stratified` to test it out.

The 459 examples have two physically separate sources. YAMNet and the acoustic
features read the existing denoised files. BirdNET reads only files under
`raw_clips_459`. Three curated examples without a recoverable raw source are not
included; the remaining 425 curated and 34 manually reviewed passive clips form
the 459-clip dataset.

### YAMNet and augmented features

The default feature-cache build directly reads the existing files in
`data_multi_class_background_cleaned` and the existing 16 January
`clips_denoised` directory. It does not apply denoising again. Those files were
originally produced by splitting at 300 Hz and applying gentler stationary noise
reduction below 300 Hz (`prop_decrease=0.40`) and stronger reduction above it
(`0.85`). Local YAMNet produces a sequence of 1024-value frame embeddings from
this existing denoised ten-second waveform.

The same denoised waveform supplies five clip-level values:

1. peak-to-mean energy from 10–300 Hz;
2. coefficient of variation from 10–300 Hz;
3. 10–37 Hz energy divided by 37–173 Hz energy;
4. 10–173 Hz energy divided by total energy;
5. longest sustained run of sub-50-Hz energy.

Inside each fold, the mean and scale of these five values
from the training clips is calculated. The five standardised values are repeated beside
every YAMNet frame, producing the requested **1029-dimensional vector**. PCA is
then fitted on the training fold's 1029-dimensional frames and reduces them to 64
dimensions. A second scaler standardises the PCA output to ensure all values are in a similar range and will not dominate.

### BirdNET

BirdNET receives the raw, pre-denoising ten-second waveform only from
`raw_clips_459`. It is divided into
eight overlapping three-second windows with a one-second jump. Each window is
resampled so that feeding it to BirdNET at 48 kHz represents 16× 
frequency increase. The accelerated sound is tiled to fill BirdNET's
three-second input. Fold-local PCA and scaling reduce these frames to 64 dimensions. The key purpose of PCA is to reduce the number of parameters which has led to greater generalisation. We have dropped our parameters to 30000 from 100,000+ earlier and our model performs better on held out recordings.

### Training and evaluation

`StratifiedKFold(n_splits=5, shuffle=True, random_state=0)` keeps the elephant and
control proportions similar across folds. Every clip is tested once. In every
fold, scalers and PCA are fitted only after the split and only on training clips to avoid data leakage;
test clips are transformed with those frozen objects. The neural network uses one
training seed (`1234`), balanced class weights, training-only frame masking, and
early stopping on training loss. Out-of-fold probabilities are evaluated at a
0.5 threshold for accuracy, AUC, F1, precision, recall, and specificity.

This split estimates performance on a familiar mixed distribution. It does not
hold entire recordings or environments out, so it should not be presented as an
unseen-deployment estimate. Also, seven control clips used to construct the fixed
noise reference remain in the requested 459-clip population. Because denoising is
unsupervised and fixed before model fitting this does not expose labels, but the
evaluation is not completely independent of those seven waveforms and that
limitation should be reported.



# three-branch-lstm
