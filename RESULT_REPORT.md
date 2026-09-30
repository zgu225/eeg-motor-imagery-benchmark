# Methodology and Evidence

## Research Question

How do spatial-filtering baselines and compact neural networks compare when
generalizing to another session or an unseen participant?

The current numerical results and plots are generated from completed folds in
[the v2 report](reports/benchmark_v2/RESULTS.md). Historical v1 scores in
`reports/results/` are retained for traceability, not as current evidence.

## Data and Input Contract

The processed dataset contains 5,184 epochs, 22 channels and 876 samples per epoch.
MNE supplies volt-valued data. The epoch spans 0.5-4.0 seconds after cue onset
with inclusive endpoints at 250 Hz and 8-30 Hz filtering.
String labels are sorted before mapping to integers; every bundle records the mapping.
No additional artifact-rejection or ICA pipeline is implemented.

For neural networks, each channel is centered and divided by its within-trial
standard deviation, clamped below at 1e-12. This uses the complete trial and is
therefore an offline operation. CSP receives filtered volt-valued epochs.

## Leakage Controls

The v2 benchmark reserves complete recording groups for validation.
Cross-session folds reserve one run from the training session and test only on
the other session. LOSO folds reserve a whole development participant for validation
and test on the designated unseen participant. The split is identical across models,
and CSP receives the same fit examples as the neural networks.

The validation partition selects neural-network checkpoints only.
Labels from the test partition are used for final metrics, not optimization.
This controls direct split leakage but does not provide an independent external
validation dataset: this benchmark was already explored during development.

## Training

Default settings are seed 42, batch size 64, at most 40 epochs, AdamW with learning
rate 0.001 and weight decay 0.0001, dropout 0.5, and early-stopping patience 8.
Weights with the highest validation accuracy are restored. A tie does not replace
the previous best checkpoint. This training budget was not exhaustively optimized.

Model construction uses evaluation mode for shape inference so synthetic zero
inputs do not update BatchNorm running statistics. Each fold resets the seed.
The run manifest records source hashes, configuration, Python and package versions.

## Architectural Scope

The Lawhern-named implementation contains temporal, depthwise spatial and separable
convolutional blocks with F1=8, D=2 and F2=16. It uses PyTorch BatchNorm and
initialization defaults and a 64-sample temporal kernel at 250 Hz.
These settings are not numerically equivalent to the reference Keras experiments.
Cross-entropy receives logits directly. The implementation should be described
as EEGNet-inspired architectural reimplementation, not full paper reproduction.

The separate EEGNet variant differs in padding as well as constraints. Comparing
the two cannot isolate the effect of max-norm. ShallowConvNet is likewise a
compact implementation inspired by power-feature decoding.

Reference: [official EEGModels.py](https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py).

## Artifacts and Verification

Each completed fold has fit/validation/test indices, per-trial predictions,
checkpoint metadata, weights, history, confusion counts, metrics and checksums.
Predictions from a reloaded checkpoint must exactly match in-memory predictions.
The reporter refuses to summarize incomplete or checksum-invalid runs.

Metrics are accuracy and macro-F1. Participants, not trials, are the unit of
aggregation. If multiple seeds are supplied, seeds are first averaged within
participant. Standard deviation describes participant variability, not a
confidence interval or uncertainty across seeds.

## Interpretation

Model rankings apply only to this dataset, split design and training budget.
A low neural-network score alone does not prove inadequate sample size or
distribution drift caused the difference. Those are hypotheses requiring
controlled experiments. Paired intervals and permutation estimates are exploratory,
especially because LOSO training sets overlap. Report effect sizes with their
uncertainty and avoid claims of statistical superiority from small differences.

## Future Work

Repeated seeds, predeclared hyperparameter selection, independent datasets,
calibration-efficient adaptation and real-time acquisition are future extensions.
They are not required to reproduce this completed offline benchmark, and are
not presented as implemented features.
