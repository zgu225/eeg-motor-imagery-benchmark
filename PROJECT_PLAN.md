# Project Scope: EEG Motor Imagery Classification

## Topic

Cross-Session and Subject-Independent EEG Motor Imagery Decoding: From CSP
Baselines to Compact Neural Networks

## Research Question

How do compact deep-learning models compare with CSP-based classifiers under
cross-session and leave-one-subject-out motor imagery evaluation?

## Dataset

The project uses BCI Competition IV Dataset 2a (`BNCI2014_001`), a standard
four-class motor imagery benchmark containing 9 subjects and 22 EEG channels.

## Completed Experiments

1. Subject-specific cross-session classification
2. Leave-one-subject-out subject-independent classification
3. Classical and neural-network model comparison
4. Per-subject metrics, aggregate statistics, confusion matrices, and learning curves

The authoritative deliverable is `scripts/run_benchmark.py` and
`reports/benchmark_v2/RESULTS.md`: all five models, two grouped protocols and
nine subjects (90 folds for seed 42). It includes checkpoint reload verification,
per-trial predictions, exact split indices and resumable artifact checksums.
Historical results under `reports/results` use a different protocol.

## Implemented Models

1. CSP-LDA
2. CSP-SVM
3. Lawhern et al. (2018) EEGNet-8,2
4. Compact EEGNet variant (not a controlled single-factor ablation)
5. ShallowConvNet

## Code Map

- `configs/bnci2014_001.yaml`: dataset, training, and output configuration
- `src/bci_project/data/load_moabb.py`: public dataset loading through MOABB
- `src/bci_project/data/preprocessing.py`: EEG standardization and subject splitting
- `src/bci_project/models/csp_baseline.py`: CSP-LDA and CSP-SVM pipelines
- `src/bci_project/models/lawhern_eegnet.py`: PyTorch EEGNet-8,2 reproduction
- `src/bci_project/models/deep_models.py`: EEGNet ablation and ShallowConvNet
- `src/bci_project/training.py`: optimization, early stopping, and inference
- `src/bci_project/evaluation/metrics.py`: accuracy, macro-F1, and confusion matrices
- `src/bci_project/evaluation/plots.py`: evaluation and learning-curve visualizations
- `src/bci_project/experiments/within_subject.py`: cross-session evaluation
- `src/bci_project/experiments/cross_subject.py`: LOSO subject-independent evaluation
- `src/bci_project/benchmark.py`: current grouped benchmark and resume workflow
- `src/bci_project/artifacts.py`: input contracts and independent model inference
- `src/bci_project/reporting.py`: current statistics and figures
- `scripts/verify_delivery.py`: independent artifact consistency audit

## Future Work

- Repeated-seed evaluation and paired statistical testing
- Calibration-efficient target-subject fine-tuning
- EEG-specific data augmentation
- Domain adaptation for inter-subject distribution shift
- External validation on an independent motor imagery dataset
