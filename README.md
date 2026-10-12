# EEG Motor Imagery Benchmark

A reproducible offline benchmark comparing classical machine learning and compact convolutional networks for four-class EEG motor imagery decoding. The project evaluates generalization across recording sessions and unseen participants using explicitly separated training, validation, and test sets.

## Dataset
The benchmark uses **BCI Competition IV Dataset 2a**, accessed through MOABB as `BNCI2014_001`: 9 participants, 2 sessions per participant, 22 EEG channels, and 5,184 trials. The four classes are left hand, right hand, feet, and tongue.
MNE applies an 8–30 Hz band-pass filter and extracts epochs from 0.5 to 4.0 seconds after the cue. At 250 Hz, each epoch contains 876 samples. Model inputs have shape `(trials, 22, 876)` and are expressed in volts.

## Models
| Model | Implementation |
|---|---|
| CSP–LDA | Multiclass CSP with 8 components and linear discriminant analysis |
| CSP–SVM | CSP features with an RBF support vector machine |
| EEGNet-8,2 | PyTorch architectural reimplementation of Lawhern et al. |
| EEGNet-inspired CNN | A variant with different padding and no max-norm constraints |
| ShallowConvNet | A shallow convolutional network with log-power features |

The EEGNet implementation reproduces the architectural design, not the paper's complete experimental setup or numerical results. The EEGNet-inspired variant is not a controlled single-factor ablation.

## Evaluation
- **Cross-session:** train and validate within the first session; test on the second session. One complete training-session recording run is reserved for validation: 240 training, 48 validation, and 288 test trials per participant.
- **Leave-one-subject-out (LOSO):** hold out one participant for testing and another for validation; train on the remaining seven: 4,032 training, 576 validation, and 576 test trials.
- Models share identical split indices within each fold. Classical baselines fit only the training set; neural checkpoints are selected using validation accuracy, never test labels.
- The completed benchmark comprises **90 folds**, covering 5 models, 2 protocols, and 9 participants with seed 42.

## Results
Accuracy is reported as the participant mean ± sample standard deviation, in percent. Four-class chance accuracy is 25%.

| Model | Cross-Session Accuracy | LOSO Accuracy |
|---|---:|---:|
| CSP–LDA | 61.65 ± 16.99 | **39.31 ± 11.90** |
| CSP–SVM | **61.69 ± 14.79** | 33.29 ± 7.30 |
| EEGNet-8,2 | 37.42 ± 10.74 | 36.98 ± 11.45 |
| EEGNet-inspired CNN | 30.09 ± 5.39 | 38.91 ± 12.98 |
| ShallowConvNet | 44.02 ± 16.69 | 37.52 ± 11.76 |

![Benchmark comparison](reports/benchmark_v2/comparison.png)

Classical CSP baselines achieved the highest mean accuracy under both protocols in this experiment. These results do not establish a universal model ranking or a causal explanation for the performance differences.
See the [full results](reports/benchmark_v2/RESULTS.md) for Macro-F1 and exploratory paired comparisons, and the [methodology report](RESULT_REPORT.md) for implementation details.

## Reproducibility
The pipeline records exact split indices, per-trial predictions, training histories, confusion counts, configuration, package versions, and source fingerprints. Completed local runs additionally include saved models and artifact checksums. Every model was reloaded, and its test predictions matched the in-memory predictions before its fold was marked complete.
**This repository contains source code and result evidence, not raw EEG, model binaries, or fold completion markers.** Run the benchmark to regenerate complete model bundles before using inference or delivery-audit commands.

## Quick Start
Tested locally with Python 3.12 and CPU PyTorch on Windows. From the project root:
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -u scripts/run_benchmark.py --output reports/reproduced
```
The first run downloads the dataset. Repeat the same benchmark command to verify and resume completed folds. Use a new output directory when changing the experiment configuration. `requirements-lock.txt` records the observed local environment; a fresh-machine installation of that lockfile has not been independently verified.

## Project Structure
`src/bci_project/` contains data processing, models, training, evaluation, artifact persistence, and reporting. `scripts/` provides benchmark, inference, and verification entry points. `configs/` contains experiment settings, `tests/` contains regression tests, and `reports/benchmark_v2/` contains the completed benchmark evidence. GitHub Actions is configured to run core tests on pushes and pull requests.

## Scope and Limitations
This is an offline research benchmark, not a real-time or clinical BCI system. Reported results cover one dataset and one random seed with a maximum 40-epoch training budget and early stopping. They do not establish state-of-the-art performance or external validity. Paired statistical comparisons are exploratory and unadjusted; LOSO training sets overlap. Multi-seed evaluation, domain adaptation, and external-dataset validation remain future work.

## References
- [BCI Competition IV](https://www.bbci.de/competition/iv/)
- [MOABB](https://github.com/NeuroTechX/moabb)
- [EEGNet reference implementation](https://github.com/vlawhern/arl-eegmodels)
- Lawhern et al. (2018). *EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces*. Journal of Neural Engineering, 15, 056013. [DOI](https://doi.org/10.1088/1741-2552/aace8c)
