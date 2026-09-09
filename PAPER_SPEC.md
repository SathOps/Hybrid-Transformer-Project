# Paper Specification

## Working Title

**CICIoT2023 CNN-Transformer Cyber Attack Detection**

## Objective

Reproduce and evaluate a CNN-Transformer-based cyber-attack detection system for IoT networks using the CICIoT2023 dataset.

## Scope of This Repository Phase

This document defines the implementation target. It does not download CICIoT2023 or implement the model and training pipeline in this phase.

## Dataset

- Dataset: CICIoT2023
- Input: 46 network traffic features
- Intended task: eight-class IoT cyber-attack detection
- Local raw-data location: `data/raw/`
- Local processed-data location: `data/processed/`
- Exact release, file list, download source, and feature names: **TBD**

### Target Classes

The original attack labels must be consolidated into these eight target classes:

1. Benign
2. BruteForce
3. DDoS
4. DoS
5. Mirai
6. Recon
7. Spoofing
8. Web-based

The exact mapping from every original CICIoT2023 label to these classes must be documented during preprocessing.

## Preprocessing

- Use a train/test separation, with stratified splitting where applicable.
- Fit MinMax normalization parameters using training data only.
- Apply the fitted normalization parameters unchanged to validation, test, and inference data.
- Consolidate the original dataset labels into the eight target classes above.
- Record the split proportions, feature ordering, missing-value handling, and any filtering decisions when preprocessing is implemented.

## Intended Experimental Pipeline

1. Acquire and document the dataset.
2. Profile the raw files and labels.
3. Define preprocessing and train/validation/test splitting.
4. Implement the specified CNN-Transformer architecture.
5. Train with recorded configuration and random seeds.
6. Evaluate with class-aware metrics and documented reporting.

The steps above describe the research sequence; implementation belongs to later phases.

## CNN Architecture

The CNN receives 46 input features. The specified CNN block is:

1. Batch Normalization
2. Conv1D: 64 filters, kernel size 3, SeLU activation
3. MaxPooling1D: pool size 2
4. Batch Normalization
5. Conv1D: 128 filters, kernel size 3, SeLU activation
6. MaxPooling1D: pool size 2
7. Batch Normalization
8. Conv1D: 256 filters, kernel size 3, SeLU activation
9. MaxPooling1D
10. Flatten
11. Dense: 256 units, SeLU activation, L2 regularization `0.001`
12. Dropout: `0.05`
13. Dense: 128 units, SeLU activation, L2 regularization `0.001`
14. Dropout: `0.05`
15. Dense: 64 units, SeLU activation, L2 regularization `0.001`
16. Dropout: `0.05`

The pool size for the third MaxPooling1D layer and other layer options not listed above remain subject to the source specification.

## Transformer

The Transformer component must include:

- Transformer encoder.
- Multi-head self-attention with 3 attention heads.
- Embedding dimension of 8.
- Post-attention dropout of `0.05`.
- Residual or skip connections.
- Layer Normalization.
- Feed-forward network.

The source specification must determine the exact CNN-to-Transformer interface, tokenization or reshaping, feed-forward dimensions, normalization order, and whether the feed-forward activation is SeLU or GELU. These details must be documented before implementation when they are not directly specified.

## Output Layer

The classifier uses a Softmax output with 8 classes.

## Training

- Optimizer: Adam
- Learning rate: `0.00005`
- Batch size: `1024`
- Epochs: `50`
- Dropout: `0.05`

Any unspecified Adam options, checkpoint selection rule, and early-stopping behavior must be recorded rather than inferred.

## Evaluation

Report the following for the eight-class task:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Per-class metrics

The averaging method for aggregate precision, recall, and F1-score must be stated explicitly in the results.

## Baselines

The implementation must compare the CNN-Transformer with:

- MLP
- XGBoost

Baseline preprocessing, split, label mapping, and evaluation must be comparable and documented.

## Reference Results

The following approximate accuracies are reported reference results from the paper, not guaranteed target results for this implementation:

| Model | Reference accuracy |
| --- | ---: |
| CNN-Transformer | approximately 99.47% |
| MLP | approximately 99.39% |
| XGBoost | approximately 99.40% |

Our implementation must report its own measured results honestly, including deviations from these references and the experimental conditions under which they were obtained.

## Reproducibility Requirements

Future phases should record:

- Dataset release, checksums, and selected files.
- Feature selection, transformations, and label mapping.
- Data split strategy and random seeds.
- Exact architecture and hyperparameters.
- Software and hardware environment.
- Checkpoint selection rule and evaluation protocol.

## Planned Outputs

- Versioned preprocessing artifacts in `data/processed/`.
- Experiment configurations and results in `experiments/`.
- Model checkpoints in `checkpoints/`.
- Reproducible notebooks in `notebooks/`.
- Focused tests in `tests/`.

The concrete file formats and naming conventions are **TBD**.

## Remaining Open Research Details

The unresolved implementation details are tracked in [ASSUMPTIONS.md](ASSUMPTIONS.md). They must be resolved from the source specification or explicitly declared as implementation decisions before results are compared with the paper.
