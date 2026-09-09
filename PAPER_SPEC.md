# Paper Specification

## Working Title

**CICIoT2023 CNN-Transformer Cyber Attack Detection**

## Objective

Reproduce and evaluate a hybrid CNN-Transformer intrusion detection system for IoT network traffic using the CICIoT2023 dataset.

## Scope of This Repository Phase

This phase defines the project boundary and repository organization. It does not download CICIoT2023, preprocess records, implement a model, or run training.

## Dataset

- Dataset: CICIoT2023
- Intended use: IoT network intrusion and cyber attack detection
- Local raw-data location: `data/raw/`
- Local processed-data location: `data/processed/`
- Exact release, file list, download source, and feature schema: **TBD**

## Intended Experimental Pipeline

1. Acquire and document the dataset.
2. Profile the raw files and labels.
3. Define preprocessing and train/validation/test splitting.
4. Implement the specified CNN-Transformer architecture.
5. Train with recorded configuration and random seeds.
6. Evaluate with class-aware metrics and documented reporting.

The steps above describe the research sequence only; implementation belongs to later phases.

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

## Open Research Details

The following details are intentionally unresolved and must be confirmed from the source paper or experimental protocol before implementation:

- Binary versus multiclass classification.
- Dataset release and exact attack categories.
- Input feature representation and tensor shape.
- CNN block design and Transformer configuration.
- Imbalance handling.
- Split methodology and leakage controls.
- Primary metrics and baseline comparisons.
- Training schedule and hardware requirements.
