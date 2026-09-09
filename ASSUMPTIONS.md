# Assumptions

This document records explicit provisional assumptions for the repository scaffold. None of these choices should be treated as the final research protocol without confirmation.

1. **Dataset availability:** CICIoT2023 will be acquired manually in a later phase; this task does not download or vendor any dataset files.
2. **Dataset location:** Raw files will be placed under `data/raw/`, and derived artifacts will be written under `data/processed/`.
3. **Task definition:** The project concerns cyber attack detection, but the final binary or multiclass target is not yet specified.
4. **Architecture:** The phrase CNN-Transformer identifies the intended model family only. CNN depth, Transformer settings, feature reshaping, and fusion method are intentionally unspecified.
5. **Preprocessing:** Feature cleaning, scaling, encoding, selection, and imbalance handling require confirmation from the source protocol and dataset inspection.
6. **Evaluation:** Accuracy alone is not assumed to be sufficient. The primary metrics, averaging method, baselines, and split protocol remain open.
7. **Reproducibility:** A default random seed of `42` is provided as a placeholder in `config.yaml`; it is not evidence of the source study's seed.
8. **Environment:** Python and PyTorch are provisionally selected because later phases are expected to implement a deep-learning model; versions and hardware support remain to be pinned.
9. **Repository policy:** Empty directories are created now to establish ownership boundaries. No implementation, notebook, checkpoint, or dataset artifact is included in this phase.
