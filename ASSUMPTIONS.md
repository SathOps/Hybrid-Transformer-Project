# Assumptions

This document records explicit ambiguities and provisional assumptions for the research implementation. None of these choices should be silently inferred or treated as part of the source protocol without confirmation.

1. **Dataset availability:** CICIoT2023 will be acquired manually in a later phase; this task does not download or vendor any dataset files.
2. **Dataset location:** Raw files will be placed under `data/raw/`, and derived artifacts will be written under `data/processed/`.
3. **Target labels:** The implementation will use the eight classes specified in `PAPER_SPEC.md`, but the exact mapping from each original CICIoT2023 attack label to those classes must be established from the source protocol.
4. **Project-level class mapping:** The paper specifies an eight-class setup, but no authoritative 34-to-8 table was found. `Backdoor_Malware` (663 rows) and `VulnerabilityScan` (7,127 rows) could not be assigned to a target class and are quarantined rather than silently mapped.
5. **Input shape:** The model receives 46 features, but the sequence/channel arrangement required by Conv1D and the CNN-to-Transformer reshaping or tokenization are not specified.
6. **Splits:** Train/test separation and stratification where applicable are required, but split proportions, validation-set construction, grouping, and leakage controls are not specified.
7. **Normalization:** MinMax parameters will be fitted on training data only and reused for validation, test, and inference data. The feature range and behavior for values outside the training range are not specified.
8. **CNN details:** The third MaxPooling1D pool size, convolution padding/stride, and other unlisted layer options are unspecified.
9. **Transformer dimensions:** The specification gives 3 attention heads and embedding dimension 8. Standard multi-head attention often requires the embedding dimension to be divisible by the head count, so the source implementation must clarify how these values are reconciled.
10. **Transformer details:** Feed-forward dimensions, normalization order, positional information, CNN-to-Transformer interface, and the location of the stated post-attention dropout are unspecified.
11. **Activations:** CNN activations are specified as SeLU, but the Transformer feed-forward activation is ambiguous between SeLU and GELU; the source specification must decide this.
12. **Regularization and optimization:** L2 regularization is specified as `0.001` for the dense CNN layers and dropout as `0.05`, but the regularization scope, Adam options, checkpoint selection, and early stopping are unspecified.
13. **Evaluation:** Accuracy, precision, recall, F1-score, confusion matrix, and per-class metrics are required. The averaging method for aggregate metrics and the exact reporting protocol are unspecified.
14. **Baselines:** MLP and XGBoost are required baselines, but their architecture, hyperparameters, and tuning protocol are unspecified.
15. **Reference results:** The paper accuracies are approximate reference values, not guaranteed targets. The implementation must report its own results without presenting them as reproduced values unless experimentally obtained.
16. **Reproducibility:** A default random seed of `42` is provided as a placeholder in `config.yaml`; it is not evidence of the source study's seed.
17. **Environment:** Python and PyTorch are provisionally selected because later phases are expected to implement a deep-learning model; versions and hardware support remain to be pinned.
18. **Repository policy:** Empty directories are created now to establish ownership boundaries. No implementation, notebook, checkpoint, or dataset artifact is included in this phase.
