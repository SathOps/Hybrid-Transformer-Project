# Real CICIoT2023 7-Class Experiment Taxonomy Documentation

**Date**: September 11, 2026  
**Status**: APPROVED & CONFIGURATION ACTIVE — READY FOR MODEL TRAINING  
**Artifact Location**: `results/real_dataset_audit/experiment_7class_taxonomy.md`  

---

## 1. Executive Summary & Taxonomy Definition

This document establishes the official class mapping configuration for all model training and evaluation experiments on the real CICIoT2023 dataset.

### Taxonomy Overview
1. **Full Research Project Taxonomy**: **8 Canonical Classes**  
   - Canonical ID 0: `Benign`
   - Canonical ID 1: `BruteForce`
   - Canonical ID 2: `DDoS`
   - Canonical ID 3: `DoS`
   - Canonical ID 4: `Mirai`
   - Canonical ID 5: `Recon`
   - Canonical ID 6: `Spoofing`
   - Canonical ID 7: `Web-based`

2. **Real CICIoT2023 Active Experiment Taxonomy**: **7 Active Classes**  
   - Empirical investigation confirmed that the official downloaded release of CICIoT2023 contains **0 Recon records** (all 4 Recon subclasses `Recon-HostDiscovery`, `Recon-OSScan`, `Recon-PingSweep`, `Recon-PortScan` are absent from disk).
   - **Recon (Canonical ID 5) remains documented in the project taxonomy, but is set to INACTIVE for model output layer dimensions.**
   - All active models (CNN-Transformer, MLP, XGBoost) output logits/probabilities for exactly **7 active classes**.

---

## 2. Canonical-to-Active Class Index Mapping

To prevent output dimension mismatches or dummy class predictions for zero-sample classes, canonical target labels stored in `.npz` shards are dynamically mapped to contiguous active target indices `[0..6]` during dataset batch iteration:

| Canonical Target Class | Canonical Class ID | Active Status | Active Target Index | Active Class Order |
| :--- | :---: | :---: | :---: | :--- |
| **Benign** | 0 | **Active** | **0** | Index 0 |
| **BruteForce** | 1 | **Active** | **1** | Index 1 |
| **DDoS** | 2 | **Active** | **2** | Index 2 |
| **DoS** | 3 | **Active** | **3** | Index 3 |
| **Mirai** | 4 | **Active** | **4** | Index 4 |
| **Recon** | 5 | *Inactive (0 samples)* | *N/A* | *Excluded from Output Layer* |
| **Spoofing** | 6 | **Active** | **5** | Index 5 |
| **Web-based** | 7 | **Active** | **6** | Index 6 |

---

## 3. Pipeline Integrity Safeguards

1. **No Canonical ID Renumbering**: Original canonical IDs `(0..7)` remain unchanged in preprocessed `.npz` data shards and raw label mappings.
2. **No Data Re-preprocessing**: Raw CSV files and prepared `.npz` files are preserved untouched.
3. **No Silent Removal**: Recon is explicitly documented as Class ID 5 in `CANONICAL_TARGET_CLASSES` and `CANONICAL_ID_TO_ACTIVE_INDEX`.
4. **Vectorized Label Translation**:
   - `canonical_to_active_labels(y)` converts canonical label arrays `[0, 1, 2, 3, 4, 6, 7]` $\rightarrow$ `[0, 1, 2, 3, 4, 5, 6]`.
   - `active_to_canonical_labels(y)` converts active predictions `[0, 1, 2, 3, 4, 5, 6]` $\rightarrow$ canonical label IDs `[0, 1, 2, 3, 4, 6, 7]` for reporting.
5. **Model Output Dimension**: Output layers in `CNN-Transformer`, `MLP`, and `XGBoost` are configured for `num_classes = 7`.

---

## 4. Summary of Configuration Updates

- `src/preprocessing/class_mapping.py`: Single authoritative active class configuration (`ACTIVE_TARGET_CLASSES`, `NUM_ACTIVE_CLASSES = 7`, `canonical_to_active_labels()`, `active_to_canonical_labels()`).
- `config.yaml`: Updated `model.num_classes = 7`.
- `src/models/cnn_transformer.py`: Output layer set to `Dense(7) -> Softmax`.
- `src/models/mlp.py`: Output layer set to `Dense(7) -> Softmax`.
- `src/models/xgboost_model.py`: Multi-class objective set to `num_class = 7`.
- `src/preprocessing/dataset.py`: Streaming iterator converts canonical IDs to active targets `[0..6]`.
- `src/evaluation/evaluate.py` & `evaluate_xgboost.py`: Evaluation metrics and confusion matrices operate on 7 x 7 active classes.
- Unit Tests: All test fixtures and assertions updated to verify 7 active classes output contract.
