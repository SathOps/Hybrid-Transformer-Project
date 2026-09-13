# Real CICIoT2023 Preprocessing & Data Splitting Plan (Updated Specification)

**Project:** Cyberattack Detection using Hybrid Transformer  
**Target Dataset:** Real CICIoT2023 Raw Dataset (37,422,631 records across 215 CSV files)  
**Date:** September 11, 2026  
**Status:** Methodological Plan Updated — Awaiting User Approval Before Execution  

---

## Executive Summary

This updated design specification defines the preprocessing, scaling, data partitioning, and imbalance handling methodology for the **37,422,631 raw flow records** of the real CICIoT2023 dataset.

### Key Methodological Directives
1. **Full Real Dataset Preservation:** The complete 37,422,631-record dataset is preserved in its full, natural distribution. Preprocessing will **NOT permanently cap or discard majority-class training data**.
2. **Untouched Natural Validation & Test Partitions:** The Validation (15% = 5,613,395 rows) and Test (15% = 5,613,395 rows) partitions retain their **exact natural class distributions**. They are **never sampled, capped, or altered**.
3. **Reproducible Training-Only Subsampled Variant:** For rapid BPSO feature optimization and training acceleration, a separate **TRAINING-ONLY** subsampled variant will be made available (`data/processed/prepared/subsampled_train/`). It retains 100% of minority training samples while capping majority training classes with fixed seed `42`. Validation and test sets remain identical to the full natural benchmark.
4. **Log-Transformed Scaling (Replacing Plain MinMaxScaler):** Plain `MinMaxScaler` is replaced with **Log1p-Transformed Normalization** ($\log(1 + x)$ followed by scaling) or `RobustScaler`. This resolves extreme right-skewed flow feature outliers and prevents >99% of normal traffic values from being compressed into $[0, 0.001]$.
5. **Zero Test-Set Data Leakage:** The hold-out test set is strictly isolated and never accessed during scaling, class weighting, sampling decisions, BPSO, or hyperparameter tuning.

---

## 1. Multi-Pass Streaming Architecture (37.4M Records)

Preprocessing processes all 215 CSV files under `data/raw/ciciot2023/` using a streaming chunked design (`chunksize = 100,000` rows) across 3 sequential passes:

```mermaid
flowchart TD
    subgraph Pass 1: Global Line Counting & Label Verification
        A1["Stream 215 Raw CSV Files"] --> A2["clean_chunk(): Map labels & Drop Infs/NaNs"]
        A2 --> A3["Build Global Class & Line Counters"]
    end
    
    subgraph Pass 2: Strict Training Scaler Fitting
        B1["Stream Training Partition Rows Only"] --> B2["Log1p Transform + Scaler.partial_fit(training_chunk)"]
        B2 --> B3["Serialize Scaler to checkpoints/scaler.pkl"]
    end
    
    subgraph Pass 3: Normalization & Binary Sharded Output
        C1["Stream Raw Rows"] --> C2["Apply Pre-Fitted Log1p Scaler"]
        C2 --> C3["Split into Train (70%) / Val (15%) / Test (15%)"]
        C3 --> C4["Write Binary Shards to data/processed/prepared/full/"]
        C3 --> C5["(Optional) Write Training-Only Subsampled Shards"]
    end
    
    Pass 1 --> Pass 2 --> Pass 3
```

- **Pass 1 (Counting & Validation):** Scans all 215 CSV files, validates 46 features, applies 32-to-8 class mapping, and records exact per-class valid row counts.
- **Pass 2 (Training Scaler Fit):** Fits the normalization parameters **strictly on training set rows** (`split == "train"`).
- **Pass 3 (Shard Writing):** Transforms features, casts arrays (`float32` features, `int64` labels), and writes compressed binary `.npz` shards (`part-XXXXX.npz`, 100,000 rows/shard).

---

## 2. 32 Subclasses to 8 Canonical Target Classes Mapping

All 32 attack subclass folders present in `data/raw/ciciot2023/` map to the 8 canonical target classes defined in `src/preprocessing/class_mapping.py`:

| Canonical Target Class | Class ID | Mapped Attack Subclasses | Total Raw Flow Records | Share % |
| :--- | :---: | :--- | ---: | ---: |
| **DDoS** | `2` | `DDoS-ACK_Fragmentation`, `DDoS-HTTP_Flood`, `DDoS-ICMP_Flood`, `DDoS-ICMP_Fragmentation`, `DDoS-PSHACK_FLOOD`, `DDoS-RSTFINFLOOD`, `DDoS-SYN_Flood`, `DDoS-SlowLoris`, `DDoS-SynonymousIP_Flood`, `DDoS-TCP_Flood`, `DDoS-UDP_Flood`, `DDoS-UDP_Fragmentation` | 27,665,412 | 73.93% |
| **DoS** | `3` | `DoS-HTTP_Flood`, `DoS-SYN_Flood`, `DoS-TCP_Flood`, `DoS-UDP_Flood` | 7,112,045 | 19.01% |
| **Mirai** | `4` | `Mirai-greeth_flood`, `Mirai-greip_flood`, `Mirai-udpplain` | 1,210,815 | 3.24% |
| **Benign** | `0` | `Benign_Final` (`BenignTraffic`) | 900,412 | 2.41% |
| **Recon** | `5` | `Recon-HostDiscovery`, `Recon-OSScan`, `Recon-PingSweep`, `Recon-PortScan` | 312,050 | 0.83% |
| **Spoofing** | `6` | `DNS_Spoofing`, `MITM-ArpSpoofing` | 190,540 | 0.51% |
| **BruteForce** | `1` | `DictionaryBruteForce` | 24,115 | 0.06% |
| **Web-based** | `7` | `BrowserHijacking`, `CommandInjection`, `SqlInjection`, `Uploading_Attack`, `XSS` | 7,242 | 0.02% |
| **Total** | | **32 Subclasses** | **37,422,631** | **100.00%** |

---

## 3. Data Partitioning & Natural Distribution Preservation

### Partition Ratios
- **Train Partition:** 70% (**26,195,841 rows**)
- **Validation Partition:** 15% (**5,613,395 rows**)
- **Test Partition (Hold-Out):** 15% (**5,613,395 rows**)

### Natural Partition Breakdown (Full Real Dataset)

| Canonical Target Class | Total Raw Rows | Train (70%) | Validation (15%) | Test (15%) |
| :--- | ---: | ---: | ---: | ---: |
| **DDoS** | 27,665,412 | 19,365,788 | 4,149,812 | 4,149,812 |
| **DoS** | 7,112,045 | 4,978,431 | 1,066,807 | 1,066,807 |
| **Mirai** | 1,210,815 | 847,570 | 181,622 | 181,623 |
| **Benign** | 900,412 | 630,288 | 135,062 | 135,062 |
| **Recon** | 312,050 | 218,435 | 46,807 | 46,808 |
| **Spoofing** | 190,540 | 133,378 | 28,581 | 28,581 |
| **BruteForce** | 24,115 | 16,880 | 3,617 | 3,618 |
| **Web-based** | 7,242 | 5,069 | 1,086 | 1,087 |
| **Total** | **37,422,631** | **26,195,841** | **5,613,395** | **5,613,395** |

> [!IMPORTANT]
> **Natural Distribution Guarantee:** Both Validation (5.61M rows) and Test (5.61M rows) partitions remain **100% full and un-sampled**. This ensures that model evaluation metrics (Macro F1, Weighted F1, PR-AUC, Confusion Matrix) accurately measure real-world performance on natural, un-manipulated network traffic.

---

## 4. Controlled Training-Only Subsampling vs Full Training

To enable both **full-scale research benchmarking** and **fast, reproducible BPSO feature optimization**, two dataset representations will be maintained:

### Variant 1: `data/processed/prepared/full/` (Primary Research Dataset)
- Contains all **37.4M records** (26.2M Train, 5.61M Val, 5.61M Test).
- Uses cost-sensitive class weighting during model training:
  $$w_k = \frac{N_{\text{train}}}{K \cdot N_{\text{train}, k}}$$
  where $N_{\text{train}} = 26,195,841$, $K = 8$, and $N_{\text{train}, k}$ is the count of class $k$ training rows.

### Variant 2: `data/processed/prepared/subsampled_train/` (Training-Only Fast Variant)
- **Validation Set:** Unchanged (Natural 5.61M rows).
- **Test Set:** Unchanged (Natural 5.61M rows).
- **Training Set:** Capped at $N_{\text{max}} = 500,000$ rows for majority classes (`DDoS`, `DoS`, `Mirai`), while retaining **100% of available minority training rows** (`Benign`: 630.2k, `Recon`: 218.4k, `Spoofing`: 133.3k, `BruteForce`: 16.8k, `Web-based`: 5.0k).
- **Selection Determinism:** Subsampling indices for majority training classes are drawn using fixed random seed `42`.
- **Purpose:** Accelerates BPSO feature selection iterations from hours to minutes while guaranteeing that BPSO and model selection are validated against the full natural validation set.

---

## 5. Re-evaluation of Scaler: Log-Transformed Normalization

An inspection of feature distributions (`results/real_dataset_audit/feature_quality.csv`) reveals that flow statistics (e.g. `flow_duration`, `Rate`, `Srate`, `Drate`, `Tot size`, `IAT`, `Variance`) exhibit **extreme right-skewed heavy-tail distributions with large outliers**:

- `flow_duration`: Min = 0.0, Max > $10^8$, Mean = 124,510.
- `Rate`: Min = 0.0, Max > $10^7$, Mean = 4,210.

### Why Plain `MinMaxScaler` is Methodologically Flawed for Skewed Traffic
Under plain `MinMaxScaler`, a single extreme outlier value (e.g. $x_{\text{max}} = 10^8$) forces over **99.5% of normal traffic flow values** to compress into a tiny uninformative sub-interval $[0, 0.001]$, destroying neural gradient sensitivity and feature resolution.

### Selected Scaler Solution: Log1p Transformation + MinMaxScaler
```python
x_transformed = np.log1p(np.maximum(0, x))
```
1. **Log1p Transformation:** $\log(1 + x)$ compresses heavy-tailed right-skewed feature ranges logarithmically while preserving exact zero values ($\log(1 + 0) = 0$).
2. **MinMaxScaler:** Scaling the log-transformed features cleanly maps all 46 inputs to $[0, 1]$, matching the activation ranges required by the **CNN-Transformer** (SeLU activation) and **MLP** without outlier compression.
3. **Training-Only Fit:** Normalization parameters (min and max of log-transformed features) are computed **strictly on training set rows** (`split == "train"`) and applied unchanged to validation and test partitions.

---

## 6. Calculation of Class Weights

When cost-sensitive class weighting is enabled during model training (e.g. cross-entropy loss), weights $w_k$ are calculated **ONLY from the training partition used in that specific experiment**:

$$w_k = \frac{N_{\text{train\_exp}}}{8 \cdot N_{\text{train\_exp}, k}}$$

- For **Full Data Experiments:** Calculated from the 26,195,841 full training rows.
- For **Subsampled Training Experiments:** Calculated from the actual subsampled training rows.
- **Leakage Prohibition:** Validation and test set row counts are NEVER included in class weight calculations.

---

## 7. Strict Hold-Out Test Set Isolation

To guarantee zero data leakage:

```
[ Raw CICIoT2023 CSV Files (37.4M Rows) ]
                   │
                   ├── Train Split (70%) ────────> Fit Scaler & Train Models / Run BPSO
                   ├── Validation Split (15%) ────> Monitor Early Stopping & Select Models
                   └── Test Split (15%) ──────────> UNTOUCHED (Evaluated ONCE at the end)
```

- **Scaler Fitting:** Computed ONLY on the 70% train split.
- **BPSO Feature Selection:** Evaluated ONLY on train and validation loss.
- **Hyperparameter Tuning:** Evaluated ONLY on validation loss.
- **Test Set Evaluation:** Executed ONCE per final trained model on `data/processed/prepared/full/test/`.

---

## 8. Exact Configuration & Reproducibility Parameters

```yaml
preprocessing:
  random_seed: 42
  chunksize: 100000
  shard_size: 100000
  split_ratios:
    train: 0.70
    validation: 0.15
    test: 0.15
  scaler:
    method: "Log1p_MinMaxScaler"     # Log1p transform followed by MinMaxScaler
    fit_split: "train"
    scaler_path: "checkpoints/scaler.pkl"
  dataset_variants:
    full:
      path: "data/processed/prepared/full"
      train_rows: 26195841
      val_rows: 5613395
      test_rows: 5613395
    subsampled_train:
      path: "data/processed/prepared/subsampled_train"
      max_majority_train_rows: 500000
      val_rows: 5613395              # Untouched natural validation set
      test_rows: 5613395             # Untouched natural test set
```

---

## 9. Disk Space & Memory Footprint

| Dataset Variant | Raw CSV Size | Prepared `.npz` Size | Peak RAM (Streaming) | Processing Time |
| :--- | :---: | :---: | :---: | :---: |
| **Full Natural Dataset (37.4M rows)** | 6.66 GB | ~6.8 GB | < 650 MB | ~6–10 mins |
| **Subsampled Train Variant** | 6.66 GB | ~2.1 GB | < 450 MB | ~3–5 mins |

---

## 10. Multi-Model & BPSO Pipeline Reusability

Both dataset variants yield standardized output arrays:
- **CNN-Transformer Input:** `(batch, 46, 1)`
- **MLP Input:** `(batch, 46)`
- **XGBoost Input:** `(N, 46)`
- **BPSO Input:** Binary feature selection mask $M \in \{0, 1\}^{46}$ applied directly across all models.

---

## 11. Methodological Changes Summary

| Aspect | Previous Plan Draft | Updated Methodological Plan | Rationale |
| :--- | :--- | :--- | :--- |
| **Full Dataset Preservation** | Majority classes capped globally | **Full 37.4M dataset preserved** in `prepared/full/` | Ensures full research dataset is available without data loss. |
| **Validation & Test Distributions** | Capped/Subsampled | **Natural class distributions preserved** (5.61M rows each) | Guarantees realistic evaluation of real-world intrusion detection performance. |
| **Subsampling Scope** | Global | **Training-Only Subsampling** in `prepared/subsampled_train/` | Accelerates BPSO and model iterations while keeping validation/test sets natural and un-manipulated. |
| **Scaling Method** | Plain `MinMaxScaler` | **Log1p Transformation + `MinMaxScaler`** | Resolves extreme right-skewed flow feature outliers ($\log(1+x)$) to prevent outlier compression. |
| **Class Weight Calculation** | Fixed global formula | **Calculated ONLY from training data used for that experiment** | Eliminates data leakage from validation/test set counts into loss weights. |
| **Test Set Isolation** | Basic hold-out | **Strict single evaluation rule** | Ensures test set is never used for scaler fitting, sampling, weights, BPSO, or hyperparameter selection. |

---

> [!STOP]
> **Methodological Plan Update Complete.** No raw dataset files were modified. No dataset shards were generated. No models were trained. Awaiting explicit user approval before running preprocessing.
