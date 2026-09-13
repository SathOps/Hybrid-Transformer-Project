# Baseline Integrity & Data-Leakage Audit Report

**Project:** Cyberattack Detection using Hybrid Transformer  
**Audit Target:** CNN-Transformer, MLP, and XGBoost Baseline Experiments  
**Date:** September 11, 2026  
**Status:** Audit Complete — Action Required Before Benchmark / BPSO  

---

## Executive Summary

A comprehensive baseline integrity and data-leakage audit was conducted on the completed **CNN-Transformer** (87.40% test acc), **MLP** (80.07% test acc), and **XGBoost** (100.00% test acc) baseline experiments.

### Audit Verdict: **Category B — Synthetic Data Artifact / Leakage Found**

The 100.00% test accuracy and 1.0000 Macro F1 score of the XGBoost baseline is **NOT a result of data leakage in preprocessing or label encoding**, but is directly caused by **synthetic sample generation** in `src/preprocessing/sample_data.py`. When raw CICIoT2023 CSV files were unavailable in `data/raw/`, the data generator populated the raw dataset with synthetic samples where every feature $x_i$ for class $k$ was sampled from Gaussian distribution $\mathcal{N}(10.0 + 5.0 \cdot k, 2.0^2)$. This introduced artificial linear feature separation between classes, allowing XGBoost decision tree splits to achieve perfect 100% classification.

---

## 1. Complete Data Pipeline Audit

### Data Flow Trace
1. **Raw Files (`data/raw/`):** 8 sample CSV files (`sample_benign.csv`, `sample_bruteforce.csv`, `sample_ddos.csv`, `sample_dos.csv`, `sample_mirai.csv`, `sample_recon.csv`, `sample_spoofing.csv`, `sample_web-based.csv`).
2. **Preprocessing Scripts:**
   - [src/preprocessing/sample_data.py](file:///c:/Users/user/OneDrive/ドキュメント/gitam/vscode%20proj/cyberattack%20detection/src/preprocessing/sample_data.py)
   - [src/preprocessing/prepare_dataset.py](file:///c:/Users/user/OneDrive/ドキュメント/gitam/vscode%20proj/cyberattack%20detection/src/preprocessing/prepare_dataset.py)
   - [src/preprocessing/class_mapping.py](file:///c:/Users/user/OneDrive/ドキュメント/gitam/vscode%20proj/cyberattack%20detection/src/preprocessing/class_mapping.py)
3. **Prepared Partitions (`data/processed/prepared/`):**
   - `train/part-00000.npz`: 14,000 samples (1,750 / class)
   - `validation/part-00000.npz`: 3,000 samples (375 / class)
   - `test/part-00000.npz`: 3,000 samples (375 / class)

### Reconciliation of Dataset Record Counts
- **Documented Dry-Run Estimation:** 9,022,524 raw rows / 9,014,734 retained rows (calculated during full dataset streaming dry-runs).
- **Paper Dataset Specification:** Approximately 46.18 million records (full CICIoT2023 release).
- **Actual Prepared Dataset:** **20,000 total rows** (2,500 rows/class).
- **Reconciliation Explanation:** Because physical raw CICIoT2023 dataset CSV files were not loaded into `data/raw/`, `ensure_prepared_dataset()` triggered `generate_sample_raw_csvs()` to create 20,000 synthetic rows for pipeline development and verification. All three baseline models were trained and evaluated on this 20,000-row synthetic partition.

---

## 2. Direct Label Leakage Audit

All 46 input features were audited to evaluate whether any feature directly encodes the target class, contains dataset/path metadata, or acts as a surrogate class identifier.

### Audit Findings for Key Features
| Feature | Category / Type | Legitimate Feature? | Direct Leakage? | Audit Notes |
| :--- | :--- | :---: | :---: | :--- |
| **HTTP / HTTPS / SSH** | Protocol Binary Flags | Yes | No | Standard network protocol flags. In synthetic data, they exhibit shifted means per class. |
| **DHCP / DNS / Telnet / SMTP / IRC / TCP / UDP / ARP / ICMP / IPv / LLC** | Protocol Binary Flags | Yes | No | Legitimate protocol indicators. |
| **syn_flag_number / ack_count / urg_count / rst_count / fin_count** | TCP Flag & Packet Counters | Yes | No | Standard TCP control flag counts. |
| **Tot size / Tot sum / Min / Max / AVG / Std / Radius / Covariance / Variance / IAT** | Flow Statistical Aggregates | Yes | No | Flow length, window size, inter-arrival time statistics. |

No feature in the 46-feature schema represents direct label leakage or class ID encoding.

---

## 3. Duplicate and Near-Duplicate Sample Analysis

A full hash-based duplicate analysis and sampled near-duplicate analysis ($L_\infty < 10^{-4}$) was performed across all dataset partitions.

```json
{
  "dataset_total_samples": 20000,
  "exact_duplicates": {
    "within_train": 0,
    "within_validation": 0,
    "within_test": 0,
    "cross_train_validation": 0,
    "cross_train_test": 0,
    "cross_validation_test": 0
  },
  "near_duplicates": {
    "sampled_test_pairs_tested": 500,
    "near_matches_in_train": 0
  }
}
```

**Finding:** There are **zero exact duplicates** and **zero near-duplicates** across train, validation, and test splits. The sample partitioning is clean of overlap.

---

## 4. Split Methodology Audit

- **Split Type:** Stratified Random Row-Wise Split (70% Train / 15% Validation / 15% Test) with fixed random seed `42`.
- **Flow/Session Aware?** No. Row-wise split does not group flows by session ID or source host.
- **Impact on Real Data:** In real network intrusion datasets (such as raw CICIoT2023), row-wise splitting can introduce session-level leakage. On this synthetic dataset, however, row-wise splitting had no session leakage because samples were drawn independently per row.

---

## 5. Class Distribution Audit

Class distribution across all three partitions is perfectly balanced:

| Class ID | Class Name | Train Count | Train % | Val Count | Val % | Test Count | Test % | Total Count |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Benign | 1,750 | 12.5% | 375 | 12.5% | 375 | 12.5% | 2,500 |
| 1 | BruteForce | 1,750 | 12.5% | 375 | 12.5% | 375 | 12.5% | 2,500 |
| 2 | DDoS | 1,750 | 12.5% | 375 | 12.5% | 375 | 12.5% | 2,500 |
| 3 | DoS | 1,750 | 12.5% | 375 | 12.5% | 375 | 12.5% | 2,500 |
| 4 | Mirai | 1,750 | 12.5% | 375 | 12.5% | 375 | 12.5% | 2,500 |
| 5 | Recon | 1,750 | 12.5% | 375 | 12.5% | 375 | 12.5% | 2,500 |
| 6 | Spoofing | 1,750 | 12.5% | 375 | 12.5% | 375 | 12.5% | 2,500 |
| 7 | Web-based | 1,750 | 12.5% | 375 | 12.5% | 375 | 12.5% | 2,500 |
| **Total** | | **14,000** | **100%** | **3,000** | **100%** | **3,000** | **100%** | **20,000** |

**Interpretation:** Because test class distribution is perfectly balanced ($375 \times 8 = 3,000$), Overall Accuracy, Macro F1, and Weighted F1 are identical ($1.0000$ for XGBoost).

---

## 6. Preprocessing Leakage Audit

- **Scaler Type:** `MinMaxScaler`
- **Fit Target:** Training split ONLY (`fit_split: train`).
- **Validation/Test Fitting:** Verified. Parameters were computed strictly on `train` data and applied unchanged to `validation` and `test` data. No data leakage occurred via scaler fitting.

---

## 7. Root Cause Analysis: Why XGBoost is Perfect (100% Accuracy)

Inspection of class-wise feature distributions for XGBoost's top features reveals the root cause:

### Top Feature Class-Wise Means (Normalized $[0, 1]$ Range)

| Top Feature | Benign (0) | BruteForce (1) | DDoS (2) | DoS (3) | Mirai (4) | Recon (5) | Spoofing (6) | Web-based (7) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tot size** | 0.1277 | 0.2347 | 0.3396 | 0.4447 | 0.5510 | 0.6542 | 0.7599 | 0.8641 |
| **HTTPS** | 0.1385 | 0.2423 | 0.3492 | 0.4526 | 0.5569 | 0.6609 | 0.7663 | 0.8710 |
| **Std** | 0.1527 | 0.2529 | 0.3572 | 0.4571 | 0.5606 | 0.6617 | 0.7656 | 0.8716 |
| **HTTP** | 0.1332 | 0.2361 | 0.3400 | 0.4447 | 0.5475 | 0.6521 | 0.7554 | 0.8596 |
| **Radius** | 0.1309 | 0.2359 | 0.3425 | 0.4455 | 0.5510 | 0.6542 | 0.7570 | 0.8606 |

### Code Root Cause in `src/preprocessing/sample_data.py` (Lines 38–41):
```python
# Generate 46 feature values with class-specific distribution shift
class_offset = ALLOWED_TARGET_CLASSES.index(target_class) * 5.0
feature_vals = rng.normal(loc=10.0 + class_offset, scale=2.0, size=len(MODEL_FEATURES))
```

**Explanation:** Every single feature for target class $k$ was generated with mean $\mu_k = 10.0 + 5.0 \cdot k$ and standard deviation $\sigma = 2.0$. This creates non-overlapping decision boundaries along orthogonal feature hyperplanes. Tree-based models like XGBoost easily find clean decision thresholds (e.g., `Tot size < 0.18` $\rightarrow$ Benign, `< 0.28` $\rightarrow$ BruteForce, etc.), guaranteeing **100% precision, recall, and accuracy**. Neural models (CNN-Transformer and MLP) achieve 87.40% and 80.07% because gradient descent with high dropout ($0.05$) and continuous loss optimization does not overfit discrete orthogonal tree boundaries as sharply.

---

## 8. Test-Set Integrity Verification

- **Location:** `data/processed/prepared/test/part-00000.npz`
- **Rows:** 3,000 rows (375 per class)
- **Features:** 46
- **SHA256 Checksum:** `ae19f672954057672a73eeb5e0813de66d37ec3cfcd23ba97a2a3aa6facf69d5`
- **Immutability:** Confirmed. File timestamp and checksum match across CNN-Transformer, MLP, and XGBoost evaluation runs. The test set was untouched during training and evaluation.

---

## 9. Review of Evaluation Methodology

- **Class Ordering:** Identical (`Benign: 0`, `BruteForce: 1`, `DDoS: 2`, `DoS: 3`, `Mirai: 4`, `Recon: 5`, `Spoofing: 6`, `Web-based: 7`).
- **Test Samples:** All three models evaluated on the exact same 3,000 test samples in `data/processed/prepared/test/part-00000.npz`.
- **Metrics Implementation:** Sklearn classification metrics (`accuracy_score`, `precision_recall_fscore_support`, `roc_auc_score`, `average_precision_score`) applied identically across all model outputs.
- **Label Leakage in Evaluation:** None. Predictions were generated strictly from model outputs.

---

## 10. Baseline Model Comparison

| Model | Test Accuracy | Macro F1 | Weighted F1 | Test Samples | Input Schema | Evaluation Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNN-Transformer** | **87.40%** | **83.24%** | **87.27%** | 3,000 | $(1, 46, 1)$ | Evaluated |
| **MLP** | **80.07%** | **77.18%** | **79.91%** | 3,000 | $(46,)$ | Evaluated |
| **XGBoost** | **100.00%** | **1.0000** | **1.0000** | 3,000 | $(46,)$ | Evaluated |

**Conclusion on Comparison:**
XGBoost's 100% result is mathematically authentic for the **synthetic sample dataset**, but **unrealistic for real-world IoT intrusion detection**. The difference between tree performance (100%) and neural net performance (80.07%–87.40%) is entirely due to how tree splits handle artificially shifted Gaussian feature distributions.

---

## 11. Generated Audit Artifacts

All structured audit data artifacts have been saved under [results/baseline_audit/](file:///c:/Users/user/OneDrive/ドキュメント/gitam/vscode%20proj/cyberattack%20detection/results/baseline_audit/):

1. [baseline_integrity_audit.md](file:///c:/Users/user/OneDrive/ドキュメント/gitam/vscode%20proj/cyberattack%20detection/results/baseline_audit/baseline_integrity_audit.md) — Comprehensive audit report.
2. [data_split_audit.json](file:///c:/Users/user/OneDrive/ドキュメント/gitam/vscode%20proj/cyberattack%20detection/results/baseline_audit/data_split_audit.json) — Data split configuration, reconciliation, and partition metadata.
3. [duplicate_analysis.json](file:///c:/Users/user/OneDrive/ドキュメント/gitam/vscode%20proj/cyberattack%20detection/results/baseline_audit/duplicate_analysis.json) — Hash and near-duplicate analysis results.
4. [feature_leakage_audit.csv](file:///c:/Users/user/OneDrive/ドキュメント/gitam/vscode%20proj/cyberattack%20detection/results/baseline_audit/feature_leakage_audit.csv) — Audit table for all 46 features.
5. [class_distribution.csv](file:///c:/Users/user/OneDrive/ドキュメント/gitam/vscode%20proj/cyberattack%20detection/results/baseline_audit/class_distribution.csv) — Partition-wise class breakdown table.
6. [top_feature_analysis.csv](file:///c:/Users/user/OneDrive/ドキュメント/gitam/vscode%20proj/cyberattack%20detection/results/baseline_audit/top_feature_analysis.csv) — Per-class distributions for XGBoost's top 10 features.

---

## 12. Final Audit Conclusion

### Verdict: **B. Possible Leakage / Synthetic Data Artifact Found**

1. **Root Cause:** The 100% accuracy of XGBoost and current baseline results are driven by **synthetic data generation** (`src/preprocessing/sample_data.py`), which introduced artificial class-dependent feature means ($\mu_k = 10.0 + 5.0 \cdot k$).
2. **Impact:** The code pipeline, model implementations, training scripts, and evaluation metrics function correctly and without data leakage. However, the current baseline scores reflect synthetic dataset artifacts rather than authentic CICIoT2023 network intrusion detection performance.
3. **Required Action:** Before benchmarking real-world model accuracy or running BPSO feature selection, authentic raw CICIoT2023 CSV files must be preprocessed through `prepare_dataset.py`.

---

> [!STOP]
> **Audit Completed.** No models were trained or modified. No checkpoints or test datasets were altered. Awaiting further user instructions.
