# Proposed Model Evaluation Report: CNN-Transformer + BPSO (27 Features) — REAL CICIoT2023

## Executive Summary

This report documents the performance of the **main proposed architecture**: the **CNN-Transformer classifier operating on the 27 BPSO-selected features** on the real CICIoT2023 dataset.

### Key Performance Summary
- **Input Feature Count**: `27` (41.3% reduction from 46 original features)
- **Epochs Completed**: `14` (Early stopped at epoch 9)
- **Best Validation Loss**: `1.724596`
- **Test Set Accuracy**: `53.80%` (`0.537973`)
- **Macro F1-Score**: `0.261354`
- **Weighted F1-Score**: `0.590346`
- **Macro Precision**: `0.269088`
- **Macro Recall**: `0.309511`
- **Macro ROC-AUC**: `0.656733`
- **Macro PR-AUC**: `0.304337`
- **Total Training Duration**: `2933.4s`

---

## Benchmark Comparison Across All 6 Research Experiments

| Experiment / Model | Features | Test Acc | Macro F1 | Weighted F1 | Macro Recall | Macro ROC-AUC | Macro PR-AUC | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. CNN-Transformer Baseline** | 46 | 56.40% | 0.5148 | 0.6063 | 0.5124 | 0.8924 | 0.5512 | ~1200s |
| **2. CNN-Transformer Class-Weighted** | 46 | 63.04% | 0.5288 | 0.6869 | 0.5747 | 0.9084 | 0.5841 | ~1250s |
| **3. MLP Baseline** | 46 | 63.46% | 0.5694 | 0.6791 | 0.5512 | 0.9204 | 0.6041 | ~450s |
| **4. XGBoost Baseline** | 46 | 74.91% | 0.7032 | 0.7746 | 0.6953 | 0.9731 | 0.7621 | ~75s |
| **5. XGBoost + BPSO** | 27 | 74.56% | 0.6960 | 0.7716 | 0.6899 | 0.9725 | 0.7585 | 44.58s |
| **6. Proposed CNN-Transformer + BPSO** | **27** | **53.80%** | **0.2614** | **0.5903** | **0.3095** | **0.6567** | **0.3043** | **2933.4s** |

---

## Per-Class Evaluation Metrics (Proposed Model)

| Class ID | Class Name | Precision | Recall | F1-Score | Support | ROC-AUC | PR-AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | **Benign** | 0.0000 | 0.0000 | 0.0000 | 164,719 | 0.2427 | 0.0177 |
| 1 | **BruteForce** | 0.0000 | 0.0000 | 0.0000 | 1,960 | 0.6822 | 0.0068 |
| 2 | **DDoS** | 0.8101 | 0.6119 | 0.6972 | 4,127,344 | 0.6513 | 0.8636 |
| 3 | **DoS** | 0.1902 | 0.2406 | 0.2124 | 985,553 | 0.5576 | 0.2087 |
| 4 | **Mirai** | 0.8501 | 0.8660 | 0.8580 | 259,041 | 0.9904 | 0.9348 |
| 5 | **Spoofing** | 0.0332 | 0.4480 | 0.0618 | 72,966 | 0.7871 | 0.0981 |
| 6 | **Web-based** | 0.0000 | 0.0000 | 0.0000 | 1,691 | 0.6858 | 0.0005 |

---

## Selected 27 BPSO Feature Subset
`['Header_Length', 'Protocol Type', 'Duration', 'Rate', 'Drate', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number', 'syn_count', 'rst_count', 'HTTPS', 'DNS', 'Telnet', 'SSH', 'ICMP', 'IPv', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight']`

---

## Conclusion & Scientific Analysis

1. **Impact of BPSO Feature Selection on CNN-Transformer**:
   - 46-Feature Baseline Test Acc: `56.40%` (Macro F1 = `0.5148`)
   - 27-Feature BPSO Test Acc: `53.80%` (Macro F1 = `0.2614`)
   - BPSO feature selection change for CNN-Transformer: **-2.6% Accuracy**, **-0.2534 Macro F1**.

2. **Overall Best Performing IDS Model**:
   - The tree-based gradient boosted models (XGBoost Baseline and XGBoost + BPSO) remain the overall top-performing algorithms on real tabular CICIoT2023 data (`74.91%` and `74.56%` accuracy).

---
*Report generated automatically at 2026-09-12 15:40:09*
