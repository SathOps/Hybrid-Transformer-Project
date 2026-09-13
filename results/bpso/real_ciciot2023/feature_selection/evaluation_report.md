# BPSO Feature Selection & XGBoost Evaluation Report — REAL CICIoT2023

## Executive Summary

Binary Particle Swarm Optimization (BPSO) was applied to select an optimal feature subset from the **46** original preprocessed CICIoT2023 features.
The BPSO fitness objective optimized **Validation Macro F1-Score** using a fixed, reproducible subset ($N=200,000$) sampled strictly from training and validation partitions with `seed=42`. Test data was kept **100% untouched** during feature selection.

### BPSO Feature Reduction Summary
- **Original Feature Count**: `46`
- **BPSO Selected Feature Count**: `27` (`58.7%` retained, `41.3%` reduction)
- **Best BPSO Iteration**: Epoch/Iteration `13` / `20`
- **Best BPSO Validation Macro F1**: `0.662942`

---

## Benchmark Comparison: BPSO Selected Subset vs. Full 46-Feature Baseline

| Metric | Full 46-Feature XGBoost Baseline | **BPSO Selected (27 Features) XGBoost** | Absolute Change |
| :--- | :---: | :---: | :---: |
| **Feature Count** | 46 | **27** | **-19 features (-41.3%)** |
| **Test Accuracy** | 74.91% (`0.749095`) | **74.56% (`0.745646`)** | **-0.0034 (-0.34%)** |
| **Test Log Loss** | 0.420332 | **0.423912** | **+0.003580** |
| **Macro F1-Score** | 0.703153 | **0.695970** | **-0.007183** |
| **Weighted F1-Score** | 0.774626 | **0.771576** | **-0.003050** |
| **Macro Precision** | 0.865403 | **0.859439** | **-0.005964** |
| **Weighted Precision** | 0.872807 | **0.872088** | **-0.000719** |
| **Macro Recall** | 0.694793 | **0.689886** | **-0.004907** |
| **Weighted Recall** | 0.749095 | **0.745646** | **-0.003449** |
| **Macro ROC-AUC** | 0.972964 | **0.972499** | **-0.000465** |
| **Macro PR-AUC** | 0.763626 | **0.758505** | **-0.005121** |
| **Training Time** | 50.68 s | **44.58 s** | **-6.10 s** |

---

## Selected Features List (27 Features)

| Selected Index | Feature Name | Original Index |
| :---: | :--- | :---: |
| **1** | `Header_Length` | 1 |
| **2** | `Protocol Type` | 2 |
| **3** | `Duration` | 3 |
| **4** | `Rate` | 4 |
| **6** | `Drate` | 6 |
| **8** | `syn_flag_number` | 8 |
| **9** | `rst_flag_number` | 9 |
| **10** | `psh_flag_number` | 10 |
| **15** | `syn_count` | 15 |
| **18** | `rst_count` | 18 |
| **20** | `HTTPS` | 20 |
| **21** | `DNS` | 21 |
| **22** | `Telnet` | 22 |
| **24** | `SSH` | 24 |
| **30** | `ICMP` | 30 |
| **31** | `IPv` | 31 |
| **34** | `Min` | 34 |
| **35** | `Max` | 35 |
| **36** | `AVG` | 36 |
| **37** | `Std` | 37 |
| **38** | `Tot size` | 38 |
| **39** | `IAT` | 39 |
| **41** | `Magnitue` | 41 |
| **42** | `Radius` | 42 |
| **43** | `Covariance` | 43 |
| **44** | `Variance` | 44 |
| **45** | `Weight` | 45 |

---

## Per-Class Test Performance Breakdown

| Class Name | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Benign** | 0.9091 | 0.9874 | 0.9466 | 164,719 |
| **BruteForce** | 0.9229 | 0.2689 | 0.4164 | 1,960 |
| **DDoS** | 0.9732 | 0.6775 | 0.7988 | 4,127,344 |
| **DoS** | 0.4060 | 0.9211 | 0.5636 | 985,553 |
| **Mirai** | 0.9855 | 0.9987 | 0.9920 | 259,041 |
| **Spoofing** | 0.9636 | 0.8142 | 0.8826 | 72,966 |
| **Web-based** | 0.8558 | 0.1614 | 0.2716 | 1,691 |
