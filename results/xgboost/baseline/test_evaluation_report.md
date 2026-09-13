# XGBoost Baseline Test Evaluation Report

## Executive Summary

The trained **XGBoost Baseline** model (`model.json`) was evaluated on the untouched test dataset partition (`data/processed/prepared/test/`).

### Dataset & Partition Metrics Comparison

| Partition / Model | Samples Evaluated | Accuracy | Loss / Log-Loss |
| :--- | :--- | :--- | :--- |
| **XGBoost Test Set (Evaluation)** | **3,000** | **100.00%** | **0.0163** |
| *CNN-Transformer Test Baseline* | *3,000* | *87.40%* | *0.5855* |
| *MLP Test Baseline* | *3,000* | *80.07%* | *1.0051* |

---

## Key Aggregate Metrics

- **Test Accuracy**: `100.00%` (`1.000000`)
- **Test Log Loss**: `0.016267`
- **Macro Precision**: `1.000000`
- **Weighted Precision**: `1.000000`
- **Macro Recall**: `1.000000`
- **Weighted Recall**: `1.000000`
- **Macro F1-Score**: `1.000000`
- **Weighted F1-Score**: `1.000000`
- **Macro ROC-AUC (OvR)**: `1.000000`
- **Weighted ROC-AUC (OvR)**: `1.000000`
- **Macro PR-AUC (OvR)**: `0.998902`
- **Weighted PR-AUC (OvR)**: `0.998902`
- **Inference Time**: `0.0028` s (`0.0009` ms/sample)

---

## Per-Class Performance Breakdown

| Class Name | Precision | Recall | F1-Score | Support | ROC-AUC (OvR) | PR-AUC (OvR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Benign** | 1.0000 | 1.0000 | 1.0000 | 375 | 1.0000 | 0.9989 |
| **BruteForce** | 1.0000 | 1.0000 | 1.0000 | 375 | 1.0000 | 0.9989 |
| **DDoS** | 1.0000 | 1.0000 | 1.0000 | 375 | 1.0000 | 0.9989 |
| **DoS** | 1.0000 | 1.0000 | 1.0000 | 375 | 1.0000 | 0.9989 |
| **Mirai** | 1.0000 | 1.0000 | 1.0000 | 375 | 1.0000 | 0.9989 |
| **Recon** | 1.0000 | 1.0000 | 1.0000 | 375 | 1.0000 | 0.9989 |
| **Spoofing** | 1.0000 | 1.0000 | 1.0000 | 375 | 1.0000 | 0.9989 |
| **Web-based** | 1.0000 | 1.0000 | 1.0000 | 375 | 1.0000 | 0.9989 |

---

## Top 10 Features by Gain Importance

| Rank | Feature Name | Gain Score |
| :---: | :--- | :---: |
| 1 | **Tot size** | 131.3154 |
| 2 | **HTTPS** | 114.4035 |
| 3 | **Std** | 105.9333 |
| 4 | **HTTP** | 72.4317 |
| 5 | **Radius** | 70.8537 |
| 6 | **syn_flag_number** | 67.4915 |
| 7 | **DHCP** | 62.2002 |
| 8 | **ack_count** | 58.4046 |
| 9 | **SSH** | 57.0477 |
| 10 | **urg_count** | 56.8228 |

---

## Confusion Matrix

```
True \ Pred	Benign	BruteF	DDoS	DoS	Mirai	Recon	Spoofi	Web-ba
Benign    	375	0	0	0	0	0	0	0
BruteForce	0	375	0	0	0	0	0	0
DDoS      	0	0	375	0	0	0	0	0
DoS       	0	0	0	375	0	0	0	0
Mirai     	0	0	0	0	375	0	0	0
Recon     	0	0	0	0	0	375	0	0
Spoofing  	0	0	0	0	0	0	375	0
Web-based 	0	0	0	0	0	0	0	375
```

---

## Verification & Integrity Safeguards

1. **Model File**: Loaded successfully from `checkpoints/xgboost/baseline/model.json`.
2. **Output Contract**: Probability shape `(3000, 8)`, finite predictions (`isfinite == True`), row sums equal `1.0`.
3. **Data Protection**: `data/processed/prepared/test/` read-only streaming pass. No files modified or deleted.
