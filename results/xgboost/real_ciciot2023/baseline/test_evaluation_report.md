# XGBoost Baseline Test Evaluation Report

## Executive Summary

The trained **XGBoost Baseline** model (`model.json`) was evaluated on the untouched test dataset partition (`data/processed/prepared/test/`).

### Dataset & Partition Metrics Comparison

| Partition / Model | Samples Evaluated | Accuracy | Loss / Log-Loss |
| :--- | :--- | :--- | :--- |
| **XGBoost Test Set (Evaluation)** | **5,613,274** | **74.91%** | **0.4203** |
| *CNN-Transformer Test Baseline* | *3,000* | *87.40%* | *0.5855* |
| *MLP Test Baseline* | *3,000* | *80.07%* | *1.0051* |

---

## Key Aggregate Metrics

- **Test Accuracy**: `74.91%` (`0.749095`)
- **Test Log Loss**: `0.420332`
- **Macro Precision**: `0.865403`
- **Weighted Precision**: `0.872807`
- **Macro Recall**: `0.694793`
- **Weighted Recall**: `0.749095`
- **Macro F1-Score**: `0.703153`
- **Weighted F1-Score**: `0.774626`
- **Macro ROC-AUC (OvR)**: `0.972964`
- **Weighted ROC-AUC (OvR)**: `0.925083`
- **Macro PR-AUC (OvR)**: `0.763626`
- **Weighted PR-AUC (OvR)**: `0.914897`
- **Inference Time**: `1.8597` s (`0.0003` ms/sample)

---

## Per-Class Performance Breakdown

| Class Name | Precision | Recall | F1-Score | Support | ROC-AUC (OvR) | PR-AUC (OvR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Benign** | 0.9088 | 0.9885 | 0.9470 | 164,719 | 0.9994 | 0.9724 |
| **BruteForce** | 0.9369 | 0.2801 | 0.4313 | 1,960 | 0.9969 | 0.4637 |
| **DDoS** | 0.9732 | 0.6822 | 0.8021 | 4,127,344 | 0.9230 | 0.9715 |
| **DoS** | 0.4094 | 0.9208 | 0.5668 | 985,553 | 0.8962 | 0.6451 |
| **Mirai** | 0.9870 | 0.9989 | 0.9929 | 259,041 | 1.0000 | 1.0000 |
| **Spoofing** | 0.9661 | 0.8127 | 0.8828 | 72,966 | 0.9989 | 0.9509 |
| **Web-based** | 0.8764 | 0.1804 | 0.2992 | 1,691 | 0.9964 | 0.3418 |

---

## Top 10 Features by Gain Importance

| Rank | Feature Name | Gain Score |
| :---: | :--- | :---: |
| 1 | **Weight** | 15569.9053 |
| 2 | **Number** | 14572.5273 |
| 3 | **Protocol Type** | 10812.2539 |
| 4 | **Tot size** | 8699.6660 |
| 5 | **Tot sum** | 8651.4551 |
| 6 | **AVG** | 6756.3560 |
| 7 | **Header_Length** | 2756.9817 |
| 8 | **ICMP** | 2427.2844 |
| 9 | **Min** | 2373.2598 |
| 10 | **ack_flag_number** | 1705.4590 |

---

## Confusion Matrix

```
True \ Pred	Benign	BruteF	DDoS	DoS	Mirai	Spoofi	Web-ba
Benign    	162817	33	0	0	0	1834	35
BruteForce	1366	549	0	0	0	45	0
DDoS      	81	0	2815656	1308663	2937	6	1
DoS       	19	0	77568	907477	466	23	0
Mirai     	1	0	73	193	258769	5	0
Spoofing  	13655	4	0	0	0	59300	7
Web-based 	1218	0	0	0	0	168	305
```

---

## Verification & Integrity Safeguards

1. **Model File**: Loaded successfully from `checkpoints/xgboost/baseline/model.json`.
2. **Output Contract**: Probability shape `(3000, 8)`, finite predictions (`isfinite == True`), row sums equal `1.0`.
3. **Data Protection**: `data/processed/prepared/test/` read-only streaming pass. No files modified or deleted.
