# CNN-Transformer Baseline Test Evaluation Report — REAL CICIoT2023

> [!IMPORTANT]
> **RESEARCH LABELING STATEMENT**
> - **Dataset**: REAL CICIoT2023
> - **Active Model Classes**: 7 ACTIVE CLASSES (Benign, BruteForce, DDoS, DoS, Mirai, Spoofing, Web-based; Recon inactive with 0 instances)
> - **Training Data**: SUBSAMPLED TRAINING PARTITION (2,626,223 rows)
> - **Validation Partition**: NATURAL VALIDATION SET (5,613,269 rows)
> - **Test Partition**: NATURAL HOLD-OUT TEST SET (5,613,274 rows)
> - **Synthetic Data**: NOT USED (0 synthetic samples)

## Executive Summary

The trained **CNN-Transformer** model checkpoint (`best_model.keras`) was evaluated on the untouched natural test dataset partition.

### Test Partition Evaluation Metrics

| Partition | Samples Evaluated | Accuracy | Loss |
| :--- | :--- | :--- | :--- |
| **Test Set (Evaluation)** | **5,613,274** | **56.40%** | **0.9554** |

> [!NOTE]
> The test set evaluation was performed strictly in inference mode (`training=False`). The test split remained completely untouched and was never used during model training or checkpoint selection.

---

## Key Aggregate Metrics

- **Test Accuracy**: `56.40%` (`0.564006`)
- **Test Loss**: `0.955435`
- **Macro Precision**: `0.166511`
- **Weighted Precision**: `0.566508`
- **Macro Recall**: `0.270536`
- **Weighted Recall**: `0.564006`
- **Macro F1-Score**: `0.190772`
- **Weighted F1-Score**: `0.561702`
- **Macro ROC-AUC (OvR)**: `0.688962`
- **Weighted ROC-AUC (OvR)**: `0.619231`
- **Macro PR-AUC (OvR)**: `0.205454`
- **Weighted PR-AUC (OvR)**: `0.662484`
- **Inference Time**: `201.7140` s (`0.0359` ms/sample)

---

## Per-Class Performance Breakdown

| Class Name | Precision | Recall | F1-Score | Support | ROC-AUC (OvR) | PR-AUC (OvR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Benign** | 0.0000 | 0.0000 | 0.0000 | 164,719 | 0.7769 | 0.0575 |
| **BruteForce** | 0.0000 | 0.0000 | 0.0000 | 1,960 | 0.9343 | 0.0025 |
| **DDoS** | 0.7334 | 0.7041 | 0.7185 | 4,127,344 | 0.6145 | 0.8459 |
| **DoS** | 0.1329 | 0.1897 | 0.1563 | 985,553 | 0.5649 | 0.1822 |
| **Mirai** | 0.0000 | 0.0000 | 0.0000 | 259,041 | 0.6988 | 0.0678 |
| **Spoofing** | 0.2993 | 1.0000 | 0.4607 | 72,966 | 0.9830 | 0.2820 |
| **Web-based** | 0.0000 | 0.0000 | 0.0000 | 1,691 | 0.2504 | 0.0002 |

---

## Confusion Matrix

```
True \ Pred	Benign	BruteF	DDoS	DoS	Mirai	Spoofi	Web-ba
Benign    	0	0	0	0	0	164719	0
BruteForce	1	0	0	0	0	1959	0
DDoS      	156	0	2906010	1219842	0	1336	0
DoS       	0	0	798272	186949	0	332	0
Mirai     	5	0	257834	399	0	803	0
Spoofing  	2	0	0	0	0	72964	0
Web-based 	0	0	0	0	0	1691	0
```

---

## Verification & Integrity Safeguards

1. **Model Checkpoint**: Loaded successfully from `checkpoints/cnn_transformer/baseline_50_epochs/best_model.keras`.
2. **Output Contract**: Shape `(batch_size, 8)`, finite outputs (`isfinite == True`), Softmax probabilities sum to `1.0`.
3. **Data Protection**: `data/processed/prepared/test/` read-only streaming pass. No files modified or deleted.
