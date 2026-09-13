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
| **Test Set (Evaluation)** | **5,613,274** | **53.80%** | **1.6029** |

> [!NOTE]
> The test set evaluation was performed strictly in inference mode (`training=False`). The test split remained completely untouched and was never used during model training or checkpoint selection.

---

## Key Aggregate Metrics

- **Test Accuracy**: `53.80%` (`0.537973`)
- **Test Loss**: `1.602948`
- **Macro Precision**: `0.269088`
- **Weighted Precision**: `0.668727`
- **Macro Recall**: `0.309511`
- **Weighted Recall**: `0.537973`
- **Macro F1-Score**: `0.261354`
- **Weighted F1-Score**: `0.590346`
- **Macro ROC-AUC (OvR)**: `0.656733`
- **Weighted ROC-AUC (OvR)**: `0.640329`
- **Macro PR-AUC (OvR)**: `0.304337`
- **Weighted PR-AUC (OvR)**: `0.716605`
- **Inference Time**: `262.4756` s (`0.0468` ms/sample)

---

## Per-Class Performance Breakdown

| Class Name | Precision | Recall | F1-Score | Support | ROC-AUC (OvR) | PR-AUC (OvR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Benign** | 0.0000 | 0.0000 | 0.0000 | 164,719 | 0.2427 | 0.0177 |
| **BruteForce** | 0.0000 | 0.0000 | 0.0000 | 1,960 | 0.6822 | 0.0068 |
| **DDoS** | 0.8101 | 0.6119 | 0.6972 | 4,127,344 | 0.6513 | 0.8636 |
| **DoS** | 0.1902 | 0.2406 | 0.2124 | 985,553 | 0.5576 | 0.2087 |
| **Mirai** | 0.8501 | 0.8660 | 0.8580 | 259,041 | 0.9904 | 0.9348 |
| **Spoofing** | 0.0332 | 0.4480 | 0.0618 | 72,966 | 0.7871 | 0.0981 |
| **Web-based** | 0.0000 | 0.0000 | 0.0000 | 1,691 | 0.6858 | 0.0005 |

---

## Confusion Matrix

```
True \ Pred	Benign	BruteF	DDoS	DoS	Mirai	Spoofi	Web-ba
Benign    	0	0	80520	51365	4	32830	0
BruteForce	0	0	990	333	0	637	0
DDoS      	0	0	2525597	949431	4935	647381	0
DoS       	0	0	453559	237163	34553	260278	0
Mirai     	0	0	23820	997	224341	9883	0
Spoofing  	0	0	32509	7714	56	32687	0
Web-based 	0	0	511	206	0	974	0
```

---

## Verification & Integrity Safeguards

1. **Model Checkpoint**: Loaded successfully from `checkpoints/cnn_transformer/baseline_50_epochs/best_model.keras`.
2. **Output Contract**: Shape `(batch_size, 8)`, finite outputs (`isfinite == True`), Softmax probabilities sum to `1.0`.
3. **Data Protection**: `data/processed/prepared/test/` read-only streaming pass. No files modified or deleted.
