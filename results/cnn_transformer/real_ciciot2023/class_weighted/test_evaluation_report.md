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
| **Test Set (Evaluation)** | **5,613,274** | **63.04%** | **1.0165** |

> [!NOTE]
> The test set evaluation was performed strictly in inference mode (`training=False`). The test split remained completely untouched and was never used during model training or checkpoint selection.

---

## Key Aggregate Metrics

- **Test Accuracy**: `63.04%` (`0.630424`)
- **Test Loss**: `1.016495`
- **Macro Precision**: `0.181308`
- **Weighted Precision**: `0.600360`
- **Macro Recall**: `0.288530`
- **Weighted Recall**: `0.630424`
- **Macro F1-Score**: `0.207886`
- **Weighted F1-Score**: `0.612664`
- **Macro ROC-AUC (OvR)**: `0.726522`
- **Weighted ROC-AUC (OvR)**: `0.616998`
- **Macro PR-AUC (OvR)**: `0.252926`
- **Weighted PR-AUC (OvR)**: `0.667637`
- **Inference Time**: `207.8061` s (`0.0370` ms/sample)

---

## Per-Class Performance Breakdown

| Class Name | Precision | Recall | F1-Score | Support | ROC-AUC (OvR) | PR-AUC (OvR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Benign** | 0.0000 | 0.0000 | 0.0000 | 164,719 | 0.9691 | 0.2994 |
| **BruteForce** | 0.0000 | 0.0000 | 0.0000 | 1,960 | 0.9906 | 0.1376 |
| **DDoS** | 0.7618 | 0.7832 | 0.7724 | 4,127,344 | 0.6117 | 0.8404 |
| **DoS** | 0.2068 | 0.2365 | 0.2207 | 985,553 | 0.5572 | 0.2006 |
| **Mirai** | 0.0000 | 0.0000 | 0.0000 | 259,041 | 0.6016 | 0.0560 |
| **Spoofing** | 0.3005 | 1.0000 | 0.4622 | 72,966 | 0.9814 | 0.2363 |
| **Web-based** | 0.0000 | 0.0000 | 0.0000 | 1,691 | 0.3742 | 0.0002 |

---

## Confusion Matrix

```
True \ Pred	Benign	BruteF	DDoS	DoS	Mirai	Spoofi	Web-ba
Benign    	0	0	0	0	0	164719	0
BruteForce	0	0	0	0	0	1960	0
DDoS      	0	0	3232682	893918	0	744	0
DoS       	0	0	752292	233100	0	161	0
Mirai     	0	0	258491	1	0	549	0
Spoofing  	0	0	3	0	0	72963	0
Web-based 	0	0	0	0	0	1691	0
```

---

## Verification & Integrity Safeguards

1. **Model Checkpoint**: Loaded successfully from `checkpoints/cnn_transformer/baseline_50_epochs/best_model.keras`.
2. **Output Contract**: Shape `(batch_size, 8)`, finite outputs (`isfinite == True`), Softmax probabilities sum to `1.0`.
3. **Data Protection**: `data/processed/prepared/test/` read-only streaming pass. No files modified or deleted.
