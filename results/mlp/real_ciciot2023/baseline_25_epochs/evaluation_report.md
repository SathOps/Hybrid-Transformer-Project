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
| **Test Set (Evaluation)** | **5,613,274** | **63.46%** | **0.8370** |

> [!NOTE]
> The test set evaluation was performed strictly in inference mode (`training=False`). The test split remained completely untouched and was never used during model training or checkpoint selection.

---

## Key Aggregate Metrics

- **Test Accuracy**: `63.46%` (`0.634618`)
- **Test Loss**: `0.836988`
- **Macro Precision**: `0.304072`
- **Weighted Precision**: `0.718302`
- **Macro Recall**: `0.453237`
- **Weighted Recall**: `0.634618`
- **Macro F1-Score**: `0.340587`
- **Weighted F1-Score**: `0.656431`
- **Macro ROC-AUC (OvR)**: `0.926279`
- **Weighted ROC-AUC (OvR)**: `0.810655`
- **Macro PR-AUC (OvR)**: `0.566948`
- **Weighted PR-AUC (OvR)**: `0.813679`
- **Inference Time**: `19.6660` s (`0.0035` ms/sample)

---

## Per-Class Performance Breakdown

| Class Name | Precision | Recall | F1-Score | Support | ROC-AUC (OvR) | PR-AUC (OvR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Benign** | 0.0000 | 0.0000 | 0.0000 | 164,719 | 0.9963 | 0.8199 |
| **BruteForce** | 0.0000 | 0.0000 | 0.0000 | 1,960 | 0.9845 | 0.2807 |
| **DDoS** | 0.8621 | 0.6594 | 0.7473 | 4,127,344 | 0.8090 | 0.9285 |
| **DoS** | 0.2772 | 0.5171 | 0.3609 | 985,553 | 0.7229 | 0.3000 |
| **Mirai** | 0.6899 | 0.9962 | 0.8152 | 259,041 | 0.9995 | 0.9942 |
| **Spoofing** | 0.2993 | 1.0000 | 0.4607 | 72,966 | 0.9922 | 0.6361 |
| **Web-based** | 0.0000 | 0.0000 | 0.0000 | 1,691 | 0.9796 | 0.0092 |

---

## Confusion Matrix

```
True \ Pred	Benign	BruteF	DDoS	DoS	Mirai	Spoofi	Web-ba
Benign    	0	0	0	0	0	164719	0
BruteForce	0	0	0	0	0	1960	0
DDoS      	0	0	2721670	1328689	75448	1537	0
DoS       	0	0	434891	509591	40557	514	0
Mirai     	0	0	344	261	258057	379	0
Spoofing  	0	0	2	0	0	72964	0
Web-based 	0	0	0	0	0	1691	0
```

---

## Verification & Integrity Safeguards

1. **Model Checkpoint**: Loaded successfully from `checkpoints/cnn_transformer/baseline_50_epochs/best_model.keras`.
2. **Output Contract**: Shape `(batch_size, 8)`, finite outputs (`isfinite == True`), Softmax probabilities sum to `1.0`.
3. **Data Protection**: `data/processed/prepared/test/` read-only streaming pass. No files modified or deleted.
