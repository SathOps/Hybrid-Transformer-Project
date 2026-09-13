# CNN-Transformer Baseline Test Evaluation Report

## Executive Summary

The trained **CNN-Transformer** model checkpoint (`best_model.keras`) was evaluated on the untouched test dataset partition (`data/processed/prepared/test/`).

### Dataset & Partition Metrics Comparison

| Partition | Samples Evaluated | Accuracy | Loss |
| :--- | :--- | :--- | :--- |
| **Training (50-epoch)** | 6,310,310 (or shard batches) | **89.92%** | 1.2145 |
| **Validation (Best Epoch 50)** | 1,352,209 (or shard batches) | **87.47%** | 1.1614 |
| **Test Set (Evaluation)** | **3,000** | **87.40%** | **0.5855** |

> [!NOTE]
> The test set evaluation was performed strictly in inference mode (`training=False`). The test split remained completely untouched and was never used during model training or checkpoint selection.

---

## Key Aggregate Metrics

- **Test Accuracy**: `87.40%` (`0.874000`)
- **Test Loss**: `0.585503`
- **Macro Precision**: `0.811257`
- **Weighted Precision**: `0.811257`
- **Macro Recall**: `0.874000`
- **Weighted Recall**: `0.874000`
- **Macro F1-Score**: `0.832390`
- **Weighted F1-Score**: `0.832390`
- **Macro ROC-AUC (OvR)**: `0.999979`
- **Weighted ROC-AUC (OvR)**: `0.999979`
- **Macro PR-AUC (OvR)**: `0.998843`
- **Weighted PR-AUC (OvR)**: `0.998843`
- **Inference Time**: `0.1530` s (`0.0510` ms/sample)

---

## Per-Class Performance Breakdown

| Class Name | Precision | Recall | F1-Score | Support | ROC-AUC (OvR) | PR-AUC (OvR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Benign** | 1.0000 | 1.0000 | 1.0000 | 375 | 1.0000 | 0.9989 |
| **BruteForce** | 0.0000 | 0.0000 | 0.0000 | 375 | 1.0000 | 0.9989 |
| **DDoS** | 1.0000 | 1.0000 | 1.0000 | 375 | 1.0000 | 0.9989 |
| **DoS** | 1.0000 | 1.0000 | 1.0000 | 375 | 1.0000 | 0.9989 |
| **Mirai** | 1.0000 | 1.0000 | 1.0000 | 375 | 1.0000 | 0.9989 |
| **Recon** | 1.0000 | 1.0000 | 1.0000 | 375 | 1.0000 | 0.9989 |
| **Spoofing** | 0.9921 | 1.0000 | 0.9960 | 375 | 1.0000 | 0.9989 |
| **Web-based** | 0.4980 | 0.9920 | 0.6631 | 375 | 0.9998 | 0.9984 |

---

## Confusion Matrix

```
True \ Pred	Benign	BruteF	DDoS	DoS	Mirai	Recon	Spoofi	Web-ba
Benign    	375	0	0	0	0	0	0	0
BruteForce	0	0	0	0	0	0	0	375
DDoS      	0	0	375	0	0	0	0	0
DoS       	0	0	0	375	0	0	0	0
Mirai     	0	0	0	0	375	0	0	0
Recon     	0	0	0	0	0	375	0	0
Spoofing  	0	0	0	0	0	0	375	0
Web-based 	0	0	0	0	0	0	3	372
```

---

## Verification & Integrity Safeguards

1. **Model Checkpoint**: Loaded successfully from `checkpoints/cnn_transformer/baseline_50_epochs/best_model.keras`.
2. **Output Contract**: Shape `(batch_size, 8)`, finite outputs (`isfinite == True`), Softmax probabilities sum to `1.0`.
3. **Data Protection**: `data/processed/prepared/test/` read-only streaming pass. No files modified or deleted.
