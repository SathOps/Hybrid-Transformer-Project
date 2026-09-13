# CNN-Transformer Baseline Test Evaluation Report

## Executive Summary

The trained **CNN-Transformer** model checkpoint (`best_model.keras`) was evaluated on the untouched test dataset partition (`data/processed/prepared/test/`).

### Dataset & Partition Metrics Comparison

| Partition | Samples Evaluated | Accuracy | Loss |
| :--- | :--- | :--- | :--- |
| **Training (50-epoch)** | 6,310,310 (or shard batches) | **89.92%** | 1.2145 |
| **Validation (Best Epoch 50)** | 1,352,209 (or shard batches) | **87.47%** | 1.1614 |
| **Test Set (Evaluation)** | **3,000** | **80.07%** | **1.0051** |

> [!NOTE]
> The test set evaluation was performed strictly in inference mode (`training=False`). The test split remained completely untouched and was never used during model training or checkpoint selection.

---

## Key Aggregate Metrics

- **Test Accuracy**: `80.07%` (`0.800667`)
- **Test Loss**: `1.005074`
- **Macro Precision**: `0.772426`
- **Weighted Precision**: `0.772426`
- **Macro Recall**: `0.800667`
- **Weighted Recall**: `0.800667`
- **Macro F1-Score**: `0.771844`
- **Weighted F1-Score**: `0.771844`
- **Macro ROC-AUC (OvR)**: `0.992991`
- **Weighted ROC-AUC (OvR)**: `0.992991`
- **Macro PR-AUC (OvR)**: `0.950396`
- **Weighted PR-AUC (OvR)**: `0.950396`
- **Inference Time**: `0.0298` s (`0.0099` ms/sample)

---

## Per-Class Performance Breakdown

| Class Name | Precision | Recall | F1-Score | Support | ROC-AUC (OvR) | PR-AUC (OvR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Benign** | 1.0000 | 1.0000 | 1.0000 | 375 | 1.0000 | 0.9989 |
| **BruteForce** | 0.9665 | 1.0000 | 0.9830 | 375 | 1.0000 | 0.9989 |
| **DDoS** | 0.9945 | 0.9627 | 0.9783 | 375 | 1.0000 | 0.9987 |
| **DoS** | 0.9971 | 0.9147 | 0.9541 | 375 | 0.9860 | 0.9226 |
| **Mirai** | 0.9148 | 0.8880 | 0.9012 | 375 | 0.9860 | 0.9206 |
| **Recon** | 0.8481 | 0.6400 | 0.7295 | 375 | 0.9942 | 0.9455 |
| **Spoofing** | 0.0000 | 0.0000 | 0.0000 | 375 | 0.9782 | 0.8216 |
| **Web-based** | 0.4584 | 1.0000 | 0.6287 | 375 | 0.9996 | 0.9965 |

---

## Confusion Matrix

```
True \ Pred	Benign	BruteF	DDoS	DoS	Mirai	Recon	Spoofi	Web-ba
Benign    	375	0	0	0	0	0	0	0
BruteForce	0	375	0	0	0	0	0	0
DDoS      	0	13	361	1	0	0	0	0
DoS       	0	0	2	343	30	0	0	0
Mirai     	0	0	0	0	333	41	1	0
Recon     	0	0	0	0	1	240	64	70
Spoofing  	0	0	0	0	0	2	0	373
Web-based 	0	0	0	0	0	0	0	375
```

---

## Verification & Integrity Safeguards

1. **Model Checkpoint**: Loaded successfully from `checkpoints/cnn_transformer/baseline_50_epochs/best_model.keras`.
2. **Output Contract**: Shape `(batch_size, 8)`, finite outputs (`isfinite == True`), Softmax probabilities sum to `1.0`.
3. **Data Protection**: `data/processed/prepared/test/` read-only streaming pass. No files modified or deleted.
