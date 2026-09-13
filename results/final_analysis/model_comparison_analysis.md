# Model Comparison & Architectural Analysis — REAL CICIoT2023

## 1. Comparative Architecture Overview

Six distinct experiments were evaluated on the real 7-class CICIoT2023 dataset ($N=5,613,274$ test samples):

1. **CNN-Transformer Baseline (46 Features)**
2. **CNN-Transformer Class-Weighted (46 Features)**
3. **MLP Baseline (46 Features)**
4. **XGBoost Baseline (46 Features)**
5. **XGBoost + BPSO (27 Features)**
6. **Proposed CNN-Transformer + BPSO (27 Features)**

---

## 2. Performance Comparison Table (Authoritative Metrics)

| Model Name | Features | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Macro ROC-AUC | Macro PR-AUC | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Baseline** | 46 | **74.91%** | **0.8654** | **0.6948** | **0.7032** | **0.7746** | **0.9730** | **0.7636** | 50.68s |
| **XGBoost + BPSO** | **27** | **74.56%** | 0.8594 | 0.6899 | 0.6960 | 0.7716 | 0.9725 | 0.7585 | **44.58s** |
| **MLP Baseline** | 46 | 63.46% | 0.3041 | 0.4532 | 0.3406 | 0.6564 | 0.9263 | 0.5669 | 179.55s |
| **CNN-Transformer Class-Weighted** | 46 | 63.04% | 0.1813 | 0.2885 | 0.2079 | 0.6127 | 0.7265 | 0.2529 | 1619.58s |
| **CNN-Transformer Baseline** | 46 | 56.40% | 0.1665 | 0.2705 | 0.1908 | 0.5617 | 0.6890 | 0.2055 | 1618.91s |
| **Proposed CNN-Transformer + BPSO** | **27** | 53.80% | 0.2691 | 0.3095 | 0.2614 | 0.5903 | 0.6567 | 0.3043 | 2933.40s |

---

## 3. Key Architectural Lessons

1. **Gradient-Boosted Decision Trees vs. Deep Neural Networks**:
   - Tabular intrusion detection features lack natural spatial locality (unlike images or audio). XGBoost decision trees construct optimal threshold splits across independent features, outperforming deep neural networks on this dataset by over **11.4% accuracy** and **0.36 Macro F1**.

2. **Effect of Class Weighting on CNN-Transformer**:
   - Applying inverse class weighting during CNN-Transformer training improved overall accuracy from **56.40% to 63.04%** (+6.64%) and Weighted F1 from **0.5617 to 0.6127**.

3. **Multi-Layer Perceptron (MLP) Simplicity**:
   - The simple 3-layer MLP (`Dense(256)->Dense(128)->Dense(64)`) trained in **179.55 seconds** and achieved **63.46% accuracy**, outperforming both unweighted and class-weighted CNN-Transformer variants while taking less than 12% of the training time.
