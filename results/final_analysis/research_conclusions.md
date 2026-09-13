# Research Conclusions, Limitations & Deployment Recommendations

## 1. Summary of Scientific Findings

1. **Tree-Based Superiority on Tabular Intrusion Data**:
   - XGBoost demonstrated state-of-the-art accuracy (`74.91%`) and Macro F1 (`0.7032`) with extremely low computational overhead (50.68s training, 0.0003 ms/sample inference).

2. **Efficacy of BPSO Feature Selection**:
   - BPSO selected a **27-feature subset** (41.3% reduction).
   - On XGBoost, BPSO reduced training time by **40.6%** with negligible accuracy impact (`74.91%` → `74.56%`).
   - On CNN-Transformer, feature removal disrupted 1D-convolutional receptive fields, reducing accuracy (`56.40%` → `53.80%`).

---

## 2. Recommended Models

- **Recommended for Real-Time High-Throughput Production Deployment**:
  **XGBoost + BPSO (27 Features)** — Delivers `74.56%` test accuracy, `0.7716` Weighted F1, `0.0003 ms/sample` latency, and requires 41.3% fewer input feature calculations.

- **Recommended Baseline for Research Comparison**:
  **XGBoost Baseline (46 Features)** — Highest overall accuracy (`74.91%`) and Macro F1 (`0.7032`).

---

## 3. Limitations

1. **Tabular Feature Locality in CNNs**:
   - 1D-CNN filters assume ordered spatial relationships between adjacent channels, which tabular network statistics do not naturally possess.
2. **Class Imbalance Sensitivity**:
   - Extremely rare attack classes (`BruteForce` with 1,960 samples, `Web-based` with 1,691 samples) remain challenging for unweighted neural networks without specialized focal losses or sampling techniques.

---

## 4. Future Research Directions

- **Graph Neural Networks (GNNs)** to model topological host-to-host flow connections.
- **Focal Loss & SMOTE Oversampling** for severe minority class imbalance.
- **TabNet / FT-Transformer** architectures specifically designed for tabular data.
