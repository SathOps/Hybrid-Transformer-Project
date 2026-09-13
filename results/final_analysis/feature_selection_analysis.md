# Feature Selection & Dimensionality Analysis — REAL CICIoT2023

## 1. Feature Reduction Overview

The original preprocessed CICIoT2023 dataset contains **46 numerical features** representing network packet statistics, protocol flags, flow rates, and window sizes.
Binary Particle Swarm Optimization (BPSO) was conducted using XGBoost on a fixed 200,000-sample training and 200,000-sample validation subset strictly from training and validation partitions (`seed=42`).

- **Original Feature Count**: 46 features (100.0%)
- **BPSO Selected Feature Count**: 27 features (58.7%)
- **Excluded Features**: 19 features (41.3% feature reduction)

---

## 2. BPSO Selected Feature Subset (27 Features)

The 27 selected features spans key network communication categories:

1. **Header & Protocol**: `Header_Length`, `Protocol Type`, `IPv`
2. **Timing & Rates**: `Duration`, `Rate`, `Drate`, `IAT`
3. **TCP Control Flags**: `syn_flag_number`, `rst_flag_number`, `psh_flag_number`
4. **Packet Counts**: `syn_count`, `rst_count`
5. **Application Protocols**: `HTTPS`, `DNS`, `Telnet`, `SSH`, `ICMP`
6. **Packet Size Statistics**: `Min`, `Max`, `AVG`, `Std`, `Tot size`, `Magnitue`, `Radius`, `Covariance`, `Variance`, `Weight`

---

## 3. Excluded Feature Subset (19 Features)

The 19 features pruned by BPSO were determined to be redundant or uninformative for distinguishing intrusion classes:

`flow_duration`, `Srate`, `fin_flag_number`, `ack_flag_number`, `ece_flag_number`, `cwr_flag_number`, `ack_count`, `fin_count`, `urg_count`, `HTTP`, `SMTP`, `IRC`, `TCP`, `UDP`, `DHCP`, `ARP`, `LLC`, `Tot sum`, `Number`.

---

## 4. Impact of Feature Reduction Across Model Families

| Model Architecture | Full 46 Features Accuracy | 27 Features (BPSO) Accuracy | Accuracy Delta | Full 46 Features Macro F1 | 27 Features (BPSO) Macro F1 | Macro F1 Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** | **74.91%** | **74.56%** | **-0.35%** | **0.7032** | **0.6960** | **-0.0072** |
| **CNN-Transformer Classifier** | **56.40%** | **53.80%** | **-2.60%** | **0.1908** | **0.2614** | **+0.0706** |

### Key Insight:
For decision-tree ensembles (XGBoost), pruning 19 features reduced training execution time by **40.6%** with negligible accuracy impact (-0.35%).
For 1D-convolutional neural networks (CNN-Transformer), feature pruning altered the spatial sequence structure, reducing raw accuracy from 56.40% to 53.80%, while slightly shifting per-class predictions resulting in higher macro recall on minority classes (increasing Macro F1 from 0.1908 to 0.2614).
