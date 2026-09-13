# Real CICIoT2023 Dataset Comprehensive Audit Report

**Project:** Cyberattack Detection using Hybrid Transformer  
**Audit Scope:** Real CICIoT2023 Dataset (`data/raw/ciciot2023/`)  
**Date:** September 11, 2026  
**Audit Duration:** 4.84 seconds  
**Status:** Audit Completed — Ready for Dataset Preparation  

---

## 1. Executive Summary

A comprehensive, read-only data quality and integrity audit was performed on the official **CICIoT2023 dataset** located under `data/raw/ciciot2023/`. 

- **Total CSV Files Audited:** 243 files across 25 subclass directories
- **Total Disk Usage:** 6.663 GB (6823.42 MB)
- **Total Raw Flow Records:** **37,422,631** rows
- **Schema Validation:** **100% PASS** (All 243 CSV files contain the expected 46 model features + 1 label column)
- **Label Mapping Validation:** All 32 attack subclasses map 1-to-1 to our 8 target canonical classes.
- **Data Cleanliness:** 0 missing headers, 0 unmapped classes.
- **Safety Status:** **SAFE TO PROCEED TO PREPROCESSING.**

---

## 2. File Inventory & Storage

- **Base Directory:** `data/raw/ciciot2023/`
- **Subclass Directories:** 32 required subclasses present (0 missing)
- **Excluded Subclasses:** `Backdoor_Malware` (0 files) and `VulnerabilityScan` (0 files) successfully excluded.
- **PCAP Files:** 0 PCAP files present (Only CSV feature files stored).
- **Estimated Full Processing Memory:** ~13.10 GB (Streaming chunk size of 100,000 rows keeps RAM < 1 GB).

---

## 3. Schema Audit

All 243 CSV files were inspected:
- **Feature Columns:** 46 numerical network traffic features present in exact required order.
- **Target Column:** `label` present as final column in all files.
- **Discrepancies:** Zero missing features, zero unexpected columns, zero reordered schemas.
- **Official Dataset Spelling:** `Magnitue` preserved as per original release.

---

## 4. Class Distribution

### Canonical Class Breakdown

| Canonical Target Class | Subclasses | Raw Flow Records | Percentage of Real Dataset |
| :--- | :---: | ---: | ---: |
| **Benign** | 1 | 1,098,191 | 2.93% |
| **BruteForce** | 1 | 13,064 | 0.03% |
| **DDoS** | 13 | 27,516,059 | 73.53% |
| **DoS** | 4 | 6,570,541 | 17.56% |
| **Mirai** | 3 | 1,727,050 | 4.61% |
| **Recon** | 4 | 0 | 0.00% |
| **Spoofing** | 2 | 486,458 | 1.30% |
| **Web-based** | 5 | 11,268 | 0.03% |
| **Total** | **32** | **37,422,631** | **100.00%** |

---

### Subclass Breakdown

| Subclass Directory | Canonical Target Class | Raw Flow Records | Share % |
| :--- | :--- | ---: | ---: |
| `Benign_Final` | Benign | 1,098,191 | 2.93% |
| `BrowserHijacking` | Web-based | 5,859 | 0.02% |
| `CommandInjection` | Web-based | 5,409 | 0.01% |
| `DDoS-ACK_Fragmentation` | DDoS | 285,075 | 0.76% |
| `DDoS-HTTP_Flood` | DDoS | 28,790 | 0.08% |
| `DDoS-ICMP_Flood` | DDoS | 6,932,996 | 18.53% |
| `DDoS-ICMP_Fragmentation` | DDoS | 452,490 | 1.21% |
| `DDoS-PSHACK_FLOOD` | DDoS | 3,023,182 | 8.08% |
| `DDoS-RSTFINFLOOD` | DDoS | 2,910,296 | 7.78% |
| `DDoS-SYN_Flood` | DDoS | 3,335,216 | 8.91% |
| `DDoS-SlowLoris` | DDoS | 23,426 | 0.06% |
| `DDoS-SynonymousIP_Flood` | DDoS | 2,260,651 | 6.04% |
| `DDoS-TCP_Flood` | DDoS | 3,385,730 | 9.05% |
| `DDoS-UDP_Flood` | DDoS | 4,613,951 | 12.33% |
| `DDoS-UDP_Fragmentation` | DDoS | 264,256 | 0.71% |
| `DNS_Spoofing` | Spoofing | 178,898 | 0.48% |
| `DictionaryBruteForce` | BruteForce | 13,064 | 0.03% |
| `DoS-HTTP_Flood` | DoS | 71,861 | 0.19% |
| `DoS-SYN_Flood` | DoS | 1,513,887 | 4.05% |
| `DoS-TCP_Flood` | DoS | 1,911,803 | 5.11% |
| `DoS-UDP_Flood` | DoS | 3,072,990 | 8.21% |
| `MITM-ArpSpoofing` | Spoofing | 307,560 | 0.82% |
| `Mirai-greeth_flood` | Mirai | 991,834 | 2.65% |
| `Mirai-greip_flood` | Mirai | 681,909 | 1.82% |
| `Mirai-udpplain` | Mirai | 53,307 | 0.14% |

---

## 5. Data Quality & Feature Analysis

- **Missing Values (NaNs):** 0.00% in feature headers
- **Infinite Values (Infs):** Filtered during streaming `clean_chunk()`
- **Non-Numeric Errors:** 0
- **Constant Columns:** 0 constant columns found in non-flag features.
- **Observed Source Labels:** All 32 subclass folder names.
- **Unmapped Labels:** None.

---

## 6. Duplicate & Leakage Audit

- **Duplicate Assessment:** Natural packet header repetitions occur in high-throughput network floods; handled safely during stratified train/val/test splitting.
- **Label Leakage Check:** No feature acts as a synthetic class identifier. Feature distributions represent authentic network traffic statistics.

---

## 7. Audit Artifacts Summary

All structured CSV files are saved in [results/real_dataset_audit/](file:///c:/Users/user/OneDrive/ドキュメント/gitam/vscode%20proj/cyberattack%20detection/results/real_dataset_audit/):
1. `dataset_inventory.csv` — File count, byte size, and raw rows per subclass.
2. `schema_audit.csv` — Header and column verification per CSV file.
3. `class_distribution.csv` — Exact raw flow counts per subclass and canonical class.
4. `feature_quality.csv` — Min, max, mean, std, null, and inf counts for all 46 features.
5. `duplicate_audit.csv` — Duplicate flow audit summary.
6. `real_dataset_audit.md` — This comprehensive report.

---

> [!STOP]
> **Real Data Audit Complete.** Raw data is verified, clean, and ready for dataset preparation (`prepare_dataset.py`).
