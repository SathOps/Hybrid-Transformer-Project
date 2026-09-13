# CICIoT2023 Reconnaissance (Recon) Class Investigation Report

**Date**: September 11, 2026  
**Status**: COMPLETE — ACTION D DETERMINED (Recon data genuinely absent from downloaded raw release)  
**Report File**: `results/real_dataset_audit/recon_class_investigation.md`  

---

## Executive Summary

A targeted investigation was conducted to determine why the canonical **Recon** class contained **0 rows** in the processed dataset reconciliation report.

The investigation inspected the local raw dataset filesystem under `data/raw/ciciot2023/`, evaluated label mapping code in `src/preprocessing/class_mapping.py`, and verified streaming preprocessing logic in `src/preprocessing/prepare_dataset.py`.

### Key Findings
1. **Recon Folders & Files on Disk**: All 4 expected Recon subclass folders (`Recon-HostDiscovery`, `Recon-OSScan`, `Recon-PingSweep`, `Recon-PortScan`) are **absent** from `data/raw/ciciot2023/` (0 folders, 0 CSV files, 0 bytes, 0 rows).
2. **In-File Label Search**: Scanning all 243 downloaded CSV files across the 25 present subclass directories revealed **0** Recon records embedded inside any CSV files.
3. **Class-Mapping Logic**: `src/preprocessing/class_mapping.py` correctly defines `SOURCE_TO_TARGET` mappings for all 4 Recon subclasses to `"Recon"`. The mapping logic is 100% functional and bug-free.
4. **Preprocessing Rejection Check**: Zero rows were rejected by `clean_chunk()` for Recon because zero Recon rows existed in the raw dataset.
5. **Taxonomy Integrity Notice**: As per research directives, **Recon is retained in the official 8-class project taxonomy**, but documented as having 0 instances in the currently downloaded release partition.

---

## Detailed Investigation (Item-by-Item)

### 1. Number of CSV Files
* `Recon-HostDiscovery`: **0 files**
* `Recon-OSScan`: **0 files**
* `Recon-PingSweep`: **0 files**
* `Recon-PortScan`: **0 files**
* **Total**: **0 CSV files**

### 2. Total File Size
* `Recon-HostDiscovery`: **0 bytes**
* `Recon-OSScan`: **0 bytes**
* `Recon-PingSweep`: **0 bytes**
* `Recon-PortScan`: **0 bytes**
* **Total**: **0 bytes**

### 3. Number of Parsed Rows
* `Recon-HostDiscovery`: **0 rows**
* `Recon-OSScan`: **0 rows**
* `Recon-PingSweep`: **0 rows**
* `Recon-PortScan`: **0 rows**
* **Total**: **0 rows**

### 4. Actual Label Values Found in CSVs
* **N/A**: No CSV files exist for any of the 4 Recon subclass directories.
* **Raw Dataset Scan**: Across all 243 downloaded CSV files in `data/raw/ciciot2023/`, zero in-file labels correspond to Recon.

### 5. Expected 46 Features + Label Schema
* **N/A**: 0 CSV files exist under Recon subclass paths to validate schema against.

### 6. Row Rejection by `clean_chunk()`
* **No**: Zero rows were rejected during preprocessing. Zero Recon rows entered `clean_chunk()`.

### 7. Rejection Reason (If Applicable)
* **N/A**: No rows were rejected.

### 8. Class-Mapping Logic Recognition
* **YES**: `map_label()` in `src/preprocessing/class_mapping.py` explicitly recognizes all 4 Recon subclasses:
  - `map_label("Recon-HostDiscovery")` $\rightarrow$ `"Recon"`
  - `map_label("Recon-OSScan")` $\rightarrow$ `"Recon"`
  - `map_label("Recon-PingSweep")` $\rightarrow$ `"Recon"`
  - `map_label("Recon-PortScan")` $\rightarrow$ `"Recon"`

### 9. Genuinely Part of Downloaded Release
* **NO**: The 4 Recon subclass folders and files were not present in the downloaded raw CSV release under `data/raw/ciciot2023/`. The downloaded release consists of 243 CSV files across 25 subclass folders (Benign, BruteForce, DDoS, DoS, Mirai, Spoofing, Web-based).

---

## Action Determination

**Determined Action: Action D**
> **D. Recon data is genuinely absent from the downloaded release $\rightarrow$ document this clearly and STOP.**

### Notes on Action D
- No bugs exist in `prepare_dataset.py` or `class_mapping.py`.
- No rows were incorrectly filtered, dropped, or remapped.
- Recon remains defined as Class ID 5 in `CANONICAL_CLASSES` within `src/preprocessing/class_mapping.py`. It is NOT removed from the project taxonomy.

---

## Verification Log
- `pytest` test suite: **52 / 52 PASSED**
- Status: **INVESTIGATION COMPLETE — STOPPED**
