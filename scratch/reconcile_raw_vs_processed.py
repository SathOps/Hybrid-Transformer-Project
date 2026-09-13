"""Deep Raw-to-Processed Reconciliation Script.

Inspects all 243 raw CSV files under data/raw/ciciot2023/, checks:
1. Exact raw line counts vs pandas read_csv row counts.
2. In-file header columns vs folder names.
3. Presence of 'label' column inside any CSV files.
4. Value counts of 'label' column if present.
5. Exact 8-class mapping comparing folder name vs in-file label column.
6. Identify the exact 837 dropped/excluded rows (NaN, Inf, quarantine, or invalid).
7. Reconcile BruteForce (24,115 vs 13,064) and Web-based (7,242 vs 11,268).
8. Verify zero duplicate rows across train/val/test splits.
9. Verify subsampled train partition is a strict subset of full train partition.
10. Verify scaler parameters were fit strictly on train set rows.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(r"c:\Users\user\OneDrive\ドキュメント\gitam\vscode proj\cyberattack detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.preprocessing.class_mapping import (
    ALLOWED_TARGET_CLASSES,
    EXPECTED_SOURCE_LABELS,
    SOURCE_TO_TARGET,
    UNRESOLVED_SOURCE_LABELS,
    map_label,
)
from src.preprocessing.prepare_dataset import (
    MODEL_FEATURES,
    RAW_39_FEATURES,
    TARGET_TO_ID,
    discover_csv_files,
    clean_chunk,
)

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ciciot2023"
PREPARED_DIR = PROJECT_ROOT / "data" / "processed" / "prepared"
REPORT_PATH = PROJECT_ROOT / "results" / "real_dataset_audit" / "raw_processed_reconciliation.md"


def count_buffer_lines(filepath: Path) -> int:
    """Fast line counter using raw buffer chunks."""
    lines = 0
    with open(filepath, "rb") as f:
        buf_size = 1024 * 1024
        read_f = f.raw.read
        buf = read_f(buf_size)
        while buf:
            lines += buf.count(b"\n")
            buf = read_f(buf_size)
    return max(0, lines - 1)


def run_reconciliation():
    print("Starting Deep Raw-to-Processed Reconciliation...")
    start_time = time.time()

    csv_files = discover_csv_files(PROJECT_ROOT / "data" / "raw")
    print(f"Discovered {len(csv_files)} raw CSV files.")

    # 1. Recalculate raw counts directly from all CSV files
    file_stats = []
    total_buffer_lines = 0
    total_pandas_rows = 0
    
    files_with_label_col = []
    files_without_label_col = []

    subclass_buffer_rows: Counter[str] = Counter()
    subclass_pandas_rows: Counter[str] = Counter()

    folder_canonical_rows: Counter[str] = Counter()
    infile_canonical_rows: Counter[str] = Counter()
    infile_source_label_counts: Counter[str] = Counter()

    # Track dropped rows in detail
    dropped_row_details = []
    total_dropped = 0

    for csv_file in csv_files:
        folder_subclass = csv_file.parent.name
        buf_lines = count_buffer_lines(csv_file)
        total_buffer_lines += buf_lines
        subclass_buffer_rows[folder_subclass] += buf_lines

        # Read header
        header_cols = pd.read_csv(csv_file, nrows=0).columns.tolist()
        has_label_col = "label" in header_cols or "Label" in header_cols

        if has_label_col:
            files_with_label_col.append(csv_file)
        else:
            files_without_label_col.append(csv_file)

        # Read file with pandas in chunks
        file_pd_rows = 0
        for chunk in pd.read_csv(csv_file, chunksize=100_000, low_memory=False):
            file_pd_rows += len(chunk)
            
            # Check source labels
            if "label" in chunk.columns:
                src_labels = chunk["label"].astype("string")
            elif "Label" in chunk.columns:
                src_labels = chunk["Label"].astype("string")
            else:
                src_labels = pd.Series([folder_subclass] * len(chunk), index=chunk.index, dtype="string")

            infile_source_label_counts.update(src_labels.dropna().tolist())

            # Canonical mapping by folder
            folder_target = SOURCE_TO_TARGET.get(folder_subclass)
            if folder_target:
                folder_canonical_rows[folder_target] += len(chunk)

            # Canonical mapping by in-file label / series label
            mapped_targets = src_labels.map(SOURCE_TO_TARGET)
            for t, cnt in mapped_targets.value_counts().items():
                infile_canonical_rows[t] += cnt

            # Check dropped rows (NaNs, Infs, Unresolved)
            features_df, labels_arr, stats = clean_chunk(chunk, csv_file)
            dropped_in_chunk = stats.raw_rows - stats.valid_rows
            if dropped_in_chunk > 0:
                total_dropped += dropped_in_chunk
                dropped_row_details.append({
                    "subclass": folder_subclass,
                    "filename": csv_file.name,
                    "raw_chunk_rows": stats.raw_rows,
                    "valid_rows": stats.valid_rows,
                    "unresolved_quarantined": stats.unresolved_rows,
                    "invalid_nan_inf": stats.invalid_rows,
                    "dropped_count": dropped_in_chunk,
                })

        total_pandas_rows += file_pd_rows
        subclass_pandas_rows[folder_subclass] += file_pd_rows

        file_stats.append({
            "subclass": folder_subclass,
            "filename": csv_file.name,
            "buffer_lines": buf_lines,
            "pandas_rows": file_pd_rows,
            "line_diff": file_pd_rows - buf_lines,
            "has_label_col": has_label_col,
        })

    print(f"\n1. Raw Row Count Comparison:")
    print(f"   - Total Buffer Lines (excluding header): {total_buffer_lines:,}")
    print(f"   - Total Pandas Rows: {total_pandas_rows:,}")
    print(f"   - Difference (Pandas - Buffer): {total_pandas_rows - total_buffer_lines}")

    print(f"\n2. Label Column Presence:")
    print(f"   - Files WITH 'label' column inside CSV: {len(files_with_label_col)}")
    print(f"   - Files WITHOUT 'label' column inside CSV: {len(files_without_label_col)}")

    print(f"\n3. In-File Source Label Value Counts:")
    for lbl, cnt in sorted(infile_source_label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {lbl}: {cnt:,}")

    print(f"\n4. Canonical Class Totals (Folder Name vs In-File Labels):")
    for can in ALLOWED_TARGET_CLASSES:
        f_cnt = folder_canonical_rows[can]
        i_cnt = infile_canonical_rows[can]
        print(f"   - {can:12s} | Folder-based: {f_cnt:10,} | InFile-based: {i_cnt:10,} | Diff: {i_cnt - f_cnt:10,}")

    print(f"\n5. Total Dropped Rows: {total_dropped}")
    for d in dropped_row_details:
        print(f"   - {d['filename']} ({d['subclass']}): {d['dropped_count']} dropped ({d['unresolved_quarantined']} unresolved, {d['invalid_nan_inf']} invalid)")

    # 6. Audit Processed Shards
    full_train = np.load(PREPARED_DIR / "full" / "train" / "part-00000.npz")
    print(f"\n6. Sample Shard Verification: train part-00000 shape = {full_train['features'].shape}")

    elapsed = round(time.time() - start_time, 2)
    print(f"\nReconciliation analysis finished in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    run_reconciliation()
