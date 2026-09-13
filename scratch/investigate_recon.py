import os
import glob
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Fix stdout encoding for Windows cp1252
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(r"c:\Users\user\OneDrive\ドキュメント\gitam\vscode proj\cyberattack detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.class_mapping import SOURCE_TO_TARGET, UNRESOLVED_SOURCE_LABELS, map_label, ALLOWED_TARGET_CLASSES
from src.preprocessing.prepare_dataset import clean_chunk

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ciciot2023"

recon_subclasses = [
    "Recon-HostDiscovery",
    "Recon-OSScan",
    "Recon-PingSweep",
    "Recon-PortScan",
]

print("=== RECON CLASS DETAILED INVESTIGATION ===")

# 1, 2, 3. Folder Check
print("\n--- 1, 2, 3. Folder, CSV Count, File Size, Parsed Rows ---")
folder_details = {}
for sub in recon_subclasses:
    sub_path = RAW_DIR / sub
    exists = sub_path.exists()
    if exists:
        csvs = list(sub_path.glob("*.csv"))
        size_bytes = sum(f.stat().st_size for f in csvs)
        parsed_rows = 0
        for f in csvs:
            parsed_rows += len(pd.read_csv(f))
        folder_details[sub] = {
            "exists": True,
            "csv_count": len(csvs),
            "size_bytes": size_bytes,
            "parsed_rows": parsed_rows,
            "csv_files": csvs
        }
    else:
        folder_details[sub] = {
            "exists": False,
            "csv_count": 0,
            "size_bytes": 0,
            "parsed_rows": 0,
            "csv_files": []
        }

for sub, info in folder_details.items():
    print(f"Folder '{sub}':")
    print(f"  Exists on disk: {info['exists']}")
    print(f"  CSV files: {info['csv_count']}")
    print(f"  Total size (bytes): {info['size_bytes']}")
    print(f"  Parsed rows: {info['parsed_rows']}")

# 4 & 5. Check if any Recon labels exist anywhere in all 243 raw CSV files
print("\n--- 4 & 5. Label Search Across All 243 Raw CSV Files ---")
all_csvs = list(RAW_DIR.glob("**/*.csv"))
print(f"Total CSV files found under raw directory: {len(all_csvs)}")

unique_labels_found = set()
files_with_recon_labels = []

for f in all_csvs:
    df_head = pd.read_csv(f, nrows=10)
    has_label_col = "label" in df_head.columns or "Label" in df_head.columns
    if has_label_col:
        col_name = "label" if "label" in df_head.columns else "Label"
        df_full = pd.read_csv(f, usecols=[col_name])
        labels = set(df_full[col_name].unique())
        unique_labels_found.update(labels)
        for sub in recon_subclasses:
            if sub in labels:
                files_with_recon_labels.append((f.name, sub))

print(f"Total unique in-file labels across all CSVs: {len(unique_labels_found)}")
print(f"Recon in-file labels found: {[l for l in unique_labels_found if 'recon' in str(l).lower()]}")

# 8. Check Class Mapping Logic
print("\n--- 8. Class Mapping Logic Check ---")
for sub in recon_subclasses:
    mapped = map_label(sub)
    mapped_dict = SOURCE_TO_TARGET.get(sub)
    print(f"Subclass '{sub}': map_label() -> {mapped!r}, SOURCE_TO_TARGET.get() -> {mapped_dict!r}")

print("\nAll Recon entries in SOURCE_TO_TARGET:")
for k, v in SOURCE_TO_TARGET.items():
    if v == "Recon" or "recon" in k.lower():
        print(f"  - '{k}': '{v}'")

print("\nInvestigation script completed.")
