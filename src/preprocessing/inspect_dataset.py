"""Inspect CICIoT2023 CSV files without loading the complete dataset into memory."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


COMMON_LABEL_NAMES = (
    "label",
    "target",
    "class",
    "attack",
    "attack_type",
    "category",
)


def discover_csv_files(raw_dir: Path) -> list[Path]:
    """Return CSV files recursively, in a stable order."""
    return sorted(path for path in raw_dir.rglob("*.csv") if path.is_file())


def read_headers(csv_files: Iterable[Path]) -> tuple[dict[str, list[str]], list[str]]:
    """Read only CSV headers and return per-file headers plus their union."""
    headers_by_file: dict[str, list[str]] = {}
    all_columns: list[str] = []
    seen_columns: set[str] = set()

    for csv_file in csv_files:
        header = pd.read_csv(csv_file, nrows=0).columns.tolist()
        headers_by_file[str(csv_file)] = header
        for column in header:
            if column not in seen_columns:
                all_columns.append(column)
                seen_columns.add(column)

    return headers_by_file, all_columns


def determine_label_column(
    columns: list[str], configured_label: str | None
) -> tuple[str, str]:
    """Determine the label column and explain how it was selected."""
    if configured_label:
        if configured_label not in columns:
            raise ValueError(
                f"Configured label column {configured_label!r} was not found. "
                f"Available columns: {columns}"
            )
        return configured_label, "configured via --label-column"

    columns_by_lower_name = {column.lower(): column for column in columns}
    for candidate in COMMON_LABEL_NAMES:
        if candidate in columns_by_lower_name:
            return (
                columns_by_lower_name[candidate],
                f"matched common label name {candidate!r}",
            )

    if not columns:
        raise ValueError("No columns were found in the CSV headers.")
    return columns[-1], "fallback to the final column; confirm against the dataset schema"


def json_safe(value: Any) -> Any:
    """Convert pandas and NumPy values into JSON-compatible values."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def inspect_dataset(
    raw_dir: Path,
    output_path: Path,
    chunksize: int,
    sample_rows: int,
    label_column: str | None = None,
) -> dict[str, Any]:
    """Inspect CSV files in chunks and save the resulting report."""
    if chunksize < 1:
        raise ValueError("chunksize must be a positive integer")
    if sample_rows < 0:
        raise ValueError("sample_rows cannot be negative")

    csv_files = discover_csv_files(raw_dir)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {raw_dir}")

    headers_by_file, columns = read_headers(csv_files)
    resolved_label, label_method = determine_label_column(columns, label_column)
    feature_columns = [column for column in columns if column != resolved_label]

    dtype_values: dict[str, set[str]] = defaultdict(set)
    missing_values: Counter[str] = Counter()
    infinite_values: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    unique_labels: set[str] = set()
    sample_data: list[dict[str, Any]] = []
    total_rows = 0
    files_with_errors: list[dict[str, str]] = []

    for csv_file in csv_files:
        try:
            header = headers_by_file[str(csv_file)]
            if resolved_label not in header:
                raise ValueError(f"missing label column {resolved_label!r}")

            for chunk in pd.read_csv(csv_file, chunksize=chunksize, low_memory=False):
                total_rows += len(chunk)

                for column, dtype in chunk.dtypes.items():
                    dtype_values[column].add(str(dtype))

                missing_values.update(
                    {column: int(count) for column, count in chunk.isna().sum().items()}
                )

                for column in chunk.columns:
                    numeric_values = pd.to_numeric(chunk[column], errors="coerce")
                    infinite_values[column] += int(np.isinf(numeric_values).sum())

                labels = chunk[resolved_label]
                label_strings = labels.map(
                    lambda value: "<NA>" if pd.isna(value) else str(value)
                )
                unique_labels.update(label_strings.unique().tolist())
                class_counts.update(label_strings.tolist())

                if len(sample_data) < sample_rows:
                    remaining = sample_rows - len(sample_data)
                    sample_data.extend(
                        json_safe(record)
                        for record in chunk.head(remaining).to_dict(orient="records")
                    )
        except Exception as error:
            files_with_errors.append({"file": str(csv_file), "error": str(error)})

    report: dict[str, Any] = {
        "raw_directory": str(raw_dir),
        "number_of_csv_files": len(csv_files),
        "files": [str(csv_file) for csv_file in csv_files],
        "columns": columns,
        "columns_by_file": headers_by_file,
        "label_column": resolved_label,
        "label_column_detection": label_method,
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "data_types": {
            column: sorted(dtype_values.get(column, set())) for column in columns
        },
        "missing_values": {
            column: missing_values.get(column, 0) for column in columns
        },
        "infinite_values": {
            column: infinite_values.get(column, 0) for column in columns
        },
        "unique_labels": sorted(unique_labels),
        "class_counts": dict(sorted(class_counts.items())),
        "rows_scanned": total_rows,
        "sample_rows": sample_data,
        "files_with_errors": files_with_errors,
        "inspection_parameters": {
            "chunksize": chunksize,
            "sample_rows": sample_rows,
            "normalization_applied": False,
            "dataset_modified": False,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect CICIoT2023 CSV files using memory-efficient chunks."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw CSV files (default: data/raw).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/dataset_inspection.json"),
        help="Inspection report path (default: experiments/dataset_inspection.json).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Rows per CSV read chunk (default: 100000).",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=10,
        help="Number of sample rows to include (default: 10).",
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help="Explicit label column; otherwise common names or the final column are used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = inspect_dataset(
        raw_dir=args.raw_dir,
        output_path=args.output,
        chunksize=args.chunksize,
        sample_rows=args.sample_rows,
        label_column=args.label_column,
    )
    print(f"Inspected {report['number_of_csv_files']} CSV file(s).")
    print(f"Columns: {len(report['columns'])}; features: {report['feature_count']}")
    print(f"Label column: {report['label_column']}")
    print(f"Rows scanned: {report['rows_scanned']}")
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()