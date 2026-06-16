#!/usr/bin/env python3
"""Prepare CADETS sidecars for unified group threshold replay."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


VALIDATION_SOURCE = "conditional_score_summary_by_target_action_type.csv"
VALIDATION_OUTPUT = "validation_group_threshold_summary.csv"
TRAIN_SIDECAR = "train_group_score_summary.csv"
MISSING_OUTPUT = "missing_unified_sidecars.json"


def prepare_cadets_sidecars(result_dir: Path) -> dict[str, Any]:
    """Prepare replay sidecars available from the CADETS result directory."""
    result_path = Path(result_dir)
    validation_rows = _load_validation_rows(result_path / VALIDATION_SOURCE)
    _write_validation_summary(result_path / VALIDATION_OUTPUT, validation_rows)

    missing: list[str] = []
    if not (result_path / TRAIN_SIDECAR).exists():
        missing.append(TRAIN_SIDECAR)
    missing_payload = {
        "missing_required_sidecars": missing,
        "notes": [
            "train_group_score_summary.csv cannot be reconstructed from test score trace",
            "generate it from train/calibration phase before final train max fallback replay",
        ],
    }
    (result_path / MISSING_OUTPUT).write_text(
        json.dumps(missing_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "result_dir": str(result_path),
        "validation_group_rows": len(validation_rows),
        "validation_group_threshold_summary": str(result_path / VALIDATION_OUTPUT),
        "train_group_score_summary_exists": (result_path / TRAIN_SIDECAR).exists(),
        "missing_sidecar_report": str(result_path / MISSING_OUTPUT),
    }
    (result_path / "cadets_unified_sidecar_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _load_validation_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "target_case": str(row.get("target_case", "")),
                    "action": str(row.get("action_name", "")),
                    "src_type": str(row.get("src_type_name", "")),
                    "dst_type": str(row.get("dst_type_name", "")),
                    "validation_count": str(row.get("validation_count", "0")),
                    "threshold": str(row.get("val_p9995") or row.get("threshold") or "0"),
                    "threshold_quantile": "0.9995",
                    "threshold_source": "validation_group_quantile",
                }
            )
    return rows


def _write_validation_summary(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "target_case",
        "action",
        "src_type",
        "dst_type",
        "validation_count",
        "threshold",
        "threshold_quantile",
        "threshold_source",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> int:
    """Prepare CADETS sidecars from an existing result directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()
    summary = prepare_cadets_sidecars(Path(args.result_dir))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
