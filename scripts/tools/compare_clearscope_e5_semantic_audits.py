#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object from disk."""
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _int_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(key): int(raw_value) for key, raw_value in value.items()}


def _event_summary(summary: dict[str, Any]) -> dict[str, Any]:
    value = summary.get("event_tuple_summary", {})
    return value if isinstance(value, dict) else {}


def _metric_delta(v31: object, v33: object) -> dict[str, object]:
    try:
        old = float(v31)
        new = float(v33)
    except (TypeError, ValueError):
        return {"v31": v31, "v33": v33, "delta": None}
    return {"v31": v31, "v33": v33, "delta": new - old}


def compare_audit_dirs(v31_dir: Path, v33_dir: Path) -> dict[str, Any]:
    """Compare v31 and v33 ClearScope E5 semantic audit report directories."""
    v31_label_free = load_json(v31_dir / "label_free_summary.json")
    v33_label_free = load_json(v33_dir / "label_free_summary.json")
    v31_label_aware = load_json(v31_dir / "label_aware_diagnostics.json")
    v33_label_aware = load_json(v33_dir / "label_aware_diagnostics.json")

    v31_fallback = _int_mapping(v31_label_free.get("fallback_counts"))
    v33_fallback = _int_mapping(v33_label_free.get("fallback_counts"))
    v31_mal_fallback = _int_mapping(v31_label_aware.get("malicious_fallback_counts"))
    v33_mal_fallback = _int_mapping(v33_label_aware.get("malicious_fallback_counts"))
    v31_events = _event_summary(v31_label_free)
    v33_events = _event_summary(v33_label_free)

    fallback_tokens = sorted(set(v31_fallback) | set(v33_fallback))
    malicious_fallback_tokens = sorted(set(v31_mal_fallback) | set(v33_mal_fallback))
    metric_keys = [
        "input_event_count",
        "event_count",
        "skipped_event_count",
        "usable_event_rate",
        "train_tuple_count",
        "val_tuple_count",
        "val_oov_tuple_count",
        "test_tuple_count",
        "test_oov_tuple_count",
        "test_seen_tuple_count",
    ]

    label_free_keys = {"malicious", "attack", "label", "ground_truth"}
    label_free_key_text = {str(key).lower() for key in v33_label_free}

    return {
        "v31_dir": str(v31_dir),
        "v33_dir": str(v33_dir),
        "node_count": _metric_delta(
            v31_label_free.get("node_count"),
            v33_label_free.get("node_count"),
        ),
        "endpoint_node_count": _metric_delta(
            v31_label_free.get("endpoint_node_count"),
            v33_label_free.get("endpoint_node_count"),
        ),
        "fallback_counts": {
            token: _metric_delta(v31_fallback.get(token, 0), v33_fallback.get(token, 0))
            for token in fallback_tokens
        },
        "malicious_fallback_counts": {
            token: _metric_delta(
                v31_mal_fallback.get(token, 0),
                v33_mal_fallback.get(token, 0),
            )
            for token in malicious_fallback_tokens
        },
        "event_tuple_summary": {
            key: _metric_delta(v31_events.get(key), v33_events.get(key))
            for key in metric_keys
        },
        "malicious_collision_group_count": _metric_delta(
            len(v31_label_aware.get("malicious_detail_collision_groups", {}) or {}),
            len(v33_label_aware.get("malicious_detail_collision_groups", {}) or {}),
        ),
        "label_free_summary_has_label_fields": bool(label_free_keys & label_free_key_text),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v31_dir", required=True)
    parser.add_argument("--v33_dir", required=True)
    parser.add_argument("--output", default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the v31/v33 ClearScope E5 audit comparison."""
    args = parse_args(argv)
    comparison = compare_audit_dirs(Path(args.v31_dir), Path(args.v33_dir))
    text = json.dumps(comparison, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
