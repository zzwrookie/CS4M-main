# ClearScope E5 V33 Android-Safe Semantics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `raw_detail_v33_e5_android_safe`, rerun the E5 audit smoke, compare
v31/v33 audit outputs, and stop at a go/no-go decision for later E5 pipeline smoke
planning.

**Architecture:** Extend the existing ClearScope Android semantic mode registry and
file-token helpers with a v33 mode that inherits v31 behavior except for
`/config/sdcardfs` package shapes and a small stable `/dev/*` allowlist. Extend the
E5 audit smoke to accept the new mode and add a comparison report helper that reads
the existing v31/v33 generated audit JSON/CSV files.

**Tech Stack:** Python 3, `unittest`, existing `cs4m.semantics.clearscope_android`
helpers, `scripts/tools/audit_clearscope_e5_semantic_smoke.py`, local PostgreSQL
for the bounded audit rerun.

---

## File Structure

- Modify `cs4m/semantics/clearscope_android.py`
  - Add `CLEARSCOPE_V33_E5_ANDROID_SAFE_SEMANTIC_MODE`.
  - Add aliases for `raw_detail_v33_e5_android_safe`.
  - Add v33 file detail/token helpers.
  - Add v33 residual text routing.
- Modify `scripts/pipeline/features/semantic_features.py`
  - Route `raw_detail_v33_e5_android_safe` to the v33 residual text helper.
- Create `tests/test_clearscope_e5_v33_semantics.py`
  - Unit tests for v33 normalization, sdcardfs rules, device allowlist, unknown
    device preservation, and v31 non-target behavior.
- Modify `scripts/tools/audit_clearscope_e5_semantic_smoke.py`
  - Accept and tokenize `raw_detail_v33_e5_android_safe`.
- Modify `tests/test_clearscope_e5_semantic_audit_smoke.py`
  - Add audit helper test for v33 semantic mode.
- Create `scripts/tools/compare_clearscope_e5_semantic_audits.py`
  - Compare generated v31/v33 audit outputs and print a JSON go/no-go summary.
- Create `tests/test_compare_clearscope_e5_semantic_audits.py`
  - Unit tests for decision logic using temporary audit fixtures.
- Generated only: `tmp/clearscope_e5_v33_semantic_audit_smoke/`
  - Do not commit generated files.

## Task 1: Register V33 Mode And Tokenizer Tests

**Files:**
- Modify: `cs4m/semantics/clearscope_android.py`
- Create: `tests/test_clearscope_e5_v33_semantics.py`

- [ ] **Step 1: Write failing v33 tokenizer tests**

Create `tests/test_clearscope_e5_v33_semantics.py`:

```python
"""Tests for ClearScope E5 v33 Android-safe semantics."""

from __future__ import annotations

import unittest

from cs4m.semantics.clearscope_android import (
    CLEARSCOPE_V33_E5_ANDROID_SAFE_SEMANTIC_MODE,
    android_file_detail_v33_e5_android_safe,
    clearscope_file_natural_tokens_v31,
    clearscope_file_natural_tokens_v33_e5_android_safe,
    normalize_clearscope_semantic_mode,
)


class ClearScopeE5V33SemanticsTests(unittest.TestCase):
    """Validate v33 E5 Android-safe ClearScope token behavior."""

    def test_v33_mode_aliases_normalize(self) -> None:
        self.assertEqual(
            normalize_clearscope_semantic_mode("raw_detail_v33_e5_android_safe"),
            CLEARSCOPE_V33_E5_ANDROID_SAFE_SEMANTIC_MODE,
        )
        self.assertEqual(
            normalize_clearscope_semantic_mode(
                "clearscope_raw_detail_v33_e5_android_safe"
            ),
            CLEARSCOPE_V33_E5_ANDROID_SAFE_SEMANTIC_MODE,
        )

    def test_v33_sdcardfs_appid_keeps_general_package_family(self) -> None:
        self.assertEqual(
            clearscope_file_natural_tokens_v33_e5_android_safe(
                "/config/sdcardfs/com.android.providers.contacts/appid"
            ),
            (
                "file",
                "android_sdcardfs_appid",
                "com_android_providers_contacts_appid",
            ),
        )
        self.assertEqual(
            clearscope_file_natural_tokens_v33_e5_android_safe(
                "/config/sdcardfs/de.belu.appstarter/appid"
            ),
            ("file", "android_sdcardfs_appid", "de_belu_appstarter_appid"),
        )
        self.assertEqual(
            clearscope_file_natural_tokens_v33_e5_android_safe(
                "/config/sdcardfs/com.bloketech.lockwatch/appid"
            ),
            ("file", "android_sdcardfs_appid", "com_bloketech_lockwatch_appid"),
        )

    def test_v33_sdcardfs_package_directory_keeps_general_package_family(self) -> None:
        self.assertEqual(
            clearscope_file_natural_tokens_v33_e5_android_safe(
                "/config/sdcardfs/de.belu.appstarter"
            ),
            ("file", "android_sdcardfs_package", "de_belu_appstarter"),
        )

    def test_v33_device_allowlist(self) -> None:
        expected = {
            "/dev/pmsg0": "dev_pmsg0",
            "/dev/ashmem": "dev_ashmem",
            "/dev/ion": "dev_ion",
            "/dev/binder": "dev_binder",
            "/dev/hwbinder": "dev_hwbinder",
            "/dev/vndbinder": "dev_vndbinder",
            "/dev/null": "dev_null",
            "/dev/zero": "dev_zero",
        }
        for path, detail in expected.items():
            with self.subTest(path=path):
                self.assertEqual(
                    android_file_detail_v33_e5_android_safe(
                        path,
                        "android_device_file",
                    ),
                    detail,
                )

    def test_v33_unknown_device_stays_dev_other(self) -> None:
        self.assertEqual(
            android_file_detail_v33_e5_android_safe(
                "/dev/msm_g711tlaw",
                "android_device_file",
            ),
            "dev_other",
        )

    def test_v31_targeted_path_is_unchanged(self) -> None:
        self.assertEqual(
            clearscope_file_natural_tokens_v31(
                "/config/sdcardfs/de.belu.appstarter/appid"
            ),
            ("file", "file_other", "other"),
        )

    def test_v33_non_target_path_matches_v31(self) -> None:
        path = "/data/data/org.mozilla.fennec_vagrant/cache/x/cache2/doomed/12345"
        self.assertEqual(
            clearscope_file_natural_tokens_v33_e5_android_safe(path),
            clearscope_file_natural_tokens_v31(path),
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python3 -m unittest tests.test_clearscope_e5_v33_semantics
```

Expected: FAIL/ERROR because v33 symbols are missing.

- [ ] **Step 3: Implement v33 constants and file helper functions**

In `cs4m/semantics/clearscope_android.py`, add near existing constants:

```python
CLEARSCOPE_V33_E5_ANDROID_SAFE_SEMANTIC_MODE = "raw_detail_v33_e5_android_safe"
CLEARSCOPE_V33_E5_ANDROID_SAFE_SEMANTIC_ALIASES = {
    "raw_detail_v33_e5_android_safe",
    "clearscope_raw_detail_v33_e5_android_safe",
}
```

Update `normalize_clearscope_semantic_mode()` before v32 checks:

```python
    if text in CLEARSCOPE_V33_E5_ANDROID_SAFE_SEMANTIC_ALIASES:
        return CLEARSCOPE_V33_E5_ANDROID_SAFE_SEMANTIC_MODE
```

Add helper functions near v31/v32 file detail helpers:

```python
def _sdcardfs_package(path: str) -> str:
    match = re.fullmatch(r"/config/sdcardfs/([^/]+)(?:/appid)?", path)
    return match.group(1) if match else ""


def _sdcardfs_detail(path: str, suffix: str = "") -> str:
    package = _sdcardfs_package(path)
    family = _refined_package_family_from_package(package)
    return normalize_refined_token("_".join(part for part in (family, suffix) if part))


def classify_android_file_nll_v33_e5_android_safe(path: object) -> str:
    """Return the v33 E5-safe ClearScope Android file coarse label."""
    raw = str(path or "").strip()
    p = raw.lower()
    if re.fullmatch(r"/config/sdcardfs/[^/]+/appid", p):
        return "android_sdcardfs_appid"
    if re.fullmatch(r"/config/sdcardfs/[^/]+", p):
        return "android_sdcardfs_package"
    return classify_android_file_nll(path)


def _device_detail_v33_e5_android_safe(path: str) -> str:
    basename = normalize_refined_token(path.rstrip("/").rsplit("/", 1)[-1], max_len=40)
    if basename in {
        "pmsg0",
        "ashmem",
        "ion",
        "binder",
        "hwbinder",
        "vndbinder",
        "null",
        "zero",
    }:
        return f"dev_{basename}"
    return _device_detail_refined(path)


def android_file_detail_v33_e5_android_safe(
    path: object,
    label: str | None = None,
) -> str:
    """Return the v33 E5-safe ClearScope Android file detail token."""
    raw = str(path or "").strip()
    p = raw.lower()
    actual_label = label or classify_android_file_nll_v33_e5_android_safe(p)
    if actual_label == "android_sdcardfs_appid":
        return _sdcardfs_detail(p, "appid")
    if actual_label == "android_sdcardfs_package":
        return _sdcardfs_detail(p)
    if actual_label == "android_device_file":
        return _device_detail_v33_e5_android_safe(p)
    return android_file_detail_v31(path, actual_label)


def clearscope_file_natural_tokens_v33_e5_android_safe(path: object) -> tuple[str, ...]:
    """Return v33 E5-safe natural ClearScope file tokens."""
    label = classify_android_file_nll_v33_e5_android_safe(path)
    return ("file", label, android_file_detail_v33_e5_android_safe(path, label))
```

- [ ] **Step 4: Run Task 1 tests**

Run:

```bash
python3 -m unittest tests.test_clearscope_e5_v33_semantics
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add cs4m/semantics/clearscope_android.py \
    tests/test_clearscope_e5_v33_semantics.py
git commit -m "feat: add clearscope e5 v33 tokenizer tests"
```

## Task 2: Route V33 Residual Text And Audit Tokenization

**Files:**
- Modify: `cs4m/semantics/clearscope_android.py`
- Modify: `scripts/pipeline/features/semantic_features.py`
- Modify: `scripts/tools/audit_clearscope_e5_semantic_smoke.py`
- Modify: `tests/test_clearscope_e5_semantic_audit_smoke.py`
- Modify: `tests/test_clearscope_e5_v33_semantics.py`

- [ ] **Step 1: Add failing residual and audit-mode tests**

Append to `tests/test_clearscope_e5_v33_semantics.py`:

```python
    def test_v33_residual_text_routes_file_tokens(self) -> None:
        from cs4m.semantics.clearscope_android import clearscope_residual_text_v33_e5_android_safe

        row = {
            "action": "EVENT_READ",
            "src_kind": "process",
            "src_process_cmd": "system_server",
            "dst_kind": "file",
            "dst_file_path": "/config/sdcardfs/de.belu.appstarter/appid",
        }

        text = clearscope_residual_text_v33_e5_android_safe(row)

        self.assertIn("android_sdcardfs_appid", text)
        self.assertIn("de_belu_appstarter_appid", text)

    def test_residual_text_entrypoint_routes_v33(self) -> None:
        from scripts.pipeline.features.semantic_features import residual_text

        row = {
            "action": "EVENT_READ",
            "src_kind": "process",
            "src_process_cmd": "system_server",
            "dst_kind": "file",
            "dst_file_path": "/config/sdcardfs/de.belu.appstarter/appid",
        }

        text = residual_text(
            row,
            dataset="CLEARSCOPE_E5",
            semantic_mode="raw_detail_v33_e5_android_safe",
        )

        self.assertIn("android_sdcardfs_appid", text)
        self.assertIn("de_belu_appstarter_appid", text)
```

Append to `tests/test_clearscope_e5_semantic_audit_smoke.py`:

```python
    def test_tokenize_file_row_accepts_v33_semantics(self) -> None:
        from scripts.tools.audit_clearscope_e5_semantic_smoke import tokenize_node_row

        node = tokenize_node_row(
            {
                "index_id": 33,
                "node_uuid": "sdcardfs-node",
                "node_type": "file",
                "path": "/config/sdcardfs/de.belu.appstarter/appid",
            },
            semantic_mode="raw_detail_v33_e5_android_safe",
        )

        self.assertEqual(
            node.semantic_tokens,
            ("file", "android_sdcardfs_appid", "de_belu_appstarter_appid"),
        )
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python3 -m unittest tests.test_clearscope_e5_v33_semantics \
    tests.test_clearscope_e5_semantic_audit_smoke
```

Expected: FAIL because v33 residual/audit routing is incomplete.

- [ ] **Step 3: Implement v33 residual routing**

In `cs4m/semantics/clearscope_android.py`, add:

```python
def _clearscope_v33_e5_android_safe_node_tokens(
    row: dict[str, object],
    side: str,
) -> tuple[str, ...]:
    kind = normalize_refined_token(row.get(f"{side}_kind", ""), max_len=30)
    if kind == "process":
        return android_process_natural_tokens_refined(row.get(f"{side}_process_cmd", ""))
    if kind == "file":
        return clearscope_file_natural_tokens_v33_e5_android_safe(
            row.get(f"{side}_file_path", "")
        )
    if kind == "netflow":
        return clearscope_netflow_natural_tokens_refined()
    return (kind or "unknown",)


def clearscope_residual_tokens_v33_e5_android_safe(
    row: dict[str, object],
) -> tuple[str, ...]:
    """Return v33 E5-safe ClearScope residual tokens."""
    return (
        "event",
        normalize_refined_token(row.get("action", "unknown"), max_len=40),
        *_clearscope_v33_e5_android_safe_node_tokens(row, "src"),
        *_clearscope_v33_e5_android_safe_node_tokens(row, "dst"),
    )


def clearscope_residual_text_v33_e5_android_safe(row: dict[str, object]) -> str:
    """Return v33 E5-safe ClearScope residual text."""
    return " ".join(clearscope_residual_tokens_v33_e5_android_safe(row))
```

In `scripts/pipeline/features/semantic_features.py`, import the new constant and
function, then route before v32:

```python
        if clearscope_mode == CLEARSCOPE_V33_E5_ANDROID_SAFE_SEMANTIC_MODE:
            return clearscope_residual_text_v33_e5_android_safe(dict(row))
```

- [ ] **Step 4: Update audit tokenization to accept v33**

In `scripts/tools/audit_clearscope_e5_semantic_smoke.py`, import:

```python
CLEARSCOPE_V33_E5_ANDROID_SAFE_SEMANTIC_MODE
clearscope_file_natural_tokens_v33_e5_android_safe
```

Update `tokenize_node_row()`:

```python
    if normalized_mode not in {
        CLEARSCOPE_V31_SEMANTIC_MODE,
        CLEARSCOPE_V33_E5_ANDROID_SAFE_SEMANTIC_MODE,
    }:
        raise ValueError(f"unsupported E5 audit semantic mode: {semantic_mode}")
```

For file rows:

```python
        if normalized_mode == CLEARSCOPE_V33_E5_ANDROID_SAFE_SEMANTIC_MODE:
            tokens = clearscope_file_natural_tokens_v33_e5_android_safe(raw_detail)
        else:
            tokens = clearscope_file_natural_tokens_v31(raw_detail)
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
python3 -m unittest tests.test_clearscope_e5_v33_semantics \
    tests.test_clearscope_e5_semantic_audit_smoke
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add cs4m/semantics/clearscope_android.py \
    scripts/pipeline/features/semantic_features.py \
    scripts/tools/audit_clearscope_e5_semantic_smoke.py \
    tests/test_clearscope_e5_semantic_audit_smoke.py \
    tests/test_clearscope_e5_v33_semantics.py
git commit -m "feat: route clearscope e5 v33 semantics"
```

## Task 3: Rerun E5 V33 Audit Smoke And Compare Audits

**Files:**
- Create: `scripts/tools/compare_clearscope_e5_semantic_audits.py`
- Create: `tests/test_compare_clearscope_e5_semantic_audits.py`
- Generated only: `tmp/clearscope_e5_v33_semantic_audit_smoke/`

- [ ] **Step 1: Write failing comparison tests**

Create `tests/test_compare_clearscope_e5_semantic_audits.py`:

```python
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.tools.compare_clearscope_e5_semantic_audits import (
    compare_audits,
    decide_go_no_go,
)


def _write_audit(root: Path, file_other: int, malicious_file_other: int, test_oov: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "label_free_summary.json").write_text(
        json.dumps(
            {
                "fallback_counts": {"file_other": file_other, "dev_other": 44, "netflow": 20000},
                "event_tuple_summary": {
                    "skipped_event_count": 0,
                    "usable_event_rate": 1.0,
                    "test_oov_tuple_count": test_oov,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "label_aware_diagnostics.json").write_text(
        json.dumps(
            {
                "warning": "label_aware_diagnostic_only_not_runtime_policy",
                "malicious_fallback_counts": {
                    "file_other": malicious_file_other,
                    "dev_other": 1,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "collision_groups.csv").write_text(
        "token,raw_detail_count,examples\nother,1,[]\n",
        encoding="utf-8",
    )


class ClearScopeE5AuditCompareTests(unittest.TestCase):
    def test_decision_passes_when_thresholds_improve(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "v31"
            candidate = Path(tmpdir) / "v33"
            _write_audit(base, file_other=1066, malicious_file_other=3, test_oov=39)
            _write_audit(candidate, file_other=700, malicious_file_other=0, test_oov=40)

            summary = compare_audits(base, candidate)

            self.assertEqual(
                decide_go_no_go(summary),
                "v33_audit_pass_pipeline_smoke_can_be_planned",
            )

    def test_decision_blocks_when_malicious_file_other_remains(self) -> None:
        with TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "v31"
            candidate = Path(tmpdir) / "v33"
            _write_audit(base, file_other=1066, malicious_file_other=3, test_oov=39)
            _write_audit(candidate, file_other=700, malicious_file_other=1, test_oov=40)

            summary = compare_audits(base, candidate)

            self.assertEqual(
                decide_go_no_go(summary),
                "v33_needs_more_general_tokenizer_adjustment",
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run comparison tests and verify failure**

Run:

```bash
python3 -m unittest tests.test_compare_clearscope_e5_semantic_audits
```

Expected: FAIL/ERROR because comparison tool is missing.

- [ ] **Step 3: Implement comparison tool**

Create `scripts/tools/compare_clearscope_e5_semantic_audits.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _audit_metrics(root: Path) -> dict[str, Any]:
    label_free = _load_json(root / "label_free_summary.json")
    label_aware = _load_json(root / "label_aware_diagnostics.json")
    events = label_free.get("event_tuple_summary", {})
    fallback = label_free.get("fallback_counts", {})
    malicious_fallback = label_aware.get("malicious_fallback_counts", {})
    return {
        "fallback_counts": fallback,
        "malicious_fallback_counts": malicious_fallback,
        "skipped_event_count": int(events.get("skipped_event_count", 0)),
        "usable_event_rate": float(events.get("usable_event_rate", 0.0)),
        "test_oov_tuple_count": int(events.get("test_oov_tuple_count", 0)),
        "has_label_keys": "malicious" in label_free or "attack" in label_free,
    }


def compare_audits(v31_dir: str | Path, v33_dir: str | Path) -> dict[str, Any]:
    """Compare v31 and v33 ClearScope E5 semantic audit outputs."""
    v31 = _audit_metrics(Path(v31_dir))
    v33 = _audit_metrics(Path(v33_dir))
    return {
        "v31": v31,
        "v33": v33,
        "deltas": {
            "file_other": (
                int(v33["fallback_counts"].get("file_other", 0))
                - int(v31["fallback_counts"].get("file_other", 0))
            ),
            "malicious_file_other": (
                int(v33["malicious_fallback_counts"].get("file_other", 0))
                - int(v31["malicious_fallback_counts"].get("file_other", 0))
            ),
            "test_oov_tuple_count": (
                int(v33["test_oov_tuple_count"]) - int(v31["test_oov_tuple_count"])
            ),
        },
    }


def decide_go_no_go(summary: dict[str, Any]) -> str:
    """Return the v33 audit go/no-go decision."""
    v31 = summary["v31"]
    v33 = summary["v33"]
    v31_test_oov = int(v31["test_oov_tuple_count"])
    v33_test_oov = int(v33["test_oov_tuple_count"])
    if int(v33["skipped_event_count"]) != 0:
        return "v33_blocks_e5_pipeline_smoke"
    if float(v33["usable_event_rate"]) < 0.95:
        return "v33_blocks_e5_pipeline_smoke"
    if bool(v33["has_label_keys"]):
        return "v33_blocks_e5_pipeline_smoke"
    if int(v33["malicious_fallback_counts"].get("dev_other", 0)) > int(
        v31["malicious_fallback_counts"].get("dev_other", 0)
    ):
        return "v33_needs_more_general_tokenizer_adjustment"
    if int(v33["malicious_fallback_counts"].get("file_other", 0)) != 0:
        return "v33_needs_more_general_tokenizer_adjustment"
    if int(v33["fallback_counts"].get("file_other", 0)) >= 800:
        return "v33_needs_more_general_tokenizer_adjustment"
    if v33_test_oov > max(v31_test_oov + 5, int(v31_test_oov * 1.25)):
        return "v33_needs_more_general_tokenizer_adjustment"
    return "v33_audit_pass_pipeline_smoke_can_be_planned"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v31_dir", default="tmp/clearscope_e5_semantic_audit_smoke")
    parser.add_argument(
        "--v33_dir",
        default="tmp/clearscope_e5_v33_semantic_audit_smoke",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = compare_audits(args.v31_dir, args.v33_dir)
    summary["decision"] = decide_go_no_go(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run comparison tests**

Run:

```bash
python3 -m unittest tests.test_compare_clearscope_e5_semantic_audits
```

Expected: tests pass.

- [ ] **Step 5: Run v33 E5 audit smoke**

Run:

```bash
CLAD_DB_HOST=127.0.0.1 \
CLAD_DB_PORT=5433 \
CLAD_DB_USER=postgres \
CLAD_DB_PASSWORD=123456 \
python3 scripts/tools/audit_clearscope_e5_semantic_smoke.py \
  --semantic_mode raw_detail_v33_e5_android_safe \
  --max_nodes_per_type 20000 \
  --max_events_per_split 50000 \
  --output_dir tmp/clearscope_e5_v33_semantic_audit_smoke
```

Expected: exits 0 and writes the four audit files under
`tmp/clearscope_e5_v33_semantic_audit_smoke/`.

- [ ] **Step 6: Compare v31 and v33 audits**

Run:

```bash
python3 scripts/tools/compare_clearscope_e5_semantic_audits.py \
  --v31_dir tmp/clearscope_e5_semantic_audit_smoke \
  --v33_dir tmp/clearscope_e5_v33_semantic_audit_smoke
```

Expected: prints JSON with a `decision` field equal to one of:

```text
v33_audit_pass_pipeline_smoke_can_be_planned
v33_needs_more_general_tokenizer_adjustment
v33_blocks_e5_pipeline_smoke
```

- [ ] **Step 7: Commit Task 3**

Run:

```bash
git add scripts/tools/compare_clearscope_e5_semantic_audits.py \
    tests/test_compare_clearscope_e5_semantic_audits.py
git commit -m "feat: compare clearscope e5 v33 audit"
```

Do not commit files under `tmp/`.

## Task 4: Final Verification And Decision Report

**Files:**
- Read: `tmp/clearscope_e5_semantic_audit_smoke/label_free_summary.json`
- Read: `tmp/clearscope_e5_semantic_audit_smoke/label_aware_diagnostics.json`
- Read: `tmp/clearscope_e5_v33_semantic_audit_smoke/label_free_summary.json`
- Read: `tmp/clearscope_e5_v33_semantic_audit_smoke/label_aware_diagnostics.json`
- Read: comparison tool JSON output

- [ ] **Step 1: Run full focused verification**

Run:

```bash
python3 -m unittest tests.test_clearscope_e5_v33_semantics \
    tests.test_clearscope_e5_semantic_audit_smoke \
    tests.test_compare_clearscope_e5_semantic_audits
python3 -m compileall cs4m/semantics/clearscope_android.py \
    scripts/pipeline/features/semantic_features.py \
    scripts/tools/audit_clearscope_e5_semantic_smoke.py \
    scripts/tools/compare_clearscope_e5_semantic_audits.py \
    tests/test_clearscope_e5_v33_semantics.py \
    tests/test_clearscope_e5_semantic_audit_smoke.py \
    tests/test_compare_clearscope_e5_semantic_audits.py
```

Expected: all tests pass and compileall exits 0.

- [ ] **Step 2: Verify generated v33 audit files exist**

Run:

```bash
test -f tmp/clearscope_e5_v33_semantic_audit_smoke/label_free_summary.json
test -f tmp/clearscope_e5_v33_semantic_audit_smoke/label_aware_diagnostics.json
test -f tmp/clearscope_e5_v33_semantic_audit_smoke/fallback_counts.csv
test -f tmp/clearscope_e5_v33_semantic_audit_smoke/collision_groups.csv
```

Expected: exit 0.

- [ ] **Step 3: Confirm no pipeline smoke or full inference ran**

Run:

```bash
find outputs -maxdepth 5 \( -path '*CLEARSCOPE_E5*' -o -path '*clearscope_e5*' \) \
    2>/dev/null | sort | head -n 50
```

Expected: no new E5 model/result artifacts from this stage. Generated audit output
under `tmp/` is allowed.

- [ ] **Step 4: Report final go/no-go decision**

Final response must include:

- changed files;
- commits created;
- commands run;
- v31/v33 audit comparison values:
  - skipped event count;
  - usable event rate;
  - label-free fallback counts;
  - malicious fallback counts;
  - test OOV tuple count;
- final decision;
- remaining risks;
- preservation statement for streaming inference, no leakage, explainability, and
  reproducibility.

Do not launch E5 pipeline smoke, build E5 artifacts, or modify runners in this
task.
