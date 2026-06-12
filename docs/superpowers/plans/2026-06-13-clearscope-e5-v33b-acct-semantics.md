# ClearScope E5 V33B Accounting Semantics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `raw_detail_v33b_e5_android_safe`, rerun the E5 audit smoke, compare
v31/v33b audit outputs, and stop at a go/no-go decision for later E5 pipeline
smoke planning.

**Architecture:** Extend the existing ClearScope Android semantic mode registry
and file-token helpers with a v33b mode that inherits v33 behavior except for
low-cardinality `/acct` accounting path shapes. Extend residual text and E5 audit
smoke routing to accept the new mode explicitly, then reuse the existing audit
comparison helper.

**Tech Stack:** Python 3, `unittest`, existing
`cs4m.semantics.clearscope_android` helpers,
`scripts/tools/audit_clearscope_e5_semantic_smoke.py`, local PostgreSQL for the
bounded audit rerun.

---

## File Structure

- Modify `cs4m/semantics/clearscope_android.py`
  - Add `CLEARSCOPE_V33B_E5_ANDROID_SAFE_SEMANTIC_MODE`.
  - Add aliases for `raw_detail_v33b_e5_android_safe`.
  - Add v33b accounting file class/detail helpers.
  - Add v33b residual text routing.
- Modify `scripts/pipeline/features/semantic_features.py`
  - Route explicit `raw_detail_v33b_e5_android_safe` to the v33b residual text
    helper.
- Modify `scripts/tools/audit_clearscope_e5_semantic_smoke.py`
  - Accept and tokenize `raw_detail_v33b_e5_android_safe`.
- Modify `tests/test_clearscope_e5_v33_semantics.py`
  - Add unit tests for v33b normalization, accounting rules, v33 preservation,
    residual text routing, and netflow preservation.
- Modify `tests/test_clearscope_e5_semantic_audit_smoke.py`
  - Add audit helper tests for v33b semantic mode.
- Generated only: `tmp/clearscope_e5_v33b_semantic_audit_smoke/`
  - Do not commit generated files.
- Generated only: `tmp/clearscope_e5_v33b_semantic_audit_comparison.json`
  - Do not commit generated files.

## Task 1: Register V33B Mode And Tokenizer Tests

**Files:**
- Modify: `cs4m/semantics/clearscope_android.py`
- Modify: `tests/test_clearscope_e5_v33_semantics.py`

- [ ] **Step 1: Write failing v33b tokenizer tests**

Add imports in `tests/test_clearscope_e5_v33_semantics.py`:

```python
    CLEARSCOPE_V33B_E5_ANDROID_SAFE_SEMANTIC_MODE,
    clearscope_file_natural_tokens_v33b_e5_android_safe,
```

Add tests:

```python
    def test_v33b_mode_aliases_normalize(self) -> None:
        self.assertEqual(
            normalize_clearscope_semantic_mode("raw_detail_v33b_e5_android_safe"),
            CLEARSCOPE_V33B_E5_ANDROID_SAFE_SEMANTIC_MODE,
        )
        self.assertEqual(
            normalize_clearscope_semantic_mode(
                "clearscope_raw_detail_v33b_e5_android_safe"
            ),
            CLEARSCOPE_V33B_E5_ANDROID_SAFE_SEMANTIC_MODE,
        )

    def test_v33b_accounting_paths_use_low_cardinality_tokens(self) -> None:
        expected = {
            "/acct": ("file", "android_accounting_root", "acct_root"),
            "/acct/uid_0": ("file", "android_accounting_uid", "acct_uid"),
            "/acct/uid_0/pid_549": (
                "file",
                "android_accounting_pid",
                "acct_pid",
            ),
            "/acct/uid_1000/pid_12345": (
                "file",
                "android_accounting_pid",
                "acct_pid",
            ),
            "/acct/uid_bad": ("file", "android_accounting_other", "acct_other"),
            "/acct/uid_0/other": (
                "file",
                "android_accounting_other",
                "acct_other",
            ),
        }
        for path, tokens in expected.items():
            with self.subTest(path=path):
                self.assertEqual(
                    clearscope_file_natural_tokens_v33b_e5_android_safe(path),
                    tokens,
                )

    def test_v33_does_not_route_accounting_paths(self) -> None:
        self.assertEqual(
            clearscope_file_natural_tokens_v33_e5_android_safe("/acct/uid_0/pid_549"),
            ("file", "file_other", "other"),
        )

    def test_v33b_preserves_v33_sdcardfs_and_device_behavior(self) -> None:
        self.assertEqual(
            clearscope_file_natural_tokens_v33b_e5_android_safe(
                "/config/sdcardfs/de.belu.appstarter/appid"
            ),
            ("file", "android_sdcardfs_appid", "de_belu_appstarter_appid"),
        )
        self.assertEqual(
            clearscope_file_natural_tokens_v33b_e5_android_safe("/dev/msm_g711tlaw"),
            ("file", "android_device_file", "dev_other"),
        )
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python3 -m unittest tests.test_clearscope_e5_v33_semantics
```

Expected: FAIL/ERROR because v33b symbols are missing.

- [ ] **Step 3: Implement v33b constants and file helpers**

In `cs4m/semantics/clearscope_android.py`, add constants near v33:

```python
CLEARSCOPE_V33B_E5_ANDROID_SAFE_SEMANTIC_MODE = "raw_detail_v33b_e5_android_safe"
CLEARSCOPE_V33B_E5_ANDROID_SAFE_SEMANTIC_ALIASES = {
    "raw_detail_v33b_e5_android_safe",
    "clearscope_raw_detail_v33b_e5_android_safe",
}
```

Update `normalize_clearscope_semantic_mode()` before the v33 check:

```python
    if text in CLEARSCOPE_V33B_E5_ANDROID_SAFE_SEMANTIC_ALIASES:
        return CLEARSCOPE_V33B_E5_ANDROID_SAFE_SEMANTIC_MODE
```

Add helpers near the v33 file detail helpers:

```python
def _accounting_detail(path: str) -> tuple[str, str] | None:
    if path == "/acct":
        return ("android_accounting_root", "acct_root")
    if re.fullmatch(r"/acct/uid_[0-9]+", path):
        return ("android_accounting_uid", "acct_uid")
    if re.fullmatch(r"/acct/uid_[0-9]+/pid_[0-9]+", path):
        return ("android_accounting_pid", "acct_pid")
    if path.startswith("/acct/"):
        return ("android_accounting_other", "acct_other")
    return None


def classify_android_file_nll_v33b_e5_android_safe(path: object) -> str:
    """Return the v33b E5-safe ClearScope Android file coarse label."""
    raw = str(path or "").strip()
    p = raw.lower()
    accounting = _accounting_detail(p)
    if accounting is not None:
        return accounting[0]
    return classify_android_file_nll_v33_e5_android_safe(path)


def android_file_detail_v33b_e5_android_safe(
    path: object,
    label: str | None = None,
) -> str:
    """Return the v33b E5-safe ClearScope Android file detail token."""
    raw = str(path or "").strip()
    p = raw.lower()
    accounting = _accounting_detail(p)
    if accounting is not None:
        return accounting[1]
    actual_label = label or classify_android_file_nll_v33b_e5_android_safe(p)
    return android_file_detail_v33_e5_android_safe(path, actual_label)


def clearscope_file_natural_tokens_v33b_e5_android_safe(path: object) -> tuple[str, ...]:
    """Return v33b E5-safe natural ClearScope file tokens."""
    label = classify_android_file_nll_v33b_e5_android_safe(path)
    return ("file", label, android_file_detail_v33b_e5_android_safe(path, label))
```

- [ ] **Step 4: Run Task 1 tests**

Run:

```bash
python3 -m unittest tests.test_clearscope_e5_v33_semantics
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add cs4m/semantics/clearscope_android.py tests/test_clearscope_e5_v33_semantics.py
git commit -m "feat: add clearscope e5 v33b accounting semantics"
```

## Task 2: Route V33B Residual Text And Audit Tokenization

**Files:**
- Modify: `cs4m/semantics/clearscope_android.py`
- Modify: `scripts/pipeline/features/semantic_features.py`
- Modify: `scripts/tools/audit_clearscope_e5_semantic_smoke.py`
- Modify: `tests/test_clearscope_e5_v33_semantics.py`
- Modify: `tests/test_clearscope_e5_semantic_audit_smoke.py`

- [ ] **Step 1: Write failing routing tests**

Add imports in `tests/test_clearscope_e5_v33_semantics.py`:

```python
    clearscope_residual_text_v33b_e5_android_safe,
```

Add tests:

```python
    def test_v33b_residual_text_uses_accounting_detail(self) -> None:
        row = {
            "action": "EVENT_READ",
            "src_kind": "process",
            "src_process_cmd": "com.android.providers.contacts",
            "dst_kind": "file",
            "dst_file_path": "/acct/uid_0/pid_549",
        }

        text = clearscope_residual_text_v33b_e5_android_safe(row)

        self.assertIn("android_accounting_pid", text.split())
        self.assertIn("acct_pid", text.split())
        self.assertNotIn("pid_549", text)

    def test_pipeline_residual_text_routes_v33b_mode(self) -> None:
        row = {
            "action": "EVENT_READ",
            "src_kind": "process",
            "src_process_cmd": "com.android.providers.contacts",
            "dst_kind": "file",
            "dst_file_path": "/acct/uid_0/pid_549",
        }

        text = residual_text(
            row,
            dataset="CLEARSCOPE_E5",
            semantic_mode="raw_detail_v33b_e5_android_safe",
        )

        self.assertIn("android_accounting_pid", text.split())
        self.assertIn("acct_pid", text.split())
```

Add tests in `tests/test_clearscope_e5_semantic_audit_smoke.py`:

```python
    def test_tokenize_file_row_v33b_uses_accounting_detail(self) -> None:
        from scripts.tools.audit_clearscope_e5_semantic_smoke import tokenize_node_row

        node = tokenize_node_row(
            {
                "index_id": 17,
                "node_uuid": "file-node-v33b",
                "node_type": "file",
                "path": "/acct/uid_0/pid_549",
            },
            semantic_mode="raw_detail_v33b_e5_android_safe",
        )

        self.assertEqual(
            node.semantic_tokens,
            ("file", "android_accounting_pid", "acct_pid"),
        )

    def test_tokenize_netflow_row_keeps_fixed_clear_scope_netflow_under_v33b(self) -> None:
        from scripts.tools.audit_clearscope_e5_semantic_smoke import tokenize_node_row

        node = tokenize_node_row(
            {
                "index_id": 18,
                "node_uuid": "netflow-node-v33b",
                "node_type": "netflow",
                "src_addr": "10.0.0.1",
                "src_port": "123",
                "dst_addr": "10.0.0.2",
                "dst_port": "443",
            },
            semantic_mode="raw_detail_v33b_e5_android_safe",
        )

        self.assertEqual(node.semantic_tokens, ("netflow",))
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
python3 -m unittest tests.test_clearscope_e5_v33_semantics \
    tests.test_clearscope_e5_semantic_audit_smoke
```

Expected: FAIL/ERROR because v33b residual and audit routing are missing.

- [ ] **Step 3: Implement v33b residual helpers**

In `cs4m/semantics/clearscope_android.py`, add:

```python
def clearscope_residual_tokens_v33b_e5_android_safe(
    row: dict[str, object],
) -> tuple[str, ...]:
    """Return v33b E5-safe ClearScope residual sentence tokens for one event row."""
    action = _refined_action_token(row.get("action", "unknown"))
    tokens = [
        *_clearscope_v33b_e5_android_safe_node_tokens(row, "src"),
        "event",
        action,
        *_clearscope_v33b_e5_android_safe_node_tokens(row, "dst"),
    ]
    return tuple(str(token) for token in tokens if str(token).strip())


def clearscope_residual_text_v33b_e5_android_safe(row: dict[str, object]) -> str:
    """Return v33b E5-safe ClearScope residual sentence text."""
    return " ".join(clearscope_residual_tokens_v33b_e5_android_safe(row))
```

Add node helper near v33 helper:

```python
def _clearscope_v33b_e5_android_safe_node_tokens(
    row: dict[str, object],
    side: str,
) -> tuple[str, ...]:
    kind = normalize_refined_token(row.get(f"{side}_kind", ""), max_len=30)
    if kind == "process":
        return android_process_natural_tokens_refined(row.get(f"{side}_process_cmd", ""))
    if kind == "file":
        return clearscope_file_natural_tokens_v33b_e5_android_safe(
            row.get(f"{side}_file_path", ""),
        )
    if kind == "netflow":
        return clearscope_netflow_natural_tokens_refined()
    return (kind or "unknown",)
```

- [ ] **Step 4: Route pipeline residual text**

In `scripts/pipeline/features/semantic_features.py`, import:

```python
    CLEARSCOPE_V33B_E5_ANDROID_SAFE_SEMANTIC_MODE,
    clearscope_residual_text_v33b_e5_android_safe,
```

Add before the v33 branch:

```python
        if clearscope_mode == CLEARSCOPE_V33B_E5_ANDROID_SAFE_SEMANTIC_MODE:
            return clearscope_residual_text_v33b_e5_android_safe(dict(row))
```

- [ ] **Step 5: Route audit tokenization**

In `scripts/tools/audit_clearscope_e5_semantic_smoke.py`, import:

```python
    CLEARSCOPE_V33B_E5_ANDROID_SAFE_SEMANTIC_MODE,
    clearscope_file_natural_tokens_v33b_e5_android_safe,
```

Add v33b to the allowed mode set and file-node branch:

```python
        CLEARSCOPE_V33B_E5_ANDROID_SAFE_SEMANTIC_MODE,
```

```python
        if normalized_mode == CLEARSCOPE_V33B_E5_ANDROID_SAFE_SEMANTIC_MODE:
            tokens = clearscope_file_natural_tokens_v33b_e5_android_safe(raw_detail)
        elif normalized_mode == CLEARSCOPE_V33_E5_ANDROID_SAFE_SEMANTIC_MODE:
            tokens = clearscope_file_natural_tokens_v33_e5_android_safe(raw_detail)
        else:
            tokens = clearscope_file_natural_tokens_v31(raw_detail)
```

- [ ] **Step 6: Run Task 2 tests**

Run:

```bash
python3 -m unittest tests.test_clearscope_e5_v33_semantics \
    tests.test_clearscope_e5_semantic_audit_smoke
```

Expected: PASS.

- [ ] **Step 7: Commit Task 2**

Run:

```bash
git add cs4m/semantics/clearscope_android.py \
    scripts/pipeline/features/semantic_features.py \
    scripts/tools/audit_clearscope_e5_semantic_smoke.py \
    tests/test_clearscope_e5_v33_semantics.py \
    tests/test_clearscope_e5_semantic_audit_smoke.py
git commit -m "feat: route clearscope e5 v33b semantics"
```

## Task 3: Run V33B Audit And Compare

**Files:**
- Generated: `tmp/clearscope_e5_v33b_semantic_audit_smoke/`
- Generated: `tmp/clearscope_e5_v33b_semantic_audit_comparison.json`

- [ ] **Step 1: Run v33b bounded audit**

Run:

```bash
CLAD_DB_HOST=127.0.0.1 \
CLAD_DB_PORT=5433 \
CLAD_DB_USER=postgres \
CLAD_DB_PASSWORD=123456 \
python3 scripts/tools/audit_clearscope_e5_semantic_smoke.py \
  --semantic_mode raw_detail_v33b_e5_android_safe \
  --max_nodes_per_type 20000 \
  --max_events_per_split 50000 \
  --output_dir tmp/clearscope_e5_v33b_semantic_audit_smoke
```

Expected: JSON paths for generated label-free, label-aware, fallback, and
collision reports.

- [ ] **Step 2: Compare v31 and v33b audits**

Run:

```bash
python3 scripts/tools/compare_clearscope_e5_semantic_audits.py \
  --v31_dir tmp/clearscope_e5_semantic_audit_smoke \
  --v33_dir tmp/clearscope_e5_v33b_semantic_audit_smoke \
  --output tmp/clearscope_e5_v33b_semantic_audit_comparison.json
```

Expected: comparison JSON is created.

- [ ] **Step 3: Print go/no-go metrics**

Run:

```bash
python3 - <<'PY'
import json

with open("tmp/clearscope_e5_v33b_semantic_audit_comparison.json", encoding="utf-8") as f:
    comparison = json.load(f)
with open("tmp/clearscope_e5_v33b_semantic_audit_smoke/label_free_summary.json", encoding="utf-8") as f:
    label_free = json.load(f)
with open("tmp/clearscope_e5_v33b_semantic_audit_smoke/label_aware_diagnostics.json", encoding="utf-8") as f:
    label_aware = json.load(f)

event_summary = label_free["event_tuple_summary"]
print("fallback_counts", label_free["fallback_counts"])
print("malicious_fallback_counts", label_aware["malicious_fallback_counts"])
print("skipped_event_count", event_summary["skipped_event_count"])
print("usable_event_rate", event_summary["usable_event_rate"])
print("test_oov_tuple_count", event_summary["test_oov_tuple_count"])
print("label_free_summary_has_label_fields", comparison["label_free_summary_has_label_fields"])
PY
```

Expected values must be evaluated against the spec go/no-go criteria.

## Task 4: Final Verification

**Files:**
- No code edits unless verification finds a bug.

- [ ] **Step 1: Run unit tests**

Run:

```bash
python3 -m unittest tests.test_clearscope_e5_v33_semantics \
    tests.test_clearscope_e5_semantic_audit_smoke \
    tests.test_compare_clearscope_e5_semantic_audits
```

Expected: PASS.

- [ ] **Step 2: Run compile check**

Run:

```bash
python3 -m compileall cs4m/semantics/clearscope_android.py \
    scripts/pipeline/features/semantic_features.py \
    scripts/tools/audit_clearscope_e5_semantic_smoke.py \
    scripts/tools/compare_clearscope_e5_semantic_audits.py \
    tests/test_clearscope_e5_v33_semantics.py \
    tests/test_clearscope_e5_semantic_audit_smoke.py \
    tests/test_compare_clearscope_e5_semantic_audits.py
```

Expected: PASS.

- [ ] **Step 3: Confirm generated audit artifacts exist**

Run:

```bash
test -f tmp/clearscope_e5_v33b_semantic_audit_smoke/label_free_summary.json
test -f tmp/clearscope_e5_v33b_semantic_audit_smoke/label_aware_diagnostics.json
test -f tmp/clearscope_e5_v33b_semantic_audit_comparison.json
```

Expected: exit code `0`.

- [ ] **Step 4: Check no E5 pipeline artifacts were created**

Run:

```bash
find outputs -maxdepth 5 \( -path '*CLEARSCOPE_E5*' -o -path '*clearscope_e5*' \) 2>/dev/null | sort | head -n 50
```

Expected: no new E5 pipeline smoke artifacts from this stage.

- [ ] **Step 5: Report decision**

Report:

- changed files;
- commits;
- commands run;
- v31/v33b comparison metrics;
- whether v33b passed strict go/no-go;
- remaining risks;
- preservation of streaming inference, no leakage, explainability, and
  reproducibility.

Do not launch E5 bounded pipeline smoke in this task.
