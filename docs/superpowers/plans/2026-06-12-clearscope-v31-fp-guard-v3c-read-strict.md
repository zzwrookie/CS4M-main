# ClearScope v31 FP Guard v3c Read-Strict Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `clearscope_v31_fp_guard_v3c`, a stricter file/process read guard that keeps v3b's online node-evidence mechanism but restores event TP by demoting fewer `EVENT_READ file->process` alerts.

**Architecture:** The policy remains inside `_action_type_alert_policy_decision`, following the existing v1/v2/v3/v3b pattern. Streaming context continues to be passed from `infer.py` through `policy_context`; v3c changes only the decision thresholds for additional file/process read demotion.

**Tech Stack:** Python, `unittest`, existing CS4M shell runner, existing ClearScope Phase3E/Phase3G pipeline.

---

## File Structure

- Modify `scripts/pipeline/config/runtime_config.py`
  - Register `clearscope_v31_fp_guard_v3c`.
- Modify `scripts/pipeline/entrypoints/arguments.py`
  - Include v3c in the human-facing validation error text if present there.
- Modify `scripts/pipeline/features/conditional_context.py`
  - Add v3c to allowed policy handling.
  - Keep v3b unchanged.
  - Implement stricter read additional gate for v3c.
- Modify `tests/test_clearscope_v31_fp_guard_policy.py`
  - Add red tests for v3c registry and strict read behavior.
- Use existing `scripts/pipeline/conditional/infer.py`
  - No planned behavioral change unless tests reveal v3c needs additional diagnostic fields already supported by policy payloads.

## Task 1: Add Failing v3c Policy Tests

**Files:**
- Modify: `tests/test_clearscope_v31_fp_guard_policy.py`

- [ ] **Step 1: Add v3c registry and strict read tests**

Add assertions and tests equivalent to:

```python
self.assertIn("clearscope_v31_fp_guard_v3c", ACTION_TYPE_ALERT_POLICIES)

def test_v3c_preserves_read_with_only_one_prior_alert(self) -> None:
    decision = _action_type_alert_policy_decision(
        config=SlimConfig(action_type_alert_policy="clearscope_v31_fp_guard_v3c"),
        row=_row(action_id=4, src_type_id=1, dst_type_id=0),
        raw_alert=True,
        score=0.061,
        threshold=0.051,
        alert_row={
            "validation_group_count": 2000,
            "validation_count_bucket": "sufficient",
            "target_case": "event_semantic_target",
        },
        policy_context={
            "src_alert_count": 0,
            "dst_alert_count": 1,
            "src_node_evidence_count": 0,
            "dst_node_evidence_count": 0,
        },
    )
    self.assertTrue(decision["final_alert"])
    self.assertFalse(decision["node_evidence"])

def test_v3c_demotes_read_with_two_prior_alerts_and_narrow_margin(self) -> None:
    decision = _action_type_alert_policy_decision(
        config=SlimConfig(action_type_alert_policy="clearscope_v31_fp_guard_v3c"),
        row=_row(action_id=4, src_type_id=1, dst_type_id=0),
        raw_alert=True,
        score=0.061,
        threshold=0.051,
        alert_row={
            "validation_group_count": 2000,
            "validation_count_bucket": "sufficient",
            "target_case": "event_semantic_target",
        },
        policy_context={
            "src_alert_count": 0,
            "dst_alert_count": 2,
            "src_node_evidence_count": 0,
            "dst_node_evidence_count": 0,
        },
    )
    self.assertFalse(decision["final_alert"])
    self.assertEqual(decision["alert_decision"], "demoted_event")
    self.assertTrue(decision["node_evidence"])

def test_v3c_preserves_read_above_strict_margin(self) -> None:
    decision = _action_type_alert_policy_decision(
        config=SlimConfig(action_type_alert_policy="clearscope_v31_fp_guard_v3c"),
        row=_row(action_id=4, src_type_id=1, dst_type_id=0),
        raw_alert=True,
        score=0.062,
        threshold=0.051,
        alert_row={
            "validation_group_count": 2000,
            "validation_count_bucket": "sufficient",
            "target_case": "event_semantic_target",
        },
        policy_context={
            "src_alert_count": 0,
            "dst_alert_count": 2,
            "src_node_evidence_count": 0,
            "dst_node_evidence_count": 0,
        },
    )
    self.assertTrue(decision["final_alert"])
    self.assertFalse(decision["node_evidence"])

def test_v3c_demotes_read_with_four_prior_evidence_events(self) -> None:
    decision = _action_type_alert_policy_decision(
        config=SlimConfig(action_type_alert_policy="clearscope_v31_fp_guard_v3c"),
        row=_row(action_id=4, src_type_id=1, dst_type_id=0),
        raw_alert=True,
        score=0.061,
        threshold=0.051,
        alert_row={
            "validation_group_count": 2000,
            "validation_count_bucket": "sufficient",
            "target_case": "event_semantic_target",
        },
        policy_context={
            "src_alert_count": 0,
            "dst_alert_count": 0,
            "src_node_evidence_count": 0,
            "dst_node_evidence_count": 4,
        },
    )
    self.assertFalse(decision["final_alert"])
    self.assertTrue(decision["node_evidence"])

def test_v3c_preserves_low_support_read_even_with_prior_support(self) -> None:
    decision = _action_type_alert_policy_decision(
        config=SlimConfig(action_type_alert_policy="clearscope_v31_fp_guard_v3c"),
        row=_row(action_id=4, src_type_id=1, dst_type_id=0),
        raw_alert=True,
        score=0.061,
        threshold=0.051,
        alert_row={
            "validation_group_count": 2000,
            "validation_count_bucket": "low",
            "target_case": "event_semantic_target",
        },
        policy_context={
            "src_alert_count": 0,
            "dst_alert_count": 2,
            "src_node_evidence_count": 0,
            "dst_node_evidence_count": 4,
        },
    )
    self.assertTrue(decision["final_alert"])
    self.assertFalse(decision["node_evidence"])
```

- [ ] **Step 2: Run tests and verify red**

Run:

```bash
python3 -m unittest tests.test_clearscope_v31_fp_guard_policy
```

Expected: fail because `clearscope_v31_fp_guard_v3c` is not registered or recognized.

## Task 2: Implement v3c Policy

**Files:**
- Modify: `scripts/pipeline/config/runtime_config.py`
- Modify: `scripts/pipeline/entrypoints/arguments.py`
- Modify: `scripts/pipeline/features/conditional_context.py`

- [ ] **Step 1: Register v3c**

Add `clearscope_v31_fp_guard_v3c` next to v3b in `ACTION_TYPE_ALERT_POLICIES`.

- [ ] **Step 2: Update CLI validation text**

If `arguments.py` lists supported policy names in an error string, add
`clearscope_v31_fp_guard_v3c` to that string.

- [ ] **Step 3: Implement v3c in `_action_type_alert_policy_decision`**

Extend v3b policy handling without changing v3b behavior:

```python
if policy == "clearscope_v31_fp_guard_v3c" and alert_row is not None:
    margin_by_group = {
        (4, 1, 0): 0.0015,
        (9, 0, 1): 0.002,
    }
    support_by_group = {
        (4, 1, 0): (2, 4),
        (9, 0, 1): (1, 2),
    }
```

The read group uses `(prior_alert >= 2 or prior_evidence >= 4)`. The write group
uses the existing v3b thresholds `(prior_alert >= 1 or prior_evidence >= 2)`.

- [ ] **Step 4: Preserve label-free guards**

Keep the existing guards:

```python
validation_group_count >= 1000
validation_count_bucket not in {"low", "low_support"}
target_case not in {"both_cold", "cold", "low_support"}
score < floor + margin
```

- [ ] **Step 5: Add diagnostic reason strings**

Return `policy_support_reason` as:

```text
v3c_read_strict_prior_alert
v3c_read_strict_prior_evidence
v3b_node_evidence_prior_alert
v3b_node_evidence_prior_evidence
```

Use v3c strings only for the read strict rule.

- [ ] **Step 6: Run focused tests and verify green**

Run:

```bash
python3 -m unittest tests.test_clearscope_v31_fp_guard_policy
```

Expected: all tests pass.

## Task 3: Repository Checks

**Files:**
- No new code beyond Task 2.

- [ ] **Step 1: Run compile check**

Run:

```bash
python3 -m compileall configs cs4m scripts
```

Expected: command exits 0.

- [ ] **Step 2: Run shell syntax check**

Run:

```bash
bash -n scripts/run/*.sh
```

Expected: command exits 0.

## Task 4: v3c Smoke Inference

**Files:**
- Generated outputs under `outputs/results/tflr_light`.

- [ ] **Step 1: Run 100k smoke**

Run:

```bash
CLAD_DB_PASSWORD=123456 timeout 30m env \
    STAGE=infer_full \
    SEMANTIC_MODE=raw_detail_v31_discriminative \
    ACTION_TYPE_ALERT_POLICY=clearscope_v31_fp_guard_v3c \
    NODE_POOL_SCORE_MODE=base_conf_v31_support \
    EVENT_THRESHOLD_MODE=conditional_target_action_type_group_quantile \
    EVENT_THRESHOLD_QUANTILE=0.999 \
    CONDITIONAL_BOTH_COLD_UNSEEN_POLICY=observation_only_no_alert \
    MAX_TRAIN_EVENTS=200000 \
    MAX_REF_EVENTS=50000 \
    MAX_TEST_EVENTS=100000 \
    OUT_TAG_OVERRIDE=CLEARSCOPE_E3_RAW_DETAIL_V31_DISCRIMINATIVE_SMOKE_GROUPQ0999_FP_GUARD_V3C_READ_STRICT \
    bash scripts/run/run_clearscope_e3_v3_phase3e_phase3g_full.sh
```

Expected: command exits 0 and writes smoke output directory.

- [ ] **Step 2: Verify smoke config**

Run:

```bash
rg -n "action_type_alert_policy|node_pool_score_mode|event_threshold_mode|event_threshold_quantile|max_test_events|semantic_mode" \
    outputs/results/tflr_light/CLEARSCOPE_E3_RAW_DETAIL_V31_DISCRIMINATIVE_SMOKE_GROUPQ0999_FP_GUARD_V3C_READ_STRICT/config_effective.yaml
```

Expected: v3c policy, v31 semantic mode, group quantile, q=0.999, and `max_test_events: 100000`.

- [ ] **Step 3: Verify smoke alerts are label-free**

Run:

```bash
python3 - <<'PY'
import csv
from pathlib import Path
p = Path("outputs/results/tflr_light/CLEARSCOPE_E3_RAW_DETAIL_V31_DISCRIMINATIVE_SMOKE_GROUPQ0999_FP_GUARD_V3C_READ_STRICT/online_event_alerts.csv")
with p.open(newline="") as f:
    header = next(csv.reader(f))
print("columns", len(header))
print("label_or_malicious_columns", [c for c in header if "label" in c.lower() or "malicious" in c.lower()])
PY
```

Expected: `label_or_malicious_columns []`.

## Task 5: Launch Full v3c Inference

**Files:**
- Create/update: `logs/clearscope_v31_fp_guard_v3c_read_strict_full.nohup.log`
- Create/update: `logs/clearscope_v31_fp_guard_v3c_read_strict_full.pid`
- Generated outputs under `outputs/results/tflr_light`.

- [ ] **Step 1: Check no duplicate v3c full job is running**

Run:

```bash
pgrep -af "CLEARSCOPE_E3_RAW_DETAIL_V31_DISCRIMINATIVE_FULL_GROUPQ0999_FP_GUARD_V3C_READ_STRICT|clearscope_v31_fp_guard_v3c"
```

Expected: no existing v3c full process, or only irrelevant completed commands.

- [ ] **Step 2: Launch full with nohup**

Run:

```bash
CLAD_DB_PASSWORD=123456 nohup timeout 30m env \
    STAGE=infer_full \
    SEMANTIC_MODE=raw_detail_v31_discriminative \
    ACTION_TYPE_ALERT_POLICY=clearscope_v31_fp_guard_v3c \
    NODE_POOL_SCORE_MODE=base_conf_v31_support \
    EVENT_THRESHOLD_MODE=conditional_target_action_type_group_quantile \
    EVENT_THRESHOLD_QUANTILE=0.999 \
    CONDITIONAL_BOTH_COLD_UNSEEN_POLICY=observation_only_no_alert \
    MAX_TRAIN_EVENTS=0 \
    MAX_REF_EVENTS=0 \
    MAX_TEST_EVENTS=0 \
    OUT_TAG_OVERRIDE=CLEARSCOPE_E3_RAW_DETAIL_V31_DISCRIMINATIVE_FULL_GROUPQ0999_FP_GUARD_V3C_READ_STRICT \
    bash scripts/run/run_clearscope_e3_v3_phase3e_phase3g_full.sh \
    > logs/clearscope_v31_fp_guard_v3c_read_strict_full.nohup.log 2>&1 &
echo $! > logs/clearscope_v31_fp_guard_v3c_read_strict_full.pid
cat logs/clearscope_v31_fp_guard_v3c_read_strict_full.pid
```

Expected: prints a PID.

- [ ] **Step 3: Confirm initial log progress**

Run:

```bash
sleep 2
tail -n 30 logs/clearscope_v31_fp_guard_v3c_read_strict_full.nohup.log
```

Expected: log shows `START infer_full` or validation/reference test progress.

## Self-Review Checklist

- Spec coverage: v3c policy name, strict read rule, v3b write inheritance, diagnostics,
  smoke, and full launch are covered.
- Placeholder scan: no TBD/TODO placeholders.
- Type consistency: policy context keys match current v3b implementation.
- Safety: no runtime use of labels, malicious nodes, attack windows, or test-derived floors.
