# ClearScope E3 v3 FP Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add safe ClearScope E3 v3 Phase3G policy controls and run gated full inference comparisons to reduce false positives without label leakage.

**Architecture:** Keep the existing v3 baseline reproducible by preserving runner defaults. Add environment-variable routing in the ClearScope v3 runner, a default-off Phase3G `both_cold` unseen-group no-alert gate in conditional inference, and report-only experiment stages that reuse existing v3 artifacts before any retraining or semantic changes.

**Tech Stack:** Bash runner scripts, Python dataclass config, argparse CLI, Phase3G conditional inference, unittest, compileall, PostgreSQL-backed CS4M pipeline, existing `clad` Python environment.

---

## File Structure

- Modify `scripts/run/run_clearscope_e3_v3_phase3e_phase3g_full.sh`
  - Expose ClearScope v3 policy knobs through environment variables.
  - Support `OUT_TAG_OVERRIDE` for single-stage `infer_full`.
  - Preserve baseline defaults.
- Modify `scripts/pipeline/config/runtime_config.py`
  - Add `conditional_both_cold_unseen_policy` with default `alert`.
- Modify `scripts/pipeline/entrypoints/arguments.py`
  - Add CLI argument and validation for `--conditional_both_cold_unseen_policy`.
- Modify `scripts/pipeline/conditional/infer.py`
  - Apply default-off `both_cold` unseen-group no-alert gate before endpoint suppression and action policy.
  - Record gate counts and policy state in the returned score summary payload.
- Create `tests/test_clearscope_v3_runner_policy.py`
  - Validate dry-run command routing and default preservation.
- Create `tests/test_phase3g_both_cold_unseen_policy.py`
  - Validate config parsing and the gate predicate behavior through a small helper.
- Create `tmp/clearscope_v3_fp_policy_report.py`
  - Summarize baseline and new run metrics after full inference.

## Task 1: Runner Policy Controls

**Files:**
- Modify: `scripts/run/run_clearscope_e3_v3_phase3e_phase3g_full.sh`
- Test: `tests/test_clearscope_v3_runner_policy.py`

- [ ] **Step 1: Write runner dry-run tests**

Create `tests/test_clearscope_v3_runner_policy.py`:

```python
"""Tests for ClearScope v3 Phase3E/Phase3G runner policy routing."""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts/run/run_clearscope_e3_v3_phase3e_phase3g_full.sh"


class ClearScopeV3RunnerPolicyTests(unittest.TestCase):
    """Validate ClearScope v3 runner dry-run command construction."""

    def _run_dry(self, **overrides: str) -> str:
        env = os.environ.copy()
        env.update(
            {
                "DRY_RUN": "1",
                "STAGE": "infer_full",
                "CLAD_DB_PASSWORD": "dummy",
            },
        )
        env.update(overrides)
        result = subprocess.run(
            ["bash", str(RUNNER)],
            cwd=REPO_ROOT,
            env=env,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout + result.stderr

    def test_runner_preserves_baseline_defaults(self) -> None:
        output = self._run_dry()
        self.assertIn("--out_tag CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_INFER_FULL", output)
        self.assertIn("--event_threshold_mode quantile", output)
        self.assertIn("--action_type_alert_policy default", output)
        self.assertIn("--conditional_endpoint_aware_suppression false", output)
        self.assertIn("--conditional_endpoint_suppression_read false", output)
        self.assertIn("--sspm_state_model s4d_complex_node", output)
        self.assertIn("--sspm_conditional_head_arch shared_lowrank_v1", output)

    def test_runner_routes_policy_overrides_to_cli(self) -> None:
        output = self._run_dry(
            OUT_TAG_OVERRIDE="CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_INFER_GROUPQ0999",
            EVENT_THRESHOLD_MODE="conditional_target_action_type_group_quantile",
            ACTION_TYPE_ALERT_POLICY="clearscope_android_v2",
            CONDITIONAL_ENDPOINT_AWARE_SUPPRESSION="true",
            CONDITIONAL_ENDPOINT_SUPPRESSION_READ="true",
            CONDITIONAL_LOW_SUPPORT_POLICY="adaptive_margin",
            CONDITIONAL_LOW_SUPPORT_MARGIN="0.07",
            SSPM_STATE_MODEL="ema_fixed",
            SSPM_CONDITIONAL_HEAD_ARCH="dual_lowrank_by_target_case_v2",
            CONDITIONAL_BOTH_COLD_UNSEEN_POLICY="observation_only_no_alert",
        )
        self.assertIn("--out_tag CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_INFER_GROUPQ0999", output)
        self.assertIn("--event_threshold_mode conditional_target_action_type_group_quantile", output)
        self.assertIn("--action_type_alert_policy clearscope_android_v2", output)
        self.assertIn("--conditional_endpoint_aware_suppression true", output)
        self.assertIn("--conditional_endpoint_suppression_read true", output)
        self.assertIn("--conditional_low_support_policy adaptive_margin", output)
        self.assertIn("--conditional_low_support_margin 0.07", output)
        self.assertIn("--sspm_state_model ema_fixed", output)
        self.assertIn("--sspm_conditional_head_arch dual_lowrank_by_target_case_v2", output)
        self.assertIn(
            "--conditional_both_cold_unseen_policy observation_only_no_alert",
            output,
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the new runner tests and confirm they fail**

Run:

```bash
python3 -m unittest tests.test_clearscope_v3_runner_policy
```

Expected: FAIL because `OUT_TAG_OVERRIDE` and several CLI flags are not yet routed.

- [ ] **Step 3: Add environment variables to the runner**

In `scripts/run/run_clearscope_e3_v3_phase3e_phase3g_full.sh`, add near the existing threshold variables:

```bash
ACTION_TYPE_ALERT_POLICY="${ACTION_TYPE_ALERT_POLICY:-default}"
CONDITIONAL_LOW_SUPPORT_POLICY="${CONDITIONAL_LOW_SUPPORT_POLICY:-conservative_max}"
CONDITIONAL_LOW_SUPPORT_MARGIN="${CONDITIONAL_LOW_SUPPORT_MARGIN:-0.05}"
CONDITIONAL_ENDPOINT_AWARE_SUPPRESSION="${CONDITIONAL_ENDPOINT_AWARE_SUPPRESSION:-false}"
CONDITIONAL_ENDPOINT_SUPPRESSION_READ="${CONDITIONAL_ENDPOINT_SUPPRESSION_READ:-false}"
CONDITIONAL_ENDPOINT_SUPPRESSION_MODE="${CONDITIONAL_ENDPOINT_SUPPRESSION_MODE:-pair_only}"
CONDITIONAL_BOTH_COLD_UNSEEN_POLICY="${CONDITIONAL_BOTH_COLD_UNSEEN_POLICY:-alert}"
SSPM_STATE_MODEL="${SSPM_STATE_MODEL:-s4d_complex_node}"
SSPM_CONDITIONAL_HEAD_ARCH="${SSPM_CONDITIONAL_HEAD_ARCH:-shared_lowrank_v1}"
```

Replace hard-coded `common_args` values:

```bash
--action_type_alert_policy "${ACTION_TYPE_ALERT_POLICY}"
--sspm_conditional_head_arch "${SSPM_CONDITIONAL_HEAD_ARCH}"
--conditional_low_support_policy "${CONDITIONAL_LOW_SUPPORT_POLICY}"
--conditional_low_support_margin "${CONDITIONAL_LOW_SUPPORT_MARGIN}"
--conditional_endpoint_aware_suppression "${CONDITIONAL_ENDPOINT_AWARE_SUPPRESSION}"
--conditional_endpoint_suppression_read "${CONDITIONAL_ENDPOINT_SUPPRESSION_READ}"
--conditional_endpoint_suppression_mode "${CONDITIONAL_ENDPOINT_SUPPRESSION_MODE}"
--conditional_both_cold_unseen_policy "${CONDITIONAL_BOTH_COLD_UNSEEN_POLICY}"
--sspm_state_model "${SSPM_STATE_MODEL}"
```

Update `run_infer_full`:

```bash
run_infer_full() {
    local out_tag="CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_INFER_FULL"
    if [[ -n "${OUT_TAG_OVERRIDE:-}" ]]; then
        out_tag="${OUT_TAG_OVERRIDE}"
    fi
    run_command "infer_full" "${out_tag}" \
        "${common_args[@]}" \
        --out_tag "${out_tag}" \
        --sspm_train_mode load_and_infer \
        --sspm_train_data_mode phase3e_memmap \
        --sspm_checkpoint_path "${BASE_CHECKPOINT}" \
        --action_head_checkpoint_path "${ACTION_HEAD_CHECKPOINT}"
}
```

Do not apply `OUT_TAG_OVERRIDE` to build/train stages.

- [ ] **Step 4: Run runner tests and syntax check**

Run:

```bash
python3 -m unittest tests.test_clearscope_v3_runner_policy
bash -n scripts/run/run_clearscope_e3_v3_phase3e_phase3g_full.sh
```

Expected: both commands PASS.

- [ ] **Step 5: Commit runner controls**

Run:

```bash
git add scripts/run/run_clearscope_e3_v3_phase3e_phase3g_full.sh \
    tests/test_clearscope_v3_runner_policy.py
git commit -m "feat: expose clearscope v3 phase3g policy controls"
```

## Task 2: Both Cold Unseen No-Alert Policy

**Files:**
- Modify: `scripts/pipeline/config/runtime_config.py`
- Modify: `scripts/pipeline/entrypoints/arguments.py`
- Modify: `scripts/pipeline/conditional/infer.py`
- Test: `tests/test_phase3g_both_cold_unseen_policy.py`

- [ ] **Step 1: Write config and helper tests**

Create `tests/test_phase3g_both_cold_unseen_policy.py`:

```python
"""Tests for Phase3G both_cold unseen-group no-alert policy."""

from __future__ import annotations

import unittest

from cs4m.phase3g.conditional_head import BOTH_COLD_ACTION_TARGET, EVENT_SEMANTIC_TARGET
from scripts.pipeline.config.runtime_config import SlimConfig
from scripts.pipeline.conditional.infer import _phase3g_should_suppress_both_cold_unseen_alert
from scripts.pipeline.entrypoints.arguments import parse_args, validate_config


class Phase3GBothColdUnseenPolicyTests(unittest.TestCase):
    """Validate default-off both_cold unseen alert suppression."""

    def test_cli_accepts_default_off_policy(self) -> None:
        args = parse_args([])
        config = SlimConfig(**vars(args))
        self.assertEqual(config.conditional_both_cold_unseen_policy, "alert")
        validate_config(config)

    def test_cli_accepts_observation_only_no_alert(self) -> None:
        args = parse_args(
            [
                "--conditional_both_cold_unseen_policy",
                "observation_only_no_alert",
            ],
        )
        config = SlimConfig(**vars(args))
        self.assertEqual(
            config.conditional_both_cold_unseen_policy,
            "observation_only_no_alert",
        )
        validate_config(config)

    def test_policy_suppresses_only_both_cold_unseen_group(self) -> None:
        config = SlimConfig(conditional_both_cold_unseen_policy="observation_only_no_alert")
        self.assertTrue(
            _phase3g_should_suppress_both_cold_unseen_alert(
                config=config,
                target_case=BOTH_COLD_ACTION_TARGET,
                threshold_level="unseen_group_extreme",
                validation_group_count=0,
            ),
        )
        self.assertFalse(
            _phase3g_should_suppress_both_cold_unseen_alert(
                config=config,
                target_case=EVENT_SEMANTIC_TARGET,
                threshold_level="unseen_group_extreme",
                validation_group_count=0,
            ),
        )
        self.assertFalse(
            _phase3g_should_suppress_both_cold_unseen_alert(
                config=config,
                target_case=BOTH_COLD_ACTION_TARGET,
                threshold_level="level1_group_quantile",
                validation_group_count=1000,
            ),
        )

    def test_default_policy_does_not_suppress(self) -> None:
        config = SlimConfig(conditional_both_cold_unseen_policy="alert")
        self.assertFalse(
            _phase3g_should_suppress_both_cold_unseen_alert(
                config=config,
                target_case=BOTH_COLD_ACTION_TARGET,
                threshold_level="unseen_group_extreme",
                validation_group_count=0,
            ),
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and confirm they fail**

Run:

```bash
python3 -m unittest tests.test_phase3g_both_cold_unseen_policy
```

Expected: FAIL because the config field, CLI argument, and helper do not exist.

- [ ] **Step 3: Add config field**

In `scripts/pipeline/config/runtime_config.py`, add to `SlimConfig` near
`conditional_unseen_group_policy`:

```python
conditional_both_cold_unseen_policy: str = "alert"
```

- [ ] **Step 4: Add CLI argument and validation**

In `scripts/pipeline/entrypoints/arguments.py`, add after
`--conditional_unseen_group_policy`:

```python
parser.add_argument(
    "--conditional_both_cold_unseen_policy",
    choices=("alert", "observation_only_no_alert"),
    default=SlimConfig.conditional_both_cold_unseen_policy,
)
```

In `validate_config`, add within the conditional-action semantic block:

```python
if str(config.conditional_both_cold_unseen_policy) not in {
    "alert",
    "observation_only_no_alert",
}:
    raise ValueError(
        "conditional_both_cold_unseen_policy must be alert or "
        "observation_only_no_alert",
    )
```

- [ ] **Step 5: Add helper and counters in conditional inference**

In `scripts/pipeline/conditional/infer.py`, add a helper near the top of the file:

```python
def _phase3g_should_suppress_both_cold_unseen_alert(
    *,
    config: SlimConfig,
    target_case: str,
    threshold_level: str,
    validation_group_count: int,
) -> bool:
    """Return true when both_cold unseen groups should remain observation-only."""
    if str(config.conditional_both_cold_unseen_policy) != "observation_only_no_alert":
        return False
    if str(target_case) != BOTH_COLD_ACTION_TARGET:
        return False
    if str(threshold_level) != "unseen_group_extreme":
        return False
    return int(validation_group_count) <= 0
```

In `_score_phase3g_conditional_fast_stream`, initialize after
`suppressed_alert_count = 0`:

```python
both_cold_unseen_suppressed_alert_count = 0
```

After `alert_row.update(threshold_trace)` and before endpoint suppression,
insert:

```python
if _phase3g_should_suppress_both_cold_unseen_alert(
    config=config,
    target_case=target_case,
    threshold_level=threshold_level,
    validation_group_count=validation_group_count,
):
    alert = False
    both_cold_unseen_suppressed_alert_count += 1
    alert_row["both_cold_unseen_policy"] = str(
        config.conditional_both_cold_unseen_policy,
    )
    alert_row["both_cold_unseen_suppressed"] = True
else:
    alert_row["both_cold_unseen_policy"] = str(
        config.conditional_both_cold_unseen_policy,
    )
    alert_row["both_cold_unseen_suppressed"] = False
```

Guard endpoint suppression so it only runs if `alert` is still true:

```python
if alert:
    suppression_started = time.perf_counter()
    suppression = _conditional_endpoint_suppression_decision(...)
```

Add to the returned payload near endpoint suppression counters:

```python
"both_cold_unseen_policy": {
    "policy_name": str(config.conditional_both_cold_unseen_policy),
    "suppressed_alert_count": int(both_cold_unseen_suppressed_alert_count),
},
```

Also add top-level:

```python
"both_cold_unseen_suppressed_alert_count": int(
    both_cold_unseen_suppressed_alert_count,
),
```

- [ ] **Step 6: Run targeted tests**

Run:

```bash
python3 -m unittest tests.test_phase3g_both_cold_unseen_policy
python3 -m unittest tests.test_clearscope_v3_runner_policy
```

Expected: PASS.

- [ ] **Step 7: Commit both_cold policy**

Run:

```bash
git add scripts/pipeline/config/runtime_config.py \
    scripts/pipeline/entrypoints/arguments.py \
    scripts/pipeline/conditional/infer.py \
    tests/test_phase3g_both_cold_unseen_policy.py
git commit -m "feat: gate unseen both-cold phase3g alerts"
```

## Task 3: Baseline and Result Report Script

**Files:**
- Create: `tmp/clearscope_v3_fp_policy_report.py`

- [ ] **Step 1: Create a report script**

Create `tmp/clearscope_v3_fp_policy_report.py`:

```python
"""Summarize ClearScope E3 v3 Phase3G FP-reduction runs."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _csv_group_fp(path: Path, action: str, src_type: str, dst_type: str) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("action") == action
                and row.get("src_type") == src_type
                and row.get("dst_type") == dst_type
            ):
                return int(float(row.get("fp", row.get("FP", 0)) or 0))
    return 0


def _target_case_alerts(path: Path, target_case: str) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("target_case") == target_case:
                alerts = int(float(row.get("alert_count", 0) or 0))
                fp = int(float(row.get("FP", row.get("fp", 0)) or 0))
                return alerts, fp
    return 0, 0


def summarize_run(run_dir: Path) -> dict[str, Any]:
    eval_payload = _load_json(run_dir / "eval_causal_semantics_slim.json")
    score_summary = _load_json(run_dir / "score_summary.json")
    event_alerts = (
        eval_payload.get("primary_online_metrics", {})
        .get("event_alerts", {})
    )
    node_coverage = (
        eval_payload.get("primary_online_metrics", {})
        .get("online_event_node_coverage", {})
    )
    both_cold_alerts, both_cold_fp = _target_case_alerts(
        run_dir / "target_case_summary.csv",
        "both_cold_action_target",
    )
    return {
        "run": run_dir.name,
        "event_alert_count": int(event_alerts.get("count", 0) or 0),
        "tp": int(event_alerts.get("tp", 0) or 0),
        "fp": int(event_alerts.get("fp", 0) or 0),
        "precision": float(event_alerts.get("ratio", 0.0) or 0.0),
        "malicious_node_recall": float(
            node_coverage.get("malicious_node_recall", 0.0) or 0.0,
        ),
        "event_read_netflow_process_fp": _csv_group_fp(
            run_dir / "event_fp_group_summary.csv",
            "EVENT_READ",
            "netflow",
            "process",
        ),
        "both_cold_alert_count": both_cold_alerts,
        "both_cold_fp": both_cold_fp,
        "threshold_mode": score_summary.get("event_threshold_mode", ""),
        "state_model": score_summary.get("state_model", ""),
        "endpoint_suppression": score_summary.get("conditional_endpoint_suppression", {}),
        "both_cold_unseen_policy": score_summary.get("both_cold_unseen_policy", {}),
        "leakage_check": eval_payload.get("leakage_check", {}),
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: clearscope_v3_fp_policy_report.py RUN_DIR [RUN_DIR ...]", file=sys.stderr)
        return 2
    summaries = [summarize_run(Path(raw)) for raw in argv[1:]]
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
```

- [ ] **Step 2: Run the report script on the frozen baseline**

Run:

```bash
python3 tmp/clearscope_v3_fp_policy_report.py \
    outputs/results/tflr_light/CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_INFER_FULL
```

Expected: JSON containing FP `410006` and threshold mode `quantile`.

- [ ] **Step 3: Compile report script**

Run:

```bash
python3 -m compileall tmp/clearscope_v3_fp_policy_report.py
```

Expected: PASS.

- [ ] **Step 4: Commit report script**

Run:

```bash
git add tmp/clearscope_v3_fp_policy_report.py
git commit -m "chore: add clearscope v3 fp policy report"
```

## Task 4: Verification Before Full Runs

**Files:**
- Modify only if previous tasks fail.

- [ ] **Step 1: Run targeted tests**

Run:

```bash
python3 -m unittest \
    tests.test_clearscope_v3_runner_policy \
    tests.test_phase3g_both_cold_unseen_policy \
    tests.test_conditional_reports_streaming_summary \
    tests.test_phase3e_train_base_routing \
    tests.test_residual_word2vec_corpus_file
```

Expected: PASS.

- [ ] **Step 2: Run syntax checks**

Run:

```bash
python3 -m compileall cs4m scripts tests tmp/clearscope_v3_fp_policy_report.py
bash -n scripts/run/run_clearscope_e3_v3_phase3e_phase3g_full.sh
```

Expected: PASS.

- [ ] **Step 3: Dry-run the group-threshold command**

Run:

```bash
DRY_RUN=1 \
STAGE=infer_full \
OUT_TAG_OVERRIDE=CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_INFER_GROUPQ0999 \
EVENT_THRESHOLD_MODE=conditional_target_action_type_group_quantile \
EVENT_THRESHOLD_QUANTILE=0.999 \
CONDITIONAL_GROUP_MIN_COUNT=1000 \
CONDITIONAL_LOW_SUPPORT_POLICY=conservative_max \
CONDITIONAL_LOW_SUPPORT_MARGIN=0.05 \
CONDITIONAL_BOTH_COLD_UNSEEN_POLICY=observation_only_no_alert \
bash scripts/run/run_clearscope_e3_v3_phase3e_phase3g_full.sh
```

Expected output includes:

```text
--out_tag CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_INFER_GROUPQ0999
--event_threshold_mode conditional_target_action_type_group_quantile
--conditional_both_cold_unseen_policy observation_only_no_alert
```

- [ ] **Step 4: Dry-run the endpoint command**

Run:

```bash
DRY_RUN=1 \
STAGE=infer_full \
OUT_TAG_OVERRIDE=CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_INFER_GROUPQ0999_ENDPOINT \
EVENT_THRESHOLD_MODE=conditional_target_action_type_group_quantile \
EVENT_THRESHOLD_QUANTILE=0.999 \
CONDITIONAL_GROUP_MIN_COUNT=1000 \
CONDITIONAL_LOW_SUPPORT_POLICY=conservative_max \
CONDITIONAL_LOW_SUPPORT_MARGIN=0.05 \
CONDITIONAL_BOTH_COLD_UNSEEN_POLICY=observation_only_no_alert \
CONDITIONAL_ENDPOINT_AWARE_SUPPRESSION=true \
CONDITIONAL_ENDPOINT_SUPPRESSION_MODE=pair_only \
CONDITIONAL_ENDPOINT_SUPPRESSION_READ=true \
bash scripts/run/run_clearscope_e3_v3_phase3e_phase3g_full.sh
```

Expected output includes:

```text
--out_tag CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_INFER_GROUPQ0999_ENDPOINT
--conditional_endpoint_aware_suppression true
--conditional_endpoint_suppression_read true
```

## Task 5: Full Run 1, Group Threshold

**Files:**
- Generated outputs only under `outputs/results/tflr_light` and logs.

- [ ] **Step 1: Confirm required artifacts exist**

Run:

```bash
test -f outputs/models/residual_word2vec/CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_LATENT64_word2vec_window3.pkl
test -f outputs/models/sspm_phase3e/CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_PHASE3E_BASE_FULL.pkl
test -f outputs/models/phase3g_action_heads/CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_PHASE3G_CONDITIONAL_HEAD.pkl
test -f outputs/cache/phase3e/event_indices/CLEARSCOPE_E3_v3/event_index_meta.json
test -f outputs/cache/phase3e/node_embeddings/CLEARSCOPE_E3_latent64_v3/node_embeddings.npy
test -f outputs/cache/phase3e/action_embeddings/CLEARSCOPE_E3_latent64_v3/action_embeddings.npy
```

Expected: all commands exit `0`.

- [ ] **Step 2: Launch full inference with a six-hour cap**

Run with the approved Python and DB environment:

```bash
timeout 6h bash -lc '
    export PYTHON_BIN="${PYTHON_BIN:-python3}"
    export CLAD_DB_HOST="${CLAD_DB_HOST:-localhost}"
    export CLAD_DB_PORT="${CLAD_DB_PORT:-5433}"
    export CLAD_DB_USER="${CLAD_DB_USER:-postgres}"
    STAGE=infer_full \
    OUT_TAG_OVERRIDE=CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_INFER_GROUPQ0999 \
    EVENT_THRESHOLD_MODE=conditional_target_action_type_group_quantile \
    EVENT_THRESHOLD_QUANTILE=0.999 \
    CONDITIONAL_GROUP_MIN_COUNT=1000 \
    CONDITIONAL_LOW_SUPPORT_POLICY=conservative_max \
    CONDITIONAL_LOW_SUPPORT_MARGIN=0.05 \
    CONDITIONAL_BOTH_COLD_UNSEEN_POLICY=observation_only_no_alert \
    bash scripts/run/run_clearscope_e3_v3_phase3e_phase3g_full.sh
'
```

Expected: run completes within six hours. If timeout occurs, stop and report log
tail plus partial artifact state.

- [ ] **Step 3: Generate and inspect report**

Run:

```bash
python3 tmp/clearscope_v3_fp_policy_report.py \
    outputs/results/tflr_light/CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_INFER_FULL \
    outputs/results/tflr_light/CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_INFER_GROUPQ0999
```

Expected:

- New run has `threshold_mode` `conditional_target_action_type_group_quantile`.
- New run FP is compared against baseline `410006`.
- `online_event_alerts.csv` exists.

- [ ] **Step 4: Check online output is label-free**

Run:

```bash
python3 - <<'PY'
import csv
from pathlib import Path
p = Path("outputs/results/tflr_light/CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_INFER_GROUPQ0999/online_event_alerts.csv")
with p.open(newline="", encoding="utf-8") as handle:
    cols = next(csv.reader(handle))
bad = [c for c in cols if any(s in c.lower() for s in ("label", "ground", "malicious", "attack"))]
print({"columns": len(cols), "bad_columns": bad})
raise SystemExit(1 if bad else 0)
PY
```

Expected: `bad_columns` is empty.

- [ ] **Step 5: Stop and report gate decision**

Report:

- Event alert count.
- TP/FP/precision.
- Malicious node recall.
- `EVENT_READ netflow->process` FP.
- `both_cold_action_target` alert count and FP.
- Whether FP is below `100000`.

Do not start endpoint run until the report is sent.

## Task 6: Full Run 2, Group Threshold Plus Endpoint Suppression

**Files:**
- Generated outputs only under `outputs/results/tflr_light`, endpoint cache, and logs.

- [ ] **Step 1: Launch endpoint full inference after Task 5 report**

Run:

```bash
timeout 6h bash -lc '
    export PYTHON_BIN="${PYTHON_BIN:-python3}"
    export CLAD_DB_HOST="${CLAD_DB_HOST:-localhost}"
    export CLAD_DB_PORT="${CLAD_DB_PORT:-5433}"
    export CLAD_DB_USER="${CLAD_DB_USER:-postgres}"
    STAGE=infer_full \
    OUT_TAG_OVERRIDE=CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_INFER_GROUPQ0999_ENDPOINT \
    EVENT_THRESHOLD_MODE=conditional_target_action_type_group_quantile \
    EVENT_THRESHOLD_QUANTILE=0.999 \
    CONDITIONAL_GROUP_MIN_COUNT=1000 \
    CONDITIONAL_LOW_SUPPORT_POLICY=conservative_max \
    CONDITIONAL_LOW_SUPPORT_MARGIN=0.05 \
    CONDITIONAL_BOTH_COLD_UNSEEN_POLICY=observation_only_no_alert \
    CONDITIONAL_ENDPOINT_AWARE_SUPPRESSION=true \
    CONDITIONAL_ENDPOINT_SUPPRESSION_MODE=pair_only \
    CONDITIONAL_ENDPOINT_SUPPRESSION_READ=true \
    bash scripts/run/run_clearscope_e3_v3_phase3e_phase3g_full.sh
'
```

Expected: run completes within six hours. If timeout occurs, stop and report.

- [ ] **Step 2: Generate comparison report**

Run:

```bash
python3 tmp/clearscope_v3_fp_policy_report.py \
    outputs/results/tflr_light/CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_INFER_FULL \
    outputs/results/tflr_light/CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_INFER_GROUPQ0999 \
    outputs/results/tflr_light/CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_INFER_GROUPQ0999_ENDPOINT
```

Expected: endpoint run has suppression metadata and reports suppressed count.

- [ ] **Step 3: Check endpoint metadata**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path
p = Path("outputs/results/tflr_light/CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_INFER_GROUPQ0999_ENDPOINT/score_summary.json")
d = json.loads(p.read_text())
payload = d.get("conditional_endpoint_suppression", {})
print(payload)
raise SystemExit(0 if payload.get("enabled") is True else 1)
PY
```

Expected: payload has `"enabled": true`.

- [ ] **Step 4: Stop and report trigger gates**

Report:

- Whether FP is below `100000`.
- Whether `EVENT_READ netflow->process` still dominates.
- Whether `both_cold_action_target` still produces large alert volume.
- Whether EMA baseline, Phase3G retraining, or semantic v3.1 is triggered.

Do not launch EMA, retraining, or v3.1 without this report checkpoint.

## Task 7: Triggered Follow-Up Plans

**Files:**
- Create a new spec/plan only if a trigger fires.

- [ ] **Step 1: EMA trigger**

If endpoint run FP remains above `100000` or `EVENT_READ netflow->process` still
dominates, write a short follow-up plan for:

```text
SSPM_STATE_MODEL=ema_fixed
EVENT_THRESHOLD_MODE=conditional_target_action_type_group_quantile
CONDITIONAL_BOTH_COLD_UNSEEN_POLICY=observation_only_no_alert
```

Use a separate out tag:

```text
CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_INFER_EMA_GROUPQ0999
```

- [ ] **Step 2: Phase3G retraining trigger**

If `both_cold_action_target` still produces large alert volume, write a follow-up
plan for a new target-case separated head checkpoint. Do not overwrite:

```text
outputs/models/phase3g_action_heads/CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_PHASE3G_CONDITIONAL_HEAD.pkl
```

- [ ] **Step 3: Semantic v3.1 trigger**

If token audit shows material FP/TP tuple collision after method fixes, write a
new v3.1 semantic spec before editing tokenizer code. The spec must define a new
semantic mode and require new Word2Vec and Phase3E caches.

## Final Verification and Reporting

- [ ] **Step 1: Final git status**

Run:

```bash
git status --short
```

Expected: source/test/doc changes are committed or clearly listed; generated
outputs remain untracked or ignored.

- [ ] **Step 2: Final response checklist**

Report:

- Changed files.
- Commits created.
- Commands run.
- Result directories produced.
- Metrics compared to baseline.
- Remaining risks and next trigger decision.
- Whether streaming inference, no leakage, explainability, and reproducibility
  are preserved.
