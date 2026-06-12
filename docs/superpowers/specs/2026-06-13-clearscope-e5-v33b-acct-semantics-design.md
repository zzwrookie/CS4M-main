# ClearScope E5 V33B Accounting Semantics Design

## Goal

Add the smallest general ClearScope Android tokenizer correction needed after the
`raw_detail_v33_e5_android_safe` audit.

The new semantic mode is:

```text
raw_detail_v33b_e5_android_safe
```

It inherits `raw_detail_v33_e5_android_safe` and only adds low-cardinality
handling for the label-free `/acct/...` Android/Linux accounting path shape.

This stage stops after:

1. implementing the v33b tokenizer mode;
2. rerunning the E5 semantic audit smoke;
3. comparing v31 and v33b audit outputs;
4. deciding whether E5 bounded artifact and pipeline smoke planning can start.

It does not build E5 artifacts, modify E5 runners, run pipeline smoke, or run full
inference.

## Current Evidence

The v33 bounded E5 audit produced:

- `input_event_count`: `150000`
- `event_count`: `150000`
- `skipped_event_count`: `0`
- `usable_event_rate`: `1.0`
- `fallback_counts`:
  - `file_other`: `801`
  - `dev_other`: `50`
  - `netflow`: `20000`
- `malicious_node_count`: `8`
- `malicious_fallback_counts`:
  - `dev_other`: `1`

The v33 comparison showed:

```text
malicious file_other: 3 -> 0
malicious collision groups: 1 -> 0
label-free file_other: 1066 -> 801
test_oov_tuple_count: 39 -> 39
```

The strict v33 gate required label-free `file_other < 800`, so v33 is still a
no-go by 2 nodes. The remaining label-free `file_other` collision row is the
general accounting path shape:

```text
/acct
/acct/uid_0
/acct/uid_0/pid_549
...
```

This is a runtime-visible Android/Linux accounting shape. It is not a malicious
package, malicious path, attack-window rule, threshold change, or evaluation
change.

## Non-Goals

This stage does not:

- modify event thresholds;
- modify group quantile behavior;
- modify `fp_guard`, v3c read gates, or alert policy;
- modify node pool score mode;
- modify evaluation or ground-truth matching;
- train Word2Vec, Phase3E, or Phase3G artifacts;
- create or change E5 runners;
- run pipeline smoke or full inference;
- reuse `CLEARSCOPE_E3_*` artifacts for E5;
- expand `/dev/__properties__` or other `/dev/*` paths;
- change netflow behavior.

## Safety Rules

The tokenizer changes must be general path-shape rules. They must not depend on:

- malicious node IDs;
- attack windows at runtime;
- hardcoded attack package names;
- hardcoded malicious file paths;
- label-derived score thresholds;
- label-derived rescue rules.

Label-aware data can only be used as a diagnostic appendix after the label-free
audit is complete. The label-free report remains the primary source for
go/no-go decisions.

## Semantic Mode Behavior

`raw_detail_v33b_e5_android_safe` must normalize through the existing ClearScope
semantic-mode normalization path and must not change existing v33 outputs unless
one of the new `/acct` accounting shapes applies.

### `/acct` Accounting Paths

Add low-cardinality file class/detail rules:

```text
/acct
-> file android_accounting_root acct_root

/acct/uid_<uid>
-> file android_accounting_uid acct_uid

/acct/uid_<uid>/pid_<pid>
-> file android_accounting_pid acct_pid

/acct/<anything_else>
-> file android_accounting_other acct_other
```

The implementation must not keep concrete `uid` or `pid` numbers in detail
tokens. This prevents high-cardinality detail and avoids OOV growth.

Examples:

```text
/acct/uid_0
-> file android_accounting_uid acct_uid

/acct/uid_0/pid_549
-> file android_accounting_pid acct_pid

/acct/uid_1000/pid_12345
-> file android_accounting_pid acct_pid

/acct/uid_bad
-> file android_accounting_other acct_other

/acct/uid_0/other
-> file android_accounting_other acct_other
```

### Inherited V33 Behavior

v33b preserves v33 behavior for:

- `/config/sdcardfs/<package>/appid`;
- `/config/sdcardfs/<package>`;
- the stable `/dev/*` allowlist;
- unknown `/dev/*` values remaining `dev_other`;
- fixed coarse netflow.

## Audit Rerun

After implementation, rerun E5 audit smoke with:

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

Compare against the current v31 audit:

```bash
python3 scripts/tools/compare_clearscope_e5_semantic_audits.py \
  --v31_dir tmp/clearscope_e5_semantic_audit_smoke \
  --v33_dir tmp/clearscope_e5_v33b_semantic_audit_smoke \
  --output tmp/clearscope_e5_v33b_semantic_audit_comparison.json
```

## Go/No-Go Decision

This stage ends with one of:

```text
v33b_audit_pass_pipeline_smoke_can_be_planned
v33b_needs_more_general_tokenizer_adjustment
v33b_blocks_e5_pipeline_smoke
```

The audit passes only if:

- `skipped_event_count = 0`;
- `usable_event_rate >= 0.95`;
- malicious `file_other = 0`;
- malicious `dev_other` does not increase;
- label-free `file_other < 800`;
- no raw package/hash explosion appears in top detail or collision rows;
- `test_oov_tuple_count` does not materially worsen from v33's `39`;
- label-free summary still has no top-level `malicious`, `attack`, `label`, or
  `ground_truth` keys;
- threshold, policy, node-pool, evaluation, and inference code remain unchanged.

If the audit passes, the next stage is a separate E5 bounded artifact and pipeline
smoke design. That future stage must still avoid reusing `CLEARSCOPE_E3_*`
artifacts.

## Acceptance Criteria

The stage is complete when:

- the new semantic mode is registered and unit-tested;
- v33 behavior is unchanged for non-`/acct` targeted paths;
- `/acct` accounting shapes produce low-cardinality detail tokens;
- uid and pid numbers are not preserved in detail tokens;
- E5 audit smoke runs with `raw_detail_v33b_e5_android_safe`;
- v31/v33b audit comparison is reported;
- a go/no-go decision is documented;
- no pipeline smoke or full inference is launched.
