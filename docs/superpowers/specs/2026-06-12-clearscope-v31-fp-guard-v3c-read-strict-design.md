# ClearScope v31 FP Guard v3c Read-Strict Design

## Goal

Add a conservative `clearscope_v31_fp_guard_v3c` policy that keeps the current
v31 semantic model and thresholding unchanged, preserves v3b online node-evidence
context, and reduces only the extra `EVENT_READ file->process` demotion that made
v3b lose too much event TP.

Current formal baseline remains:

```text
CLEARSCOPE_E3_RAW_DETAIL_V31_DISCRIMINATIVE_FULL_GROUPQ0999_FP_GUARD_V2_CONSERVATIVE_RERUN1
```

v3b remains a useful ablation:

```text
CLEARSCOPE_E3_RAW_DETAIL_V31_DISCRIMINATIVE_FULL_GROUPQ0999_FP_GUARD_V3B_NODE_EVIDENCE
```

## Success Criteria

- `covered_malicious_nodes >= 40/41`
- `top3000 node TP >= 40`
- `event FP < 13,657`
- `event TP` should move back toward `550+`
- Runtime policy must remain streaming and label-free

## Baseline Evidence

The full-run comparison motivating v3c:

| Policy | Event TP | Event FP | Alerts | Covered Nodes | Top3000 TP | Strict Node FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v31 groupq0999 raw | 961 | 74,679 | 75,640 | 40/41 | 40 | 6,661 |
| v2 conservative | 623 | 13,657 | 14,280 | 40/41 | 40 | 3,401 |
| v3b node evidence | 485 | 10,518 | 11,003 | 40/41 | 40 | 3,325 |

v3b preserved node recall and top3000 TP, but lost 138 event TP relative to v2
conservative. The loss is concentrated in `EVENT_READ file->process`, where v3b
demoted more near-floor alerts than intended.

## Non-Leakage Boundary

v3c may use only fields visible during streaming inference or validation-derived
thresholding:

- action, source type, destination type
- event score
- current threshold and existing v2 conservative floor
- validation group count
- validation count bucket
- target case
- prior online alert counts for source and destination nodes
- prior online node-evidence counts for source and destination nodes

v3c must not use:

- test labels
- attack windows
- malicious node lists
- test ranking
- test TP/FP as runtime rule inputs
- label-derived rescue rules
- dataset-specific attack strings

Ground truth remains post-inference evaluation only.

## Policy Name

Add:

```text
clearscope_v31_fp_guard_v3c
```

Keep `clearscope_v31_fp_guard_v3b` unchanged so the previous ablation remains
reproducible.

## Runtime Configuration

v3c inherits the v2/v3b baseline configuration:

```text
SEMANTIC_MODE=raw_detail_v31_discriminative
ACTION_TYPE_ALERT_POLICY=clearscope_v31_fp_guard_v3c
NODE_POOL_SCORE_MODE=base_conf_v31_support
EVENT_THRESHOLD_MODE=conditional_target_action_type_group_quantile
EVENT_THRESHOLD_QUANTILE=0.999
CONDITIONAL_BOTH_COLD_UNSEEN_POLICY=observation_only_no_alert
CONDITIONAL_ENDPOINT_AWARE_SUPPRESSION=false
CONDITIONAL_ENDPOINT_SUPPRESSION_READ=false
```

No training logic, semantic tokenization, threshold mode, or floor value changes are
part of v3c.

## Decision Rule

v3c inherits v1/v2 behavior first:

- demote `EVENT_OPEN process->process` to node evidence
- demote `EVENT_READ process->process` to node evidence
- demote `EVENT_READ file->process` when it is below the existing read floor
- demote `EVENT_WRITE process->file` when it is below the existing write floor

Then v3c applies additional label-free node-evidence demotion:

### EVENT_WRITE process->file

Use the existing v3b rule unchanged. This group had low observed downside in v3b.

### EVENT_READ file->process

Use a stricter rule than v3b:

```text
score < read_floor + 0.0015
validation_group_count >= 1000
validation_count_bucket not in {low, low_support}
target_case not in {both_cold, cold, low_support}
max(src_alert_count, dst_alert_count) >= 2
or max(src_node_evidence_count, dst_node_evidence_count) >= 4
```

The rule intentionally tightens v3b:

```text
margin: 0.003 -> 0.0015
prior alert support: >= 1 -> >= 2
prior evidence support: >= 2 -> >= 4
```

This keeps higher-score file/process alerts visible while suppressing repeated,
near-floor alerts for nodes that already have online evidence.

## Diagnostics

Node-evidence rows should identify the v3c reason where feasible:

```text
policy_support_reason = v3c_read_strict_prior_alert
policy_support_reason = v3c_read_strict_prior_evidence
policy_margin_used = 0.0015
src_prior_alert_count
dst_prior_alert_count
src_prior_node_evidence_count
dst_prior_node_evidence_count
```

Do not add these fields to `online_event_alerts.csv` unless the current writer already
supports them without changing the alert schema. Keeping them in
`action_type_node_evidence_events.csv` and policy summaries is sufficient.

## Test Requirements

Focused policy tests must cover:

- policy registry includes `clearscope_v31_fp_guard_v3c`
- v3c preserves v1 process/process demotion
- v3c preserves v2 floor demotion
- v3c keeps first near-floor `EVENT_READ file->process`
- v3c keeps read when prior alert count is only `1`
- v3c demotes read when prior alert count is `2` and score is narrow
- v3c keeps read when score is `read_floor + 0.002`
- v3c demotes read when prior node-evidence count is `4`
- v3c keeps low-support or both-cold read events
- v3c keeps v3b behavior for `EVENT_WRITE process->file`

Existing v3b tests must keep passing.

## Run Plan

Smoke tag:

```text
CLEARSCOPE_E3_RAW_DETAIL_V31_DISCRIMINATIVE_SMOKE_GROUPQ0999_FP_GUARD_V3C_READ_STRICT
```

Full tag:

```text
CLEARSCOPE_E3_RAW_DETAIL_V31_DISCRIMINATIVE_FULL_GROUPQ0999_FP_GUARD_V3C_READ_STRICT
```

Run smoke first with:

```text
MAX_TRAIN_EVENTS=200000
MAX_REF_EVENTS=50000
MAX_TEST_EVENTS=100000
```

If smoke passes syntax/config/header checks, launch the full run with:

```text
nohup timeout 30m
MAX_TRAIN_EVENTS=0
MAX_REF_EVENTS=0
MAX_TEST_EVENTS=0
```

Do not overwrite existing v2, v3, or v3b output directories.

## Expected Outcome

Expected v3c range:

- event FP between `10,518` and `13,657`, target below `12,500`
- event TP between `485` and `623`, target `550+`
- covered nodes remains `40/41`
- top3000 TP remains `40`
- strict node FP remains near or below v2 conservative's `3,401`

If full v3c still has `event TP < 520`, the read gate remains too aggressive. The next
ablation should remove or further narrow the extra file/process read demotion while
keeping the write-side v3b behavior.

If full v3c FP rises close to v2 conservative, the read gate is too weak. The next
ablation should consider `read margin = 0.002` or prior evidence threshold `3`, still
without using test labels as runtime inputs.
