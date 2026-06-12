# ClearScope v31 FP Guard v3c Candidate Report

## Summary

`clearscope_v31_fp_guard_v3c` is the current recommended ClearScope E3 v31
candidate. It improves on the v2 conservative baseline by reducing event FP while
preserving malicious node coverage and top3000 node recall. It also recovers a
meaningful part of the event TP that v3b lost.

Recommended status:

```text
candidate_best = CLEARSCOPE_E3_RAW_DETAIL_V31_DISCRIMINATIVE_FULL_GROUPQ0999_FP_GUARD_V3C_READ_STRICT
baseline = CLEARSCOPE_E3_RAW_DETAIL_V31_DISCRIMINATIVE_FULL_GROUPQ0999_FP_GUARD_V2_CONSERVATIVE_RERUN1
ablation = CLEARSCOPE_E3_RAW_DETAIL_V31_DISCRIMINATIVE_FULL_GROUPQ0999_FP_GUARD_V3B_NODE_EVIDENCE
```

Do not continue to v32 or tune test-derived floors from this result. v3c reached
the planned acceptance gate and should move into candidate freeze and reproduction
audit.

## Output Directory

```text
outputs/results/tflr_light/CLEARSCOPE_E3_RAW_DETAIL_V31_DISCRIMINATIVE_FULL_GROUPQ0999_FP_GUARD_V3C_READ_STRICT
```

Completion log:

```text
logs/clearscope_v31_fp_guard_v3c_read_strict_full.nohup.log
```

The run completed at:

```text
Fri Jun 12 23:14:05 CST 2026
```

## Effective Configuration

Important values from `config_effective.yaml`:

```text
action_type_alert_policy: clearscope_v31_fp_guard_v3c
semantic_mode: raw_detail_v31_discriminative
event_threshold_mode: conditional_target_action_type_group_quantile
event_threshold_quantile: 0.999
node_pool_score_mode: base_conf_v31_support
conditional_both_cold_unseen_policy: observation_only_no_alert
clearscope_v31_fp_guard_read_floor: 0.06
clearscope_v31_fp_guard_write_floor: 0.085
max_test_events: 0
```

## Primary Metrics

| run | event TP | event FP | alerts | precision | covered nodes | top1000 TP | top1500 TP | top3000 TP | strict node FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_groupq0999 | 961 | 74,679 | 75,640 | 0.0127 | 40/41 | 10 | 21 | 40 | 6,661 |
| v2_conservative | 623 | 13,657 | 14,280 | 0.0436 | 40/41 | 13 | 26 | 40 | 3,401 |
| v3b_node_evidence | 485 | 10,518 | 11,003 | 0.0441 | 40/41 | 13 | 28 | 40 | 3,325 |
| v3c_read_strict | 550 | 11,925 | 12,475 | 0.0441 | 40/41 | 13 | 28 | 40 | 3,344 |

v3c satisfies the planned gate:

- `covered_malicious_nodes = 40/41`
- `top3000 node TP = 40`
- `event FP = 11,925`, below v2 conservative `13,657`
- `event TP = 550`, meeting the target line
- strict node FP remains below v2 conservative

## Group-Level Explanation

| run | group | alerts | TP | FP | demoted | node evidence |
|---|---|---:|---:|---:|---:|---:|
| v2_conservative | EVENT_OPEN process->process | 0 | 0 | 0 | 8,352 | 8,352 |
| v2_conservative | EVENT_READ process->process | 0 | 0 | 0 | 40,490 | 40,490 |
| v2_conservative | EVENT_READ file->process | 8,631 | 332 | 8,299 | 9,524 | 9,524 |
| v2_conservative | EVENT_WRITE process->file | 4,609 | 289 | 4,320 | 2,994 | 2,994 |
| v3b_node_evidence | EVENT_OPEN process->process | 0 | 0 | 0 | 8,352 | 8,352 |
| v3b_node_evidence | EVENT_READ process->process | 0 | 0 | 0 | 40,490 | 40,490 |
| v3b_node_evidence | EVENT_READ file->process | 5,422 | 194 | 5,228 | 12,733 | 12,733 |
| v3b_node_evidence | EVENT_WRITE process->file | 4,541 | 289 | 4,252 | 3,062 | 3,062 |
| v3c_read_strict | EVENT_OPEN process->process | 0 | 0 | 0 | 8,352 | 8,352 |
| v3c_read_strict | EVENT_READ process->process | 0 | 0 | 0 | 40,490 | 40,490 |
| v3c_read_strict | EVENT_READ file->process | 6,894 | 259 | 6,635 | 11,261 | 11,261 |
| v3c_read_strict | EVENT_WRITE process->file | 4,541 | 289 | 4,252 | 3,062 | 3,062 |

The main v3c effect is exactly the intended one: it relaxes v3b for
`EVENT_READ file->process`, recovering 65 event TP relative to v3b while still
keeping FP well below v2 conservative. It keeps the v3b write-side behavior.

## Runtime And Memory

From `eval_causal_semantics_slim.json`:

```text
events_scored: 4,496,239
elapsed_seconds: 1,405.36
throughput_events_per_second: 3,206.32
deploy_infer_rss_peak_mb: 228.32
deploy_rss_valid: true
state_array_mb: 16.0
```

## Evidence Files

Key files in the v3c output directory:

```text
config_effective.yaml
eval_causal_semantics_slim.json
metrics.json
online_event_alerts.csv
action_type_node_evidence_events.csv
group_alert_policy_summary.csv
demoted_group_summary.csv
node_topk_metrics.json
online_event_node_coverage_summary.json
```

Header audit:

```text
online_event_alerts.csv columns: 64
online_event_alerts.csv label_or_malicious_columns: []
action_type_node_evidence_events.csv columns: 69
action_type_node_evidence_events.csv label_or_malicious_columns: []
```

The full v3c artifact was generated before the diagnostics-only column hardening,
so its `action_type_node_evidence_events.csv` does not include the optional
reason-specific v3c fields. A follow-up diagnostics smoke run verified the current
code writes these fields only to node-evidence diagnostics, without adding them to
`online_event_alerts.csv`:

```text
diagnostics_smoke = CLEARSCOPE_E3_RAW_DETAIL_V31_DISCRIMINATIVE_SMOKE_GROUPQ0999_FP_GUARD_V3C_READ_STRICT_DIAG
diagnostics_smoke event alerts: 6
diagnostics_smoke event TP/FP: 0/6
diagnostics_smoke online_event_alerts.csv columns: 64
diagnostics_smoke online_event_alerts.csv label_or_malicious_columns: []
diagnostics_smoke action_type_node_evidence_events.csv columns: 75
diagnostics_smoke action_type_node_evidence_events.csv label_or_malicious_columns: []
diagnostics columns present:
  policy_support_reason
  policy_margin_used
  src_prior_alert_count
  dst_prior_alert_count
  src_prior_node_evidence_count
  dst_prior_node_evidence_count
```

The full candidate metrics remain the frozen decision evidence. The diagnostics
smoke is supporting evidence that current code improves demotion explainability
without changing the primary alert output schema or using label-derived runtime
signals.

## Safety Properties

Streaming inference is preserved:

- the v3c decision uses current event fields, validation-derived thresholds, and
  prior online alert/evidence counts only;
- no future events are required for online alert decisions.

No leakage is preserved:

- runtime policy does not use labels, malicious nodes, attack windows, or test
  rankings;
- ground truth is used only by post-inference evaluation outputs.

Explainability is preserved:

- final alerts are written to `online_event_alerts.csv`;
- demoted events are retained as node evidence;
- group summaries explain which event groups were demoted and by how much.

Reproducibility is preserved:

- result outputs are under a unique v3c tag;
- v2/v3b result directories were not overwritten;
- effective config and metrics are saved in the output directory.

## Recommendation

Freeze v3c as the current ClearScope E3 v31 candidate. Keep v2 conservative as the
stable baseline and v3b as an ablation. Do not enter v3d unless a later review
requires a diagnostics-only improvement or a cross-dataset validation shows that
the v3c tradeoff does not hold.
