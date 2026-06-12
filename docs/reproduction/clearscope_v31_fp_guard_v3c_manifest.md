# ClearScope v31 FP Guard v3c Reproduction Manifest

## Candidate

```text
candidate_tag = CLEARSCOPE_E3_RAW_DETAIL_V31_DISCRIMINATIVE_FULL_GROUPQ0999_FP_GUARD_V3C_READ_STRICT
policy = clearscope_v31_fp_guard_v3c
baseline_tag = CLEARSCOPE_E3_RAW_DETAIL_V31_DISCRIMINATIVE_FULL_GROUPQ0999_FP_GUARD_V2_CONSERVATIVE_RERUN1
ablation_tag = CLEARSCOPE_E3_RAW_DETAIL_V31_DISCRIMINATIVE_FULL_GROUPQ0999_FP_GUARD_V3B_NODE_EVIDENCE
completed_at = Fri Jun 12 23:14:05 CST 2026
```

## Runtime Configuration

```text
STAGE=infer_full
SEMANTIC_MODE=raw_detail_v31_discriminative
ACTION_TYPE_ALERT_POLICY=clearscope_v31_fp_guard_v3c
NODE_POOL_SCORE_MODE=base_conf_v31_support
EVENT_THRESHOLD_MODE=conditional_target_action_type_group_quantile
EVENT_THRESHOLD_QUANTILE=0.999
CONDITIONAL_BOTH_COLD_UNSEEN_POLICY=observation_only_no_alert
OUT_TAG_OVERRIDE=CLEARSCOPE_E3_RAW_DETAIL_V31_DISCRIMINATIVE_FULL_GROUPQ0999_FP_GUARD_V3C_READ_STRICT
```

The effective configuration is recorded in:

```text
outputs/results/tflr_light/CLEARSCOPE_E3_RAW_DETAIL_V31_DISCRIMINATIVE_FULL_GROUPQ0999_FP_GUARD_V3C_READ_STRICT/config_effective.yaml
```

## Primary Metrics

```text
event_tp = 550
event_fp = 11925
event_alerts = 12475
covered_malicious_nodes = 40/41
top1000_node_tp = 13
top1500_node_tp = 28
top3000_node_tp = 40
strict_node_fp = 3344
events_scored = 4496239
elapsed_seconds = 1405.36
throughput_events_per_second = 3206.32
deploy_infer_rss_peak_mb = 228.32
```

## Artifact Inventory

All paths are relative to the repository root.

| artifact | bytes | mtime | sha256 |
|---|---:|---|---|
| `outputs/results/tflr_light/CLEARSCOPE_E3_RAW_DETAIL_V31_DISCRIMINATIVE_FULL_GROUPQ0999_FP_GUARD_V3C_READ_STRICT/config_effective.yaml` | 10,435 | 2026-06-12 23:14:05 +0800 | `a69b407f09f9190bf6f40c6c70022c67eecda84ab07b6fe0a9d1ee874986b6be` |
| `outputs/results/tflr_light/CLEARSCOPE_E3_RAW_DETAIL_V31_DISCRIMINATIVE_FULL_GROUPQ0999_FP_GUARD_V3C_READ_STRICT/eval_causal_semantics_slim.json` | 153,438 | 2026-06-12 23:14:05 +0800 | `103fa247c1bd35eec9a2579511e2345f02eee56a38ffbf25507b427603e7cd0d` |
| `outputs/results/tflr_light/CLEARSCOPE_E3_RAW_DETAIL_V31_DISCRIMINATIVE_FULL_GROUPQ0999_FP_GUARD_V3C_READ_STRICT/metrics.json` | 232,968 | 2026-06-12 23:14:05 +0800 | `60e34dbe3678ae9bcbba52d4048a72443394091915c16f966168097235323407` |
| `outputs/results/tflr_light/CLEARSCOPE_E3_RAW_DETAIL_V31_DISCRIMINATIVE_FULL_GROUPQ0999_FP_GUARD_V3C_READ_STRICT/online_event_alerts.csv` | 8,037,743 | 2026-06-12 23:12:22 +0800 | `d6008e53d969179658acf2f705444e5cc1c01b2534388274695c90c734a08cdc` |
| `outputs/results/tflr_light/CLEARSCOPE_E3_RAW_DETAIL_V31_DISCRIMINATIVE_FULL_GROUPQ0999_FP_GUARD_V3C_READ_STRICT/action_type_node_evidence_events.csv` | 45,444,365 | 2026-06-12 23:12:22 +0800 | `2f3d00e3b3fd1ec2766a1f5c2744a93f0e4ea10e9c962c069dd113c35e53767f` |
| `outputs/results/tflr_light/CLEARSCOPE_E3_RAW_DETAIL_V31_DISCRIMINATIVE_FULL_GROUPQ0999_FP_GUARD_V3C_READ_STRICT/group_alert_policy_summary.csv` | 4,019 | 2026-06-12 23:12:25 +0800 | `ec4e31de8e725228b5d2d24df17f4d508eb2c590b760e3e5ea133009b062b797` |
| `outputs/results/tflr_light/CLEARSCOPE_E3_RAW_DETAIL_V31_DISCRIMINATIVE_FULL_GROUPQ0999_FP_GUARD_V3C_READ_STRICT/demoted_group_summary.csv` | 1,201 | 2026-06-12 23:12:27 +0800 | `8deba8896e51de8cc6cf3bb46580eccee40f7aca73458b92162f4401ffad0324` |
| `outputs/results/tflr_light/CLEARSCOPE_E3_RAW_DETAIL_V31_DISCRIMINATIVE_SMOKE_GROUPQ0999_FP_GUARD_V3C_READ_STRICT_DIAG/action_type_node_evidence_events.csv` | 33,862 | 2026-06-12 23:53:09 +0800 | `1cc626a76d91bf5c49a49629f09949861df2aa74c488d3e31a021d34997c4369` |

## Diagnostics Smoke

```text
diagnostics_tag = CLEARSCOPE_E3_RAW_DETAIL_V31_DISCRIMINATIVE_SMOKE_GROUPQ0999_FP_GUARD_V3C_READ_STRICT_DIAG
event_alerts = 6
event_tp = 0
event_fp = 6
online_event_alerts_columns = 64
online_event_alerts_label_or_malicious_columns = []
node_evidence_columns = 75
node_evidence_label_or_malicious_columns = []
```

Diagnostics fields present in `action_type_node_evidence_events.csv`:

```text
policy_support_reason
policy_margin_used
src_prior_alert_count
dst_prior_alert_count
src_prior_node_evidence_count
dst_prior_node_evidence_count
```

## Verification Commands

```text
python3 -m unittest tests.test_clearscope_v31_fp_guard_policy
python3 -m compileall configs cs4m scripts
bash -n scripts/run/*.sh
```

## Safety Notes

Streaming inference is preserved because v3c uses only current event fields,
validation-derived thresholds, and prior online alert/evidence counters. Runtime
policy decisions do not use labels, attack windows, malicious nodes, or test
rankings. Full candidate outputs were written under a unique tag and existing
v2/v3b outputs were not overwritten.
