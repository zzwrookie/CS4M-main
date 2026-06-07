# CS4M Backup Confirmation Manifest

Created: 2026-06-07

## Status

No files have been copied yet.

Backup target exists:

```text
../CS4M-main
```

Because `../CS4M-main` is outside the writable sandbox root, the actual copy
will require explicit approval before it can start.

Recommended target root after approval:

```text
../CS4M-main/backups/tflr_best_e3_artifacts_20260607/
```

Estimated size of the selected existing artifacts is about 32G before any
filesystem-level deduplication or compression.

## Can The Backup Follow The Requested Standard?

Yes, with two explicit gaps:

1. `outputs/cache/phase3g_action_validation/THEIA_E3/` is missing. The THEIA_E3
   rerun result references this path in `config_effective.yaml` and
   `metrics.json`, so direct rerun in `CS4M-main` may need that cache
   regenerated.
2. The THEIA_E3 rerun result directory does not contain `target_case_summary.csv`.
   No renamed `*target*case*` file was found in that result directory. This is
   consistent with the THEIA shared-head v1 path, while CADETS dual-head v2 has
   `target_case_summary.csv`.

The requested residual Word2Vec wildcard artifacts do exist as `.json` and
`.pkl` files. Gensim-style `.model` and `.npy` sidecars are not present, but
they were not required by the requested `word2vec_window3.*` wildcard.

## Result Artifacts

### CADETS_E3 Primary Best Result

Ready:

```text
<HISTORICAL_TMP_ROOT>/tflr_light_phase3g_v2_Q09995_full/CADETS_E3_PHASE3E_E4_NONE/
```

Size: 12M

Key files present:

- `config_effective.yaml`
- `eval_causal_semantics_slim.json`
- `metrics.json`
- `online_event_alerts.csv`
- `online_node_alerts.csv`
- `node_topk_metrics.json`
- `score_summary.json`
- `conditional_score_summary_by_target_action_type.csv`
- `target_case_summary.csv`
- `profiling.csv`
- `state_memory_storage_audit.json`
- `rss_breakdown_online_event_core.json`
- `rss_timeline.csv`

Configuration signature:

- `action_type_alert_policy: default`
- `sspm_conditional_head_arch: dual_lowrank_by_target_case_v2`
- `target_case_head_mode: target_case_specific`
- `event_threshold_quantile: 0.9995`
- action head:
  `outputs/models/phase3g_action_heads_v2/CADETS_E3_E4_PHASE3G_CONDITIONAL_DUAL_HEAD.pkl`
- validation cache:
  `<HISTORICAL_TMP_ROOT>/phase3g_action_validation_v2_Q09995/CADETS_E3`

### CADETS_E3 Same-Level Ablations

Ready:

```text
<HISTORICAL_TMP_ROOT>/tflr_light_phase3g_v2_Q09995_full/CADETS_E3_PHASE3E_E2_NONE/
<HISTORICAL_TMP_ROOT>/tflr_light_phase3g_v2_Q09995_full/CADETS_E3_PHASE3E_E5_NONE/
```

Sizes:

- E2: 13M
- E5: 13M

Both contain the same requested key file set as the E4 primary result,
including `target_case_summary.csv`, `rss_breakdown_online_event_core.json`,
and `rss_timeline.csv`.

### THEIA_E3 Primary Best Result

Ready with one missing optional/architecture-specific file:

```text
outputs/results/phase3g_theia_e3_e4_theia_v1_lazy100k_pair_only_full_rerun_20260602_142411/THEIA_E3_PHASE3E_E4_NONE/
```

Size: 535M

Key files present:

- `config_effective.yaml`
- `eval_causal_semantics_slim.json`
- `metrics.json`
- `online_event_alerts.csv`
- `online_node_alerts.csv`
- `node_topk_metrics.json`
- `score_summary.json`
- `conditional_score_summary_by_target_action_type.csv`
- `profiling.csv`
- `state_memory_storage_audit.json`
- `rss_breakdown_online_event_core.json`
- `rss_timeline.csv`

Missing from this result directory:

- `target_case_summary.csv`

Configuration signature:

- `action_type_alert_policy: theia_v1`
- `sspm_conditional_head_arch: shared_lowrank_v1`
- `event_threshold_quantile: 0.999`
- action head:
  `outputs/models/phase3g_action_heads/THEIA_E3_E4_PHASE3G_CONDITIONAL_HEAD.pkl`
- validation cache referenced by config:
  `outputs/cache/phase3g_action_validation/THEIA_E3`

## Code To Back Up

Ready:

```text
scripts/tools/run_tflr_light_db_lowrank.py
cs4m/
scripts/tools/
scripts/data/
scripts/eval/
cs4m/config/
configs/
```

Requested runners are present:

```text
scripts/run/run_cadets_e3_phase3e_node_action_semantic_full.sh
scripts/run/run_phase3g_cadets_e3_dual_head_v2_q09995_e2_e4_e5_full.sh
legacy/runners/run_phase3g_cadets_e3_dual_head_v2_e2_e4_e5_full.sh
scripts/run/run_phase3g_cadets_e3_train_dual_head_v2_full.sh
scripts/run/run_phase3g_theia_e3_e4_theia_v1_lazy100k_pair_only_full.sh
```

Note: `run_tflr_light_db_lowrank.py` is already inside `scripts/tools/`.
It remains listed separately because the user called it out explicitly.

## Models

Ready:

```text
outputs/models/sspm_phase3e/CADETS_E3_E4_PHASE3E_BASE_FULL.pkl
outputs/models/sspm_phase3e/THEIA_E3_E4_PHASE3E_BASE_FULL.pkl
outputs/models/phase3g_action_heads_v2/CADETS_E3_E4_PHASE3G_CONDITIONAL_DUAL_HEAD.pkl
outputs/models/phase3g_action_heads/THEIA_E3_E4_PHASE3G_CONDITIONAL_HEAD.pkl
outputs/models/residual_word2vec/CADETS_E3_RAW_DETAIL_RULES_V1_LATENT64_word2vec_window3.json
outputs/models/residual_word2vec/CADETS_E3_RAW_DETAIL_RULES_V1_LATENT64_word2vec_window3.pkl
outputs/models/residual_word2vec/THEIA_E3_RAW_DETAIL_RULES_V1_LATENT64_word2vec_window3.json
outputs/models/residual_word2vec/THEIA_E3_RAW_DETAIL_RULES_V1_LATENT64_word2vec_window3.pkl
```

Useful ClearScope semantic-model artifact also present and recommended for the
future `CLEARSCOPE_E3` work:

```text
outputs/models/residual_word2vec/CLEARSCOPE_E3_RAW_DETAIL_V2_REFINED_WORD2VEC_LATENT64_word2vec_window3.json
outputs/models/residual_word2vec/CLEARSCOPE_E3_RAW_DETAIL_V2_REFINED_WORD2VEC_LATENT64_word2vec_window3.pkl
```

## Caches

Ready:

```text
outputs/cache/phase3e/event_indices/CADETS_E3/
outputs/cache/phase3e/event_indices/THEIA_E3/
outputs/cache/phase3e/x_context/CADETS_E3/
outputs/cache/phase3e/x_context/THEIA_E3/
outputs/cache/phase3e/node_embeddings/CADETS_E3_latent64/
outputs/cache/phase3e/node_embeddings/THEIA_E3_latent64/
outputs/cache/phase3e/action_embeddings/CADETS_E3_latent64/
outputs/cache/phase3e/action_embeddings/THEIA_E3_latent64/
outputs/cache/phase3e/compact_used_node_embeddings/CADETS_E3/
outputs/cache/phase3e/compact_used_node_embeddings/THEIA_E3/
outputs/cache/phase3g_endpoint_suppression_compact/CADETS_E3/
outputs/cache/phase3g_endpoint_suppression_compact/THEIA_E3/
<HISTORICAL_TMP_ROOT>/phase3g_action_validation_v2_Q09995/CADETS_E3/
<HISTORICAL_TMP_ROOT>/phase3g_action_validation_v2/CADETS_E3/
```

Approximate large-cache sizes:

- CADETS x-context: 12G
- THEIA x-context: 17G
- CADETS event indices: 357M
- THEIA event indices: 622M
- CADETS compact used node embeddings: 503M
- THEIA compact used node embeddings: 903M

Missing:

```text
outputs/cache/phase3g_action_validation/THEIA_E3/
```

Nearby cache root exists but only contains CADETS:

```text
outputs/cache/phase3g_action_validation/conditional_train_memmaps/CADETS_E3/
```

## Ground Truth

Ready:

```text
ground_truth/
```

Size: 1.5M

This includes:

- `ground_truth/E3-CADETS/`
- `ground_truth/E3-CLEARSCOPE/`
- `ground_truth/E3-THEIA/`
- `ground_truth/E5-CADETS/`
- `ground_truth/E5-CLEARSCOPE/`
- `ground_truth/E5-THEIA/`

Backing up the full directory is recommended because future work includes both
`CLEARSCOPE_E3` and `THEIA_E5`, and the directory is small.

## Supplemental Documentation

Ready:

```text
docs/method/CLEARSCOPE_E3_E5_RAW_DETAIL_V2_REFINED_RULE_BACKUP.md
docs/reproduction/CS4M_BACKUP_CONFIRMATION_MANIFEST_20260607.md
```

Recommended optional diagnostic outputs for the ClearScope semantic backup:

```text
<HISTORICAL_TMP_ROOT>/clearscope_detail_semantic_audit_v2_refined/
<HISTORICAL_TMP_ROOT>/clearscope_file_raw_detail_compressed_candidate_audit_min/
```

These are small and useful for later E3/E5 tokenizer comparison.

## Items Needing User Decision

1. Confirm target root:
   `../CS4M-main/backups/tflr_best_e3_artifacts_20260607/`
2. Confirm whether to include the optional ClearScope diagnostic audit outputs.
3. Confirm that missing `outputs/cache/phase3g_action_validation/THEIA_E3/`
   should be recorded as a regeneration requirement rather than blocking the
   backup.
4. Confirm that missing THEIA `target_case_summary.csv` should be recorded as
   architecture-specific/not available rather than blocking the backup.

## Safety Notes

- Copying preserves result artifacts only; it does not rerun inference.
- Ground truth is backed up as reference data only. It must remain post-stream
  evaluation input, not runtime scoring or thresholding input.
- The ClearScope rule document preserves leakage-safe tokenizer behavior and
  calls out where diagnostic audits used ground truth after inventory.
