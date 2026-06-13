# ClearScope E5 V33B Full Run Design

## Goal

Run a full ClearScope E5 pipeline using the approved E5 semantic mode:

```text
raw_detail_v33b_e5_android_safe
```

The run must produce independent full-run artifacts named with
`CLEARSCOPE_E5_V33B_FULL` and must not reuse E3 or bounded smoke artifacts.

## Starting Evidence

The preceding bounded smoke passed with independent E5 artifacts:

- Word2Vec trained on E5 train split only;
- Phase3E artifacts built with E5/v33b routing;
- Phase3E base and Phase3G head trained with bounded E5 data;
- bounded online inference produced `online_event_alerts.csv`;
- artifact and result manifests did not contain `CLEARSCOPE_E3`.

Bounded artifacts were only smoke evidence. Full run must regenerate artifacts
under full names and directories.

## Required Semantic And Safety Boundaries

Full run must preserve:

- streaming online inference;
- no test-label leakage;
- explainable alert outputs;
- reproducible artifact provenance.

Rules:

- `SEMANTIC_MODE` must be `raw_detail_v33b_e5_android_safe`;
- tokenizer rules are frozen;
- threshold, alert policy, evaluation, and label matching are frozen;
- train split can train Word2Vec and model artifacts;
- validation/reference split can calibrate validation-only thresholds and caches;
- test split can only be used for streaming inference followed by evaluation;
- ground truth can only be loaded after inference output exists;
- attack windows, test labels, test rankings, and full-test statistics cannot
  affect tokenization, thresholds, policy, or artifact selection.

## Artifact Names And Directories

All full artifacts must use full-run identities.

Word2Vec:

```text
outputs/models/residual_word2vec/CLEARSCOPE_E5_V33B_FULL_LATENT64_word2vec_window3.pkl
outputs/models/residual_word2vec/CLEARSCOPE_E5_V33B_FULL_LATENT64_word2vec_window3.json
outputs/models/residual_word2vec/CLEARSCOPE_E5_V33B_FULL_LATENT64/
```

Phase3E caches:

```text
outputs/cache/phase3e/node_embeddings/CLEARSCOPE_E5_v33b_full_latent64/
outputs/cache/phase3e/action_embeddings/CLEARSCOPE_E5_v33b_full_latent64/
outputs/cache/phase3e/event_indices/CLEARSCOPE_E5_v33b_full/
outputs/cache/phase3e/compact_used_node_embeddings/CLEARSCOPE_E5_v33b_full/
```

Phase3G validation caches:

```text
outputs/cache/phase3g_action_validation/CLEARSCOPE_E5_v33b_full/
outputs/cache/phase3g_endpoint_suppression/CLEARSCOPE_E5_v33b_full/
```

Checkpoints:

```text
outputs/models/sspm_phase3e/CLEARSCOPE_E5_V33B_FULL_PHASE3E_BASE.pkl
outputs/models/phase3g_action_heads/CLEARSCOPE_E5_V33B_FULL_PHASE3G_CONDITIONAL_HEAD.pkl
```

Results and logs:

```text
outputs/results/tflr_light/CLEARSCOPE_E5_V33B_FULL_PIPELINE/
logs/clearscope_e5_v33b_full/
```

No full-run read path or output manifest may reference:

```text
CLEARSCOPE_E3
CLEARSCOPE_E5_V33B_BOUNDED
CLEARSCOPE_E5_v33b_bounded
```

## Runner

Add:

```text
scripts/run/run_clearscope_e5_v33b_full_phase3e_phase3g.sh
```

Supported stages:

```text
train_word2vec
build_phase3e_artifacts
train_phase3e_base
train_phase3g_head
infer_full
all
```

`all` runs stages in that order.

## Full Defaults

Full run defaults:

```text
DATASET=CLEARSCOPE_E5
SEMANTIC_MODE=raw_detail_v33b_e5_android_safe
MAX_TRAIN_EVENTS=0
MAX_REF_EVENTS=0
MAX_TEST_EVENTS=0
WORD2VEC_EPOCHS=10
WORD2VEC_WORKERS=4
SSPM_EPOCHS=1
SSPM_CONDITIONAL_MAX_EPOCHS=80
```

`0` means no bounded event limit, following the existing runner convention.

Policy and evaluation defaults remain unchanged from the E5 bounded smoke
runner:

```text
EVENT_THRESHOLD_MODE=quantile
EVENT_THRESHOLD_QUANTILE=0.999
ACTION_TYPE_ALERT_POLICY=default
NODE_POOL_SCORE_MODE=base_conf
CONDITIONAL_LOW_SUPPORT_POLICY=conservative_max
CONDITIONAL_ENDPOINT_SUPPRESSION_MODE=pair_only
```

## Required Preflight Checks

The full runner must fail before execution when:

- `DATASET` is not `CLEARSCOPE_E5`;
- `SEMANTIC_MODE` is not `raw_detail_v33b_e5_android_safe`;
- `CLAD_DB_PASSWORD` is missing for non-dry-run execution;
- any read path contains `CLEARSCOPE_E3`;
- any read path contains `CLEARSCOPE_E5_V33B_BOUNDED`;
- any read path contains `CLEARSCOPE_E5_v33b_bounded`;
- `OUT_TAG_OVERRIDE` is set for a stage other than `infer_full`;
- `OUT_TAG_OVERRIDE` contains slash, backslash, or `..`;
- a requested downstream stage lacks its required full upstream artifact.

Dry runs must print commands without creating artifacts.

## Stage Acceptance

### Train Word2Vec

Acceptance:

- full Word2Vec pickle exists;
- full Word2Vec JSON sidecar exists;
- metadata contains `dataset_id=CLEARSCOPE_E5`;
- metadata contains `semantic_mode=raw_detail_v33b_e5_android_safe`;
- metadata records train-only leakage contract;
- train vocabulary size is greater than `0`;
- metadata and logs contain no E3 or bounded artifact references.

### Build Phase3E Artifacts

Acceptance:

- full event index metadata exists;
- full node embeddings exist;
- full action embeddings exist;
- full compact used-node cache exists when required;
- metadata contains E5/v33b identity;
- metadata contains no E3 or bounded artifact references.

### Train Phase3E Base

Acceptance:

- full Phase3E base checkpoint exists;
- training uses full E5 train/reference data according to runner defaults;
- output metadata contains no E3 or bounded artifact references.

### Train Phase3G Head

Acceptance:

- full Phase3G conditional head checkpoint exists;
- full action validation cache exists;
- full endpoint suppression cache exists if required by selected options;
- validation cache is validation-only;
- output metadata contains no E3 or bounded artifact references.

### Infer Full

Acceptance:

- full inference exits `0`;
- `online_event_alerts.csv` exists;
- `metrics.json` exists;
- `eval_causal_semantics_slim.json` exists;
- alert CSV contains event, score, threshold, and entity-location fields;
- evaluation flags show labels are only attached after streaming;
- output metadata contains no E3 or bounded artifact references.

## Go/No-Go

Full run passes only if:

```text
E5_v33b_full_pipeline_passed
```

with all of:

- all stages exit `0`;
- all full artifacts exist;
- all result outputs exist;
- full output manifests contain E5/v33b identity;
- full output manifests do not contain E3 or bounded identities;
- no tokenizer, threshold, policy, evaluation, or label logic changed.

Failure handling:

- artifact missing: fix runner path or preflight only;
- semantic mismatch: fix semantic propagation, not tokenizer rules;
- cache fingerprint mismatch: fix metadata/fingerprint before rerun;
- OOM or throughput issue: reduce workers or batch size, not semantics/policy;
- poor metric: record it, do not tune tokenizer/policy during full run;
- evaluation mismatch: review evaluation separately.
