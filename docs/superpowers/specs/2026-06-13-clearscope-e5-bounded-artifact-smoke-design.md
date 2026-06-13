# ClearScope E5 Bounded Artifact Smoke Design

## Goal

Build and run a real bounded ClearScope E5 artifact and pipeline smoke using the
approved E5 v33b semantics:

```text
raw_detail_v33b_e5_android_safe
```

This stage creates independent E5 smoke artifacts and runs bounded online
inference. It is not a full experiment and does not optimize final detection
metrics.

The stage covers:

1. E5 residual Word2Vec artifact contract;
2. E5 Phase3E artifact contract;
3. E5 Phase3G action-head and validation-cache contract;
4. E5 bounded pipeline smoke with online event alerts;
5. a go/no-go decision for later E5 full-run planning.

## Current Evidence

The `raw_detail_v33b_e5_android_safe` audit passed the semantic go gate:

```text
input_event_count: 150000
event_count: 150000
skipped_event_count: 0
usable_event_rate: 1.0
label-free file_other: 480
label-free dev_other: 50
netflow: 20000
malicious file_other: 0
malicious dev_other: 1
test_oov_tuple_count: 39
malicious collision groups: 0
```

The current repository has no committed or discovered `CLEARSCOPE_E5` pipeline
artifacts under `outputs/`. Existing ClearScope runner support is E3-specific,
including `scripts/run/run_clearscope_e3_v3_phase3e_phase3g_full.sh`, which is
restricted to `CLEARSCOPE_E3`.

## Design Choice

Use a new independent E5 bounded runner instead of generalizing the existing E3
runner.

The new runner should be named:

```text
scripts/run/run_clearscope_e5_v33b_bounded_phase3e_phase3g_smoke.sh
```

It must hard-code or defensively validate:

```text
DATASET=CLEARSCOPE_E5
SEMANTIC_MODE=raw_detail_v33b_e5_android_safe
```

It must not accept or silently normalize E3 defaults. Any path or artifact name
containing `CLEARSCOPE_E3` is a smoke blocker.

## Bounded Defaults

The runner uses these defaults unless the caller explicitly overrides them for a
smaller bounded smoke:

```text
MAX_TRAIN_EVENTS=50000
MAX_REF_EVENTS=50000
MAX_TEST_EVENTS=50000
SSPM_EPOCHS=1
SSPM_CONDITIONAL_MAX_EPOCHS=3
```

The defaults are intended to verify artifact contracts, cache routing, online
inference, and post-inference evaluation without approaching a full run.

## Artifact Names And Directories

All generated E5 smoke artifacts must include the E5/v33b/bounded identity.

Recommended names:

```text
CLEARSCOPE_E5_V33B_BOUNDED_WORD2VEC
CLEARSCOPE_E5_V33B_BOUNDED_PHASE3E
CLEARSCOPE_E5_V33B_BOUNDED_PHASE3G
CLEARSCOPE_E5_V33B_BOUNDED_PIPELINE_SMOKE
```

Recommended directories, aligned with existing cache roots:

```text
outputs/models/residual_word2vec/
outputs/cache/phase3e/node_embeddings/CLEARSCOPE_E5_v33b_bounded_latent64/
outputs/cache/phase3e/action_embeddings/CLEARSCOPE_E5_v33b_bounded_latent64/
outputs/cache/phase3e/event_indices/CLEARSCOPE_E5_v33b_bounded/
outputs/cache/phase3e/compact_used_node_embeddings/CLEARSCOPE_E5_v33b_bounded/
outputs/cache/phase3g_action_validation/CLEARSCOPE_E5_v33b_bounded/
outputs/cache/phase3g_endpoint_suppression/CLEARSCOPE_E5_v33b_bounded/
outputs/models/sspm_phase3e/
outputs/models/phase3g_action_heads/
outputs/results/tflr_light/CLEARSCOPE_E5_V33B_BOUNDED_PIPELINE_SMOKE/
```

The exact file names should follow existing runner conventions, with E5/v33b
substituted for E3 mode labels. They must not overwrite or read
`CLEARSCOPE_E3_*` artifacts.

## Stages

The runner supports the same high-level stages as the existing ClearScope E3
runner, plus the missing Word2Vec bootstrap stage needed for independent E5
artifacts:

```text
train_word2vec
build_phase3e_artifacts
train_phase3e_base
train_phase3g_head
infer_smoke
all
```

`all` runs the stages in the listed order.

### Train Word2Vec

Purpose:

- train a bounded E5 residual Word2Vec embedder from the E5 train split only;
- use `raw_detail_v33b_e5_android_safe`;
- write E5/v33b/bounded model and metadata under `outputs/models/residual_word2vec/`;
- avoid validation, test, ground-truth labels, attack windows, and E3 artifacts.

Acceptance:

- the Word2Vec pickle exists;
- the JSON sidecar exists;
- metadata contains `CLEARSCOPE_E5`;
- metadata contains `raw_detail_v33b_e5_android_safe`;
- metadata records the train-only leakage contract;
- vocabulary size is greater than `0`;
- no model path, sidecar, log, or summary references `CLEARSCOPE_E3`.

### Build Phase3E Artifacts

Purpose:

- build bounded E5 event indices and node/action embedding caches;
- use `raw_detail_v33b_e5_android_safe`;
- use the E5 bounded residual Word2Vec embedder;
- preserve event ordering and streaming-compatible cache metadata.

Acceptance:

- event-index metadata exists;
- node embeddings exist;
- action embeddings exist;
- compact used-node embeddings exist when required by lazy backing;
- cache metadata references `CLEARSCOPE_E5`;
- cache metadata references `raw_detail_v33b_e5_android_safe`;
- cache metadata does not reference `CLEARSCOPE_E3`.

### Train Phase3E Base

Purpose:

- train the bounded Phase3E base checkpoint from E5 train/reference data only;
- use the Phase3E memmaps from the E5 bounded artifact stage.

Acceptance:

- bounded Phase3E base checkpoint exists;
- checkpoint/config metadata contains `CLEARSCOPE_E5`;
- checkpoint/config metadata contains `raw_detail_v33b_e5_android_safe`;
- no E3 artifact path is referenced.

### Train Phase3G Head

Purpose:

- train the bounded Phase3G conditional/action head using validation-safe data;
- create validation caches under E5-specific directories.

Acceptance:

- bounded Phase3G head checkpoint exists;
- action validation cache exists;
- endpoint suppression cache exists if required by the selected options;
- cache/checkpoint metadata points to E5 bounded artifacts;
- no test labels, attack windows, or full-test rankings are used.

### Infer Smoke

Purpose:

- load E5 bounded Word2Vec, Phase3E, Phase3G, and cache artifacts;
- run bounded online inference over the E5 test stream;
- write online alerts and post-inference evaluation outputs.

Acceptance:

- process exits with status `0`;
- result directory exists;
- `online_event_alerts.csv` exists;
- alerts include event identity, score, threshold or threshold basis, and entity
  fields when alerts are emitted;
- metrics/evaluation outputs are generated only after online inference output;
- no missing-cache fallback reads E3 artifacts;
- no output manifest, config, or log references `CLEARSCOPE_E3`.

## Safety Boundaries

This stage must preserve:

- streaming online inference;
- no test-label leakage;
- explainable alert outputs;
- reproducible artifact provenance.

Rules:

- train split can train Word2Vec and model artifacts;
- validation/reference split can calibrate validation-only thresholds and caches;
- test split can only be used for online inference followed by evaluation;
- ground truth can only be loaded after inference output exists;
- test labels, attack windows, test rankings, and full-test statistics cannot
  affect training, tokenization, thresholds, policy, or artifact selection.

## Non-Goals

This stage does not:

- run a full ClearScope E5 experiment;
- optimize detection metrics;
- modify tokenizer rules;
- modify threshold, alert policy, evaluation, or label matching;
- reuse `CLEARSCOPE_E3_*` artifacts;
- silently fall back to E3 caches or checkpoints;
- change historical E3 runners.

## Required Preflight Checks

The runner must fail before real execution when:

- `DATASET` is not `CLEARSCOPE_E5`;
- `SEMANTIC_MODE` is not `raw_detail_v33b_e5_android_safe`;
- `CLAD_DB_PASSWORD` is missing for non-dry-run execution;
- an artifact path that will be read contains `CLEARSCOPE_E3`;
- required upstream artifacts are missing for the requested stage;
- `OUT_TAG_OVERRIDE` is unsafe or would mask baseline output protection.

Dry runs must print commands without creating artifacts.

## Go/No-Go Decision

The stage passes only if:

```text
E5_bounded_pipeline_smoke_passed
```

with all of:

- all bounded stages exit `0`;
- E5 bounded artifacts exist;
- result output exists;
- `online_event_alerts.csv` exists;
- metadata/manifests/logs contain `CLEARSCOPE_E5` and v33b identity;
- metadata/manifests/logs do not contain `CLEARSCOPE_E3`;
- no threshold, policy, tokenizer, or evaluation logic was changed.

Failure categories:

1. artifact missing: fix runner/config path only;
2. semantic mode missing: fix propagation or metadata validation;
3. cache fingerprint mismatch: fix metadata/fingerprint before rerunning;
4. OOV or vocabulary too small: return to label-free train corpus audit;
5. evaluation mismatch: review evaluation separately, without changing tokenizer
   or smoke artifacts.

## Acceptance Criteria

The stage is complete when:

- the E5 bounded runner is implemented and syntax-tested;
- dry-run command output shows E5/v33b paths and no E3 artifact reads;
- the real bounded smoke runs with `50k/50k/50k` default limits;
- generated artifacts and outputs are checked;
- a go/no-go decision is reported;
- no full run is launched.
