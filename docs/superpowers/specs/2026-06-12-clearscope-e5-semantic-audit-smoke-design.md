# ClearScope E5 Semantic Audit And Smoke Design

## Goal

Validate whether the current ClearScope E3 semantic detail rules can safely transfer
to `CLEARSCOPE_E5` before running full E5 experiments.

The next stage uses a two-phase gate:

1. Run an E5 PostgreSQL semantic audit smoke.
2. If the audit is acceptable, prepare an independent E5 bounded pipeline smoke.

The audit uses a label-free primary analysis plus a label-aware diagnostic appendix.
Labels may help observe collision risk, but must not define runtime semantic rules,
thresholds, rescue strings, or policy shortcuts.

## Current Context

`CLEARSCOPE_E5` is configured with:

- database: `clearscope_e5`
- train splits: `day_8`, `day_9`
- validation split: `day_11`
- test splits: `day_14`, `day_15`, `day_17`
- unused splits: `day_10`, `day_12`, `day_13`, `day_16`

The PostgreSQL database has four source tables:

- `event_table`
- `file_node_table`
- `netflow_node_table`
- `subject_node_table`

Observed table sizes:

- `event_table`: `198794211`
- `file_node_table`: `324210`
- `netflow_node_table`: `104022`
- `subject_node_table`: `73097`

The existing E3 runner is not safe to reuse directly for E5 because it is restricted
to `CLEARSCOPE_E3` and hardcodes `CLEARSCOPE_E3_*` artifact names. E5 smoke must
use E5-specific artifact names and cache roots.

## Non-Goals

This stage does not:

- run full E5 inference;
- train full E5 Word2Vec, Phase3E, or Phase3G artifacts;
- change ClearScope semantic rules before audit evidence exists;
- use labels to choose thresholds, package-specific rescue rules, or online policy;
- reuse `CLEARSCOPE_E3_*` checkpoints, Word2Vec models, or caches for E5.

## Safety Rules

The primary audit is label-free. It may use:

- split membership;
- event operation;
- source and destination node type;
- raw file path, subject path, subject command, and netflow fields;
- current tokenizer output;
- token support, collision, unknown-bucket, and OOV statistics.

The diagnostic appendix may use E5 ground-truth nodes and attack windows only to
observe risk. It must not create rules that depend on:

- specific malicious node IDs;
- specific attack package names;
- specific attack paths;
- attack-window timing at runtime;
- label-derived score thresholds;
- label-derived alert policies.

Any proposed semantic rule must be a general Android path or command shape rule,
such as package path normalization, cache/hash compression, timestamp compression,
or generic device/socket categorization.

## Phase 1: E5 Semantic Audit Smoke

Create a bounded read-only audit that samples and summarizes E5 semantic detail
from PostgreSQL. The audit should be reproducible and should write small reports
under `tmp/` or another generated-output location.

The audit should compute:

- node counts by node type and split;
- operation counts by split;
- raw file path, subject path, subject command, and netflow shape distributions;
- tokenizer output token distributions;
- tuple distributions for operation, source type, destination type, source detail,
  and destination detail;
- train to validation/test OOV rates;
- token support buckets across splits;
- collision groups where many raw details collapse into the same token;
- high-rate fallback buckets, including `file_other`, `dev_other`, `socket_other`,
  `unknown_file`, and coarse `netflow` tokens;
- E3-v31/v3c transfer risks, especially where E5 has new Android package/cache,
  device, socket, or temporary-file shapes.

The audit should avoid scanning all `198M` events when a bounded sample or indexed
aggregation is enough. If a query would be expensive, it should support limits by
event count, day range, or per-operation sampling.

## Phase 2: Label-Aware Diagnostic Appendix

After the label-free report is generated, add a separate appendix using:

- `ground_truth/E5-CLEARSCOPE/node_clearscope_e5_appstarter_0515.csv`
- `ground_truth/E5-CLEARSCOPE/node_clearscope_e5_lockwatch_0517.csv`
- `ground_truth/E5-CLEARSCOPE/node_clearscope_e5_tester_0517.csv`
- E5 attack windows from dataset config

The appendix should answer:

- whether malicious nodes collapse into high-frequency benign buckets;
- whether malicious raw details are represented by generic fallback tokens;
- whether malicious event tuples are present in train/validation support or are
  mostly OOV;
- whether coarse buckets such as `netflow`, `file_other`, `dev_other`, and
  `socket_other` hide meaningful E5 distinctions;
- whether candidate general shape rules would reduce collision without using
  attack-specific strings.

The appendix is diagnostic only. It can recommend a general rule for later review,
but cannot make that rule acceptable by itself.

## Phase 3: Independent E5 Pipeline Smoke Gate

Proceed to E5 bounded pipeline smoke only if Phase 1 and Phase 2 show no blocking
semantic mismatch.

The E5 smoke must:

- use `DATASET=CLEARSCOPE_E5` or an E5-specific runner path;
- write E5-specific output tags;
- use E5-specific Word2Vec, Phase3E, Phase3G, node embedding, action embedding,
  event index, action validation, and endpoint suppression caches;
- fail fast if an artifact path contains `CLEARSCOPE_E3`;
- run with bounded limits, for example `MAX_TRAIN_EVENTS=200000`,
  `MAX_REF_EVENTS=50000`, and `MAX_TEST_EVENTS=100000`;
- verify `config_effective.yaml`;
- verify `online_event_alerts.csv` does not contain label or malicious fields;
- avoid launching full inference automatically.

If E5 artifacts do not exist, the implementation plan should define the smallest
bounded artifact build needed for smoke instead of falling back to E3 artifacts.

## Acceptance Criteria

The stage is complete when:

- a reproducible E5 audit report is generated;
- the report separates label-free findings from label-aware diagnostics;
- fallback bucket and collision risks are explicitly listed;
- any proposed semantic changes are general shape rules, not attack-specific rules;
- the decision to proceed or not proceed to E5 pipeline smoke is documented;
- if pipeline smoke proceeds, it uses only E5-specific artifacts and bounded event
  limits;
- streaming inference, no leakage, explainability, and reproducibility constraints
  remain preserved.

## Expected Decision Outcomes

The audit can end in one of three decisions:

1. E5 can reuse current v31/v3c semantics for bounded pipeline smoke.
2. E5 needs a small general tokenizer adjustment before pipeline smoke.
3. E5 has too much semantic mismatch, so pipeline smoke should wait for a separate
   E5 semantic design.

Full E5 experiments are out of scope for this spec.
