# ClearScope E3 v3 FP Reduction Design

## Goal

Reduce ClearScope E3 v3 false positives while preserving streaming online
inference, no label leakage, label-free online alert outputs, v3 fingerprint
isolation, and post-inference-only use of ground truth.

The frozen baseline is:

- Result directory:
  `outputs/results/tflr_light/CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_INFER_FULL`
- Event alerts: `415,263`
- False positives: `410,006`
- Precision: `1.27%`
- Malicious node recall: `2/41 = 4.88%`
- Phase3G state model: `s4d_complex_node`
- Score head: `conditional_action_semantic`
- Threshold mode: global `quantile`
- Final threshold: `0.043265`
- Endpoint suppression: disabled
- Action policy: `default`

The baseline result directory must not be overwritten or modified.

## Scope

This project uses a gated, evidence-driven sequence:

1. Freeze and summarize the existing baseline.
2. Add ClearScope v3 runner/config support for policy experiments without
   changing the baseline defaults.
3. Add a default-off `both_cold` unseen-group alert gate.
4. Run inference-only full comparisons using existing v3 artifacts.
5. Continue to EMA, Phase3G retraining, or semantic v3.1 only when result gates
   justify them.

The first implementation round must not retrain models unless the inference-only
results show that retraining is needed.

## Architecture

The ClearScope v3 runner remains the orchestration boundary. It will expose
policy knobs through environment variables and pass them into
`scripts.pipeline.entrypoints.conditional_e4`. Existing defaults remain the
current baseline behavior so historical results stay reproducible.

The `both_cold` unseen-group gate is implemented in the Phase3G conditional
inference path, not in post-evaluation. It is label-free and depends only on the
runtime target case and validation-derived threshold metadata.

Endpoint suppression remains the existing Phase3G endpoint suppression mechanism.
This work only makes it runnable from the ClearScope v3 runner and verifies that
its cache fingerprint and metadata are written.

## Runner Requirements

Modify `scripts/run/run_clearscope_e3_v3_phase3e_phase3g_full.sh` to support:

- `OUT_TAG_OVERRIDE`
- `ACTION_TYPE_ALERT_POLICY`
- `CONDITIONAL_ENDPOINT_AWARE_SUPPRESSION`
- `CONDITIONAL_ENDPOINT_SUPPRESSION_READ`
- `CONDITIONAL_ENDPOINT_SUPPRESSION_MODE`
- `CONDITIONAL_LOW_SUPPORT_POLICY`
- `CONDITIONAL_LOW_SUPPORT_MARGIN`
- `SSPM_STATE_MODEL`
- `SSPM_CONDITIONAL_HEAD_ARCH`

Default values must preserve the existing baseline:

- `EVENT_THRESHOLD_MODE=quantile`
- `ACTION_TYPE_ALERT_POLICY=default`
- `CONDITIONAL_ENDPOINT_AWARE_SUPPRESSION=false`
- `CONDITIONAL_ENDPOINT_SUPPRESSION_READ=false`
- `CONDITIONAL_ENDPOINT_SUPPRESSION_MODE=pair_only`
- `CONDITIONAL_LOW_SUPPORT_POLICY=conservative_max`
- `CONDITIONAL_LOW_SUPPORT_MARGIN=0.05`
- `SSPM_STATE_MODEL=s4d_complex_node`
- `SSPM_CONDITIONAL_HEAD_ARCH=shared_lowrank_v1`

`OUT_TAG_OVERRIDE` is allowed for single-stage runs and must be used for new full
inference outputs. The baseline out tag must never be reused for a policy
experiment.

## Both Cold Gate

Add a new default-off config:

```text
conditional_both_cold_unseen_policy=alert
```

Supported values:

- `alert`: current behavior.
- `observation_only_no_alert`: suppress online alert emission for
  `both_cold_action_target` events whose resolved threshold metadata has
  `validation_group_count == 0` and `threshold_level == unseen_group_extreme`.

When the gate suppresses an event:

- Do not write it to `online_event_alerts.csv`.
- Do count it in score summaries and policy diagnostics.
- Write policy state to `score_summary.json`.
- Do not read labels or malicious-node information.

The gate only applies to `both_cold_action_target`; it must not suppress
`event_semantic_target`.

## First Full Runs

Run 1: group-threshold inference only.

```text
OUT_TAG_OVERRIDE=CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_INFER_GROUPQ0999
STAGE=infer_full
EVENT_THRESHOLD_MODE=conditional_target_action_type_group_quantile
EVENT_THRESHOLD_QUANTILE=0.999
CONDITIONAL_GROUP_MIN_COUNT=1000
CONDITIONAL_LOW_SUPPORT_POLICY=conservative_max
CONDITIONAL_LOW_SUPPORT_MARGIN=0.05
CONDITIONAL_BOTH_COLD_UNSEEN_POLICY=observation_only_no_alert
```

Run 2: group threshold plus endpoint suppression.

```text
OUT_TAG_OVERRIDE=CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_INFER_GROUPQ0999_ENDPOINT
STAGE=infer_full
EVENT_THRESHOLD_MODE=conditional_target_action_type_group_quantile
EVENT_THRESHOLD_QUANTILE=0.999
CONDITIONAL_GROUP_MIN_COUNT=1000
CONDITIONAL_LOW_SUPPORT_POLICY=conservative_max
CONDITIONAL_LOW_SUPPORT_MARGIN=0.05
CONDITIONAL_BOTH_COLD_UNSEEN_POLICY=observation_only_no_alert
CONDITIONAL_ENDPOINT_AWARE_SUPPRESSION=true
CONDITIONAL_ENDPOINT_SUPPRESSION_MODE=pair_only
CONDITIONAL_ENDPOINT_SUPPRESSION_READ=true
```

Each full inference run has a six-hour limit. After each full run, stop and
report metrics before starting the next long stage.

## Trigger Gates

After Run 1:

- If FP is below `100,000`, continue to Run 2 but do not retrain yet.
- If FP remains above `100,000`, inspect group summaries before Run 2. Continue
  to Run 2 if `EVENT_READ netflow->process` or other endpoint-heavy groups still
  dominate.

After Run 2:

- If FP remains above `100,000` or `EVENT_READ netflow->process` still dominates,
  run an EMA baseline.
- If `both_cold_action_target` still produces large alert volume, start Phase3G
  target-case-head or both-cold-weighting retraining.
- If FP/TP token audit shows concrete semantic tuple collision, design and run
  semantic v3.1.

## EMA Baseline

EMA baseline is a comparison, not a default replacement:

```text
SSPM_STATE_MODEL=ema_fixed
SSPM_SCORE_HEAD=conditional_action_semantic
EVENT_THRESHOLD_MODE=conditional_target_action_type_group_quantile
```

The baseline answers whether ClearScope Android streams are over-sensitive under
S4D state dynamics. It must use separate output tags and metadata.

## Phase3G Retraining

If retraining is triggered, use a new checkpoint/tag and do not overwrite the
existing shared head. Allowed directions:

- Switch to or add a target-case separated head such as
  `dual_lowrank_by_target_case_v2`.
- Add both-cold weighting or sampling if needed.

Training remains label-free. Target case is runtime-derived and may be used;
ground truth labels must not be used for training, calibration, thresholding, or
alert decisions.

## Semantic v3.1

Semantic v3.1 is only allowed after audit evidence shows token tuple collision
is still material after method fixes.

Allowed endpoint class token candidates:

- `netflow public_external`
- `netflow private`
- `netflow loopback`
- `netflow unknown`
- Optional stable service class tokens such as `service_http`,
  `service_https`, `service_dns`, and `service_other`

Forbidden:

- Exact IP tokens.
- Exact Firefox cache hashes.
- Attack package rescue tokens.
- Label-derived shortcuts.

Any v3.1 semantic change requires a new semantic mode, new Word2Vec training, new
Phase3E cache, new fingerprints, and no reuse of v3 Word2Vec.

## Verification

Unit and syntax checks:

- `bash -n scripts/run/run_clearscope_e3_v3_phase3e_phase3g_full.sh`
- `python3 -m compileall cs4m scripts tests`
- Targeted tests for runner/config argument handling.
- Targeted tests for `both_cold` unseen no-alert behavior.

Output checks after each run:

- `score_summary.json` writes semantic mode, threshold mode, state model, and
  both-cold policy.
- `online_event_alerts.csv` has no label, ground-truth, malicious, or attack
  columns.
- `conditional_score_summary_by_target_action_type.csv` shows group threshold
  behavior for `conditional_target_action_type_group_quantile`.
- Endpoint suppression run writes suppression metadata and cache fingerprint.
- Evaluation confirms ground truth is used only after streaming inference.

## Reporting

After each full run, report:

- Event alert count.
- TP and FP.
- Precision.
- Malicious node recall.
- `EVENT_READ netflow->process` FP count.
- `both_cold_action_target` alert count and FP count.
- Whether endpoint suppression was enabled and how many alerts it suppressed.
- Leakage and label-free output checks.
- Whether the next trigger gate is met.

Generated models, caches, and result directories must not be committed. Commit
only source, tests, docs, and small metadata summaries when explicitly useful.
