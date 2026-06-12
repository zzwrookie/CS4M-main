# ClearScope E5 V33 Android-Safe Semantics Design

## Goal

Add the smallest general ClearScope Android tokenizer correction needed before
`CLEARSCOPE_E5` pipeline smoke planning.

The new semantic mode is:

```text
raw_detail_v33_e5_android_safe
```

It inherits the current v31 discriminative ClearScope behavior and only adds
general Android shape handling for E5 audit risks found in the previous
`clearscope_e5` audit.

This stage stops after:

1. implementing the tokenizer mode;
2. rerunning the E5 semantic audit smoke;
3. comparing v31 and v33 audit outputs;
4. deciding whether E5 pipeline smoke can be planned later.

It does not build E5 artifacts, modify E5 runners, run pipeline smoke, or run full
inference.

## Current Evidence

The bounded E5 audit smoke produced:

- `input_event_count`: `150000`
- `event_count`: `150000`
- `skipped_event_count`: `0`
- `usable_event_rate`: `1.0`
- `fallback_counts`:
  - `file_other`: `1066`
  - `dev_other`: `44`
  - `netflow`: `20000`
- `malicious_node_count`: `8`
- `malicious_fallback_counts`:
  - `file_other`: `3`
  - `dev_other`: `1`

The label-aware diagnostic appendix showed these malicious paths collapse to the
same `other` detail token:

```text
/config/sdcardfs/com.android.providers.contacts/appid
/config/sdcardfs/com.bloketech.lockwatch/appid
/config/sdcardfs/de.belu.appstarter/appid
```

This is a general Android path-shape issue, not an attack-specific rule request.
The runtime-visible path form is:

```text
/config/sdcardfs/<package>/appid
```

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
- reuse `CLEARSCOPE_E3_*` artifacts for E5.

## Safety Rules

The tokenizer changes must be general Android shape rules. They must not depend
on:

- malicious node IDs;
- attack windows at runtime;
- hardcoded attack package names;
- hardcoded malicious file paths;
- label-derived score thresholds;
- label-derived rescue rules.

Label-aware data can only be used to verify whether the general shape rule fixes
the previously observed collision risk. The label-free report remains the primary
source for go/no-go decisions.

## Semantic Mode Behavior

`raw_detail_v33_e5_android_safe` must normalize through the existing ClearScope
semantic-mode normalization path and must not change existing v31 outputs unless
one of the new Android-safe shapes applies.

### `/config/sdcardfs/<package>/appid`

Add a file class/detail for Android sdcardfs app-id records.

Examples:

```text
/config/sdcardfs/com.android.providers.contacts/appid
-> file android_sdcardfs_appid com_android_providers_contacts_appid

/config/sdcardfs/de.belu.appstarter/appid
-> file android_sdcardfs_appid de_belu_appstarter_appid

/config/sdcardfs/com.bloketech.lockwatch/appid
-> file android_sdcardfs_appid com_bloketech_lockwatch_appid
```

Package names are runtime-visible Android identifiers. Keeping package-family
detail is allowed because the rule applies to every package and does not single
out attack packages.

### `/config/sdcardfs/<package>`

Add a matching directory-level class/detail so the package directory and its
`appid` file do not split inconsistently.

Example:

```text
/config/sdcardfs/de.belu.appstarter
-> file android_sdcardfs_package de_belu_appstarter
```

### `/dev/*`

Keep unknown device names coarse. Only allow stable Android device names that are
general platform shapes:

```text
/dev/pmsg0 -> dev_pmsg0
/dev/ashmem -> dev_ashmem
/dev/ion -> dev_ion
/dev/binder -> dev_binder
/dev/hwbinder -> dev_hwbinder
/dev/vndbinder -> dev_vndbinder
/dev/null -> dev_null
/dev/zero -> dev_zero
```

Unknown device names remain:

```text
/dev/<unknown> -> dev_other
```

The initial test must assert:

```text
/dev/msm_g711tlaw -> dev_other
```

This prevents accidental raw device expansion from creating high-cardinality or
label-derived detail.

### Netflow

Do not change netflow behavior. The current coarse `netflow` token is preserved
until a separate label-free audit proves endpoint detail reduces collision without
creating unacceptable OOV or overlap.

## Audit Rerun

After implementation, rerun E5 audit smoke with:

```bash
CLAD_DB_HOST=127.0.0.1 \
CLAD_DB_PORT=5433 \
CLAD_DB_USER=postgres \
CLAD_DB_PASSWORD=123456 \
python3 scripts/tools/audit_clearscope_e5_semantic_smoke.py \
  --semantic_mode raw_detail_v33_e5_android_safe \
  --max_nodes_per_type 20000 \
  --max_events_per_split 50000 \
  --output_dir tmp/clearscope_e5_v33_semantic_audit_smoke
```

Compare against the current v31 audit:

```text
tmp/clearscope_e5_semantic_audit_smoke
tmp/clearscope_e5_v33_semantic_audit_smoke
```

The comparison must include:

- skipped event count and usable event rate;
- label-free fallback counts;
- malicious fallback counts from the diagnostic appendix;
- top collision rows;
- validation/test OOV tuple counts;
- a package/hash explosion check using top detail and collision rows.

## Go/No-Go Decision

This stage ends with one of:

```text
v33_audit_pass_pipeline_smoke_can_be_planned
v33_needs_more_general_tokenizer_adjustment
v33_blocks_e5_pipeline_smoke
```

The audit passes only if:

- `skipped_event_count = 0`;
- `usable_event_rate >= 0.95`;
- malicious `file_other` drops from `3` to `0`, or any remaining item is clearly
  explained without a label-derived rule;
- malicious `dev_other` does not increase;
- label-free `file_other` drops from `1066` to less than `800`;
- no raw package/hash explosion appears in top detail or collision rows;
- `test_oov_tuple_count` does not materially worsen;
- label-free summary still has no top-level `malicious` or `attack` keys;
- threshold, policy, node-pool, evaluation, and inference code remain unchanged.

If the audit passes, the next stage is a separate E5 bounded artifact and pipeline
smoke design. That future stage must still avoid reusing `CLEARSCOPE_E3_*`
artifacts.

## Acceptance Criteria

The stage is complete when:

- the new semantic mode is registered and unit-tested;
- v31 behavior is unchanged for non-targeted paths;
- sdcardfs package and appid shapes produce package-family detail tokens;
- stable allowlisted `/dev/*` paths produce specific device detail tokens;
- unknown `/dev/*` paths remain `dev_other`;
- E5 audit smoke runs with `raw_detail_v33_e5_android_safe`;
- v31/v33 audit comparison is reported;
- a go/no-go decision is documented;
- no pipeline smoke or full inference is launched.
