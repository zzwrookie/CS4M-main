# CS4M Backup Execution Spec

Created: 2026-06-07

## Confirmed Requirements

- Copy directly into `../CS4M-main`, not under a `backups/` subdirectory.
- Preserve source-relative structure.
- Absolute `/tmp/...` sources become `../CS4M-main/tmp/...`.
- Relative repo sources become matching paths under `../CS4M-main/`.
- Existing same-name files may be overwritten/updated.
- Include `ground_truth/`.
- Include `docs/reproduction/CS4M_BACKUP_CONFIRMATION_MANIFEST_20260607.md`.
- Include `docs/method/CLEARSCOPE_E3_E5_RAW_DETAIL_V2_REFINED_RULE_BACKUP.md`.
- Include the optional ClearScope diagnostic audit directories from the manifest.

## Source List

The executable source list is:

```text
docs/reproduction/CS4M_BACKUP_SOURCE_PATHS_20260607.txt
```

Missing items are intentionally not in the source list:

- `outputs/cache/phase3g_action_validation/THEIA_E3/`
- `outputs/results/phase3g_theia_e3_e4_theia_v1_lazy100k_pair_only_full_rerun_20260602_142411/THEIA_E3_PHASE3E_E4_NONE/target_case_summary.csv`

They remain documented in:

```text
docs/reproduction/CS4M_BACKUP_CONFIRMATION_MANIFEST_20260607.md
```

## Execution Method

Use `rsync -aR` from `<HISTORICAL_SOURCE_REPO>`.

`-a` preserves directories, timestamps, symlinks, and file metadata where
possible. `-R` preserves source-relative path structure under `../CS4M-main`.

Planned dry-run command shape:

```bash
mapfile -t sources < docs/reproduction/CS4M_BACKUP_SOURCE_PATHS_20260607.txt
rsync -aR --dry-run --stats "${sources[@]}" ../CS4M-main/
```

Planned copy command shape:

```bash
mapfile -t sources < docs/reproduction/CS4M_BACKUP_SOURCE_PATHS_20260607.txt
rsync -aR --stats "${sources[@]}" ../CS4M-main/
```

## Verification

After copying, verify:

- both backup docs exist under `../CS4M-main/docs/`;
- `ground_truth/` exists under `../CS4M-main/`;
- CADETS result paths exist under `../CS4M-main<HISTORICAL_TMP_ROOT>/...`;
- THEIA result path exists under `../CS4M-main/outputs/results/...`;
- representative model/cache files exist under `../CS4M-main/outputs/...`;
- missing THEIA action-validation cache remains documented as a regeneration
  requirement.

## Safety

This backup copies artifacts only. It does not rerun training, inference, or
evaluation. It does not change tokenizer/runtime code. Ground truth is copied
for later post-stream evaluation and must not be used for runtime scoring,
thresholding, or filtering.
