# ClearScope E3/E5 Raw Detail V2 Refined Rule Backup

Created: 2026-06-07

## Scope

This note backs up the latest ClearScope semantic raw-token splitting rules for
`CLEARSCOPE_E3` and `CLEARSCOPE_E5`.

The runtime source of truth is `cs4m/clearscope_semantics.py`. ClearScope
E3 and E5 both enter the same `is_clearscope_dataset()` path, so the tokenizer
rules are shared across both datasets. The canonical active mode is:

```text
raw_detail_v2_refined
```

Legacy mode is available only when explicitly requested as
`clearscope_android_semantics_v1` or `legacy`. Empty/default mode resolves to
`raw_detail_v2_refined`.

## Leakage Contract

Runtime tokenization uses only event-visible fields:

- process command/package string;
- file path;
- netflow endpoint fields when audit mode explicitly asks for them;
- event action.

Ground truth is not used during tokenization, model training, thresholding,
alert generation, or filtering. Some diagnostic audits listed below used
ground truth only after streaming/inventory construction to measure overlap.

## Runtime Token Shape

Node token tuples are natural semantic tuples:

```text
process <android_process_label> <process_detail>
file <android_file_label> <file_detail>
netflow
```

Event residual sentences use:

```text
<src_node_tokens> event event_<normalized_action> <dst_node_tokens>
```

For Phase3E/Phase3G, node embeddings are built from these node token tuples.
The model fingerprint records `clearscope_semantic_rules_version` as the
normalized semantic mode, so cache/model reuse is blocked when the semantic
mode changes.

## Process Rules

`classify_android_cmd_nll_refined()` assigns a coarse Android process label:

- `providers` for `com.android.providers.*`
- `com_android` for `com.android.*`
- `android_process` for `android.process.*`
- `org_mozilla` for `org.mozilla.*`
- `com_google` for `com.google.*`
- `com_other` for other `com.*`
- `package_other` for other package-like names
- `native_process` for native/non-package names

`android_cmd_detail_refined()` then keeps the meaningful package suffix and
preserves process suffixes after `:`. Examples:

```text
com.android.settings:CryptKeeper -> process com_android settings_cryptkeeper
com.android.email -> process com_android email
org.mozilla.fennec_firefox_dev -> process org_mozilla fennec_firefox_dev
android.process.acore -> process android_process acore
```

This keeps Android package families visible without retaining full package
strings as independent high-cardinality surfaces.

## File Coarse Labels

`classify_android_file_nll()` assigns the coarse file label before detail
selection. Important label families are:

- `android_apk_file` for `.apk`
- `native_library_file` for `.so`
- `android_app_cache_file` for `/data/data/<package>/cache/...`
- `android_app_private_file` for `/data/data/<package>/...`
- `android_external_app_cache_file` for external app cache paths
- `android_user_media_file` for DCIM/media paths
- `android_user_storage_file` for other `/storage/emulated/0/...`
- `android_tmp_file`, `android_app_install_file`, `android_dalvik_cache_file`
- `android_misc_file`, `android_backup_file`, `android_tombstone_file`
- `android_recent_task_file`, `android_recent_image_file`
- `android_system_dropbox_file`, `android_procstats_file`
- `android_usagestats_file`, `android_user_state_file`
- `android_system_state_file`, `android_system_file`, `android_vendor_file`
- `proc_cmdline`, `proc_file`
- `android_socket_file`, `android_device_file`, `android_sysfs_file`
- `android_cache_file`, `android_ashmem_file`, `android_policy_file`
- `filesystem_root`, `android_root_dir`, `file_other`

## File Detail Rules

`android_file_detail_refined()` applies targeted Android refinements before
falling back to the legacy coarse-detail mapping.

### Proc Files

PIDs are removed. Stable proc suffixes remain:

```text
/proc/<pid>/cmdline -> file proc_cmdline cmdline
/proc/<pid>/net/xt_qtaguid/ctrl -> file proc_file net_xt_qtaguid_ctrl
/proc/<pid>/net/xt_qtaguid/iface_stat/... -> file proc_file net_xt_qtaguid_iface_stat
/proc/<pid>/net/tcp -> file proc_file net_tcp
/proc/<pid>/task/<tid>/maps -> file proc_file task_maps
```

### Device And Socket Files

Known device/socket names are preserved as semantic details:

```text
/dev/ashmem -> file android_device_file dev_ashmem
/dev/ion -> file android_device_file dev_ion
/dev/socket/logdw -> file android_socket_file socket_logdw
/dev/socket/lmkd -> file android_socket_file socket_lmkd
/dev/socket/fwmarkd -> file android_socket_file socket_fwmarkd
```

Unknown devices collapse to `dev_other`; unknown sockets collapse to
`socket_other`.

### APK And Install Paths

APK details distinguish system apps, privileged apps, temporary installs, and
known package install roles:

```text
/system/app/Camera2/Camera2.apk -> file android_apk_file system_app_camera2_apk
/system/priv-app/<app>/<app>.apk -> file android_apk_file system_priv_app_<app>_apk
/data/app/vmdl<digits>.tmp/base.apk -> file android_apk_file data_app_vmdl_base_apk
/data/app/com.metasploit.stage-*/base.apk -> file android_apk_file data_app_metasploit_stage_apk
/data/local/tmp/<mozilla-or-fennec>.apk -> file android_apk_file tmp_org_mozilla_apk
```

### System Files

System paths keep coarse functional areas instead of full path expansion:

```text
/system/media/audio/ui/... -> system_media_ui
/system/media/audio/notifications/... -> system_media_notifications
/system/fonts/... -> system_fonts
/system/etc/... -> system_etc
/system/framework/... -> system_framework
/system/bin/... -> system_bin
/system/lib/... -> system_lib
```

### App Private Files

App-private paths keep package family plus meaningful subdirectory where useful:

```text
/data/data/com.android.email/files/body/.../*.txt -> email_body_txt
/data/data/com.android.email/files/body/.../*.html -> email_body_html
/data/data/<package>/app_webview/... -> <family>_app_webview
/data/data/<package>/files/mozilla/... -> <family>_mozilla_profile
/data/data/<package>/shared_files/... -> <family>_shared_files
/data/data/<package>/databases/... -> <family>_databases
/data/data/<package>/shared_prefs/... -> <family>_shared_prefs
```

For `org.mozilla.fennec_firefox_dev`, the family is normalized to
`fennec_firefox_dev`.

### App Cache Files

High-cardinality browser/cache identities are shape-compressed:

```text
/data/data/org.mozilla.fennec_firefox_dev/cache/.../cache2/doomed/<digits>
  -> fennec_firefox_dev_doomed_num

/data/data/org.mozilla.fennec_firefox_dev/cache/.../cache2/entries/<hex>
  -> fennec_firefox_dev_entries_hexblob

/data/data/org.mozilla.fennec_firefox_dev/cache/.../safebrowsing/...
  -> fennec_firefox_dev_safebrowsing_cache

/data/data/com.android.camera2/cache/image_manager_disk_cache/<mixed-id>
  -> camera2_image_manager_disk_cache_mixedid

/data/data/com.android.browser/cache/... -> com_android_browser_cache
```

Other package caches use `<family>_cache`.

### Media And External Cache

Media and external cache paths are intentionally coarse:

```text
*.eml under external app cache -> email_eml
DCIM *.tc-md -> camera_metadata
DCIM *.jpg/*.jpeg/*.png -> camera_image
/data/system/recent_images/... -> thumbnail
```

Camera timestamps are not preserved in the current runtime tokenizer.

## Netflow Rules

Runtime refined ClearScope netflow tokens are fixed by default:

```text
netflow
```

The audit helper can optionally include source endpoint detail for measuring
overlap, but the E3 audit recommended keeping runtime netflow fixed because
source-detail tokens overlap benign and malicious references.

## E3 Collision Evidence

Primary audit output:

```text
<HISTORICAL_TMP_ROOT>/clearscope_detail_semantic_audit_v2_refined/
```

Summary:

- database: `clearscope_e3`
- event filter: `Orthrus10`
- semantic mode: `raw_detail_v2_refined`
- node inventory: 369101 nodes
- inventory by type: 76938 file, 279694 netflow, 12469 process
- unique refined tokens: 500
- unique refined token tuples: 468
- current v1 token tuples: 191
- train endpoint references: 23260197
- ground truth used for inventory: false
- ground truth used for netflow overlap: true
- recommendation: keep netflow source detail disabled

Top intentional collision patterns:

| Token tuple | Raw values | Nodes | Train refs | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `netflow` | 56142 | 279694 | 1711662 | fixed endpoint bucket |
| `file proc_cmdline cmdline` | 20227 | 20227 | 21647 | PID removed |
| `file proc_file stat` | 15600 | 15600 | 12 | PID removed |
| `file android_app_cache_file fennec_firefox_dev_doomed_num` | 10955 | 10955 | 8 | numeric cache IDs compressed |
| `file android_app_cache_file fennec_firefox_dev_entries_hexblob` | 5019 | 5019 | 532 | Firefox cache hashes compressed |
| `file android_app_cache_file com_android_browser_cache` | 4344 | 4344 | 279890 | browser cache shape |
| `file android_app_cache_file com_android_email_cache` | 3251 | 3251 | 8646 | email cache shape |
| `file android_app_cache_file camera2_image_manager_disk_cache_mixedid` | 3029 | 3029 | 9799 | camera cache IDs compressed |
| `file proc_file task_maps` | 2800 | 2800 | 11422 | task IDs removed |
| `file android_user_media_file camera_image` | 1516 | 1516 | 263577 | camera timestamp hidden |
| `file android_app_private_file email_body_txt` | 1118 | 1118 | 18661 | email body numeric path hidden |
| `file android_recent_task_file task_xml` | 599 | 599 | 8193 | recent task IDs hidden |

These collisions are mostly deliberate: they remove volatile PIDs, timestamps,
hashes, numeric IDs, and cache entry identities while retaining the stable
Android role and package family.

## Candidate Compression Audit

The diagnostic candidate audit is separate from the runtime tokenizer. It
tested a compressed candidate-detail design for future runtime hardening.

Output:

```text
<HISTORICAL_TMP_ROOT>/clearscope_file_raw_detail_compressed_candidate_audit_min/
```

Observed reductions:

- candidate vocab tokens: 9918 -> 620
- candidate file tuples: 9894 -> 597
- candidate detail tokens: 9892 -> 594
- singleton details: 6127 -> 231
- hash-like details: 5228 -> 3
- timestamp-like details: 624 -> 0
- numeric-id-like details: 2677 -> 13
- test malicious file refs seen in train/val: 15866 / 15881
- recommendation: `should_implement_compressed_candidate_tokenizer = true`
- recommendation: `should_compress_more = false`
- recommendation: `should_split_more = false`

Candidate compression patterns:

```text
vmdl<digits>_tmp_base_apk -> data_app_vmdl_base_apk
body_<num>..._(txt|html) -> body_num_..._(txt|html)
img_YYYYMMDD_HHMMSS_(jpg|jpeg|png) -> img_timestamp_<ext>
<digits>_task_thumbnail_png -> recent_task_thumbnail
<digits>_task_xml -> recent_task_xml
..._doomed_<digits> -> ..._doomed_num
..._entries_<hex> -> ..._entries_hexblob
cache tails with hex/mixed IDs -> cache_hexblob / cache_mixedid shape
generic long hex/alnum/numeric parts -> hexblob / mixedid / num
```

This audit supports promoting additional targeted compression only after a
bounded ClearScope E3/E5 regression run confirms no online recall loss.

## Known Collision Risks

Current intentional coarse buckets can still hide detail:

- `netflow` hides all endpoints. E3 audit says this is safer than source-detail
  splitting, but E5 should be re-audited before final runs.
- `camera_image`, `thumbnail`, `email_eml`, and `email_body` merge many raw
  files. This is acceptable for E3 because timestamps/numeric IDs caused high
  cardinality, but E5 may need a package/action-aware diagnostic check.
- `com_android_browser_cache` and `com_android_email_cache` merge cache
  contents by app family. This prevents cache hash explosion.
- `socket_other` and `dev_other` merge rare unknown endpoints. If E5 attacks
  rely on a new specific device/socket name, add it only after label-free
  support/collision analysis.
- `file_other` remains a fallback and should not be used as a rescue rule.

## E5 Status

The E5 rule path is the same runtime code path as E3. This document therefore
backs up the E5 tokenizer contract, but the collision statistics above are from
`clearscope_e3`.

Before final `CLEARSCOPE_E5` full experiments, run the same refined-token audit
against `clearscope_e5` and compare:

- unique token and tuple counts;
- top collision rows;
- netflow source-detail overlap;
- unknown `dev_other`, `socket_other`, and `file_other` rates;
- attack-window recall only after streaming inference completes.

## Improvement Suggestions

1. Add a small regression test fixture for the required examples in the E3
   audit summary.
2. Promote the candidate-compression rules into runtime only behind a new
   semantic mode name, for example `raw_detail_v3_compressed`.
3. Keep `raw_detail_v2_refined` frozen for reproducibility of existing E3
   artifacts.
4. Add an E5 collision audit before training/running E5 full experiments.
5. Keep netflow fixed unless an E5 audit proves source detail reduces collision
   without benign/malicious overlap.
6. Add cache metadata checks so Word2Vec, Phase3E, and Phase3G artifacts cannot
   mix legacy ClearScope semantics with refined ClearScope semantics.
7. Track `file_other`, `dev_other`, `socket_other`, and `unknown_file` rates in
   every ClearScope full result summary.
8. Do not add attack-string, package-name rescue, or label-derived rules.

## Files To Preserve With This Rule Backup

Recommended backup set for semantic-rule reproducibility:

- `cs4m/clearscope_semantics.py`
- `scripts/tools/causal_semantics_slim.py`
- `scripts/tools/train_residual_word2vec_models.py`
- `scripts/tools/smoke_clearscope_refined_tokenizer_path.py`
- `scripts/tools/smoke_clearscope_phase3g_refined_token_path.py`
- `scripts/diagnostics/clearscope_raw_detail_v2_refined_audit.py`
- `scripts/diagnostics/clearscope_file_raw_detail_candidate_audit.py`
- `<HISTORICAL_TMP_ROOT>/clearscope_detail_semantic_audit_v2_refined/`
- `<HISTORICAL_TMP_ROOT>/clearscope_file_raw_detail_compressed_candidate_audit_min/`
- this document
