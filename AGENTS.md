# AGENTS.md

## Role

This repository is a CS4M backup/redeploy artifact for streaming provenance-based threat detection.

Codex should make minimal, reviewable changes and must preserve:

* streaming online inference;
* no test-label leakage;
* explainable alerts;
* reproducible backed-up evidence.

Before non-trivial changes, clarify the goal, scope, constraints, and acceptance criteria.

## Repository Structure

Current layout:

* `cs4m/`: core Python package, model logic, and importable config helpers.
* `configs/`: YAML experiment and process-semantic presets.
* `scripts/tools/`: training, scoring, diagnostics, and smoke-check scripts.
* `scripts/run/`: shell runners.
* `scripts/data/`: dataset helpers.
* `scripts/eval/`: evaluation helpers.
* `docs/`: notes and documentation.
* `ground_truth/`: labels used only for post-inference evaluation.
* `outputs/`: generated models, caches, and results.
* `tmp/`: temporary artifacts.
* `legacy/`: migrated historical diagnostics, old runners, and experiments.

## Portability And New-Server Deployment

Do not commit or depend on current-server absolute paths. Active code, runners,
configs, and deployment docs must be portable across hosts.

Use environment variables for machine-specific settings:

```bash
export CS4M_ROOT="$(pwd)"
export PYTHON_BIN="python3"
export TMP_ROOT="${CS4M_ROOT}/tmp"
export RESULT_ROOT="${CS4M_ROOT}/outputs/results/tflr_light"
export PHASE3E_CACHE_ROOT="${CS4M_ROOT}/outputs/cache/phase3e"
export ACTION_VALIDATION_CACHE_DIR="${CS4M_ROOT}/outputs/cache/phase3g_action_validation/<DATASET>"
export CONDITIONAL_ENDPOINT_SUPPRESSION_CACHE_DIR="${CS4M_ROOT}/outputs/cache/phase3g_endpoint_suppression/<DATASET>"
export CLAD_DB_HOST="127.0.0.1"
export CLAD_DB_PORT="5433"
export CLAD_DB_USER="postgres"
export CLAD_DB_PASSWORD="<set-outside-repo>"
```

New-server checklist:

* Restore or configure PostgreSQL and provide `CLAD_DB_HOST`, `CLAD_DB_PORT`,
  `CLAD_DB_USER`, and `CLAD_DB_PASSWORD` through the environment only.
* Create/activate the Python or conda environment, then set `PYTHON_BIN` if
  `python3` is not the desired interpreter.
* Restore required checkpoints, residual Word2Vec models, caches, and result
  artifacts under `outputs/`, or override `RESULT_ROOT`, `PHASE3E_CACHE_ROOT`,
  `ACTION_VALIDATION_CACHE_DIR`, and
  `CONDITIONAL_ENDPOINT_SUPPRESSION_CACHE_DIR`.
* Treat paths in `tmp/` reports and `docs/reproduction/` manifests as historical
  evidence records, not deployment defaults.

No `backups/` directory is present in this cleanup copy. Treat `outputs/`, `tmp/`,
and any future `backups/` directory as generated artifacts, not source code,
unless the user explicitly asks to inspect, restore, compare, or package them.

## Backup Boundary

If backup directories are reintroduced, backups such as the following are read-only
evidence by default:

```text
backups/tflr_phase3g_backup_*
```

Do not delete, rewrite, reformat, overwrite, or silently replace backed-up CSV/JSON/YAML files, checkpoints, caches, or result directories.

Important backed-up result groups:

* CADETS_E3 best result: v2/Q09995, especially `CADETS_E3_PHASE3E_E4_NONE`.
* CADETS_E3 ablations: `CADETS_E3_PHASE3E_E2_NONE`, `CADETS_E3_PHASE3E_E5_NONE`.
* THEIA_E3 best result: rerun directory containing `THEIA_E3_PHASE3E_E4_NONE`.

Notes:

* CADETS v2/Q09995 uses dual-head conditional semantic scoring.
* THEIA `theia_v1` means alert policy, not necessarily an old model.
* If THEIA rerun fails because a validation cache is missing, report the missing cache. Do not change evaluation logic to bypass it.

## Core Safety Rules

### Streaming inference

* Score events in arrival order or timestamp order.
* Use only current and past information.
* Do not use future events, full-test statistics, full-test sorting, or labels before alert generation.

### No leakage

During training, thresholding, feature design, model selection, calibration, filtering, or alert policy design, do not use:

* test labels;
* attack windows;
* ground-truth malicious entities;
* test rankings;
* test performance;
* label-derived rescue rules.

Runtime-visible raw fields may be used, such as process names, paths, commands, remote IPs, filenames, and event types.

Do not hardcode attack strings or dataset-specific shortcuts derived from labels.

### Explainable alerts

Primary online output:

```text
online_event_alerts.csv
```

Alerts should preserve score, threshold/ranking basis, triggering event/entity, and main contributing components when available.

### Evaluation safety

Ground truth may be used only after streaming inference is complete.

Any change to label loading, matching, filtering, top-k, thresholding, aggregation, or metric computation is high risk and must be explained separately.

## Change Rules

* Prefer small, reviewable diffs.
* Do not perform unrelated refactoring, formatting, deletion, renaming, or result regeneration.
* Do not launch full experiments unless explicitly requested.
* Prefer bounded smoke checks before full runs.

## Common Checks

Python syntax check:

```bash
python3 -m compileall configs cs4m scripts
```

Shell syntax check:

```bash
bash -n scripts/run/*.sh
```

List CS4M scripts:

```bash
find scripts/tools -maxdepth 2 -type f | sort
```

List key result files:

```bash
find outputs -maxdepth 4 -type f \( -name "metrics.json" -o -name "online_event_alerts.csv" -o -name "eval_*.json" \) 2>/dev/null | sort
```

Use the `clad` conda environment when available:

```bash
conda activate clad
```

Database settings must come from environment variables, not committed files:

```bash
export CLAD_DB_HOST=127.0.0.1
export CLAD_DB_USER=postgres
export CLAD_DB_PORT=5433
export CLAD_DB_PASSWORD='<set-outside-repo>'
```

Do not write passwords, private hostnames/IPs, `.env` files, raw datasets, checkpoints, or large generated outputs into source-controlled files.

## Final Response Requirement

After making changes, report:

* changed files;
* commands run;
* outputs checked;
* remaining risks;
* whether streaming inference, no leakage, explainability, and reproducibility are still preserved.
