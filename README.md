# CS4M Cleanup Reproduction Project

CS4M is a streaming provenance-based threat detection project. The current tree is a
backup/redeploy cleanup copy focused on preserving the validated CADETS_E3 and THEIA_E3
Phase3E/Phase3G reproduction chains while making the source layout less ambiguous.

## Safety Contract

Online inference must score events in arrival or timestamp order and use only current and past
runtime-visible information. Do not use test labels, attack windows, malicious entity lists, test
rankings, or test performance for training, thresholding, feature design, model selection,
calibration, filtering, or online alerting. Ground truth is only for post-inference evaluation.

Primary explainable alert output is `online_event_alerts.csv`. It should preserve score,
threshold/ranking basis, triggering event/entity, and component information when available.

## Supported Datasets

The active config helpers define E3/E5 splits for:

- `CADETS_E3`, `CADETS_E5`
- `THEIA_E3`, `THEIA_E3_MINI`, `THEIA_E5`
- `CLEARSCOPE_E3`, `CLEARSCOPE_E5`

Dataset-specific semantic implementations are intentional variants, not duplicate code:

- CADETS uses FreeBSD process/path semantics in `cs4m/semantics/cadets_freebsd.py`.
- THEIA uses Linux process/path/netflow semantics in `cs4m/semantics/theia_linux.py`.
- ClearScope uses Android semantics in `cs4m/semantics/clearscope_android.py`, with both legacy
  `clearscope_android_semantics_v1` and refined `raw_detail_v2_refined` support.

## Current Best Reproduction Chains

CADETS_E3 best chain:

- Wrapper: `scripts/run/run_phase3g_cadets_e3_dual_head_v2_q09995_e2_e4_e5_full.sh`
- Base runner: `scripts/run/run_cadets_e3_phase3e_node_action_semantic_full.sh`
- Python entrypoint: `scripts/tools/causal_semantics_slim.py`
- Best family: `CADETS_E3_PHASE3E_E4_NONE`
- Conditional head: `dual_lowrank_by_target_case_v2`
- Threshold policy: `conditional_target_action_type_group_quantile`
- Threshold quantile: `0.9995`

THEIA_E3 best chain:

- Wrapper: `scripts/run/run_phase3g_theia_e3_e4_theia_v1_lazy100k_pair_only_full.sh`
- Base runner: `scripts/run/run_cadets_e3_phase3e_node_action_semantic_full.sh`
- Python entrypoint: `scripts/tools/causal_semantics_slim.py`
- Best family: `THEIA_E3_PHASE3E_E4_NONE`
- Conditional head: `shared_lowrank_v1`
- Alert policy: `theia_v1`
- Threshold policy: `conditional_target_action_type_group_quantile`
- Threshold quantile: `0.999`

`theia_v1` is an alert policy name, not evidence that the model is obsolete.

## Directory Structure

- `cs4m/`: importable core package. The root contains only `__init__.py`; implementation
  modules live in functional subpackages.
- `cs4m/config/`: importable dataset/database/runtime config helpers and `clad.yml`.
- `cs4m/semantics/`: dataset/OS semantic adapters, semantic dispatch, and residual tokenizers.
- `cs4m/embeddings/`: residual embedding builders and state-dict loaders.
- `cs4m/phase3e/`: event-index memmaps, Word2Vec adapters, semantic tables, X-context, and
  Torch head training helpers.
- `cs4m/phase3g/`: compact used-node artifacts and active Phase3G conditional heads.
- `cs4m/scoring/`: score-target builders, calibration, and gates.
- `cs4m/state/`: streaming state dynamics, bounded state memory, and online state merging.
- `cs4m/models/`: CS4M low-rank streaming model implementations.
- `cs4m/utils/`: shared hashing, type, row, cache, and profiling utilities.
- `configs/`: YAML experiment and process-semantic presets only.
- `scripts/run/`: shell wrappers for bounded preflight, dry-run, training, and inference flows.
- `scripts/tools/`: Python entrypoints, smoke checks, export tools, and summarizers.
- `scripts/data/`, `scripts/eval/`: dataset split helpers and post-inference evaluation helpers.
- `legacy/`: diagnostics, historical baseline models, old runners, and experiments.
- `outputs/`, `tmp/`, `logs/`: generated artifacts and local run logs.

## `cs4m/` Module Map

Active modules in the current CADETS/THEIA or common E3/E5 pipeline:

- `cs4m/utils/common.py`: hashing, typed row helpers, relation/type utilities.
- `cs4m/utils/profiling.py`: RSS/profiling helper used by the active runner.
- `cs4m/utils/residual_embed_cache.py`: SQLite residual embedding cache and fingerprinting.
- `cs4m/semantics/semantic_router.py`: dispatches process/file/netflow semantics by dataset
  profile.
- `cs4m/semantics/cadets_freebsd.py`, `theia_linux.py`, `clearscope_android.py`:
  dataset/OS semantic adapters.
- `cs4m/semantics/residual_tokens.py`: residual/netflow token parsing used by the SSPM chain.
- `cs4m/embeddings/residual.py`: residual semantic embedding builders.
- `cs4m/phase3e/event_index.py`: compact event-index dtype, fingerprints, and memmap IO.
- `cs4m/phase3e/word2vec_adapter.py`: Word2Vec adapter fingerprinting.
- `cs4m/phase3e/node_action_tables.py`: node/action embedding tables and coverage audits.
- `cs4m/phase3e/context_memmap.py`: Phase3E `X_context` memmaps.
- `cs4m/phase3e/head_training.py`: Torch low-rank head training helper.
- `cs4m/phase3g/compact_node_embeddings.py`: compact used-node embeddings and remapped indexes.
- `cs4m/phase3g/conditional_head.py`: shared/dual conditional semantic heads and thresholds.
- `cs4m/scoring/target_builder.py`: event/action and node-pair score targets.
- `cs4m/models/cs4m_lowrank.py`: current SSPM low-rank streaming model.
- `cs4m/state/state_models.py`, `state_memory.py`, `online_state_merging.py`: state dynamics,
  bounded memory, and online state merging.
- `cs4m/scoring/calibration.py`, `simple_gates.py`: residual calibration and update gates.
- `cs4m/config/config.py`, `provnet_utils.py`: dataset splits, runtime YAML, DB utilities.

Historical baselines and optional experiments:

- `legacy/baselines/lowrank.py`, `semantic_sketch.py`, and `chain_profile.py`: CSSM/low-rank
  baseline internals. They are separate from the active SSPM chain.
- `legacy/tools/run_tflr_light_db_lowrank.py` and `run_tflr_cssm_unified.py`: historical
  baseline entrypoints.
- `legacy/experiments/phase3g_no_action_semantic.py`: experimental no-action head. Active
  node-pair target construction remains in `cs4m/scoring/target_builder.py`.
- `legacy/experiments/phase3g_action_head.py`: historical Phase3G action-predict head. Current
  CADETS/THEIA best chains use `cs4m/phase3g/conditional_head.py`.
- `legacy/experiments/residual_hash_doc2vec.py`: historical signed hash-sketch and Doc2Vec
  residual embedders. Active CADETS/THEIA best chains use residual Word2Vec.

No root compatibility shims are kept. Active and legacy code import from the final subpackage
or `legacy/` paths directly.

Moved to legacy:

- `legacy/experiments/ltpnm.py`: LTPNM P1/P2 experiment implementation, used only by
  `legacy/experiments/run_ltpnm.py`.

## Config Organization

`cs4m/config/` is the importable Python config package. `configs/` is intentionally retained for YAML:

- `configs/common/process_semantics.yaml`
- `configs/common/tflr_cssm_unified.yaml`
- `configs/experimental/process_semantics_v2_coarse_nll.yaml`
- `configs/experimental/process_semantics_v3_process_kind_nll.yaml`

Database host/user/port/password must come from environment variables, not committed files.

## New-Server Deployment Checklist

Set machine-specific paths and credentials through environment variables. Do not edit source files
or commit `.env` files for a single server.

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

Deployment checklist:

- Restore PostgreSQL data on the target server and provide DB connection settings with
  `CLAD_DB_HOST`, `CLAD_DB_PORT`, `CLAD_DB_USER`, and `CLAD_DB_PASSWORD`.
- Create the Python/conda environment used by the project; set `PYTHON_BIN` if `python3` is not
  the desired interpreter.
- Restore required artifacts listed in `docs/ARTIFACT_MANIFEST.md`, especially checkpoints,
  residual Word2Vec models, Phase3E caches, Phase3G validation caches, and reproduction results.
- Override `RESULT_ROOT`, `PHASE3E_CACHE_ROOT`, `ACTION_VALIDATION_CACHE_DIR`, or
  `CONDITIONAL_ENDPOINT_SUPPRESSION_CACHE_DIR` when artifacts live outside `outputs/`.
- Treat `tmp/` reports and `docs/reproduction/` manifests as historical evidence records, not
  deployment path defaults.

## Checks

Static checks:

```bash
python3 -m compileall -q cs4m scripts utils legacy
bash -n scripts/run/*.sh legacy/runners/*.sh
PYTHONDONTWRITEBYTECODE=1 python3 -c "import cs4m; print('cs4m-import-ok')"
PYTHONDONTWRITEBYTECODE=1 python3 scripts/tools/causal_semantics_slim.py --help
```

Focused tests added in this cleanup:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v tests.test_phase3g_memmap_only
```

CADETS dry-run/preflight:

```bash
DRY_RUN=1 DATASET=CADETS_E3 RUN_ONLY=E4_NONE STAGE=infer_ablation \
    bash scripts/run/run_cadets_e3_phase3e_node_action_semantic_full.sh
```

## Missing THEIA_E3 Memmaps

The THEIA_E3 full wrapper requires these action-validation train memmaps:

```text
outputs/cache/phase3g_action_validation/THEIA_E3/conditional_train_memmaps/X_conditional_s4d_complex_node.memmap
outputs/cache/phase3g_action_validation/THEIA_E3/conditional_train_memmaps/Y_conditional_s4d_complex_node.memmap
outputs/cache/phase3g_action_validation/THEIA_E3/conditional_train_memmaps/target_case_s4d_complex_node.memmap
```

Generate only those cache artifacts with:

```bash
setsid nohup bash scripts/run/build_theia_e3_phase3g_conditional_memmaps.sh \
    > logs/theia_e3_phase3g_conditional_memmaps/nohup.log 2>&1 < /dev/null &
echo $! > logs/theia_e3_phase3g_conditional_memmaps/nohup.pid
```

Check status:

```bash
pid="$(cat logs/theia_e3_phase3g_conditional_memmaps/nohup.pid)"
ps -p "$pid" -o pid,etime,stat,cmd
tail -80 logs/theia_e3_phase3g_conditional_memmaps/nohup.log
find outputs/cache/phase3g_action_validation/THEIA_E3/conditional_train_memmaps \
    -maxdepth 1 -type f -printf '%p %s\n' | sort
```

The runner uses `dataset=THEIA_E3`, `state_family=E4`, `alert_policy=theia_v1`,
`conditional_head=shared_lowrank_v1`, and `threshold_quantile=0.999`. It stops after building
train-only conditional memmaps; it does not train, infer, evaluate, fabricate files, use CADETS
artifacts, or read ground truth.
