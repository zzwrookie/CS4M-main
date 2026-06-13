# ClearScope E5 Bounded Artifact Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run independent ClearScope E5 v33b bounded Word2Vec, Phase3E,
Phase3G, and pipeline smoke artifacts without reading or overwriting E3 artifacts.

**Architecture:** Add one E5-only shell runner that orchestrates the existing
train-only Word2Vec utility and the existing `conditional_e4` Phase3E/Phase3G
entrypoint. Keep tokenizer, threshold, policy, evaluation, and training logic
unchanged except for a minimal syntax fix in the Word2Vec utility if tests expose
it.

**Tech Stack:** Bash runners, Python `unittest`/`pytest`, `legacy/tools/train_residual_word2vec_models.py`, `scripts.pipeline.entrypoints.conditional_e4`, PostgreSQL-backed ClearScope dataset streaming.

---

## Files

- Modify: `legacy/tools/train_residual_word2vec_models.py`
  - Responsibility: train train-only residual Word2Vec artifacts and write sidecar metadata.
- Create: `scripts/run/run_clearscope_e5_v33b_bounded_phase3e_phase3g_smoke.sh`
  - Responsibility: E5-only bounded artifact and pipeline smoke orchestration.
- Create: `tests/test_clearscope_e5_bounded_runner.py`
  - Responsibility: runner syntax, dry-run routing, E5/v33b identity, and E3 rejection tests.
- Modify: `docs/superpowers/plans/2026-06-13-clearscope-e5-bounded-artifact-smoke.md`
  - Responsibility: track execution status.
- Generated, not committed: `outputs/models/residual_word2vec/`, `outputs/cache/phase3e/`, `outputs/cache/phase3g_action_validation/`, `outputs/cache/phase3g_endpoint_suppression/`, `outputs/models/sspm_phase3e/`, `outputs/models/phase3g_action_heads/`, `outputs/results/tflr_light/CLEARSCOPE_E5_V33B_BOUNDED_PIPELINE_SMOKE/`, `logs/clearscope_e5_v33b_bounded_smoke/`.

## Task 1: Expose And Fix Word2Vec Utility Syntax

**Files:**
- Modify: `legacy/tools/train_residual_word2vec_models.py`

- [ ] **Step 1: Run syntax check to verify the current failure**

Run:

```bash
python3 -m py_compile legacy/tools/train_residual_word2vec_models.py
```

Expected: FAIL if the current extra `)` is still present, with `SyntaxError:
unmatched ')'`.

- [ ] **Step 2: Remove only the stray closing parenthesis**

In `legacy/tools/train_residual_word2vec_models.py`, change:

```python
    print(
        f"[WORD2VEC][{dataset_name}] saved model={model_path} meta={meta_path}",
        file=sys.stderr,
        flush=True,
    )
    )
```

to:

```python
    print(
        f"[WORD2VEC][{dataset_name}] saved model={model_path} meta={meta_path}",
        file=sys.stderr,
        flush=True,
    )
```

- [ ] **Step 3: Verify syntax passes**

Run:

```bash
python3 -m py_compile legacy/tools/train_residual_word2vec_models.py
```

Expected: PASS with exit code `0`.

- [ ] **Step 4: Commit the syntax fix if the file changed**

Run:

```bash
git add legacy/tools/train_residual_word2vec_models.py
git commit -m "fix: restore residual word2vec trainer syntax"
```

Expected: commit succeeds if the syntax fix was needed.

## Task 2: Add Failing E5 Runner Tests

**Files:**
- Create: `tests/test_clearscope_e5_bounded_runner.py`

- [ ] **Step 1: Write tests for the missing E5 runner**

Create `tests/test_clearscope_e5_bounded_runner.py` with:

```python
import os
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts/run/run_clearscope_e5_v33b_bounded_phase3e_phase3g_smoke.sh"


def _run_runner(extra_env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "CS4M_ROOT": str(REPO_ROOT),
            "DRY_RUN": "1",
            "PYTHON_BIN": "python3",
        },
    )
    env.update(extra_env)
    return subprocess.run(
        ["bash", str(RUNNER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


class ClearScopeE5BoundedRunnerTests(unittest.TestCase):
    def test_dry_run_all_uses_e5_v33b_bounded_artifacts(self) -> None:
        result = _run_runner({"STAGE": "all"})

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        combined = result.stdout + result.stderr
        self.assertIn("train_word2vec", combined)
        self.assertIn("--datasets CLEARSCOPE_E5", combined)
        self.assertIn("--dataset CLEARSCOPE_E5", combined)
        self.assertIn("--semantic_mode raw_detail_v33b_e5_android_safe", combined)
        self.assertIn("--max_train_events 50000", combined)
        self.assertIn("--max_ref_events 50000", combined)
        self.assertIn("--max_test_events 50000", combined)
        self.assertIn("CLEARSCOPE_E5_V33B_BOUNDED", combined)
        self.assertNotIn("CLEARSCOPE_E3", combined)

    def test_rejects_e3_dataset_even_in_dry_run(self) -> None:
        result = _run_runner({"DATASET": "CLEARSCOPE_E3", "STAGE": "all"})

        self.assertEqual(result.returncode, 2)
        self.assertIn("restricted to CLEARSCOPE_E5", result.stderr)

    def test_rejects_non_v33b_semantic_mode_even_in_dry_run(self) -> None:
        result = _run_runner(
            {
                "SEMANTIC_MODE": "raw_detail_v33_e5_android_safe",
                "STAGE": "all",
            },
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("raw_detail_v33b_e5_android_safe", result.stderr)

    def test_shell_syntax_is_valid(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(RUNNER)],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
pytest tests/test_clearscope_e5_bounded_runner.py -q
```

Expected: FAIL because the runner file does not exist.

## Task 3: Implement The E5 Bounded Runner

**Files:**
- Create: `scripts/run/run_clearscope_e5_v33b_bounded_phase3e_phase3g_smoke.sh`

- [ ] **Step 1: Create the runner**

Create `scripts/run/run_clearscope_e5_v33b_bounded_phase3e_phase3g_smoke.sh` with:

```bash
#!/usr/bin/env bash
set -euo pipefail

# ClearScope E5 v33b bounded Phase3E/Phase3G smoke runner.
# Labels are only used by the pipeline's post-stream evaluation/reporting path
# after online inference.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${CS4M_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export CLAD_DB_HOST="${CLAD_DB_HOST:-localhost}"
export CLAD_DB_USER="${CLAD_DB_USER:-postgres}"
export CLAD_DB_PORT="${CLAD_DB_PORT:-5433}"

DATASET="${DATASET:-CLEARSCOPE_E5}"
SEMANTIC_MODE="${SEMANTIC_MODE:-raw_detail_v33b_e5_android_safe}"
STAGE="${STAGE:-all}"
DRY_RUN="${DRY_RUN:-0}"

SEMANTIC_LABEL="V33B_BOUNDED"
SEMANTIC_SUFFIX="v33b_bounded"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/outputs/results/tflr_light}"
PHASE3E_CACHE_ROOT="${PHASE3E_CACHE_ROOT:-${REPO_ROOT}/outputs/cache/phase3e}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/clearscope_e5_v33b_bounded_smoke}"
CHECKPOINT_ROOT="${SSPM_BASE_CHECKPOINT_ROOT:-${REPO_ROOT}/outputs/models/sspm_phase3e}"
ACTION_HEAD_ROOT="${ACTION_HEAD_CHECKPOINT_ROOT:-${REPO_ROOT}/outputs/models/phase3g_action_heads}"
WORD2VEC_ROOT="${WORD2VEC_ROOT:-${REPO_ROOT}/outputs/models/residual_word2vec}"
WORD2VEC_TAG="V33B_BOUNDED_LATENT64"
EMBEDDER_NAME="CLEARSCOPE_E5_${WORD2VEC_TAG}_word2vec_window3.pkl"
EMBEDDER_DEFAULT="${WORD2VEC_ROOT}/${EMBEDDER_NAME}"
EMBEDDER_PATH="${PRETRAINED_RESIDUAL_EMBEDDER_PATH:-${EMBEDDER_DEFAULT}}"

NODE_CACHE_DEFAULT="${PHASE3E_CACHE_ROOT}/node_embeddings/CLEARSCOPE_E5_${SEMANTIC_SUFFIX}_latent64"
ACTION_CACHE_DEFAULT="${PHASE3E_CACHE_ROOT}/action_embeddings/CLEARSCOPE_E5_${SEMANTIC_SUFFIX}_latent64"
NODE_CACHE_DIR="${NODE_EMBEDDING_CACHE_DIR:-${NODE_CACHE_DEFAULT}}"
ACTION_CACHE_DIR="${ACTION_EMBEDDING_CACHE_DIR:-${ACTION_CACHE_DEFAULT}}"
EVENT_INDEX_CACHE_DIR="${EVENT_INDEX_CACHE_DIR:-${PHASE3E_CACHE_ROOT}/event_indices/CLEARSCOPE_E5_${SEMANTIC_SUFFIX}}"
COMPACT_USED_NODE_DEFAULT="${PHASE3E_CACHE_ROOT}/compact_used_node_embeddings/CLEARSCOPE_E5_${SEMANTIC_SUFFIX}"
COMPACT_USED_NODE_CACHE_DIR="${COMPACT_USED_NODE_CACHE_DIR:-${COMPACT_USED_NODE_DEFAULT}}"
ACTION_VALIDATION_DEFAULT="${REPO_ROOT}/outputs/cache/phase3g_action_validation/CLEARSCOPE_E5_${SEMANTIC_SUFFIX}"
ENDPOINT_CACHE_DEFAULT="${REPO_ROOT}/outputs/cache/phase3g_endpoint_suppression/CLEARSCOPE_E5_${SEMANTIC_SUFFIX}"
ACTION_VALIDATION_CACHE_DIR="${ACTION_VALIDATION_CACHE_DIR:-${ACTION_VALIDATION_DEFAULT}}"
ENDPOINT_CACHE_DIR="${CONDITIONAL_ENDPOINT_SUPPRESSION_CACHE_DIR:-${ENDPOINT_CACHE_DEFAULT}}"

BASE_CHECKPOINT_NAME="CLEARSCOPE_E5_${SEMANTIC_LABEL}_PHASE3E_BASE.pkl"
ACTION_HEAD_NAME="CLEARSCOPE_E5_${SEMANTIC_LABEL}_PHASE3G_CONDITIONAL_HEAD.pkl"
BASE_CHECKPOINT_DEFAULT="${CHECKPOINT_ROOT}/${BASE_CHECKPOINT_NAME}"
ACTION_HEAD_CHECKPOINT_DEFAULT="${ACTION_HEAD_ROOT}/${ACTION_HEAD_NAME}"
BASE_CHECKPOINT="${SSPM_CHECKPOINT_PATH:-${BASE_CHECKPOINT_DEFAULT}}"
ACTION_HEAD_CHECKPOINT="${ACTION_HEAD_CHECKPOINT_PATH:-${ACTION_HEAD_CHECKPOINT_DEFAULT}}"

MAX_TRAIN_EVENTS="${MAX_TRAIN_EVENTS:-50000}"
MAX_REF_EVENTS="${MAX_REF_EVENTS:-50000}"
MAX_TEST_EVENTS="${MAX_TEST_EVENTS:-50000}"
PROGRESS_INTERVAL_EVENTS="${PROGRESS_INTERVAL_EVENTS:-10000}"
FETCH_SIZE="${FETCH_SIZE:-10000}"
SSPM_EPOCHS="${SSPM_EPOCHS:-1}"
SSPM_TRAIN_BATCH_EVENTS="${SSPM_TRAIN_BATCH_EVENTS:-8192}"
SSPM_CONDITIONAL_MAX_EPOCHS="${SSPM_CONDITIONAL_MAX_EPOCHS:-3}"
SSPM_TORCH_BATCH_EVENTS="${SSPM_TORCH_BATCH_EVENTS:-8192}"
WORD2VEC_EPOCHS="${WORD2VEC_EPOCHS:-1}"
WORD2VEC_WORKERS="${WORD2VEC_WORKERS:-2}"
CONDITIONAL_GROUP_MIN_COUNT="${CONDITIONAL_GROUP_MIN_COUNT:-1000}"
EVENT_THRESHOLD_MODE="${EVENT_THRESHOLD_MODE:-quantile}"
EVENT_THRESHOLD_QUANTILE="${EVENT_THRESHOLD_QUANTILE:-0.999}"
ACTION_TYPE_ALERT_POLICY="${ACTION_TYPE_ALERT_POLICY:-default}"
NODE_POOL_SCORE_MODE="${NODE_POOL_SCORE_MODE:-base_conf}"
CLEARSCOPE_V31_FP_GUARD_WRITE_FLOOR="${CLEARSCOPE_V31_FP_GUARD_WRITE_FLOOR:-0.085}"
CLEARSCOPE_V31_FP_GUARD_READ_FLOOR="${CLEARSCOPE_V31_FP_GUARD_READ_FLOOR:-0.060}"
CONDITIONAL_LOW_SUPPORT_POLICY="${CONDITIONAL_LOW_SUPPORT_POLICY:-conservative_max}"
CONDITIONAL_LOW_SUPPORT_MARGIN="${CONDITIONAL_LOW_SUPPORT_MARGIN:-0.05}"
CONDITIONAL_ENDPOINT_AWARE_SUPPRESSION="${CONDITIONAL_ENDPOINT_AWARE_SUPPRESSION:-false}"
CONDITIONAL_ENDPOINT_SUPPRESSION_READ="${CONDITIONAL_ENDPOINT_SUPPRESSION_READ:-false}"
CONDITIONAL_ENDPOINT_SUPPRESSION_MODE="${CONDITIONAL_ENDPOINT_SUPPRESSION_MODE:-pair_only}"
CONDITIONAL_BOTH_COLD_UNSEEN_POLICY="${CONDITIONAL_BOTH_COLD_UNSEEN_POLICY:-alert}"
SSPM_STATE_MODEL="${SSPM_STATE_MODEL:-s4d_complex_node}"
SSPM_CONDITIONAL_HEAD_ARCH="${SSPM_CONDITIONAL_HEAD_ARCH:-shared_lowrank_v1}"

is_true() {
    case "${1:-0}" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

print_command() {
    printf '%q' "$1"
    shift
    for arg in "$@"; do
        printf ' %q' "${arg}"
    done
    printf '\n'
}

require_file() {
    local path="$1"
    local label="$2"
    if [[ ! -f "${path}" ]]; then
        echo "error: missing ${label}: ${path}" >&2
        exit 2
    fi
}

require_writable_dir() {
    local path="$1"
    local label="$2"
    mkdir -p "${path}"
    if [[ ! -d "${path}" || ! -w "${path}" ]]; then
        echo "error: ${label} is not writable: ${path}" >&2
        exit 2
    fi
}

reject_e3_path() {
    local path="$1"
    local label="$2"
    if [[ "${path}" == *CLEARSCOPE_E3* ]]; then
        echo "error: ${label} must not reference CLEARSCOPE_E3: ${path}" >&2
        exit 2
    fi
}

validate_out_tag_override() {
    if [[ -z "${OUT_TAG_OVERRIDE:-}" ]]; then
        return 0
    fi
    if [[ "${STAGE}" != "infer_smoke" ]]; then
        echo "error: OUT_TAG_OVERRIDE is only supported with STAGE=infer_smoke" >&2
        exit 2
    fi
    case "${OUT_TAG_OVERRIDE}" in
        */*|*\\*|*..*)
            echo "error: OUT_TAG_OVERRIDE must not contain slash, backslash, or '..'" >&2
            exit 2
            ;;
    esac
}

preflight() {
    if [[ "${DATASET}" != "CLEARSCOPE_E5" ]]; then
        echo "error: this runner is restricted to CLEARSCOPE_E5" >&2
        exit 2
    fi
    if [[ "${SEMANTIC_MODE}" != "raw_detail_v33b_e5_android_safe" ]]; then
        echo "error: SEMANTIC_MODE must be raw_detail_v33b_e5_android_safe" >&2
        exit 2
    fi
    case "${STAGE}" in
        all|train_word2vec|build_phase3e_artifacts|train_phase3e_base|train_phase3g_head|infer_smoke) ;;
        *)
            echo "error: unsupported STAGE=${STAGE}" >&2
            exit 2
            ;;
    esac
    validate_out_tag_override
    reject_e3_path "${EMBEDDER_PATH}" "Word2Vec embedder path"
    reject_e3_path "${NODE_CACHE_DIR}" "node cache path"
    reject_e3_path "${ACTION_CACHE_DIR}" "action cache path"
    reject_e3_path "${EVENT_INDEX_CACHE_DIR}" "event index cache path"
    reject_e3_path "${COMPACT_USED_NODE_CACHE_DIR}" "compact node cache path"
    reject_e3_path "${ACTION_VALIDATION_CACHE_DIR}" "action validation cache path"
    reject_e3_path "${ENDPOINT_CACHE_DIR}" "endpoint cache path"
    reject_e3_path "${BASE_CHECKPOINT}" "Phase3E checkpoint path"
    reject_e3_path "${ACTION_HEAD_CHECKPOINT}" "Phase3G checkpoint path"
    if is_true "${DRY_RUN}"; then
        return 0
    fi
    if [[ -z "${CLAD_DB_PASSWORD:-}" ]]; then
        echo "error: set CLAD_DB_PASSWORD before running ClearScope E5 bounded smoke" >&2
        exit 2
    fi
    if [[ "${STAGE}" != "train_word2vec" && "${STAGE}" != "all" ]]; then
        require_file "${EMBEDDER_PATH}" "ClearScope E5 v33b bounded residual Word2Vec embedder"
    fi
    if [[ "${STAGE}" == "train_phase3e_base" || "${STAGE}" == "train_phase3g_head" || \
        "${STAGE}" == "infer_smoke" ]]; then
        require_file "${EVENT_INDEX_CACHE_DIR}/event_index_meta.json" "Phase3E event index meta"
        require_file "${NODE_CACHE_DIR}/node_embeddings.npy" "Phase3E node embeddings"
        require_file "${ACTION_CACHE_DIR}/action_embeddings.npy" "Phase3E action embeddings"
    fi
    if [[ "${STAGE}" == "train_phase3g_head" || "${STAGE}" == "infer_smoke" ]]; then
        require_file "${BASE_CHECKPOINT}" "Phase3E base checkpoint"
    fi
    if [[ "${STAGE}" == "infer_smoke" ]]; then
        require_file "${ACTION_HEAD_CHECKPOINT}" "Phase3G conditional head checkpoint"
    fi
    require_writable_dir "${RESULT_ROOT}" "result root"
    require_writable_dir "${LOG_DIR}" "log directory"
    require_writable_dir "${WORD2VEC_ROOT}" "Word2Vec root"
    require_writable_dir "${CHECKPOINT_ROOT}" "checkpoint root"
    require_writable_dir "${ACTION_HEAD_ROOT}" "action head checkpoint root"
    require_writable_dir "${ACTION_VALIDATION_CACHE_DIR}" "action validation cache directory"
    require_writable_dir "${ENDPOINT_CACHE_DIR}" "endpoint suppression cache directory"
    require_writable_dir "${NODE_CACHE_DIR}" "node embedding cache directory"
    require_writable_dir "${ACTION_CACHE_DIR}" "action embedding cache directory"
    require_writable_dir "${EVENT_INDEX_CACHE_DIR}" "event index cache directory"
    require_writable_dir "${COMPACT_USED_NODE_CACHE_DIR}" "compact used-node cache directory"
}

run_logged_command() {
    local label="$1"
    local out_tag="$2"
    shift 2
    local log_path="${LOG_DIR}/${out_tag}.log"
    local cmd=("$@")
    if is_true "${DRY_RUN}"; then
        echo "===== DRY_RUN ${label} ${out_tag} ====="
        print_command "${cmd[@]}"
        return 0
    fi
    echo "===== START ${label} ${out_tag} $(date) ====="
    "${cmd[@]}" 2>&1 | tee "${log_path}"
    echo "===== DONE ${label} ${out_tag} $(date) ====="
}

pipeline_args=(
    --dataset "${DATASET}"
    --result_root "${RESULT_ROOT}"
    --max_train_events "${MAX_TRAIN_EVENTS}"
    --max_ref_events "${MAX_REF_EVENTS}"
    --max_test_events "${MAX_TEST_EVENTS}"
    --event_score_mode res_only
    --event_threshold_mode "${EVENT_THRESHOLD_MODE}"
    --event_threshold_quantile "${EVENT_THRESHOLD_QUANTILE}"
    --action_type_alert_policy "${ACTION_TYPE_ALERT_POLICY}"
    --node_pool_score_mode "${NODE_POOL_SCORE_MODE}"
    --clearscope_v31_fp_guard_write_floor "${CLEARSCOPE_V31_FP_GUARD_WRITE_FLOOR}"
    --clearscope_v31_fp_guard_read_floor "${CLEARSCOPE_V31_FP_GUARD_READ_FLOOR}"
    --pretrained_residual_embedder_path "${EMBEDDER_PATH}"
    --semantic_embedding_method word2vec
    --semantic_mode "${SEMANTIC_MODE}"
    --word2vec_window 3
    --sspm_target_dim 64
    --sspm_state_dim 64
    --rank 32
    --sspm_context_mode with_action
    --sspm_global_context_mode no_global
    --state_memory_mode bounded
    --sspm_state_memory_policy probationary_lru
    --sspm_target_mode node_action_semantic_mean
    --node_word2vec_source residual_pretrained
    --node_embedding_lookup_mode global_memmap
    --node_embedding_lazy_backing compact_used_nodes
    --node_embedding_cache_max_nodes 50000
    --node_embedding_cache_evict_policy lru
    --node_embedding_cache_dir "${NODE_CACHE_DIR}"
    --compact_used_node_cache_dir "${COMPACT_USED_NODE_CACHE_DIR}"
    --action_embedding_cache_dir "${ACTION_CACHE_DIR}"
    --event_index_cache_mode auto
    --event_index_cache_dir "${EVENT_INDEX_CACHE_DIR}"
    --sspm_infer_backend numpy
    --sspm_score_head conditional_action_semantic
    --node_repr_fusion simple_mean
    --conditional_semantic_loss cosine
    --sspm_conditional_head_arch "${SSPM_CONDITIONAL_HEAD_ARCH}"
    --conditional_group_min_count "${CONDITIONAL_GROUP_MIN_COUNT}"
    --conditional_low_support_policy "${CONDITIONAL_LOW_SUPPORT_POLICY}"
    --conditional_low_support_margin "${CONDITIONAL_LOW_SUPPORT_MARGIN}"
    --conditional_unseen_group_policy observation_only
    --conditional_global_extreme_quantile 0.9999
    --conditional_adaptive_margin_n1 50
    --conditional_adaptive_margin_n2 200
    --conditional_adaptive_margin_low 0.15
    --conditional_adaptive_margin_mid 0.05
    --conditional_adaptive_margin_high 0.02
    --conditional_endpoint_aware_suppression "${CONDITIONAL_ENDPOINT_AWARE_SUPPRESSION}"
    --conditional_endpoint_suppression_read "${CONDITIONAL_ENDPOINT_SUPPRESSION_READ}"
    --conditional_endpoint_suppression_mode "${CONDITIONAL_ENDPOINT_SUPPRESSION_MODE}"
    --conditional_both_cold_unseen_policy "${CONDITIONAL_BOTH_COLD_UNSEEN_POLICY}"
    --conditional_endpoint_suppression_summary_mode online_minimal
    --conditional_endpoint_suppression_cache_dir "${ENDPOINT_CACHE_DIR}"
    --action_validation_cache_dir "${ACTION_VALIDATION_CACHE_DIR}"
    --sspm_base_checkpoint_root "${CHECKPOINT_ROOT}"
    --sspm_state_model "${SSPM_STATE_MODEL}"
    --sspm_update_gate_mode none
    --sspm_residual_score_mode legacy
    --sspm_residual_calibration none
    --sspm_state_merge_mode none
    --sspm_state_merge_threshold 0.98
    --write_raw_alerts true
    --write_analysis_outputs true
    --sspm_state_merge_write_diagnostics false
    --progress_interval_events "${PROGRESS_INTERVAL_EVENTS}"
    --fetch_size "${FETCH_SIZE}"
    --print_summary
)

run_train_word2vec() {
    run_logged_command "train_word2vec" "CLEARSCOPE_E5_V33B_BOUNDED_WORD2VEC" \
        "${PYTHON_BIN}" legacy/tools/train_residual_word2vec_models.py \
        --datasets "${DATASET}" \
        --out_dir "${WORD2VEC_ROOT}" \
        --out_tag "${WORD2VEC_TAG}" \
        --max_train_events "${MAX_TRAIN_EVENTS}" \
        --fetch_size "${FETCH_SIZE}" \
        --progress_interval_events "${PROGRESS_INTERVAL_EVENTS}" \
        --semantic_mode "${SEMANTIC_MODE}" \
        --word2vec_window 3 \
        --word2vec_min_count 1 \
        --word2vec_epochs "${WORD2VEC_EPOCHS}" \
        --word2vec_workers "${WORD2VEC_WORKERS}"
}

run_pipeline_stage() {
    local label="$1"
    local out_tag="$2"
    shift 2
    run_logged_command "${label}" "${out_tag}" \
        "${PYTHON_BIN}" -m scripts.pipeline.entrypoints.conditional_e4 \
        "${pipeline_args[@]}" \
        --out_tag "${out_tag}" \
        "$@"
}

run_build_phase3e_artifacts() {
    run_pipeline_stage "build_phase3e_artifacts" "CLEARSCOPE_E5_V33B_BOUNDED_ARTIFACT" \
        --sspm_train_mode build_phase3e_artifacts \
        --sspm_train_data_mode phase3e_memmap
}

run_train_phase3e_base() {
    run_pipeline_stage "train_phase3e_base" "CLEARSCOPE_E5_V33B_BOUNDED_BASE" \
        --sspm_train_mode train_phase3e_base \
        --sspm_train_data_mode phase3e_memmap \
        --sspm_checkpoint_path "${BASE_CHECKPOINT}" \
        --sspm_epochs "${SSPM_EPOCHS}" \
        --sspm_train_batch_events "${SSPM_TRAIN_BATCH_EVENTS}"
}

run_train_phase3g_head() {
    run_pipeline_stage "train_phase3g_head" "CLEARSCOPE_E5_V33B_BOUNDED_HEAD" \
        --sspm_train_mode train_conditional_and_save \
        --sspm_train_data_mode phase3e_memmap \
        --sspm_checkpoint_path "${BASE_CHECKPOINT}" \
        --action_head_checkpoint_path "${ACTION_HEAD_CHECKPOINT}" \
        --sspm_conditional_train_data_mode memmap \
        --sspm_conditional_max_epochs "${SSPM_CONDITIONAL_MAX_EPOCHS}" \
        --sspm_torch_batch_events "${SSPM_TORCH_BATCH_EVENTS}"
}

run_infer_smoke() {
    local out_tag="CLEARSCOPE_E5_V33B_BOUNDED_PIPELINE_SMOKE"
    if [[ -n "${OUT_TAG_OVERRIDE:-}" ]]; then
        out_tag="${OUT_TAG_OVERRIDE}"
    fi
    run_pipeline_stage "infer_smoke" "${out_tag}" \
        --sspm_train_mode load_and_infer \
        --sspm_train_data_mode phase3e_memmap \
        --sspm_checkpoint_path "${BASE_CHECKPOINT}" \
        --action_head_checkpoint_path "${ACTION_HEAD_CHECKPOINT}"
}

preflight

case "${STAGE}" in
    train_word2vec)
        run_train_word2vec
        ;;
    build_phase3e_artifacts)
        run_build_phase3e_artifacts
        ;;
    train_phase3e_base)
        run_train_phase3e_base
        ;;
    train_phase3g_head)
        run_train_phase3g_head
        ;;
    infer_smoke)
        run_infer_smoke
        ;;
    all)
        run_train_word2vec
        run_build_phase3e_artifacts
        run_train_phase3e_base
        run_train_phase3g_head
        run_infer_smoke
        ;;
esac
```

- [ ] **Step 2: Make the runner executable**

Run:

```bash
chmod +x scripts/run/run_clearscope_e5_v33b_bounded_phase3e_phase3g_smoke.sh
```

- [ ] **Step 3: Run runner tests to verify GREEN**

Run:

```bash
pytest tests/test_clearscope_e5_bounded_runner.py -q
```

Expected: PASS.

- [ ] **Step 4: Run shell syntax check**

Run:

```bash
bash -n scripts/run/run_clearscope_e5_v33b_bounded_phase3e_phase3g_smoke.sh
```

Expected: PASS with exit code `0`.

- [ ] **Step 5: Commit runner and tests**

Run:

```bash
git add scripts/run/run_clearscope_e5_v33b_bounded_phase3e_phase3g_smoke.sh tests/test_clearscope_e5_bounded_runner.py
git commit -m "feat: add clearscope e5 bounded smoke runner"
```

Expected: commit succeeds.

## Task 4: Run Real E5 Bounded Smoke

**Files:**
- Generated only: `outputs/`, `logs/`.

- [ ] **Step 1: Run the real bounded smoke**

Run:

```bash
CLAD_DB_HOST="${CLAD_DB_HOST:-127.0.0.1}" \
CLAD_DB_PORT="${CLAD_DB_PORT:-5433}" \
CLAD_DB_USER="${CLAD_DB_USER:-postgres}" \
CLAD_DB_PASSWORD="${CLAD_DB_PASSWORD:-123456}" \
STAGE=all \
scripts/run/run_clearscope_e5_v33b_bounded_phase3e_phase3g_smoke.sh
```

Expected: all stages exit `0`; no full run is launched because the runner defaults are bounded at `50000/50000/50000`.

- [ ] **Step 2: Check required artifacts and outputs exist**

Run:

```bash
test -f outputs/models/residual_word2vec/CLEARSCOPE_E5_V33B_BOUNDED_LATENT64_word2vec_window3.pkl
test -f outputs/models/residual_word2vec/CLEARSCOPE_E5_V33B_BOUNDED_LATENT64_word2vec_window3.json
test -f outputs/cache/phase3e/event_indices/CLEARSCOPE_E5_v33b_bounded/event_index_meta.json
test -f outputs/cache/phase3e/node_embeddings/CLEARSCOPE_E5_v33b_bounded_latent64/node_embeddings.npy
test -f outputs/cache/phase3e/action_embeddings/CLEARSCOPE_E5_v33b_bounded_latent64/action_embeddings.npy
test -f outputs/models/sspm_phase3e/CLEARSCOPE_E5_V33B_BOUNDED_PHASE3E_BASE.pkl
test -f outputs/models/phase3g_action_heads/CLEARSCOPE_E5_V33B_BOUNDED_PHASE3G_CONDITIONAL_HEAD.pkl
test -f outputs/results/tflr_light/CLEARSCOPE_E5_V33B_BOUNDED_PIPELINE_SMOKE/online_event_alerts.csv
```

Expected: all `test -f` checks exit `0`.

- [ ] **Step 3: Check provenance contains E5/v33b and no E3 references**

Run:

```bash
rg -n "CLEARSCOPE_E3" \
  outputs/models/residual_word2vec/CLEARSCOPE_E5_V33B_BOUNDED_LATENT64_word2vec_window3.json \
  outputs/models/residual_word2vec/CLEARSCOPE_E5_V33B_BOUNDED_LATENT64/ \
  outputs/cache/phase3e/event_indices/CLEARSCOPE_E5_v33b_bounded/ \
  outputs/results/tflr_light/CLEARSCOPE_E5_V33B_BOUNDED_PIPELINE_SMOKE/ \
  logs/clearscope_e5_v33b_bounded_smoke/
```

Expected: no matches, exit code `1`.

Run:

```bash
rg -n "CLEARSCOPE_E5|raw_detail_v33b_e5_android_safe" \
  outputs/models/residual_word2vec/CLEARSCOPE_E5_V33B_BOUNDED_LATENT64_word2vec_window3.json \
  outputs/models/residual_word2vec/CLEARSCOPE_E5_V33B_BOUNDED_LATENT64/word2vec_training_summary.json \
  outputs/cache/phase3e/event_indices/CLEARSCOPE_E5_v33b_bounded/ \
  outputs/results/tflr_light/CLEARSCOPE_E5_V33B_BOUNDED_PIPELINE_SMOKE/ \
  logs/clearscope_e5_v33b_bounded_smoke/
```

Expected: matches exist.

- [ ] **Step 4: Inspect online alert schema**

Run:

```bash
python3 - <<'PY'
import csv
from pathlib import Path

path = Path("outputs/results/tflr_light/CLEARSCOPE_E5_V33B_BOUNDED_PIPELINE_SMOKE/online_event_alerts.csv")
with path.open(newline="", encoding="utf-8") as handle:
    reader = csv.reader(handle)
    header = next(reader)
required_any = [
    ("event", "event_id", "event_index", "event_idx", "idx"),
    ("score", "event_score", "residual_score", "final_score"),
    ("threshold", "event_threshold", "threshold_basis"),
    ("entity", "node", "subject", "object"),
]
missing = [group for group in required_any if not any(name in header for name in group)]
if missing:
    raise SystemExit(f"missing alert header groups: {missing}; header={header}")
print({"header": header})
PY
```

Expected: exits `0` and prints the alert header. If the CSV is header-only with no alerts, the schema still satisfies the explainability contract.

## Task 5: Final Verification And Report

**Files:**
- Modify: `docs/superpowers/plans/2026-06-13-clearscope-e5-bounded-artifact-smoke.md`

- [ ] **Step 1: Run focused tests**

Run:

```bash
pytest tests/test_clearscope_e5_bounded_runner.py tests/test_residual_word2vec_corpus_file.py -q
```

Expected: PASS.

- [ ] **Step 2: Run syntax checks for touched executable code**

Run:

```bash
python3 -m py_compile legacy/tools/train_residual_word2vec_models.py
bash -n scripts/run/run_clearscope_e5_v33b_bounded_phase3e_phase3g_smoke.sh
```

Expected: both commands exit `0`.

- [ ] **Step 3: Review git diff**

Run:

```bash
git diff -- legacy/tools/train_residual_word2vec_models.py scripts/run/run_clearscope_e5_v33b_bounded_phase3e_phase3g_smoke.sh tests/test_clearscope_e5_bounded_runner.py docs/superpowers/plans/2026-06-13-clearscope-e5-bounded-artifact-smoke.md
```

Expected: only the planned changes appear. Generated `outputs/` and `logs/` remain uncommitted.

- [ ] **Step 4: Commit final plan status if changed**

Run:

```bash
git add docs/superpowers/plans/2026-06-13-clearscope-e5-bounded-artifact-smoke.md
git commit -m "docs: record clearscope e5 bounded smoke plan"
```

Expected: commit succeeds if the plan file was not already committed.

