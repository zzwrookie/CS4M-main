#!/usr/bin/env bash
set -euo pipefail

# CADETS_E3 FreeBSD v3 safe lexical full E4 dual-head serial runner.
# This starts after Word2Vec is already trained; it never retrains Word2Vec.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${CS4M_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export CLAD_DB_HOST="${CLAD_DB_HOST:-localhost}"
export CLAD_DB_USER="${CLAD_DB_USER:-postgres}"
export CLAD_DB_PORT="${CLAD_DB_PORT:-5433}"

DATASET="${DATASET:-CADETS_E3}"
SEMANTIC_MODE="${SEMANTIC_MODE:-cadets_freebsd_raw_detail_v3_safe_lexical}"
STAGE="${STAGE:-all}"
DRY_RUN="${DRY_RUN:-0}"

SEMANTIC_LABEL="FREEBSD_V3_SAFE_LEXICAL"
SEMANTIC_SUFFIX="freebsd_v3_safe_lexical"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/outputs/results/tflr_light}"
PHASE3E_CACHE_ROOT="${PHASE3E_CACHE_ROOT:-${REPO_ROOT}/outputs/cache/phase3e}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/cadets_e3_v3_safe_lexical_full}"
CHECKPOINT_ROOT="${SSPM_BASE_CHECKPOINT_ROOT:-${REPO_ROOT}/outputs/models/sspm_phase3e}"
ACTION_HEAD_ROOT="${ACTION_HEAD_CHECKPOINT_ROOT:-${REPO_ROOT}/outputs/models/phase3g_action_heads_v3}"
WORD2VEC_ROOT="${WORD2VEC_ROOT:-${REPO_ROOT}/outputs/models/residual_word2vec}"
WORD2VEC_TAG="FREEBSD_V3_SAFE_LEXICAL_LATENT64"
EMBEDDER_NAME="CADETS_E3_${WORD2VEC_TAG}_word2vec_window3.pkl"
EMBEDDER_DEFAULT="${WORD2VEC_ROOT}/${EMBEDDER_NAME}"
EMBEDDER_PATH="${PRETRAINED_RESIDUAL_EMBEDDER_PATH:-${EMBEDDER_DEFAULT}}"

NODE_CACHE_DEFAULT="${PHASE3E_CACHE_ROOT}/node_embeddings/CADETS_E3_${SEMANTIC_SUFFIX}_latent64"
ACTION_CACHE_DEFAULT="${PHASE3E_CACHE_ROOT}/action_embeddings/CADETS_E3_${SEMANTIC_SUFFIX}_latent64"
NODE_CACHE_DIR="${NODE_EMBEDDING_CACHE_DIR:-${NODE_CACHE_DEFAULT}}"
ACTION_CACHE_DIR="${ACTION_EMBEDDING_CACHE_DIR:-${ACTION_CACHE_DEFAULT}}"
EVENT_INDEX_CACHE_DIR="${EVENT_INDEX_CACHE_DIR:-${PHASE3E_CACHE_ROOT}/event_indices/CADETS_E3_${SEMANTIC_SUFFIX}}"
COMPACT_USED_NODE_DEFAULT="${PHASE3E_CACHE_ROOT}/compact_used_node_embeddings/CADETS_E3_${SEMANTIC_SUFFIX}"
COMPACT_USED_NODE_CACHE_DIR="${COMPACT_USED_NODE_CACHE_DIR:-${COMPACT_USED_NODE_DEFAULT}}"
ACTION_VALIDATION_DEFAULT="${REPO_ROOT}/outputs/cache/phase3g_action_validation_v3_Q09995/CADETS_E3_${SEMANTIC_SUFFIX}"
ENDPOINT_CACHE_DEFAULT="${REPO_ROOT}/outputs/cache/phase3g_endpoint_suppression_compact/CADETS_E3_${SEMANTIC_SUFFIX}"
ACTION_VALIDATION_CACHE_DIR="${ACTION_VALIDATION_CACHE_DIR:-${ACTION_VALIDATION_DEFAULT}}"
ENDPOINT_CACHE_DIR="${CONDITIONAL_ENDPOINT_SUPPRESSION_CACHE_DIR:-${ENDPOINT_CACHE_DEFAULT}}"

BASE_CHECKPOINT_NAME="CADETS_E3_${SEMANTIC_LABEL}_E4_PHASE3E_BASE.pkl"
ACTION_HEAD_NAME="CADETS_E3_${SEMANTIC_LABEL}_E4_PHASE3G_CONDITIONAL_DUAL_HEAD.pkl"
BASE_CHECKPOINT_DEFAULT="${CHECKPOINT_ROOT}/${BASE_CHECKPOINT_NAME}"
ACTION_HEAD_CHECKPOINT_DEFAULT="${ACTION_HEAD_ROOT}/${ACTION_HEAD_NAME}"
BASE_CHECKPOINT="${SSPM_CHECKPOINT_PATH:-${BASE_CHECKPOINT_DEFAULT}}"
ACTION_HEAD_CHECKPOINT="${ACTION_HEAD_CHECKPOINT_PATH:-${ACTION_HEAD_CHECKPOINT_DEFAULT}}"

MAX_TRAIN_EVENTS="${MAX_TRAIN_EVENTS:-0}"
MAX_REF_EVENTS="${MAX_REF_EVENTS:-0}"
MAX_TEST_EVENTS="${MAX_TEST_EVENTS:-0}"
PROGRESS_INTERVAL_EVENTS="${PROGRESS_INTERVAL_EVENTS:-10000}"
FETCH_SIZE="${FETCH_SIZE:-10000}"
SSPM_EPOCHS="${SSPM_EPOCHS:-1}"
SSPM_TRAIN_BATCH_EVENTS="${SSPM_TRAIN_BATCH_EVENTS:-8192}"
SSPM_CONDITIONAL_MAX_EPOCHS="${SSPM_CONDITIONAL_MAX_EPOCHS:-80}"
SSPM_TORCH_BATCH_EVENTS="${SSPM_TORCH_BATCH_EVENTS:-8192}"
CONDITIONAL_GROUP_MIN_COUNT="${CONDITIONAL_GROUP_MIN_COUNT:-1000}"
EVENT_THRESHOLD_MODE="${EVENT_THRESHOLD_MODE:-target_case_quantile}"
EVENT_THRESHOLD_QUANTILE="${EVENT_THRESHOLD_QUANTILE:-0.9995}"
ACTION_TYPE_ALERT_POLICY="${ACTION_TYPE_ALERT_POLICY:-cadets_e4_v2_group_v1}"
CONDITIONAL_LOW_SUPPORT_POLICY="${CONDITIONAL_LOW_SUPPORT_POLICY:-conservative_max}"
CONDITIONAL_LOW_SUPPORT_MARGIN="${CONDITIONAL_LOW_SUPPORT_MARGIN:-0.05}"
CONDITIONAL_ENDPOINT_AWARE_SUPPRESSION="${CONDITIONAL_ENDPOINT_AWARE_SUPPRESSION:-false}"
CONDITIONAL_ENDPOINT_SUPPRESSION_READ="${CONDITIONAL_ENDPOINT_SUPPRESSION_READ:-false}"
CONDITIONAL_ENDPOINT_SUPPRESSION_MODE="${CONDITIONAL_ENDPOINT_SUPPRESSION_MODE:-pair_only}"
SSPM_STATE_MODEL="${SSPM_STATE_MODEL:-s4d_complex_node}"
SSPM_CONDITIONAL_HEAD_ARCH="${SSPM_CONDITIONAL_HEAD_ARCH:-dual_lowrank_by_target_case_v2}"
SSPM_INFER_BACKEND="${SSPM_INFER_BACKEND:-numpy}"
NODE_EMBEDDING_LOOKUP_MODE="${NODE_EMBEDDING_LOOKUP_MODE:-compact_used_nodes}"
NODE_EMBEDDING_LAZY_BACKING="${NODE_EMBEDDING_LAZY_BACKING:-compact_used_nodes}"

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

capture_command_if_requested() {
    if [[ -z "${RUNNER_ARG_CAPTURE+x}" ]]; then
        return 0
    fi
    printf '__CALL__\n' >> "${RUNNER_ARG_CAPTURE}"
    for arg in "$@"; do
        printf '%s\n' "${arg}" >> "${RUNNER_ARG_CAPTURE}"
    done
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

reject_forbidden_path() {
    local path="$1"
    local label="$2"
    case "${path}" in
        *RAW_DETAIL_RULES_V1*|*raw_detail_v2*|*CADETS_E3_latent64*|*phase3g_action_heads_v2*)
            echo "error: ${label} must not reference CADETS v2 artifacts: ${path}" >&2
            exit 2
            ;;
    esac
}

preflight() {
    if [[ "${DATASET}" != "CADETS_E3" ]]; then
        echo "error: this runner is restricted to CADETS_E3" >&2
        exit 2
    fi
    if [[ "${SEMANTIC_MODE}" != "cadets_freebsd_raw_detail_v3_safe_lexical" ]]; then
        echo "error: SEMANTIC_MODE must be cadets_freebsd_raw_detail_v3_safe_lexical" >&2
        exit 2
    fi
    case "${STAGE}" in
        all|build_phase3e_artifacts|train_phase3e_base|train_phase3g_head|infer_full) ;;
        *)
            echo "error: unsupported STAGE=${STAGE}" >&2
            exit 2
            ;;
    esac
    reject_forbidden_path "${EMBEDDER_PATH}" "Word2Vec embedder path"
    reject_forbidden_path "${NODE_CACHE_DIR}" "node cache path"
    reject_forbidden_path "${ACTION_CACHE_DIR}" "action cache path"
    reject_forbidden_path "${EVENT_INDEX_CACHE_DIR}" "event index cache path"
    reject_forbidden_path "${COMPACT_USED_NODE_CACHE_DIR}" "compact node cache path"
    reject_forbidden_path "${ACTION_VALIDATION_CACHE_DIR}" "action validation cache path"
    reject_forbidden_path "${ENDPOINT_CACHE_DIR}" "endpoint cache path"
    reject_forbidden_path "${BASE_CHECKPOINT}" "Phase3E checkpoint path"
    reject_forbidden_path "${ACTION_HEAD_CHECKPOINT}" "Phase3G checkpoint path"
    if [[ "${SSPM_CONDITIONAL_HEAD_ARCH}" != "dual_lowrank_by_target_case_v2" ]]; then
        echo "error: CADETS_E3 v3 E4 mainline requires dual_lowrank_by_target_case_v2" >&2
        exit 2
    fi
    if [[ "${ACTION_TYPE_ALERT_POLICY}" != "cadets_e4_v2_group_v1" ]]; then
        echo "error: CADETS_E3 v3 E4 mainline requires cadets_e4_v2_group_v1" >&2
        exit 2
    fi
    if is_true "${DRY_RUN}"; then
        require_file "${EMBEDDER_PATH}" "CADETS_E3 v3 residual Word2Vec embedder"
        return 0
    fi
    if [[ -z "${CLAD_DB_PASSWORD:-}" ]]; then
        echo "error: set CLAD_DB_PASSWORD before running CADETS_E3 v3 full serial" >&2
        exit 2
    fi
    require_file "${EMBEDDER_PATH}" "CADETS_E3 v3 residual Word2Vec embedder"
    if [[ "${STAGE}" == "train_phase3e_base" || "${STAGE}" == "train_phase3g_head" || \
        "${STAGE}" == "infer_full" ]]; then
        require_file "${EVENT_INDEX_CACHE_DIR}/event_index_meta.json" "Phase3E event index meta"
        require_file "${NODE_CACHE_DIR}/node_embeddings.npy" "Phase3E node embeddings"
        require_file "${ACTION_CACHE_DIR}/action_embeddings.npy" "Phase3E action embeddings"
    fi
    if [[ "${STAGE}" == "train_phase3g_head" || "${STAGE}" == "infer_full" ]]; then
        require_file "${BASE_CHECKPOINT}" "Phase3E base checkpoint"
    fi
    if [[ "${STAGE}" == "infer_full" ]]; then
        require_file "${ACTION_HEAD_CHECKPOINT}" "Phase3G conditional dual head checkpoint"
    fi
    require_writable_dir "${RESULT_ROOT}" "result root"
    require_writable_dir "${LOG_DIR}" "log directory"
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
        capture_command_if_requested "${cmd[@]}"
        return 0
    fi
    echo "===== START ${label} ${out_tag} $(date) ====="
    set +e
    "${cmd[@]}" 2>&1 | tee "${log_path}"
    local pipeline_status=("${PIPESTATUS[@]}")
    set -e
    local python_status="${pipeline_status[0]:-1}"
    local tee_status="${pipeline_status[1]:-1}"
    local status=0
    if [[ "${python_status}" -ne 0 ]]; then
        status="${python_status}"
    elif [[ "${tee_status}" -ne 0 ]]; then
        status="${tee_status}"
    fi
    if [[ "${status}" -ne 0 ]]; then
        echo "===== FAIL ${label} status=${status} $(date) ====="
        exit "${status}"
    fi
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
    --node_embedding_lookup_mode "${NODE_EMBEDDING_LOOKUP_MODE}"
    --node_embedding_lazy_backing "${NODE_EMBEDDING_LAZY_BACKING}"
    --node_embedding_cache_max_nodes 50000
    --node_embedding_cache_evict_policy lru
    --node_embedding_cache_dir "${NODE_CACHE_DIR}"
    --compact_used_node_cache_dir "${COMPACT_USED_NODE_CACHE_DIR}"
    --action_embedding_cache_dir "${ACTION_CACHE_DIR}"
    --event_index_cache_mode auto
    --event_index_cache_dir "${EVENT_INDEX_CACHE_DIR}"
    --sspm_infer_backend "${SSPM_INFER_BACKEND}"
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
    --conditional_endpoint_suppression_summary_mode online_minimal
    --conditional_endpoint_suppression_cache_dir "${ENDPOINT_CACHE_DIR}"
    --action_validation_cache_dir "${ACTION_VALIDATION_CACHE_DIR}"
    --sspm_base_checkpoint_root "${CHECKPOINT_ROOT}"
    --sspm_state_model "${SSPM_STATE_MODEL}"
    --sspm_update_gate_mode none
    --sspm_update_gate_score_space conditional_event_score
    --sspm_residual_score_mode legacy
    --sspm_residual_calibration none
    --sspm_state_merge_mode none
    --sspm_state_merge_threshold 0.98
    --sspm_score_target_mode event_action_semantic
    --write_raw_alerts true
    --write_analysis_outputs true
    --sspm_state_merge_write_diagnostics false
    --progress_interval_events "${PROGRESS_INTERVAL_EVENTS}"
    --fetch_size "${FETCH_SIZE}"
    --print_summary
)

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
    run_pipeline_stage "build_phase3e_artifacts" "CADETS_E3_${SEMANTIC_LABEL}_E4_ARTIFACT" \
        --sspm_train_mode build_phase3e_artifacts \
        --sspm_train_data_mode phase3e_memmap
}

run_train_phase3e_base() {
    run_pipeline_stage "train_phase3e_base" "CADETS_E3_${SEMANTIC_LABEL}_E4_BASE" \
        --sspm_train_mode train_phase3e_base \
        --sspm_train_data_mode phase3e_memmap \
        --sspm_checkpoint_path "${BASE_CHECKPOINT}" \
        --sspm_epochs "${SSPM_EPOCHS}" \
        --sspm_train_batch_events "${SSPM_TRAIN_BATCH_EVENTS}"
}

run_train_phase3g_head() {
    run_pipeline_stage "train_phase3g_head" "CADETS_E3_${SEMANTIC_LABEL}_E4_HEAD" \
        --sspm_train_mode train_conditional_and_save \
        --sspm_train_data_mode phase3e_memmap \
        --sspm_checkpoint_path "${BASE_CHECKPOINT}" \
        --action_head_checkpoint_path "${ACTION_HEAD_CHECKPOINT}" \
        --sspm_conditional_train_data_mode memmap \
        --sspm_conditional_max_epochs "${SSPM_CONDITIONAL_MAX_EPOCHS}" \
        --sspm_torch_batch_events "${SSPM_TORCH_BATCH_EVENTS}"
}

run_infer_full() {
    local out_tag="CADETS_E3_${SEMANTIC_LABEL}_E4_FULL"
    if [[ -n "${OUT_TAG_OVERRIDE:-}" ]]; then
        out_tag="${OUT_TAG_OVERRIDE}"
    fi
    run_pipeline_stage "infer_full" "${out_tag}" \
        --sspm_train_mode load_and_infer \
        --sspm_train_data_mode phase3e_memmap \
        --sspm_checkpoint_path "${BASE_CHECKPOINT}" \
        --action_head_checkpoint_path "${ACTION_HEAD_CHECKPOINT}"
}

preflight

case "${STAGE}" in
    build_phase3e_artifacts)
        run_build_phase3e_artifacts
        ;;
    train_phase3e_base)
        run_train_phase3e_base
        ;;
    train_phase3g_head)
        run_train_phase3g_head
        ;;
    infer_full)
        run_infer_full
        ;;
    all)
        run_build_phase3e_artifacts
        run_train_phase3e_base
        run_train_phase3g_head
        run_infer_full
        ;;
esac
