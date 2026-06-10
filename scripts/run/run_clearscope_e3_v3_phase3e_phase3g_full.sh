#!/usr/bin/env bash
set -euo pipefail

# ClearScope E3 raw_detail_v3_discriminative Phase3E/Phase3G runner.
# Training stages use train/validation event streams only; labels are used only by
# the pipeline's post-stream evaluation/reporting path after online inference.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${CS4M_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export CLAD_DB_HOST="${CLAD_DB_HOST:-localhost}"
export CLAD_DB_USER="${CLAD_DB_USER:-postgres}"
export CLAD_DB_PORT="${CLAD_DB_PORT:-5433}"

DATASET="${DATASET:-CLEARSCOPE_E3}"
SEMANTIC_MODE="${SEMANTIC_MODE:-raw_detail_v3_discriminative}"
STAGE="${STAGE:-all}"
DRY_RUN="${DRY_RUN:-0}"
BASELINE_INFER_OUT_TAG="CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_INFER_FULL"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/outputs/results/tflr_light}"
PHASE3E_CACHE_ROOT="${PHASE3E_CACHE_ROOT:-${REPO_ROOT}/outputs/cache/phase3e}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/clearscope_e3_v3_phase3e_phase3g}"
CHECKPOINT_ROOT="${SSPM_BASE_CHECKPOINT_ROOT:-${REPO_ROOT}/outputs/models/sspm_phase3e}"
ACTION_HEAD_ROOT="${ACTION_HEAD_CHECKPOINT_ROOT:-${REPO_ROOT}/outputs/models/phase3g_action_heads}"
ACTION_VALIDATION_DEFAULT="${REPO_ROOT}/outputs/cache/phase3g_action_validation/${DATASET}_v3"
ENDPOINT_CACHE_DEFAULT="${REPO_ROOT}/outputs/cache/phase3g_endpoint_suppression/${DATASET}_v3"
ACTION_VALIDATION_CACHE_DIR="${ACTION_VALIDATION_CACHE_DIR:-${ACTION_VALIDATION_DEFAULT}}"
ENDPOINT_CACHE_DIR="${CONDITIONAL_ENDPOINT_SUPPRESSION_CACHE_DIR:-${ENDPOINT_CACHE_DEFAULT}}"

EMBEDDER_NAME="CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_LATENT64_word2vec_window3.pkl"
EMBEDDER_DEFAULT="${REPO_ROOT}/outputs/models/residual_word2vec/${EMBEDDER_NAME}"
EMBEDDER_PATH="${PRETRAINED_RESIDUAL_EMBEDDER_PATH:-${EMBEDDER_DEFAULT}}"
NODE_CACHE_DEFAULT="${PHASE3E_CACHE_ROOT}/node_embeddings/${DATASET}_latent64_v3"
ACTION_CACHE_DEFAULT="${PHASE3E_CACHE_ROOT}/action_embeddings/${DATASET}_latent64_v3"
NODE_CACHE_DIR="${NODE_EMBEDDING_CACHE_DIR:-${NODE_CACHE_DEFAULT}}"
ACTION_CACHE_DIR="${ACTION_EMBEDDING_CACHE_DIR:-${ACTION_CACHE_DEFAULT}}"
EVENT_INDEX_CACHE_DIR="${EVENT_INDEX_CACHE_DIR:-${PHASE3E_CACHE_ROOT}/event_indices/${DATASET}_v3}"
COMPACT_USED_NODE_DEFAULT="${PHASE3E_CACHE_ROOT}/compact_used_node_embeddings/${DATASET}_v3"
COMPACT_USED_NODE_CACHE_DIR="${COMPACT_USED_NODE_CACHE_DIR:-${COMPACT_USED_NODE_DEFAULT}}"

BASE_CHECKPOINT_NAME="CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_PHASE3E_BASE_FULL.pkl"
ACTION_HEAD_NAME="CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_PHASE3G_CONDITIONAL_HEAD.pkl"
BASE_CHECKPOINT_DEFAULT="${CHECKPOINT_ROOT}/${BASE_CHECKPOINT_NAME}"
ACTION_HEAD_CHECKPOINT_DEFAULT="${ACTION_HEAD_ROOT}/${ACTION_HEAD_NAME}"
BASE_CHECKPOINT="${SSPM_CHECKPOINT_PATH:-${BASE_CHECKPOINT_DEFAULT}}"
ACTION_HEAD_CHECKPOINT="${ACTION_HEAD_CHECKPOINT_PATH:-${ACTION_HEAD_CHECKPOINT_DEFAULT}}"

MAX_TRAIN_EVENTS="${MAX_TRAIN_EVENTS:-0}"
MAX_REF_EVENTS="${MAX_REF_EVENTS:-0}"
MAX_TEST_EVENTS="${MAX_TEST_EVENTS:-0}"
PROGRESS_INTERVAL_EVENTS="${PROGRESS_INTERVAL_EVENTS:-500000}"
SSPM_EPOCHS="${SSPM_EPOCHS:-1}"
SSPM_TRAIN_BATCH_EVENTS="${SSPM_TRAIN_BATCH_EVENTS:-8192}"
SSPM_CONDITIONAL_MAX_EPOCHS="${SSPM_CONDITIONAL_MAX_EPOCHS:-80}"
SSPM_TORCH_BATCH_EVENTS="${SSPM_TORCH_BATCH_EVENTS:-8192}"
CONDITIONAL_GROUP_MIN_COUNT="${CONDITIONAL_GROUP_MIN_COUNT:-1000}"
EVENT_THRESHOLD_MODE="${EVENT_THRESHOLD_MODE:-quantile}"
EVENT_THRESHOLD_QUANTILE="${EVENT_THRESHOLD_QUANTILE:-0.999}"
ACTION_TYPE_ALERT_POLICY="${ACTION_TYPE_ALERT_POLICY:-default}"
CONDITIONAL_LOW_SUPPORT_POLICY="${CONDITIONAL_LOW_SUPPORT_POLICY:-conservative_max}"
CONDITIONAL_LOW_SUPPORT_MARGIN="${CONDITIONAL_LOW_SUPPORT_MARGIN:-0.05}"
CONDITIONAL_ENDPOINT_AWARE_SUPPRESSION="${CONDITIONAL_ENDPOINT_AWARE_SUPPRESSION:-false}"
CONDITIONAL_ENDPOINT_SUPPRESSION_READ="${CONDITIONAL_ENDPOINT_SUPPRESSION_READ:-false}"
CONDITIONAL_ENDPOINT_SUPPRESSION_MODE="${CONDITIONAL_ENDPOINT_SUPPRESSION_MODE:-pair_only}"
CONDITIONAL_BOTH_COLD_UNSEEN_POLICY="${CONDITIONAL_BOTH_COLD_UNSEEN_POLICY:-alert}"
SSPM_STATE_MODEL="${SSPM_STATE_MODEL:-s4d_complex_node}"
SSPM_CONDITIONAL_HEAD_ARCH="${SSPM_CONDITIONAL_HEAD_ARCH:-shared_lowrank_v1}"

policy_settings_differ_from_baseline() {
    [[ "${EVENT_THRESHOLD_MODE}" != "quantile" ]] && return 0
    [[ "${EVENT_THRESHOLD_QUANTILE}" != "0.999" ]] && return 0
    [[ "${ACTION_TYPE_ALERT_POLICY}" != "default" ]] && return 0
    [[ "${CONDITIONAL_LOW_SUPPORT_POLICY}" != "conservative_max" ]] && return 0
    [[ "${CONDITIONAL_LOW_SUPPORT_MARGIN}" != "0.05" ]] && return 0
    [[ "${CONDITIONAL_ENDPOINT_AWARE_SUPPRESSION}" != "false" ]] && return 0
    [[ "${CONDITIONAL_ENDPOINT_SUPPRESSION_READ}" != "false" ]] && return 0
    [[ "${CONDITIONAL_ENDPOINT_SUPPRESSION_MODE}" != "pair_only" ]] && return 0
    [[ "${CONDITIONAL_BOTH_COLD_UNSEEN_POLICY}" != "alert" ]] && return 0
    [[ "${SSPM_STATE_MODEL}" != "s4d_complex_node" ]] && return 0
    [[ "${SSPM_CONDITIONAL_HEAD_ARCH}" != "shared_lowrank_v1" ]] && return 0
    return 1
}

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

preflight() {
    if [[ "${DATASET}" != "CLEARSCOPE_E3" ]]; then
        echo "error: this runner is restricted to CLEARSCOPE_E3" >&2
        exit 2
    fi
    if [[ "${SEMANTIC_MODE}" != "raw_detail_v3_discriminative" ]]; then
        echo "error: this runner requires SEMANTIC_MODE=raw_detail_v3_discriminative" >&2
        exit 2
    fi
    case "${STAGE}" in
        all|build_phase3e_artifacts|train_phase3e_base|train_phase3g_head|infer_full) ;;
        *)
            echo "error: unsupported STAGE=${STAGE}" >&2
            exit 2
            ;;
    esac
    if [[ -n "${OUT_TAG_OVERRIDE:-}" && "${STAGE}" != "infer_full" ]]; then
        echo "error: OUT_TAG_OVERRIDE is only supported with STAGE=infer_full" >&2
        exit 2
    fi
    if [[ -n "${OUT_TAG_OVERRIDE:-}" ]]; then
        case "${OUT_TAG_OVERRIDE}" in
            */*|*\\*|*..*)
                echo "error: OUT_TAG_OVERRIDE must not contain slash, backslash, or '..'" >&2
                exit 2
                ;;
        esac
    fi
    if [[ ( "${STAGE}" == "infer_full" || "${STAGE}" == "all" ) && \
        -z "${OUT_TAG_OVERRIDE:-}" ]]; then
        if policy_settings_differ_from_baseline; then
            echo "error: baseline infer_full out_tag protection requires OUT_TAG_OVERRIDE" >&2
            exit 2
        fi
    fi
    if is_true "${DRY_RUN}"; then
        return 0
    fi
    require_file "${EMBEDDER_PATH}" "ClearScope E3 v3 residual Word2Vec embedder"
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
        require_file "${ACTION_HEAD_CHECKPOINT}" "Phase3G conditional head checkpoint"
    fi
    if ! is_true "${DRY_RUN}" && [[ -z "${CLAD_DB_PASSWORD:-}" ]]; then
        echo "error: set CLAD_DB_PASSWORD before running ClearScope E3 v3 full stages" >&2
        exit 2
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

common_args=(
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
    --print_summary
)

run_command() {
    local label="$1"
    local out_tag="$2"
    shift 2
    local log_path="${LOG_DIR}/${out_tag}.log"
    local cmd=("${PYTHON_BIN}" -m scripts.pipeline.entrypoints.conditional_e4 "$@")
    if is_true "${DRY_RUN}"; then
        echo "===== DRY_RUN ${label} ${out_tag} ====="
        print_command "${cmd[@]}"
        return 0
    fi
    echo "===== START ${label} ${out_tag} $(date) ====="
    "${cmd[@]}" 2>&1 | tee "${log_path}"
    echo "===== DONE ${label} ${out_tag} $(date) ====="
}

run_build_phase3e_artifacts() {
    local out_tag="CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_ARTIFACT_FULL"
    run_command "build_phase3e_artifacts" "${out_tag}" \
        "${common_args[@]}" \
        --out_tag "${out_tag}" \
        --sspm_train_mode build_phase3e_artifacts \
        --sspm_train_data_mode phase3e_memmap
}

run_train_phase3e_base() {
    local out_tag="CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_BASE_FULL"
    run_command "train_phase3e_base" "${out_tag}" \
        "${common_args[@]}" \
        --out_tag "${out_tag}" \
        --sspm_train_mode train_phase3e_base \
        --sspm_train_data_mode phase3e_memmap \
        --sspm_checkpoint_path "${BASE_CHECKPOINT}" \
        --sspm_epochs "${SSPM_EPOCHS}" \
        --sspm_train_batch_events "${SSPM_TRAIN_BATCH_EVENTS}"
}

run_train_phase3g_head() {
    local out_tag="CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_HEAD_FULL"
    run_command "train_phase3g_head" "${out_tag}" \
        "${common_args[@]}" \
        --out_tag "${out_tag}" \
        --sspm_train_mode train_conditional_and_save \
        --sspm_train_data_mode phase3e_memmap \
        --sspm_checkpoint_path "${BASE_CHECKPOINT}" \
        --action_head_checkpoint_path "${ACTION_HEAD_CHECKPOINT}" \
        --sspm_conditional_train_data_mode memmap \
        --sspm_conditional_max_epochs "${SSPM_CONDITIONAL_MAX_EPOCHS}" \
        --sspm_torch_batch_events "${SSPM_TORCH_BATCH_EVENTS}"
}

run_infer_full() {
    local out_tag="${BASELINE_INFER_OUT_TAG}"
    if [[ -n "${OUT_TAG_OVERRIDE:-}" ]]; then
        out_tag="${OUT_TAG_OVERRIDE}"
    fi
    run_command "infer_full" "${out_tag}" \
        "${common_args[@]}" \
        --out_tag "${out_tag}" \
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
