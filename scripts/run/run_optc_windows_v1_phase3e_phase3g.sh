#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

is_true() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

print_command() {
    printf '%q ' "$@"
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

RUN_PROFILE="${RUN_PROFILE:-smoke}"
STAGE="${STAGE:-all}"
RUN_DATASETS="${RUN_DATASETS:-OPTC_051 OPTC_201 OPTC_501}"
SEMANTIC_MODE="${SEMANTIC_MODE:-optc_windows_v1_detail}"
if [[ -z "${SEMANTIC_SUFFIX:-}" ]]; then
    case "${SEMANTIC_MODE}" in
        optc_windows_v1_3c_detail) SEMANTIC_SUFFIX="OPTC_WINDOWS_V1_3C_DETAIL" ;;
        optc_windows_v1_3c_coarse) SEMANTIC_SUFFIX="OPTC_WINDOWS_V1_3C_COARSE" ;;
        optc_windows_v1_3b_detail) SEMANTIC_SUFFIX="OPTC_WINDOWS_V1_3B_DETAIL" ;;
        optc_windows_v1_3b_coarse) SEMANTIC_SUFFIX="OPTC_WINDOWS_V1_3B_COARSE" ;;
        optc_windows_v1_3_detail) SEMANTIC_SUFFIX="OPTC_WINDOWS_V1_3_DETAIL" ;;
        optc_windows_v1_3_coarse) SEMANTIC_SUFFIX="OPTC_WINDOWS_V1_3_COARSE" ;;
        optc_windows_v1_coarse) SEMANTIC_SUFFIX="OPTC_WINDOWS_V1_COARSE" ;;
        *) SEMANTIC_SUFFIX="OPTC_WINDOWS_V1_DETAIL" ;;
    esac
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${SEMANTIC_SUFFIX}" == *"V1_3C"* ]]; then
    DEFAULT_RESULT_ROOT="${REPO_ROOT}/outputs/results/tflr_light/optc_windows_v1_3c_${RUN_PROFILE}"
    DEFAULT_CHECKPOINT_ROOT="${REPO_ROOT}/outputs/models/sspm_phase3e_optc_windows_v1_3c"
    DEFAULT_ACTION_HEAD_ROOT="${REPO_ROOT}/outputs/models/phase3g_action_heads_optc_windows_v1_3c"
    DEFAULT_PHASE3E_CACHE_ROOT="${REPO_ROOT}/outputs/cache/phase3e_optc_windows_v1_3c"
    DEFAULT_ACTION_VALIDATION_ROOT="${REPO_ROOT}/outputs/cache/phase3g_action_validation_optc_windows_v1_3c"
    DEFAULT_ENDPOINT_CACHE_ROOT="${REPO_ROOT}/outputs/cache/phase3g_endpoint_suppression_optc_windows_v1_3c"
    DEFAULT_WORD2VEC_CORPUS_ROOT="${REPO_ROOT}/tmp/word2vec_corpus_optc_windows_v1_3c"
elif [[ "${SEMANTIC_SUFFIX}" == *"V1_3B"* ]]; then
    DEFAULT_RESULT_ROOT="${REPO_ROOT}/outputs/results/tflr_light/optc_windows_v1_3b_${RUN_PROFILE}"
    DEFAULT_CHECKPOINT_ROOT="${REPO_ROOT}/outputs/models/sspm_phase3e_optc_windows_v1_3b"
    DEFAULT_ACTION_HEAD_ROOT="${REPO_ROOT}/outputs/models/phase3g_action_heads_optc_windows_v1_3b"
    DEFAULT_PHASE3E_CACHE_ROOT="${REPO_ROOT}/outputs/cache/phase3e_optc_windows_v1_3b"
    DEFAULT_ACTION_VALIDATION_ROOT="${REPO_ROOT}/outputs/cache/phase3g_action_validation_optc_windows_v1_3b"
    DEFAULT_ENDPOINT_CACHE_ROOT="${REPO_ROOT}/outputs/cache/phase3g_endpoint_suppression_optc_windows_v1_3b"
    DEFAULT_WORD2VEC_CORPUS_ROOT="${REPO_ROOT}/tmp/word2vec_corpus_optc_windows_v1_3b"
elif [[ "${SEMANTIC_SUFFIX}" == *"V1_3"* ]]; then
    DEFAULT_RESULT_ROOT="${REPO_ROOT}/outputs/results/tflr_light/optc_windows_v1_3_${RUN_PROFILE}"
    DEFAULT_CHECKPOINT_ROOT="${REPO_ROOT}/outputs/models/sspm_phase3e_optc_windows_v1_3"
    DEFAULT_ACTION_HEAD_ROOT="${REPO_ROOT}/outputs/models/phase3g_action_heads_optc_windows_v1_3"
    DEFAULT_PHASE3E_CACHE_ROOT="${REPO_ROOT}/outputs/cache/phase3e_optc_windows_v1_3"
    DEFAULT_ACTION_VALIDATION_ROOT="${REPO_ROOT}/outputs/cache/phase3g_action_validation_optc_windows_v1_3"
    DEFAULT_ENDPOINT_CACHE_ROOT="${REPO_ROOT}/outputs/cache/phase3g_endpoint_suppression_optc_windows_v1_3"
    DEFAULT_WORD2VEC_CORPUS_ROOT="${REPO_ROOT}/tmp/word2vec_corpus_optc_windows_v1_3"
else
    DEFAULT_RESULT_ROOT="${REPO_ROOT}/outputs/results/tflr_light/optc_windows_v1_${RUN_PROFILE}"
    DEFAULT_CHECKPOINT_ROOT="${REPO_ROOT}/outputs/models/sspm_phase3e_optc_windows_v1"
    DEFAULT_ACTION_HEAD_ROOT="${REPO_ROOT}/outputs/models/phase3g_action_heads_optc_windows_v1"
    DEFAULT_PHASE3E_CACHE_ROOT="${REPO_ROOT}/outputs/cache/phase3e_optc_windows_v1"
    DEFAULT_ACTION_VALIDATION_ROOT="${REPO_ROOT}/outputs/cache/phase3g_action_validation_optc_windows_v1"
    DEFAULT_ENDPOINT_CACHE_ROOT="${REPO_ROOT}/outputs/cache/phase3g_endpoint_suppression_optc_windows_v1"
    DEFAULT_WORD2VEC_CORPUS_ROOT="${REPO_ROOT}/tmp/word2vec_corpus_optc_windows_v1"
fi

RESULT_ROOT="${RESULT_ROOT:-${DEFAULT_RESULT_ROOT}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/optc_windows_v1_phase3e_phase3g/${RUN_PROFILE}}"
WORD2VEC_ROOT="${WORD2VEC_ROOT:-${REPO_ROOT}/outputs/models/residual_word2vec}"
CHECKPOINT_ROOT="${SSPM_BASE_CHECKPOINT_ROOT:-${DEFAULT_CHECKPOINT_ROOT}}"
ACTION_HEAD_ROOT="${ACTION_HEAD_CHECKPOINT_ROOT:-${DEFAULT_ACTION_HEAD_ROOT}}"
PHASE3E_CACHE_ROOT="${PHASE3E_CACHE_ROOT:-${DEFAULT_PHASE3E_CACHE_ROOT}}"
ACTION_VALIDATION_ROOT="${ACTION_VALIDATION_ROOT:-${DEFAULT_ACTION_VALIDATION_ROOT}}"
ENDPOINT_CACHE_ROOT="${ENDPOINT_CACHE_ROOT:-${DEFAULT_ENDPOINT_CACHE_ROOT}}"
WORD2VEC_CORPUS_ROOT="${WORD2VEC_CORPUS_ROOT:-${DEFAULT_WORD2VEC_CORPUS_ROOT}}"

case "${RUN_PROFILE}" in
    smoke)
        MAX_TRAIN_EVENTS="${MAX_TRAIN_EVENTS:-20000}"
        MAX_REF_EVENTS="${MAX_REF_EVENTS:-10000}"
        MAX_TEST_EVENTS="${MAX_TEST_EVENTS:-20000}"
        WORD2VEC_EPOCHS="${WORD2VEC_EPOCHS:-2}"
        SSPM_EPOCHS="${SSPM_EPOCHS:-1}"
        SSPM_CONDITIONAL_MAX_EPOCHS="${SSPM_CONDITIONAL_MAX_EPOCHS:-2}"
        ;;
    full)
        MAX_TRAIN_EVENTS="${MAX_TRAIN_EVENTS:-0}"
        MAX_REF_EVENTS="${MAX_REF_EVENTS:-0}"
        MAX_TEST_EVENTS="${MAX_TEST_EVENTS:-0}"
        WORD2VEC_EPOCHS="${WORD2VEC_EPOCHS:-10}"
        SSPM_EPOCHS="${SSPM_EPOCHS:-1}"
        SSPM_CONDITIONAL_MAX_EPOCHS="${SSPM_CONDITIONAL_MAX_EPOCHS:-80}"
        ;;
    *)
        echo "error: RUN_PROFILE must be smoke or full, got ${RUN_PROFILE}" >&2
        exit 2
        ;;
esac

FETCH_SIZE="${FETCH_SIZE:-10000}"
PROGRESS_INTERVAL_EVENTS="${PROGRESS_INTERVAL_EVENTS:-100000}"
WORD2VEC_WORKERS="${WORD2VEC_WORKERS:-4}"
SSPM_TRAIN_BATCH_EVENTS="${SSPM_TRAIN_BATCH_EVENTS:-8192}"
SSPM_TORCH_BATCH_EVENTS="${SSPM_TORCH_BATCH_EVENTS:-8192}"
EVENT_THRESHOLD_MODE="${EVENT_THRESHOLD_MODE:-conditional_target_action_type_group_quantile}"
EVENT_THRESHOLD_QUANTILE="${EVENT_THRESHOLD_QUANTILE:-0.999}"
ACTION_TYPE_ALERT_POLICY="${ACTION_TYPE_ALERT_POLICY:-default}"
NODE_POOL_SCORE_MODE="${NODE_POOL_SCORE_MODE:-base_conf}"
SSPM_STATE_MODEL="${SSPM_STATE_MODEL:-s4d_complex_node}"
SSPM_CONDITIONAL_HEAD_ARCH="${SSPM_CONDITIONAL_HEAD_ARCH:-dual_lowrank_by_target_case_v2}"
CONDITIONAL_GROUP_MIN_COUNT="${CONDITIONAL_GROUP_MIN_COUNT:-1000}"
CONDITIONAL_LOW_SUPPORT_POLICY="${CONDITIONAL_LOW_SUPPORT_POLICY:-adaptive_margin}"
CONDITIONAL_LOW_SUPPORT_MARGIN="${CONDITIONAL_LOW_SUPPORT_MARGIN:-0.02}"
CONDITIONAL_ENDPOINT_AWARE_SUPPRESSION="${CONDITIONAL_ENDPOINT_AWARE_SUPPRESSION:-false}"
CONDITIONAL_ENDPOINT_SUPPRESSION_READ="${CONDITIONAL_ENDPOINT_SUPPRESSION_READ:-false}"
CONDITIONAL_ENDPOINT_SUPPRESSION_MODE="${CONDITIONAL_ENDPOINT_SUPPRESSION_MODE:-pair_only}"
CONDITIONAL_BOTH_COLD_UNSEEN_POLICY="${CONDITIONAL_BOTH_COLD_UNSEEN_POLICY:-observation_only_no_alert}"
OPTC_NETFLOW_NODE_CANONICALIZATION="${OPTC_NETFLOW_NODE_CANONICALIZATION:-none}"
PHASE3E_BASE_PROFILE="${PHASE3E_BASE_PROFILE:-false}"
PHASE3E_PROFILE_INTERVAL_EVENTS="${PHASE3E_PROFILE_INTERVAL_EVENTS:-100000}"
SLIM_SPLIT_OVERRIDE="${SLIM_SPLIT_OVERRIDE:-false}"

dataset_db_name() {
    case "$1" in
        OPTC_051) echo "optc_051" ;;
        OPTC_201) echo "optc_201" ;;
        OPTC_501) echo "optc_501" ;;
        *)
            echo "error: unsupported OpTC dataset: $1" >&2
            exit 2
            ;;
    esac
}

preflight() {
    case "${STAGE}" in
        all|train_word2vec|build_phase3e_artifacts|train_phase3e_base|train_phase3g_head|infer) ;;
        *)
            echo "error: unsupported STAGE=${STAGE}" >&2
            exit 2
            ;;
    esac
    if [[ "${SEMANTIC_MODE}" != "optc_windows_v1_detail" && \
        "${SEMANTIC_MODE}" != "optc_windows_v1_coarse" && \
        "${SEMANTIC_MODE}" != "optc_windows_v1_3_detail" && \
        "${SEMANTIC_MODE}" != "optc_windows_v1_3_coarse" && \
        "${SEMANTIC_MODE}" != "optc_windows_v1_3b_detail" && \
        "${SEMANTIC_MODE}" != "optc_windows_v1_3b_coarse" && \
        "${SEMANTIC_MODE}" != "optc_windows_v1_3c_detail" && \
        "${SEMANTIC_MODE}" != "optc_windows_v1_3c_coarse" ]]; then
        echo "error: unsupported OpTC semantic mode: ${SEMANTIC_MODE}" >&2
        exit 2
    fi
    if ! is_true "${DRY_RUN}" && [[ -z "${CLAD_DB_PASSWORD:-}" ]]; then
        echo "error: set CLAD_DB_PASSWORD before running OpTC Windows v1 stages" >&2
        exit 2
    fi
    for dataset in ${RUN_DATASETS}; do
        dataset_db_name "${dataset}" >/dev/null
    done
    require_writable_dir "${RESULT_ROOT}" "result root"
    require_writable_dir "${LOG_DIR}" "log directory"
    require_writable_dir "${WORD2VEC_ROOT}" "Word2Vec root"
    require_writable_dir "${CHECKPOINT_ROOT}" "checkpoint root"
    require_writable_dir "${ACTION_HEAD_ROOT}" "action head checkpoint root"
    require_writable_dir "${PHASE3E_CACHE_ROOT}" "Phase3E cache root"
    require_writable_dir "${ACTION_VALIDATION_ROOT}" "action validation root"
    require_writable_dir "${ENDPOINT_CACHE_ROOT}" "endpoint suppression root"
    require_writable_dir "${WORD2VEC_CORPUS_ROOT}" "Word2Vec corpus root"
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

common_pipeline_args() {
    local dataset="$1"
    local embedder_path="$2"
    local node_cache_dir="$3"
    local compact_cache_dir="$4"
    local action_cache_dir="$5"
    local event_index_cache_dir="$6"
    local action_validation_cache_dir="$7"
    local endpoint_cache_dir="$8"
    printf '%s\n' \
        --dataset "${dataset}" \
        --result_root "${RESULT_ROOT}" \
        --max_train_events "${MAX_TRAIN_EVENTS}" \
        --max_ref_events "${MAX_REF_EVENTS}" \
        --max_test_events "${MAX_TEST_EVENTS}" \
        --event_score_mode res_only \
        --event_threshold_mode "${EVENT_THRESHOLD_MODE}" \
        --event_threshold_quantile "${EVENT_THRESHOLD_QUANTILE}" \
        --action_type_alert_policy "${ACTION_TYPE_ALERT_POLICY}" \
        --node_pool_score_mode "${NODE_POOL_SCORE_MODE}" \
        --pretrained_residual_embedder_path "${embedder_path}" \
        --semantic_embedding_method word2vec \
        --semantic_mode "${SEMANTIC_MODE}" \
        --word2vec_window 3 \
        --sspm_target_dim 64 \
        --sspm_state_dim 64 \
        --rank 32 \
        --sspm_context_mode with_action \
        --sspm_global_context_mode no_global \
        --state_memory_mode bounded \
        --sspm_state_memory_policy probationary_lru \
        --sspm_target_mode node_action_semantic_mean \
        --node_word2vec_source residual_pretrained \
        --node_embedding_lookup_mode global_memmap \
        --node_embedding_lazy_backing compact_used_nodes \
        --node_embedding_cache_max_nodes 50000 \
        --node_embedding_cache_evict_policy lru \
        --node_embedding_cache_dir "${node_cache_dir}" \
        --compact_used_node_cache_dir "${compact_cache_dir}" \
        --action_embedding_cache_dir "${action_cache_dir}" \
        --event_index_cache_mode auto \
        --event_index_cache_dir "${event_index_cache_dir}" \
        --optc_netflow_node_canonicalization "${OPTC_NETFLOW_NODE_CANONICALIZATION}" \
        --phase3e_profile_interval_events "${PHASE3E_PROFILE_INTERVAL_EVENTS}" \
        --sspm_infer_backend numpy \
        --sspm_score_head conditional_action_semantic \
        --node_repr_fusion simple_mean \
        --conditional_semantic_loss cosine \
        --sspm_conditional_head_arch "${SSPM_CONDITIONAL_HEAD_ARCH}" \
        --conditional_group_min_count "${CONDITIONAL_GROUP_MIN_COUNT}" \
        --conditional_low_support_policy "${CONDITIONAL_LOW_SUPPORT_POLICY}" \
        --conditional_low_support_margin "${CONDITIONAL_LOW_SUPPORT_MARGIN}" \
        --conditional_unseen_group_policy observation_only \
        --conditional_global_extreme_quantile 0.9999 \
        --conditional_adaptive_margin_n1 50 \
        --conditional_adaptive_margin_n2 200 \
        --conditional_adaptive_margin_low 0.15 \
        --conditional_adaptive_margin_mid 0.05 \
        --conditional_adaptive_margin_high 0.02 \
        --conditional_endpoint_aware_suppression "${CONDITIONAL_ENDPOINT_AWARE_SUPPRESSION}" \
        --conditional_endpoint_suppression_read "${CONDITIONAL_ENDPOINT_SUPPRESSION_READ}" \
        --conditional_endpoint_suppression_mode "${CONDITIONAL_ENDPOINT_SUPPRESSION_MODE}" \
        --conditional_both_cold_unseen_policy "${CONDITIONAL_BOTH_COLD_UNSEEN_POLICY}" \
        --conditional_endpoint_suppression_summary_mode online_minimal \
        --conditional_endpoint_suppression_cache_dir "${endpoint_cache_dir}" \
        --action_validation_cache_dir "${action_validation_cache_dir}" \
        --sspm_base_checkpoint_root "${CHECKPOINT_ROOT}" \
        --sspm_state_model "${SSPM_STATE_MODEL}" \
        --sspm_update_gate_mode none \
        --sspm_residual_score_mode legacy \
        --sspm_residual_calibration none \
        --sspm_state_merge_mode none \
        --sspm_state_merge_threshold 0.98 \
        --write_raw_alerts true \
        --write_analysis_outputs true \
        --sspm_state_merge_write_diagnostics false \
        --progress_interval_events "${PROGRESS_INTERVAL_EVENTS}" \
        --fetch_size "${FETCH_SIZE}" \
        --print_summary
    if is_true "${PHASE3E_BASE_PROFILE}"; then
        printf '%s\n' --phase3e_base_profile
    fi
    if is_true "${SLIM_SPLIT_OVERRIDE}"; then
        printf '%s\n' --slim_split_override
    fi
    return 0
}

run_for_dataset() {
    local dataset="$1"
    local profile_suffix
    profile_suffix="$(printf '%s' "${RUN_PROFILE}" | tr '[:lower:]' '[:upper:]')"
    local word2vec_tag="${SEMANTIC_SUFFIX}_${profile_suffix}"
    local embedder_path="${WORD2VEC_ROOT}/${dataset}_${word2vec_tag}_word2vec_window3.pkl"
    local node_cache_dir="${PHASE3E_CACHE_ROOT}/${dataset}_${SEMANTIC_SUFFIX}_${RUN_PROFILE}/node"
    local compact_cache_dir="${PHASE3E_CACHE_ROOT}/${dataset}_${SEMANTIC_SUFFIX}_${RUN_PROFILE}/compact_used_nodes"
    local action_cache_dir="${PHASE3E_CACHE_ROOT}/${dataset}_${SEMANTIC_SUFFIX}_${RUN_PROFILE}/action"
    local event_index_cache_dir="${PHASE3E_CACHE_ROOT}/${dataset}_${SEMANTIC_SUFFIX}_${RUN_PROFILE}/event_index"
    local action_validation_cache_dir="${ACTION_VALIDATION_ROOT}/${dataset}_${SEMANTIC_SUFFIX}_${RUN_PROFILE}"
    local endpoint_cache_dir="${ENDPOINT_CACHE_ROOT}/${dataset}_${SEMANTIC_SUFFIX}_${RUN_PROFILE}"
    local base_checkpoint="${CHECKPOINT_ROOT}/${dataset}_${SEMANTIC_SUFFIX}_${profile_suffix}_PHASE3E_BASE.pkl"
    local action_head_checkpoint="${ACTION_HEAD_ROOT}/${dataset}_${SEMANTIC_SUFFIX}_${profile_suffix}_PHASE3G_HEAD.pkl"
    local artifact_tag="${dataset}_${SEMANTIC_SUFFIX}_${profile_suffix}_ARTIFACT"
    local base_tag="${dataset}_${SEMANTIC_SUFFIX}_${profile_suffix}_BASE"
    local head_tag="${dataset}_${SEMANTIC_SUFFIX}_${profile_suffix}_HEAD"
    local infer_tag="${dataset}_${SEMANTIC_SUFFIX}_${profile_suffix}_INFER"

    local pipeline_args=()
    while IFS= read -r arg; do
        pipeline_args+=("${arg}")
    done < <(
        common_pipeline_args \
            "${dataset}" \
            "${embedder_path}" \
            "${node_cache_dir}" \
            "${compact_cache_dir}" \
            "${action_cache_dir}" \
            "${event_index_cache_dir}" \
            "${action_validation_cache_dir}" \
            "${endpoint_cache_dir}"
    )

    if [[ "${STAGE}" == "train_word2vec" || "${STAGE}" == "all" ]]; then
        run_logged_command "train_word2vec" "${dataset}_${word2vec_tag}_WORD2VEC" \
            "${PYTHON_BIN}" legacy/tools/train_residual_word2vec_models.py \
            --datasets "${dataset}" \
            --out_dir "${WORD2VEC_ROOT}" \
            --out_tag "${word2vec_tag}" \
            --max_train_events "${MAX_TRAIN_EVENTS}" \
            --fetch_size "${FETCH_SIZE}" \
            --progress_interval_events "${PROGRESS_INTERVAL_EVENTS}" \
            --semantic_mode "${SEMANTIC_MODE}" \
            --word2vec_window 3 \
            --word2vec_min_count 1 \
            --word2vec_epochs "${WORD2VEC_EPOCHS}" \
            --word2vec_workers "${WORD2VEC_WORKERS}" \
            --word2vec_corpus_file_dir "${WORD2VEC_CORPUS_ROOT}"
    fi

    if ! is_true "${DRY_RUN}" && [[ "${STAGE}" != "train_word2vec" ]]; then
        require_file "${embedder_path}" "OpTC residual Word2Vec embedder"
    fi

    if [[ "${STAGE}" == "build_phase3e_artifacts" || "${STAGE}" == "all" ]]; then
        run_logged_command "build_phase3e_artifacts" "${artifact_tag}" \
            "${PYTHON_BIN}" -m scripts.pipeline.entrypoints.conditional_e4 \
            "${pipeline_args[@]}" \
            --out_tag "${artifact_tag}" \
            --sspm_train_mode build_phase3e_artifacts \
            --sspm_train_data_mode phase3e_memmap
    fi

    if [[ "${STAGE}" == "train_phase3e_base" || "${STAGE}" == "all" ]]; then
        run_logged_command "train_phase3e_base" "${base_tag}" \
            "${PYTHON_BIN}" -m scripts.pipeline.entrypoints.conditional_e4 \
            "${pipeline_args[@]}" \
            --out_tag "${base_tag}" \
            --sspm_train_mode train_phase3e_base \
            --sspm_train_data_mode phase3e_memmap \
            --sspm_checkpoint_path "${base_checkpoint}" \
            --sspm_epochs "${SSPM_EPOCHS}" \
            --sspm_train_batch_events "${SSPM_TRAIN_BATCH_EVENTS}"
    fi

    if ! is_true "${DRY_RUN}" && \
        [[ "${STAGE}" == "train_phase3g_head" || "${STAGE}" == "infer" ]]; then
        require_file "${base_checkpoint}" "OpTC Phase3E base checkpoint"
    fi

    if [[ "${STAGE}" == "train_phase3g_head" || "${STAGE}" == "all" ]]; then
        run_logged_command "train_phase3g_head" "${head_tag}" \
            "${PYTHON_BIN}" -m scripts.pipeline.entrypoints.conditional_e4 \
            "${pipeline_args[@]}" \
            --out_tag "${head_tag}" \
            --sspm_train_mode train_conditional_and_save \
            --sspm_train_data_mode phase3e_memmap \
            --sspm_checkpoint_path "${base_checkpoint}" \
            --action_head_checkpoint_path "${action_head_checkpoint}" \
            --sspm_conditional_train_data_mode memmap \
            --sspm_conditional_max_epochs "${SSPM_CONDITIONAL_MAX_EPOCHS}" \
            --sspm_torch_batch_events "${SSPM_TORCH_BATCH_EVENTS}"
    fi

    if ! is_true "${DRY_RUN}" && [[ "${STAGE}" == "infer" ]]; then
        require_file "${action_head_checkpoint}" "OpTC Phase3G conditional head"
    fi

    if [[ "${STAGE}" == "infer" || "${STAGE}" == "all" ]]; then
        run_logged_command "infer" "${infer_tag}" \
            "${PYTHON_BIN}" -m scripts.pipeline.entrypoints.conditional_e4 \
            "${pipeline_args[@]}" \
            --out_tag "${infer_tag}" \
            --sspm_train_mode load_and_infer \
            --sspm_train_data_mode phase3e_memmap \
            --sspm_checkpoint_path "${base_checkpoint}" \
            --action_head_checkpoint_path "${action_head_checkpoint}"
    fi
}

preflight

echo "RUN_PROFILE=${RUN_PROFILE}"
echo "STAGE=${STAGE}"
echo "RUN_DATASETS=${RUN_DATASETS}"
echo "SEMANTIC_MODE=${SEMANTIC_MODE}"
echo "SLIM_SPLIT_OVERRIDE=${SLIM_SPLIT_OVERRIDE}"

for dataset in ${RUN_DATASETS}; do
    echo "===== DATASET ${dataset} START $(date) ====="
    run_for_dataset "${dataset}"
    echo "===== DATASET ${dataset} DONE $(date) ====="
done
