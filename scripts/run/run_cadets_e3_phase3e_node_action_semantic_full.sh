#!/usr/bin/env bash
set -euo pipefail

# CADETS_E3 Phase3E node/action semantic runner.
# This cycle allows precompute, explicitly approved train_base, and approved infer_ablation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${CS4M_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export CLAD_DB_HOST="${CLAD_DB_HOST:-localhost}"
export CLAD_DB_USER="${CLAD_DB_USER:-postgres}"
export CLAD_DB_PORT="${CLAD_DB_PORT:-5433}"

DATASET="${DATASET:-CADETS_E3}"
SEMANTIC_MODE="${SEMANTIC_MODE:-raw_detail_v2_refined}"
STAGE="${STAGE:-precompute}"
RUN_ONLY="${RUN_ONLY:-ALL}"
DRY_RUN="${DRY_RUN:-0}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/outputs/results/tflr_light}"
PHASE3E_CACHE_ROOT="${PHASE3E_CACHE_ROOT:-${REPO_ROOT}/outputs/cache/phase3e}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/cadets_e3_phase3e_node_action_semantic}"
RSS_PROFILE_MODE="${RSS_PROFILE_MODE:-deploy_light}"
SSPM_INFER_FAST_PATH="${SSPM_INFER_FAST_PATH:-false}"
SSPM_INFER_CHUNK_EVENTS="${SSPM_INFER_CHUNK_EVENTS:-8192}"
SSPM_TARGET_MODE="${SSPM_TARGET_MODE:-node_action_semantic_mean}"
NODE_WORD2VEC_SOURCE="${NODE_WORD2VEC_SOURCE:-residual_pretrained}"
NODE_EMBEDDING_LOOKUP_MODE="${NODE_EMBEDDING_LOOKUP_MODE:-global_memmap}"
NODE_EMBEDDING_LAZY_BACKING="${NODE_EMBEDDING_LAZY_BACKING:-compact_used_nodes}"
NODE_EMBEDDING_CACHE_MAX_NODES="${NODE_EMBEDDING_CACHE_MAX_NODES:-50000}"
NODE_EMBEDDING_CACHE_EVICT_POLICY="${NODE_EMBEDDING_CACHE_EVICT_POLICY:-lru}"
SSPM_SCORE_HEAD="${SSPM_SCORE_HEAD:-semantic_residual}"
NODE_REPR_FUSION="${NODE_REPR_FUSION:-simple_mean}"
CONDITIONAL_SEMANTIC_LOSS="${CONDITIONAL_SEMANTIC_LOSS:-cosine}"
SSPM_CONDITIONAL_HEAD_ARCH="${SSPM_CONDITIONAL_HEAD_ARCH:-shared_lowrank_v1}"
CONDITIONAL_GROUP_MIN_COUNT="${CONDITIONAL_GROUP_MIN_COUNT:-1000}"
CONDITIONAL_LOW_SUPPORT_POLICY="${CONDITIONAL_LOW_SUPPORT_POLICY:-conservative_max}"
CONDITIONAL_LOW_SUPPORT_MARGIN="${CONDITIONAL_LOW_SUPPORT_MARGIN:-0.05}"
CONDITIONAL_UNSEEN_GROUP_POLICY="${CONDITIONAL_UNSEEN_GROUP_POLICY:-observation_only}"
CONDITIONAL_GLOBAL_EXTREME_QUANTILE="${CONDITIONAL_GLOBAL_EXTREME_QUANTILE:-0.9999}"
CONDITIONAL_ADAPTIVE_MARGIN_N1="${CONDITIONAL_ADAPTIVE_MARGIN_N1:-50}"
CONDITIONAL_ADAPTIVE_MARGIN_N2="${CONDITIONAL_ADAPTIVE_MARGIN_N2:-200}"
CONDITIONAL_ADAPTIVE_MARGIN_LOW="${CONDITIONAL_ADAPTIVE_MARGIN_LOW:-0.15}"
CONDITIONAL_ADAPTIVE_MARGIN_MID="${CONDITIONAL_ADAPTIVE_MARGIN_MID:-0.05}"
CONDITIONAL_ADAPTIVE_MARGIN_HIGH="${CONDITIONAL_ADAPTIVE_MARGIN_HIGH:-0.02}"
CONDITIONAL_ENDPOINT_AWARE_SUPPRESSION="${CONDITIONAL_ENDPOINT_AWARE_SUPPRESSION:-false}"
CONDITIONAL_ENDPOINT_SUPPRESSION_READ="${CONDITIONAL_ENDPOINT_SUPPRESSION_READ:-false}"
CONDITIONAL_ENDPOINT_SUPPRESSION_MODE="${CONDITIONAL_ENDPOINT_SUPPRESSION_MODE:-pair_only}"
CONDITIONAL_KNOWN_PAIR_MIN_COUNT="${CONDITIONAL_KNOWN_PAIR_MIN_COUNT:-5}"
CONDITIONAL_PAIR_SUPPRESSION_MARGIN="${CONDITIONAL_PAIR_SUPPRESSION_MARGIN:-0.02}"
CONDITIONAL_SAME_PROCESS_ENDPOINT_MIN_COUNT="${CONDITIONAL_SAME_PROCESS_ENDPOINT_MIN_COUNT:-5}"
CONDITIONAL_SAME_PROCESS_ENDPOINT_MARGIN="${CONDITIONAL_SAME_PROCESS_ENDPOINT_MARGIN:-0.01}"
CONDITIONAL_ENDPOINT_SUPPRESSION_SUMMARY_MODE="${CONDITIONAL_ENDPOINT_SUPPRESSION_SUMMARY_MODE:-online_minimal}"
if [[ -z "${CONDITIONAL_ENDPOINT_SUPPRESSION_CACHE_DIR+x}" ]]; then
    if [[ "${NODE_EMBEDDING_LOOKUP_MODE}" == "compact_used_nodes" ]] || \
        [[ "${NODE_EMBEDDING_LOOKUP_MODE}" == "lazy_mmap_lru" && \
            "${NODE_EMBEDDING_LAZY_BACKING}" == "compact_used_nodes" ]]; then
        CONDITIONAL_ENDPOINT_SUPPRESSION_CACHE_DIR="${REPO_ROOT}/outputs/cache/phase3g_endpoint_suppression_compact/${DATASET}"
    else
        CONDITIONAL_ENDPOINT_SUPPRESSION_CACHE_DIR="${REPO_ROOT}/outputs/cache/phase3g_endpoint_suppression/${DATASET}"
    fi
fi
SSPM_CONDITIONAL_TRAIN_DATA_MODE="${SSPM_CONDITIONAL_TRAIN_DATA_MODE:-memmap}"
SSPM_CONDITIONAL_MAX_EPOCHS="${SSPM_CONDITIONAL_MAX_EPOCHS:-80}"
SSPM_CONDITIONAL_E3_MAX_EPOCHS="${SSPM_CONDITIONAL_E3_MAX_EPOCHS:-2}"
SSPM_TRAIN_BACKEND="${SSPM_TRAIN_BACKEND:-torch}"
SSPM_INFER_BACKEND="${SSPM_INFER_BACKEND:-numpy}"
EVENT_THRESHOLD_MODE="${EVENT_THRESHOLD_MODE:-quantile}"
EVENT_THRESHOLD_QUANTILE="${EVENT_THRESHOLD_QUANTILE:-0.999}"
SSPM_UPDATE_GATE_THRESHOLD_MODE="${SSPM_UPDATE_GATE_THRESHOLD_MODE:-}"
SSPM_UPDATE_GATE_SCORE_SPACE="${SSPM_UPDATE_GATE_SCORE_SPACE:-checkpoint}"
SSPM_UPDATE_GATE_QUANTILE="${SSPM_UPDATE_GATE_QUANTILE:-0.999}"
SSPM_UPDATE_GATE_Q_MIN="${SSPM_UPDATE_GATE_Q_MIN:-0.05}"
SSPM_UPDATE_GATE_ETA="${SSPM_UPDATE_GATE_ETA:-1.0}"
SSPM_SCORE_TARGET_MODE="${SSPM_SCORE_TARGET_MODE:-event_action_semantic}"
ACTION_TYPE_ALERT_POLICY="${ACTION_TYPE_ALERT_POLICY:-default}"
MAX_TRAIN_EVENTS="${MAX_TRAIN_EVENTS:-0}"
MAX_REF_EVENTS="${MAX_REF_EVENTS:-0}"
MAX_TEST_EVENTS="${MAX_TEST_EVENTS:-0}"
SSPM_TORCH_BATCH_EVENTS="${SSPM_TORCH_BATCH_EVENTS:-8192}"
SSPM_TORCH_DEVICE="${SSPM_TORCH_DEVICE:-auto}"
PHASE3E_MIN_FREE_KB="${PHASE3E_MIN_FREE_KB:-15000000}"
OUT_TAG="${OUT_TAG:-${DATASET}_PHASE3E_NODE_ACTION_SEMANTIC_PRECOMPUTE}"
CHECKPOINT_ROOT="${SSPM_BASE_CHECKPOINT_ROOT:-${REPO_ROOT}/outputs/models/sspm_phase3e}"
ACTION_HEAD_CHECKPOINT_ROOT="${ACTION_HEAD_CHECKPOINT_ROOT:-${REPO_ROOT}/outputs/models/phase3g_action_heads}"
ACTION_VALIDATION_CACHE_DIR="${ACTION_VALIDATION_CACHE_DIR:-${REPO_ROOT}/outputs/cache/phase3g_action_validation/${DATASET}}"
SSPM_EPOCHS="${SSPM_EPOCHS:-80}"
SSPM_EARLY_STOP_MIN_DELTA="${SSPM_EARLY_STOP_MIN_DELTA:-0.0002}"
SSPM_EARLY_STOP_PATIENCE="${SSPM_EARLY_STOP_PATIENCE:-5}"
SSPM_LEARNING_RATE="${SSPM_LEARNING_RATE:-0.005}"
SSPM_TORCH_LR="${SSPM_TORCH_LR:-${SSPM_LEARNING_RATE}}"
SSPM_TORCH_WEIGHT_DECAY="${SSPM_TORCH_WEIGHT_DECAY:-0.0}"
REAL_DIAG_GAMMA_LR="${REAL_DIAG_GAMMA_LR:-0.001}"
REAL_DIAG_GAMMA_WEIGHT_DECAY="${REAL_DIAG_GAMMA_WEIGHT_DECAY:-0.0}"
REAL_DIAG_GAMMA_GRAD_CLIP="${REAL_DIAG_GAMMA_GRAD_CLIP:-1.0}"
REAL_DIAG_MAX_SENSITIVITY_NODES="${REAL_DIAG_MAX_SENSITIVITY_NODES:-500000}"
PRETRAINED_DEFAULT="outputs/models/residual_word2vec"
PRETRAINED_DEFAULT+="/CADETS_E3_RAW_DETAIL_RULES_V1_LATENT64_word2vec_window3.pkl"
EMBEDDER_PATH="${PRETRAINED_RESIDUAL_EMBEDDER_PATH:-${PRETRAINED_DEFAULT}}"
TORCH_CHECK_SCRIPT="${TORCH_CHECK_SCRIPT:-scripts/run/check_phase3e_torch_cuda.sh}"

DEFAULT_NODE_CACHE_DIR="${PHASE3E_CACHE_ROOT}/node_embeddings/${DATASET}_latent64"
DEFAULT_ACTION_CACHE_DIR="${PHASE3E_CACHE_ROOT}/action_embeddings/${DATASET}_latent64"
NODE_EMBEDDING_CACHE_DIR="${NODE_EMBEDDING_CACHE_DIR:-${DEFAULT_NODE_CACHE_DIR}}"
ACTION_EMBEDDING_CACHE_DIR="${ACTION_EMBEDDING_CACHE_DIR:-${DEFAULT_ACTION_CACHE_DIR}}"
EVENT_INDEX_CACHE_DIR="${EVENT_INDEX_CACHE_DIR:-${PHASE3E_CACHE_ROOT}/event_indices/${DATASET}}"
X_CONTEXT_CACHE_DIR="${X_CONTEXT_CACHE_DIR:-${PHASE3E_CACHE_ROOT}/x_context/${DATASET}}"
COMPACT_USED_NODE_CACHE_DIR="${COMPACT_USED_NODE_CACHE_DIR:-${PHASE3E_CACHE_ROOT}/compact_used_node_embeddings/${DATASET}}"
SSPM_OFSM_MATCH_BACKEND="${SSPM_OFSM_MATCH_BACKEND:-exact}"
SSPM_OFSM_CANDIDATE_CAP="${SSPM_OFSM_CANDIDATE_CAP:-64}"
SSPM_OFSM_MERGE_INTERVAL="${SSPM_OFSM_MERGE_INTERVAL:-1}"
SSPM_OFSM_MIN_COUNT="${SSPM_OFSM_MIN_COUNT:-1}"
SSPM_OFSM_DIAGNOSTICS="${SSPM_OFSM_DIAGNOSTICS:-true}"

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

variant_enabled() {
    local key="$1"
    [[ "${RUN_ONLY}" == "ALL" || "${RUN_ONLY}" == "${key}" ]]
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

run_torch_check() {
    "${TORCH_CHECK_SCRIPT}"
}

check_disk_free() {
    local available_kb
    available_kb="$(df -Pk "${REPO_ROOT}" | awk 'NR==2 {print $4}')"
    if [[ -z "${available_kb}" || ! "${available_kb}" =~ ^[0-9]+$ ]]; then
        echo "error: unable to determine disk space for Phase3E precompute" >&2
        exit 2
    fi
    if (( available_kb < PHASE3E_MIN_FREE_KB )); then
        echo "error: insufficient disk for Phase3E precompute: " \
            "available_kb=${available_kb} required_kb=${PHASE3E_MIN_FREE_KB}" >&2
        exit 2
    fi
}

preflight() {
    echo "[phase3e preflight] start" >&2
    if [[ ! -f "${EMBEDDER_PATH}" ]]; then
        echo "error: pretrained residual Word2Vec not found: ${EMBEDDER_PATH}" >&2
        exit 2
    fi
    echo "[phase3e preflight] embedder ok" >&2
    if [[ "${SSPM_TARGET_MODE}" != "node_action_semantic_mean" ]]; then
        echo "error: Phase3E runner requires SSPM_TARGET_MODE=node_action_semantic_mean" >&2
        exit 2
    fi
    if [[ "${NODE_WORD2VEC_SOURCE}" != "residual_pretrained" ]]; then
        echo "error: Phase3E v1 runner requires NODE_WORD2VEC_SOURCE=residual_pretrained" >&2
        exit 2
    fi
    if [[ "${SSPM_TRAIN_BACKEND}" != "torch" ]]; then
        echo "error: Phase3E precompute runner requires SSPM_TRAIN_BACKEND=torch" >&2
        exit 2
    fi
    if [[ "${SSPM_INFER_BACKEND}" != "numpy" ]]; then
        echo "error: Phase3E deploy inference requires SSPM_INFER_BACKEND=numpy" >&2
        exit 2
    fi
    if [[ "${RSS_PROFILE_MODE}" != "deploy_light" && "${RSS_PROFILE_MODE}" != "online_minimal" ]]; then
        echo "error: Phase3E runner requires RSS_PROFILE_MODE=deploy_light or online_minimal" >&2
        exit 2
    fi
    if is_true "${SSPM_INFER_FAST_PATH}" && [[ "${RSS_PROFILE_MODE}" != "online_minimal" ]]; then
        echo "error: SSPM_INFER_FAST_PATH requires RSS_PROFILE_MODE=online_minimal" >&2
        exit 2
    fi
    if [[ ! -x "${TORCH_CHECK_SCRIPT}" ]]; then
        echo "error: Phase3E torch CUDA check script is unavailable: ${TORCH_CHECK_SCRIPT}" >&2
        exit 2
    fi
    if ! is_true "${DRY_RUN}"; then
        if [[ -z "${CLAD_DB_PASSWORD+x}" ]]; then
            echo "error: set CLAD_DB_PASSWORD before running real Phase3E precompute" >&2
            exit 2
        fi
        if [[ -z "${CLAD_DB_PASSWORD}" ]]; then
            echo "error: set CLAD_DB_PASSWORD before running real Phase3E precompute" >&2
            exit 2
        fi
    fi
    echo "[phase3e preflight] config/env ok" >&2

    if is_true "${DRY_RUN}"; then
        echo "[phase3e preflight] writable dir checks skipped for dry-run" >&2
        echo "[phase3e preflight] disk check skipped for dry-run" >&2
    else
        require_writable_dir "${RESULT_ROOT}" "result root"
        require_writable_dir "${CHECKPOINT_ROOT}" "checkpoint root"
        require_writable_dir "${ACTION_HEAD_CHECKPOINT_ROOT}" "action head checkpoint root"
        require_writable_dir "${ACTION_VALIDATION_CACHE_DIR}" "action validation cache directory"
        require_writable_dir "${CONDITIONAL_ENDPOINT_SUPPRESSION_CACHE_DIR}" \
            "conditional endpoint suppression cache directory"
        require_writable_dir "${LOG_DIR}" "log directory"
        require_writable_dir "${NODE_EMBEDDING_CACHE_DIR}" "node embedding cache directory"
        require_writable_dir "${ACTION_EMBEDDING_CACHE_DIR}" "action embedding cache directory"
        require_writable_dir "${EVENT_INDEX_CACHE_DIR}" "event index cache directory"
        require_writable_dir "${X_CONTEXT_CACHE_DIR}" "X_context cache directory"
        require_writable_dir "${COMPACT_USED_NODE_CACHE_DIR}" "compact used-node cache directory"
        echo "[phase3e preflight] writable dirs ok" >&2
        if [[ "${STAGE}" == "precompute" ]]; then
            check_disk_free
            echo "[phase3e preflight] disk ok" >&2
        else
            echo "[phase3e preflight] precompute disk floor skipped for STAGE=${STAGE}" >&2
        fi
    fi
    if is_true "${DRY_RUN}"; then
        echo "[phase3e preflight] torch check skipped for dry-run" >&2
    elif [[ "${STAGE}" == "infer_ablation" && "${SSPM_INFER_BACKEND}" == "numpy" ]]; then
        echo "[phase3e preflight] torch CUDA check skipped for numpy infer_ablation" >&2
    else
        run_torch_check
        echo "[phase3e preflight] torch ok" >&2
    fi
}

case "${STAGE}" in
    precompute|train_base|train_conditional_base|infer_ablation|all) ;;
    *)
        echo "error: unknown STAGE=${STAGE}" >&2
        exit 2
        ;;
esac

case "${STAGE}" in
    all)
        echo "error: STAGE=all is not allowed in this approval cycle" >&2
        exit 2
        ;;
esac

common_args=(
    --dataset "${DATASET}"
    --result_root "${RESULT_ROOT}"
    --max_train_events "${MAX_TRAIN_EVENTS}"
    --max_ref_events "${MAX_REF_EVENTS}"
    --max_test_events "${MAX_TEST_EVENTS}"
    --event_score_mode res_only
    --event_threshold_mode "${EVENT_THRESHOLD_MODE}"
    --event_threshold_quantile "${EVENT_THRESHOLD_QUANTILE}"
    --sspm_update_gate_score_space "${SSPM_UPDATE_GATE_SCORE_SPACE}"
    --sspm_update_gate_quantile "${SSPM_UPDATE_GATE_QUANTILE}"
    --sspm_update_gate_q_min "${SSPM_UPDATE_GATE_Q_MIN}"
    --sspm_update_gate_eta "${SSPM_UPDATE_GATE_ETA}"
    --sspm_score_target_mode "${SSPM_SCORE_TARGET_MODE}"
    --action_type_alert_policy "${ACTION_TYPE_ALERT_POLICY}"
    --pretrained_residual_embedder_path "${EMBEDDER_PATH}"
    --semantic_embedding_method word2vec
    --semantic_mode "${SEMANTIC_MODE}"
    --word2vec_window 3
    --sspm_target_dim 64
    --sspm_state_dim 64
    --rank 32
    --sspm_context_mode with_action
    --sspm_context_action_mode raw_orthrus10
    --sspm_global_context_mode no_global
    --state_memory_mode bounded
    --sspm_state_memory_policy probationary_lru
    --sspm_target_mode "${SSPM_TARGET_MODE}"
    --node_word2vec_source "${NODE_WORD2VEC_SOURCE}"
    --node_embedding_lookup_mode "${NODE_EMBEDDING_LOOKUP_MODE}"
    --node_embedding_lazy_backing "${NODE_EMBEDDING_LAZY_BACKING}"
    --node_embedding_cache_max_nodes "${NODE_EMBEDDING_CACHE_MAX_NODES}"
    --node_embedding_cache_evict_policy "${NODE_EMBEDDING_CACHE_EVICT_POLICY}"
    --node_embedding_cache_dir "${NODE_EMBEDDING_CACHE_DIR}"
    --compact_used_node_cache_dir "${COMPACT_USED_NODE_CACHE_DIR}"
    --action_embedding_cache_dir "${ACTION_EMBEDDING_CACHE_DIR}"
    --event_index_cache_mode auto
    --event_index_cache_dir "${EVENT_INDEX_CACHE_DIR}"
    --x_context_cache_dir "${X_CONTEXT_CACHE_DIR}"
    --x_context_memmap_enabled
    --sspm_train_backend "${SSPM_TRAIN_BACKEND}"
    --sspm_infer_backend "${SSPM_INFER_BACKEND}"
    --sspm_score_head "${SSPM_SCORE_HEAD}"
    --node_repr_fusion "${NODE_REPR_FUSION}"
    --conditional_semantic_loss "${CONDITIONAL_SEMANTIC_LOSS}"
    --sspm_conditional_head_arch "${SSPM_CONDITIONAL_HEAD_ARCH}"
    --conditional_group_min_count "${CONDITIONAL_GROUP_MIN_COUNT}"
    --conditional_low_support_policy "${CONDITIONAL_LOW_SUPPORT_POLICY}"
    --conditional_low_support_margin "${CONDITIONAL_LOW_SUPPORT_MARGIN}"
    --conditional_unseen_group_policy "${CONDITIONAL_UNSEEN_GROUP_POLICY}"
    --conditional_global_extreme_quantile "${CONDITIONAL_GLOBAL_EXTREME_QUANTILE}"
    --conditional_adaptive_margin_n1 "${CONDITIONAL_ADAPTIVE_MARGIN_N1}"
    --conditional_adaptive_margin_n2 "${CONDITIONAL_ADAPTIVE_MARGIN_N2}"
    --conditional_adaptive_margin_low "${CONDITIONAL_ADAPTIVE_MARGIN_LOW}"
    --conditional_adaptive_margin_mid "${CONDITIONAL_ADAPTIVE_MARGIN_MID}"
    --conditional_adaptive_margin_high "${CONDITIONAL_ADAPTIVE_MARGIN_HIGH}"
    --conditional_endpoint_aware_suppression "${CONDITIONAL_ENDPOINT_AWARE_SUPPRESSION}"
    --conditional_endpoint_suppression_read "${CONDITIONAL_ENDPOINT_SUPPRESSION_READ}"
    --conditional_endpoint_suppression_mode "${CONDITIONAL_ENDPOINT_SUPPRESSION_MODE}"
    --conditional_known_pair_min_count "${CONDITIONAL_KNOWN_PAIR_MIN_COUNT}"
    --conditional_pair_suppression_margin "${CONDITIONAL_PAIR_SUPPRESSION_MARGIN}"
    --conditional_same_process_endpoint_min_count "${CONDITIONAL_SAME_PROCESS_ENDPOINT_MIN_COUNT}"
    --conditional_same_process_endpoint_margin "${CONDITIONAL_SAME_PROCESS_ENDPOINT_MARGIN}"
    --conditional_endpoint_suppression_summary_mode "${CONDITIONAL_ENDPOINT_SUPPRESSION_SUMMARY_MODE}"
    --conditional_endpoint_suppression_cache_dir "${CONDITIONAL_ENDPOINT_SUPPRESSION_CACHE_DIR}"
    --sspm_conditional_train_data_mode "${SSPM_CONDITIONAL_TRAIN_DATA_MODE}"
    --sspm_conditional_max_epochs "${SSPM_CONDITIONAL_MAX_EPOCHS}"
    --sspm_conditional_e3_max_epochs "${SSPM_CONDITIONAL_E3_MAX_EPOCHS}"
    --action_validation_cache_dir "${ACTION_VALIDATION_CACHE_DIR}"
    --sspm_torch_batch_events "${SSPM_TORCH_BATCH_EVENTS}"
    --sspm_torch_lr "${SSPM_TORCH_LR}"
    --sspm_torch_weight_decay "${SSPM_TORCH_WEIGHT_DECAY}"
    --sspm_torch_device "${SSPM_TORCH_DEVICE}"
    --sspm_learning_rate "${SSPM_LEARNING_RATE}"
    --sspm_epochs "${SSPM_EPOCHS}"
    --sspm_early_stop_min_delta "${SSPM_EARLY_STOP_MIN_DELTA}"
    --sspm_early_stop_patience "${SSPM_EARLY_STOP_PATIENCE}"
    --sspm_base_checkpoint_root "${CHECKPOINT_ROOT}"
    --rss_profile_mode "${RSS_PROFILE_MODE}"
    --write_raw_alerts false
    --write_analysis_outputs false
    --sspm_state_merge_write_diagnostics false
    --sspm_ofsm_match_backend "${SSPM_OFSM_MATCH_BACKEND}"
    --sspm_ofsm_candidate_cap "${SSPM_OFSM_CANDIDATE_CAP}"
    --sspm_ofsm_merge_interval "${SSPM_OFSM_MERGE_INTERVAL}"
    --sspm_ofsm_min_count "${SSPM_OFSM_MIN_COUNT}"
    --sspm_ofsm_diagnostics "${SSPM_OFSM_DIAGNOSTICS}"
    --print_summary
)

if is_true "${SSPM_INFER_FAST_PATH}" && [[ "${STAGE}" == "infer_ablation" ]]; then
    common_args+=(
        --sspm_infer_fast_path true
        --sspm_infer_chunk_events "${SSPM_INFER_CHUNK_EVENTS}"
    )
fi

base_checkpoint_path() {
    local family="$1"
    printf '%s/%s_%s.pkl' "${CHECKPOINT_ROOT}" "${DATASET}" "${family}"
}

action_head_checkpoint_path() {
    local family="$1"
    local override_var="ACTION_HEAD_CHECKPOINT_PATH_${family}"
    local override_value="${!override_var:-}"
    if [[ -n "${override_value}" ]]; then
        printf '%s' "${override_value}"
        return 0
    fi
    local family_root="${family%%_*}"
    local root_override_var="ACTION_HEAD_CHECKPOINT_PATH_${family_root}"
    local root_override_value="${!root_override_var:-}"
    if [[ -n "${root_override_value}" ]]; then
        printf '%s' "${root_override_value}"
        return 0
    fi
    if [[ "${SSPM_SCORE_HEAD}" == "conditional_action_semantic" ]]; then
        printf '%s/%s_%s_CONDITIONAL_HEAD.pkl' \
            "${ACTION_HEAD_CHECKPOINT_ROOT}" "${DATASET}" "${family}"
    else
        printf '%s/%s_%s_ACTION_HEAD.pkl' "${ACTION_HEAD_CHECKPOINT_ROOT}" "${DATASET}" "${family}"
    fi
}

run_command() {
    local key="$1"
    local label="$2"
    local out_tag="$3"
    shift 3
    local log_path="${LOG_DIR}/${out_tag}.log"
    local cmd=("${PYTHON_BIN}" scripts/tools/causal_semantics_slim.py "$@")
    if is_true "${DRY_RUN}"; then
        echo "===== DRY_RUN ${key} ${label} ${out_tag} ====="
        print_command "${cmd[@]}"
        capture_command_if_requested "${cmd[@]}"
        return 0
    fi
    echo "===== START ${key} ${label} ${out_tag} $(date) ====="
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
        echo "===== FAIL ${key} status=${status} $(date) ====="
        exit "${status}"
    fi
    echo "===== DONE ${key} ${out_tag} $(date) ====="
}

run_base() {
    local family="$1"
    local state_model="$2"
    local update_gate_mode="$3"
    local residual_score_mode="$4"
    local residual_calibration="$5"
    local train_gamma="$6"
    local key="${family}_BASE_FULL"
    if ! variant_enabled "${key}" && ! variant_enabled "${family}_NONE"; then
        return 0
    fi
    local out_tag="${DATASET}_${family}_PHASE3E_BASE_FULL"
    local checkpoint_path
    checkpoint_path="$(base_checkpoint_path "${family}_PHASE3E_BASE_FULL")"
    local action_checkpoint_path
    action_checkpoint_path="$(action_head_checkpoint_path "${family}_PHASE3G")"
    local args=(
        "${common_args[@]}"
        --out_tag "${out_tag}"
        --sspm_train_mode train_and_save
        --sspm_train_data_mode phase3e_memmap
        --sspm_checkpoint_path "${checkpoint_path}"
        --action_head_checkpoint_path "${action_checkpoint_path}"
        --sspm_state_model "${state_model}"
        --sspm_update_gate_mode "${update_gate_mode}"
        --sspm_residual_score_mode "${residual_score_mode}"
        --sspm_residual_calibration "${residual_calibration}"
        --sspm_state_merge_mode none
        --sspm_state_merge_threshold 0.98
    )
    if [[ "${update_gate_mode}" == "quantile" ]]; then
        args+=(--sspm_update_gate_threshold_mode "${SSPM_UPDATE_GATE_THRESHOLD_MODE:-validation_max}")
    else
        args+=(--sspm_update_gate_threshold_mode "${SSPM_UPDATE_GATE_THRESHOLD_MODE:-quantile}")
    fi
    if [[ "${residual_calibration}" == "action_diag" ]]; then
        args+=(--sspm_residual_calibration_min_count 100)
    fi
    if is_true "${train_gamma}"; then
        args+=(
            --real_diag_train_gamma
            --real_diag_gamma_lr "${REAL_DIAG_GAMMA_LR}"
            --real_diag_gamma_weight_decay "${REAL_DIAG_GAMMA_WEIGHT_DECAY}"
            --real_diag_gamma_grad_clip "${REAL_DIAG_GAMMA_GRAD_CLIP}"
            --real_diag_sensitivity_mode online_stop_message
            --real_diag_max_sensitivity_nodes "${REAL_DIAG_MAX_SENSITIVITY_NODES}"
        )
    fi
    run_command "${key}" "${family} Phase3E base train" "${out_tag}" "${args[@]}"
}

run_conditional_base() {
    local family="$1"
    local state_model="$2"
    local update_gate_mode="$3"
    local residual_score_mode="$4"
    local residual_calibration="$5"
    local train_gamma="$6"
    local key="${family}_CONDITIONAL_BASE_FULL"
    if ! variant_enabled "${key}" && ! variant_enabled "${family}_NONE"; then
        return 0
    fi
    local out_tag="${DATASET}_${family}_CONDITIONAL_BASE_FULL"
    local checkpoint_path
    checkpoint_path="$(base_checkpoint_path "${family}_PHASE3E_BASE_FULL")"
    local action_checkpoint_path
    action_checkpoint_path="$(action_head_checkpoint_path "${family}_PHASE3G")"
    local args=(
        "${common_args[@]}"
        --out_tag "${out_tag}"
        --sspm_train_mode train_conditional_and_save
        --sspm_train_data_mode phase3e_memmap
        --sspm_checkpoint_path "${checkpoint_path}"
        --action_head_checkpoint_path "${action_checkpoint_path}"
        --sspm_state_model "${state_model}"
        --sspm_update_gate_mode "${update_gate_mode}"
        --sspm_residual_score_mode "${residual_score_mode}"
        --sspm_residual_calibration "${residual_calibration}"
        --sspm_state_merge_mode none
        --sspm_state_merge_threshold 0.98
    )
    if [[ "${update_gate_mode}" == "quantile" ]]; then
        args+=(--sspm_update_gate_threshold_mode "${SSPM_UPDATE_GATE_THRESHOLD_MODE:-validation_max}")
    else
        args+=(--sspm_update_gate_threshold_mode "${SSPM_UPDATE_GATE_THRESHOLD_MODE:-quantile}")
    fi
    if [[ "${residual_calibration}" == "action_diag" ]]; then
        args+=(--sspm_residual_calibration_min_count 100)
    fi
    if is_true "${train_gamma}"; then
        args+=(
            --real_diag_train_gamma
            --real_diag_gamma_lr "${REAL_DIAG_GAMMA_LR}"
            --real_diag_gamma_weight_decay "${REAL_DIAG_GAMMA_WEIGHT_DECAY}"
            --real_diag_gamma_grad_clip "${REAL_DIAG_GAMMA_GRAD_CLIP}"
            --real_diag_sensitivity_mode online_stop_message
            --real_diag_max_sensitivity_nodes "${REAL_DIAG_MAX_SENSITIVITY_NODES}"
        )
    fi
    run_command "${key}" "${family} Phase3G conditional base train" "${out_tag}" "${args[@]}"
}

run_infer() {
    local key="$1"
    local family="$2"
    local out_tag="$3"
    local state_model="$4"
    local merge_mode="$5"
    local merge_threshold="$6"
    local update_gate_mode="$7"
    local residual_score_mode="$8"
    local residual_calibration="$9"
    if ! variant_enabled "${key}"; then
        return 0
    fi
    if [[ -n "${OUT_TAG_OVERRIDE:-}" ]]; then
        if [[ "${RUN_ONLY}" != "${key}" || "${RUN_ONLY}" == "ALL" ]]; then
            echo "error: OUT_TAG_OVERRIDE is allowed only for a single explicit RUN_ONLY" >&2
            exit 2
        fi
        out_tag="${OUT_TAG_OVERRIDE}"
    fi
    local checkpoint_path
    checkpoint_path="$(base_checkpoint_path "${family}_PHASE3E_BASE_FULL")"
    local action_checkpoint_path
    action_checkpoint_path="$(action_head_checkpoint_path "${family}_PHASE3G")"
    if [[ ! -f "${checkpoint_path}" ]] && ! is_true "${DRY_RUN}"; then
        echo "error: required Phase3E checkpoint not found for ${key}: ${checkpoint_path}" >&2
        exit 2
    fi
    if [[ ( "${SSPM_SCORE_HEAD}" == "action_predict" \
        || "${SSPM_SCORE_HEAD}" == "conditional_action_semantic" ) \
        && ! -f "${action_checkpoint_path}" ]] \
        && ! is_true "${DRY_RUN}"; then
        echo "error: required Phase3G score head not found for ${key}: ${action_checkpoint_path}" >&2
        exit 2
    fi
    local args=(
        "${common_args[@]}"
        --out_tag "${out_tag}"
        --sspm_train_mode load_and_infer
        --sspm_train_data_mode phase3e_memmap
        --sspm_checkpoint_path "${checkpoint_path}"
        --action_head_checkpoint_path "${action_checkpoint_path}"
        --sspm_state_model "${state_model}"
        --sspm_update_gate_mode "${update_gate_mode}"
        --sspm_residual_score_mode "${residual_score_mode}"
        --sspm_residual_calibration "${residual_calibration}"
        --sspm_state_merge_mode "${merge_mode}"
        --sspm_state_merge_threshold "${merge_threshold}"
    )
    if [[ "${update_gate_mode}" == "quantile" ]]; then
        args+=(--sspm_update_gate_threshold_mode "${SSPM_UPDATE_GATE_THRESHOLD_MODE:-validation_max}")
    else
        args+=(--sspm_update_gate_threshold_mode "${SSPM_UPDATE_GATE_THRESHOLD_MODE:-quantile}")
    fi
    if [[ "${residual_calibration}" == "action_diag" ]]; then
        args+=(--sspm_residual_calibration_min_count 100)
    fi
    run_command "${key}" "${family} Phase3E inference" "${out_tag}" "${args[@]}"
}

echo "STAGE=${STAGE}"
preflight

if [[ "${STAGE}" == "precompute" ]]; then
    cmd=(
        "${PYTHON_BIN}"
        scripts/tools/causal_semantics_slim.py
        "${common_args[@]}"
        --out_tag "${OUT_TAG}"
        --phase3e_precompute_only
    )
    if is_true "${DRY_RUN}"; then
        print_command "${cmd[@]}"
        capture_command_if_requested "${cmd[@]}"
        exit 0
    fi
    "${cmd[@]}"
elif [[ "${STAGE}" == "train_base" ]]; then
    run_base E2 ema_fixed none legacy none 0
    run_base E3 real_diag_learnable none legacy none 1
    run_base E4 s4d_complex_node none legacy none 0
    run_base E5 ema_fixed quantile var_calibrated action_diag 0
elif [[ "${STAGE}" == "train_conditional_base" ]]; then
    if [[ "${SSPM_SCORE_HEAD}" != "conditional_action_semantic" ]]; then
        echo "error: STAGE=train_conditional_base requires SSPM_SCORE_HEAD=conditional_action_semantic" >&2
        exit 2
    fi
    run_conditional_base E2 ema_fixed none legacy none 0
    run_conditional_base E3 real_diag_learnable none legacy none 1
    run_conditional_base E4 s4d_complex_node none legacy none 0
    run_conditional_base E5 ema_fixed quantile var_calibrated action_diag 0
elif [[ "${STAGE}" == "infer_ablation" ]]; then
    run_infer E2_NONE E2 "${DATASET}_PHASE3E_E2_NONE" \
        ema_fixed none 0.98 none legacy none
    run_infer E2_OFSM_TIME_DOMAIN_T098 E2 "${DATASET}_PHASE3E_E2_OFSM_TIME_DOMAIN_T098" \
        ema_fixed online_time_domain 0.98 none legacy none
    run_infer E2_OFSM_RANDOM E2 "${DATASET}_PHASE3E_E2_OFSM_RANDOM" \
        ema_fixed random 0.98 none legacy none
    run_infer E2_OFSM_FOURIER_T085 E2 "${DATASET}_PHASE3E_E2_OFSM_FOURIER_T085" \
        ema_fixed online_fourier 0.85 none legacy none
    run_infer E2_OFSM_FOURIER_T090 E2 "${DATASET}_PHASE3E_E2_OFSM_FOURIER_T090" \
        ema_fixed online_fourier 0.90 none legacy none
    run_infer E2_OFSM_FOURIER_T095 E2 "${DATASET}_PHASE3E_E2_OFSM_FOURIER_T095" \
        ema_fixed online_fourier 0.95 none legacy none
    run_infer E2_OFSM_FOURIER_T098 E2 "${DATASET}_PHASE3E_E2_OFSM_FOURIER_T098" \
        ema_fixed online_fourier 0.98 none legacy none
    run_infer E3_NONE E3 "${DATASET}_PHASE3E_E3_NONE" \
        real_diag_learnable none 0.98 none legacy none
    run_infer E3_OFSM_FOURIER_T098 E3 "${DATASET}_PHASE3E_E3_OFSM_FOURIER_T098" \
        real_diag_learnable online_fourier 0.98 none legacy none
    run_infer E4_NONE E4 "${DATASET}_PHASE3E_E4_NONE" \
        s4d_complex_node none 0.98 none legacy none
    run_infer E4_OFSM_FOURIER_T098 E4 "${DATASET}_PHASE3E_E4_OFSM_FOURIER_T098" \
        s4d_complex_node online_fourier 0.98 none legacy none
    run_infer E5_NONE E5 "${DATASET}_PHASE3E_E5_NONE" \
        ema_fixed none 0.98 quantile var_calibrated action_diag
    run_infer E5_OFSM_FOURIER_T098 E5 "${DATASET}_PHASE3E_E5_OFSM_FOURIER_T098" \
        ema_fixed online_fourier 0.98 quantile var_calibrated action_diag
fi

if [[ "${RUN_ONLY}" != "ALL" && "${STAGE}" == "infer_ablation" ]]; then
    case "${RUN_ONLY}" in
        E2_NONE|E2_OFSM_TIME_DOMAIN_T098|E2_OFSM_RANDOM|E2_OFSM_FOURIER_T085|\
E2_OFSM_FOURIER_T090|E2_OFSM_FOURIER_T095|E2_OFSM_FOURIER_T098|E3_NONE|\
E3_OFSM_FOURIER_T098|E4_NONE|E4_OFSM_FOURIER_T098|E5_NONE|E5_OFSM_FOURIER_T098)
            ;;
        *)
            echo "error: unknown RUN_ONLY=${RUN_ONLY}" >&2
            exit 2
            ;;
    esac
fi
