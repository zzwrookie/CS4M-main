#!/usr/bin/env bash
set -euo pipefail

# Build only THEIA_E3 E4 Phase3G conditional train memmaps.
# This does not train a head, run inference, run evaluation, or use ground truth.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${CS4M_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

DATASET="THEIA_E3"
STATE_FAMILY="${STATE_FAMILY:-E4}"
STATE_MODEL="s4d_complex_node"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/outputs/results/theia_e3_memmap_build}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/theia_e3_phase3g_conditional_memmaps}"
PHASE3E_CACHE_ROOT="${PHASE3E_CACHE_ROOT:-${REPO_ROOT}/outputs/cache/phase3e}"
CHECKPOINT_ROOT="${SSPM_BASE_CHECKPOINT_ROOT:-${REPO_ROOT}/outputs/models/sspm_phase3e}"
ACTION_HEAD_ROOT="${ACTION_HEAD_CHECKPOINT_ROOT:-${REPO_ROOT}/outputs/models/phase3g_action_heads}"
DEFAULT_ACTION_VALIDATION_CACHE="${REPO_ROOT}/outputs/cache/phase3g_action_validation/${DATASET}"
ACTION_VALIDATION_CACHE_DIR="${ACTION_VALIDATION_CACHE_DIR:-${DEFAULT_ACTION_VALIDATION_CACHE}}"
DEFAULT_ENDPOINT_CACHE="${REPO_ROOT}/outputs/cache/phase3g_endpoint_suppression_compact/${DATASET}"
ENDPOINT_CACHE_DIR="${CONDITIONAL_ENDPOINT_SUPPRESSION_CACHE_DIR:-${DEFAULT_ENDPOINT_CACHE}}"
DEFAULT_PRETRAINED_EMBEDDER="${REPO_ROOT}/outputs/models/residual_word2vec"
DEFAULT_PRETRAINED_EMBEDDER+="/THEIA_E3_RAW_DETAIL_RULES_V1_LATENT64_word2vec_window3.pkl"
PRETRAINED_EMBEDDER="${PRETRAINED_RESIDUAL_EMBEDDER_PATH:-${DEFAULT_PRETRAINED_EMBEDDER}}"
OUT_TAG="${OUT_TAG:-THEIA_E3_PHASE3G_E4_CONDITIONAL_MEMMAP_ONLY}"
DRY_RUN="${DRY_RUN:-0}"

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

missing_paths=()

record_missing_file() {
    local path="$1"
    local label="$2"
    if [[ ! -f "${path}" ]]; then
        missing_paths+=("${label}: ${path}")
    fi
}

record_missing_dir() {
    local path="$1"
    local label="$2"
    if [[ ! -d "${path}" ]]; then
        missing_paths+=("${label}: ${path}")
    fi
}

preflight() {
    if [[ "${DATASET}" != "THEIA_E3" || "${STATE_FAMILY}" != "E4" ]]; then
        echo "error: this runner is restricted to THEIA_E3 E4" >&2
        exit 2
    fi
    record_missing_file "${PRETRAINED_EMBEDDER}" "THEIA_E3 residual Word2Vec embedder"
    record_missing_file "${CHECKPOINT_ROOT}/THEIA_E3_E4_PHASE3E_BASE_FULL.pkl" \
        "THEIA_E3 E4 base checkpoint"
    record_missing_file "${ACTION_HEAD_ROOT}/THEIA_E3_E4_PHASE3G_CONDITIONAL_HEAD.pkl" \
        "THEIA_E3 E4 conditional head reference"
    record_missing_file "${PHASE3E_CACHE_ROOT}/event_indices/THEIA_E3/event_index_train.memmap" \
        "THEIA_E3 train event index"
    record_missing_file "${PHASE3E_CACHE_ROOT}/event_indices/THEIA_E3/event_index_meta.json" \
        "THEIA_E3 event index metadata"
    record_missing_file \
        "${PHASE3E_CACHE_ROOT}/node_embeddings/THEIA_E3_latent64/node_embeddings.npy" \
        "THEIA_E3 node embeddings"
    record_missing_file \
        "${PHASE3E_CACHE_ROOT}/action_embeddings/THEIA_E3_latent64/action_embeddings.npy" \
        "THEIA_E3 action embeddings"
    record_missing_dir "${ENDPOINT_CACHE_DIR}" "THEIA_E3 endpoint suppression cache"
    if (( ${#missing_paths[@]} > 0 )); then
        echo "error: THEIA_E3 memmap prerequisites are missing:" >&2
        printf '  - %s\n' "${missing_paths[@]}" >&2
        exit 2
    fi
    mkdir -p "${RESULT_ROOT}" "${LOG_DIR}" "${ACTION_VALIDATION_CACHE_DIR}"
}

preflight

cmd=(
    "${PYTHON_BIN}"
    scripts/tools/causal_semantics_slim.py
    --dataset "${DATASET}"
    --result_root "${RESULT_ROOT}"
    --out_tag "${OUT_TAG}"
    --max_train_events 0
    --max_ref_events 0
    --max_test_events 0
    --progress_interval_events "${PROGRESS_INTERVAL_EVENTS:-819200}"
    --event_score_mode res_only
    --event_threshold_mode conditional_target_action_type_group_quantile
    --event_threshold_quantile 0.999
    --action_type_alert_policy theia_v1
    --pretrained_residual_embedder_path "${PRETRAINED_EMBEDDER}"
    --semantic_embedding_method word2vec
    --semantic_mode theia_linux_raw_detail_v2
    --word2vec_window 3
    --sspm_target_dim 64
    --sspm_state_dim 64
    --rank 32
    --sspm_context_mode with_action
    --sspm_context_action_mode raw_orthrus10
    --sspm_global_context_mode no_global
    --state_memory_mode bounded
    --sspm_state_memory_policy probationary_lru
    --sspm_target_mode node_action_semantic_mean
    --node_word2vec_source residual_pretrained
    --node_embedding_lookup_mode lazy_mmap_lru
    --node_embedding_lazy_backing compact_used_nodes
    --node_embedding_cache_max_nodes 100000
    --node_embedding_cache_evict_policy lru
    --node_embedding_cache_dir "${PHASE3E_CACHE_ROOT}/node_embeddings/THEIA_E3_latent64"
    --compact_used_node_cache_dir "${PHASE3E_CACHE_ROOT}/compact_used_node_embeddings/THEIA_E3"
    --action_embedding_cache_dir "${PHASE3E_CACHE_ROOT}/action_embeddings/THEIA_E3_latent64"
    --event_index_cache_mode auto
    --event_index_cache_dir "${PHASE3E_CACHE_ROOT}/event_indices/THEIA_E3"
    --x_context_cache_dir "${PHASE3E_CACHE_ROOT}/x_context/THEIA_E3"
    --x_context_memmap_enabled
    --sspm_train_backend torch
    --sspm_infer_backend numpy
    --sspm_train_mode train_conditional_and_save
    --sspm_train_data_mode phase3e_memmap
    --sspm_score_head conditional_action_semantic
    --node_repr_fusion simple_mean
    --conditional_semantic_loss cosine
    --sspm_conditional_head_arch shared_lowrank_v1
    --conditional_group_min_count 1000
    --conditional_low_support_policy conservative_max
    --conditional_low_support_margin 0.05
    --conditional_unseen_group_policy observation_only
    --conditional_global_extreme_quantile 0.9999
    --conditional_adaptive_margin_n1 50
    --conditional_adaptive_margin_n2 200
    --conditional_adaptive_margin_low 0.15
    --conditional_adaptive_margin_mid 0.05
    --conditional_adaptive_margin_high 0.02
    --conditional_endpoint_aware_suppression true
    --conditional_endpoint_suppression_read false
    --conditional_endpoint_suppression_mode pair_only
    --conditional_known_pair_min_count 5
    --conditional_pair_suppression_margin 0.02
    --conditional_same_process_endpoint_min_count 5
    --conditional_same_process_endpoint_margin 0.01
    --conditional_endpoint_suppression_summary_mode online_minimal
    --conditional_endpoint_suppression_cache_dir "${ENDPOINT_CACHE_DIR}"
    --sspm_conditional_train_data_mode memmap
    --phase3g_build_conditional_memmap_only
    --action_validation_cache_dir "${ACTION_VALIDATION_CACHE_DIR}"
    --sspm_torch_batch_events "${SSPM_TORCH_BATCH_EVENTS:-8192}"
    --sspm_base_checkpoint_root "${CHECKPOINT_ROOT}"
    --sspm_checkpoint_path "${CHECKPOINT_ROOT}/THEIA_E3_E4_PHASE3E_BASE_FULL.pkl"
    --action_head_checkpoint_path "${ACTION_HEAD_ROOT}/THEIA_E3_E4_PHASE3G_CONDITIONAL_HEAD.pkl"
    --sspm_state_model "${STATE_MODEL}"
    --sspm_update_gate_mode none
    --sspm_residual_score_mode legacy
    --sspm_residual_calibration none
    --sspm_state_merge_mode none
    --sspm_state_merge_threshold 0.98
    --rss_profile_mode online_minimal
    --write_raw_alerts false
    --write_analysis_outputs false
    --sspm_state_merge_write_diagnostics false
    --print_summary
)

if is_true "${DRY_RUN}"; then
    print_command "${cmd[@]}"
    exit 0
fi

"${cmd[@]}"
