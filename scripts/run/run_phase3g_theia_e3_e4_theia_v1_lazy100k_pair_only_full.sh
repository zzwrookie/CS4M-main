#!/usr/bin/env bash
set -euo pipefail

# THEIA_E3 Phase3G E4 full inference with action/type alert policy v1.
# This wrapper intentionally runs only E4_NONE and refuses ALL/OFSM variants.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_RUNNER="${REPO_ROOT}/scripts/run/run_cadets_e3_phase3e_node_action_semantic_full.sh"
cd "${REPO_ROOT}"

DATASET="THEIA_E3"
RUN_ONLY="E4_NONE"
STAGE="infer_ablation"

DEFAULT_RESULT_ROOT="${REPO_ROOT}/outputs/results"
DEFAULT_RESULT_ROOT+="/phase3g_theia_e3_e4_theia_v1_lazy100k_pair_only_full"
RESULT_ROOT="${RESULT_ROOT:-${DEFAULT_RESULT_ROOT}}"

DEFAULT_LOG_DIR="${REPO_ROOT}/logs"
DEFAULT_LOG_DIR+="/theia_e3_phase3g_e4_theia_v1_lazy100k_pair_only_full"
LOG_DIR="${LOG_DIR:-${DEFAULT_LOG_DIR}}"

PHASE3E_CACHE_ROOT="${PHASE3E_CACHE_ROOT:-${REPO_ROOT}/outputs/cache/phase3e}"
CHECKPOINT_ROOT="${SSPM_BASE_CHECKPOINT_ROOT:-${REPO_ROOT}/outputs/models/sspm_phase3e}"
ACTION_HEAD_ROOT="${ACTION_HEAD_CHECKPOINT_ROOT:-${REPO_ROOT}/outputs/models/phase3g_action_heads}"
ACTION_VALIDATION_CACHE_DIR="${ACTION_VALIDATION_CACHE_DIR:-${REPO_ROOT}/outputs/cache/phase3g_action_validation/${DATASET}}"
ENDPOINT_CACHE_DIR="${CONDITIONAL_ENDPOINT_SUPPRESSION_CACHE_DIR:-${REPO_ROOT}/outputs/cache/phase3g_endpoint_suppression_compact/${DATASET}}"
PRETRAINED_EMBEDDER="${PRETRAINED_RESIDUAL_EMBEDDER_PATH:-${REPO_ROOT}/outputs/models/residual_word2vec/THEIA_E3_RAW_DETAIL_RULES_V1_LATENT64_word2vec_window3.pkl}"
SSPM_CONDITIONAL_HEAD_ARCH="${SSPM_CONDITIONAL_HEAD_ARCH:-shared_lowrank_v1}"

case "${SSPM_CONDITIONAL_HEAD_ARCH}" in
    shared_lowrank_v1)
        ACTION_HEAD_NAME="THEIA_E3_E4_PHASE3G_CONDITIONAL_HEAD.pkl"
        ACTION_HEAD_LABEL="THEIA_E3 E4 conditional head"
        ;;
    dual_lowrank_by_target_case_v2)
        ACTION_HEAD_NAME="THEIA_E3_E4_PHASE3G_CONDITIONAL_DUAL_HEAD.pkl"
        ACTION_HEAD_LABEL="THEIA_E3 E4 conditional dual head"
        ;;
    *)
        echo "error: unsupported THEIA_E3 SSPM_CONDITIONAL_HEAD_ARCH: " \
            "${SSPM_CONDITIONAL_HEAD_ARCH}" >&2
        exit 2
        ;;
esac

ACTION_HEAD_PATH="${ACTION_HEAD_CHECKPOINT_PATH_E4_PHASE3G:-${ACTION_HEAD_ROOT}/${ACTION_HEAD_NAME}}"

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
    echo "[theia e4 theia_v1] preflight start"
    record_missing_file "${BASE_RUNNER}" "base Phase3E runner"
    record_missing_file "${PRETRAINED_EMBEDDER}" "THEIA_E3 residual Word2Vec embedder"
    record_missing_file "${CHECKPOINT_ROOT}/THEIA_E3_E4_PHASE3E_BASE_FULL.pkl" \
        "THEIA_E3 E4 base checkpoint"
    record_missing_file "${ACTION_HEAD_PATH}" "${ACTION_HEAD_LABEL}"
    record_missing_file \
        "${PHASE3E_CACHE_ROOT}/compact_used_node_embeddings/THEIA_E3/compact_node_embeddings.npy" \
        "THEIA_E3 compact node embeddings"
    record_missing_file \
        "${PHASE3E_CACHE_ROOT}/compact_used_node_embeddings/THEIA_E3/compact_embedding_meta.json" \
        "THEIA_E3 compact embedding metadata"
    record_missing_file "${PHASE3E_CACHE_ROOT}/event_indices/THEIA_E3/event_index_test.memmap" \
        "THEIA_E3 test event index"
    record_missing_file "${PHASE3E_CACHE_ROOT}/event_indices/THEIA_E3/event_index_validation.memmap" \
        "THEIA_E3 validation event index"
    record_missing_file \
        "${ACTION_VALIDATION_CACHE_DIR}/conditional_train_memmaps/X_conditional_s4d_complex_node.memmap" \
        "THEIA_E3 E4 conditional X memmap"
    record_missing_file \
        "${ACTION_VALIDATION_CACHE_DIR}/conditional_train_memmaps/Y_conditional_s4d_complex_node.memmap" \
        "THEIA_E3 E4 conditional Y memmap"
    record_missing_file \
        "${ACTION_VALIDATION_CACHE_DIR}/conditional_train_memmaps/target_case_s4d_complex_node.memmap" \
        "THEIA_E3 E4 conditional target-case memmap"
    record_missing_dir "${ENDPOINT_CACHE_DIR}" "THEIA_E3 compact endpoint suppression cache root"

    if (( ${#missing_paths[@]} > 0 )); then
        echo "error: THEIA_E3 E4 full prerequisites are missing:" >&2
        printf '  - %s\n' "${missing_paths[@]}" >&2
        echo "error: refusing to infer to avoid reusing CADETS_E3 artifacts" >&2
        exit 2
    fi

    if [[ "${DATASET}" != "THEIA_E3" || "${RUN_ONLY}" != "E4_NONE" ]]; then
        echo "error: wrapper must run only DATASET=THEIA_E3 RUN_ONLY=E4_NONE" >&2
        exit 2
    fi
    if [[ "${CONDITIONAL_ENDPOINT_SUPPRESSION_MODE:-pair_only}" != "pair_only" ]]; then
        echo "error: endpoint suppression mode must remain pair_only" >&2
        exit 2
    fi
    if [[ "${ACTION_TYPE_ALERT_POLICY:-theia_v1}" != "theia_v1" ]]; then
        echo "error: ACTION_TYPE_ALERT_POLICY must be theia_v1" >&2
        exit 2
    fi
    mkdir -p "${RESULT_ROOT}" "${LOG_DIR}"
    echo "[theia e4 theia_v1] preflight ok"
}

preflight

env \
    DATASET="${DATASET}" \
    RUN_ONLY="${RUN_ONLY}" \
    STAGE="${STAGE}" \
    RESULT_ROOT="${RESULT_ROOT}" \
    LOG_DIR="${LOG_DIR}" \
    PHASE3E_CACHE_ROOT="${PHASE3E_CACHE_ROOT}" \
    SSPM_BASE_CHECKPOINT_ROOT="${CHECKPOINT_ROOT}" \
    ACTION_HEAD_CHECKPOINT_ROOT="${ACTION_HEAD_ROOT}" \
    ACTION_HEAD_CHECKPOINT_PATH_E4_PHASE3G="${ACTION_HEAD_PATH}" \
    ACTION_VALIDATION_CACHE_DIR="${ACTION_VALIDATION_CACHE_DIR}" \
    CONDITIONAL_ENDPOINT_SUPPRESSION_CACHE_DIR="${ENDPOINT_CACHE_DIR}" \
    PRETRAINED_RESIDUAL_EMBEDDER_PATH="${PRETRAINED_EMBEDDER}" \
    SSPM_SCORE_HEAD="conditional_action_semantic" \
    SSPM_CONDITIONAL_HEAD_ARCH="${SSPM_CONDITIONAL_HEAD_ARCH}" \
    SSPM_TRAIN_MODE="load_and_infer" \
    SSPM_INFER_FAST_PATH="true" \
    RSS_PROFILE_MODE="online_minimal" \
    NODE_EMBEDDING_LOOKUP_MODE="lazy_mmap_lru" \
    NODE_EMBEDDING_LAZY_BACKING="compact_used_nodes" \
    NODE_EMBEDDING_CACHE_MAX_NODES="100000" \
    NODE_EMBEDDING_CACHE_EVICT_POLICY="lru" \
    CONDITIONAL_ENDPOINT_AWARE_SUPPRESSION="true" \
    CONDITIONAL_ENDPOINT_SUPPRESSION_MODE="pair_only" \
    CONDITIONAL_LOW_SUPPORT_POLICY="conservative_max" \
    CONDITIONAL_LOW_SUPPORT_MARGIN="0.05" \
    EVENT_THRESHOLD_MODE="conditional_target_action_type_group_quantile" \
    EVENT_THRESHOLD_QUANTILE="0.999" \
    ACTION_TYPE_ALERT_POLICY="theia_v1" \
    MAX_TRAIN_EVENTS="0" \
    MAX_REF_EVENTS="0" \
    MAX_TEST_EVENTS="0" \
    bash "${BASE_RUNNER}"
