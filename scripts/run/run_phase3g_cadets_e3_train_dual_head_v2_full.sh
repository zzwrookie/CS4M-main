#!/usr/bin/env bash
set -euo pipefail

# Train CADETS_E3 Phase3G conditional_action_semantic dual-head v2 checkpoints.
# This wrapper only trains E2/E4/E5 conditional heads. It never launches inference.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_RUNNER="${REPO_ROOT}/scripts/run/run_cadets_e3_phase3e_node_action_semantic_full.sh"
cd "${REPO_ROOT}"

DATASET="CADETS_E3"
STAGE="train_conditional_base"

DEFAULT_RESULT_ROOT="${REPO_ROOT}/outputs/results"
DEFAULT_RESULT_ROOT+="/phase3g_cadets_e3_train_dual_head_v2_full"
RESULT_ROOT="${RESULT_ROOT:-${DEFAULT_RESULT_ROOT}}"

DEFAULT_LOG_DIR="${REPO_ROOT}/logs"
DEFAULT_LOG_DIR+="/cadets_e3_phase3g_train_dual_head_v2_full"
LOG_DIR="${LOG_DIR:-${DEFAULT_LOG_DIR}}"

PHASE3E_CACHE_ROOT="${PHASE3E_CACHE_ROOT:-${REPO_ROOT}/outputs/cache/phase3e}"
CHECKPOINT_ROOT="${SSPM_BASE_CHECKPOINT_ROOT:-${REPO_ROOT}/outputs/models/sspm_phase3e}"
ACTION_HEAD_V2_ROOT="${ACTION_HEAD_V2_ROOT:-${REPO_ROOT}/outputs/models/phase3g_action_heads_v2}"
ACTION_VALIDATION_CACHE_DIR="${ACTION_VALIDATION_CACHE_DIR:-${REPO_ROOT}/outputs/cache/phase3g_action_validation/${DATASET}}"
PRETRAINED_EMBEDDER="${PRETRAINED_RESIDUAL_EMBEDDER_PATH:-${REPO_ROOT}/outputs/models/residual_word2vec/CADETS_E3_RAW_DETAIL_RULES_V1_LATENT64_word2vec_window3.pkl}"

missing_paths=()

record_missing_file() {
    local path="$1"
    local label="$2"
    if [[ ! -f "${path}" ]]; then
        missing_paths+=("${label}: ${path}")
    fi
}

require_no_forbidden_modes() {
    if [[ "${RUN_ONLY:-}" == "ALL" ]]; then
        echo "error: RUN_ONLY=ALL is forbidden for dual-head v2 training" >&2
        exit 2
    fi
    if [[ "${STAGE}" != "train_conditional_base" ]]; then
        echo "error: wrapper must stay in STAGE=train_conditional_base" >&2
        exit 2
    fi
}

preflight_memmap() {
    local state_model="$1"
    local required="$2"
    local key="ema_fixed"
    if [[ "${state_model}" == "s4d_complex_node" ]]; then
        key="s4d_complex_node"
    elif [[ "${state_model}" == "ema_fixed_update_gate_quantile" ]]; then
        key="ema_fixed_update_gate_quantile"
    fi
    local paths=(
        "${ACTION_VALIDATION_CACHE_DIR}/conditional_train_memmaps/X_conditional_${key}.memmap"
        "${ACTION_VALIDATION_CACHE_DIR}/conditional_train_memmaps/Y_conditional_${key}.memmap"
        "${ACTION_VALIDATION_CACHE_DIR}/conditional_train_memmaps/target_case_${key}.memmap"
        "${ACTION_VALIDATION_CACHE_DIR}/conditional_train_memmaps/conditional_memmap_${key}_meta.json"
    )
    local missing=0
    local path
    for path in "${paths[@]}"; do
        if [[ ! -f "${path}" ]]; then
            missing=1
            if [[ "${required}" == "required" ]]; then
                missing_paths+=("CADETS_E3 ${state_model} conditional memmap: ${path}")
            fi
        fi
    done
    if [[ "${missing}" -eq 0 ]]; then
        echo "[cadets dual-head v2] reuse conditional memmap key=${key}"
    elif [[ "${required}" == "optional_build" ]]; then
        echo "[cadets dual-head v2] conditional memmap key=${key} missing; Phase3G train step may build it"
    fi
}

preflight() {
    echo "[cadets dual-head v2] preflight start"
    require_no_forbidden_modes
    record_missing_file "${BASE_RUNNER}" "base Phase3E runner"
    record_missing_file "${PRETRAINED_EMBEDDER}" "CADETS_E3 residual Word2Vec embedder"
    record_missing_file "${CHECKPOINT_ROOT}/CADETS_E3_E2_PHASE3E_BASE_FULL.pkl" \
        "CADETS_E3 E2 Phase3E base checkpoint"
    record_missing_file "${CHECKPOINT_ROOT}/CADETS_E3_E4_PHASE3E_BASE_FULL.pkl" \
        "CADETS_E3 E4 Phase3E base checkpoint"
    record_missing_file "${CHECKPOINT_ROOT}/CADETS_E3_E5_PHASE3E_BASE_FULL.pkl" \
        "CADETS_E3 E5 Phase3E base checkpoint"
    record_missing_file "${PHASE3E_CACHE_ROOT}/node_embeddings/CADETS_E3_latent64/node_embeddings.npy" \
        "CADETS_E3 node embeddings"
    record_missing_file "${PHASE3E_CACHE_ROOT}/action_embeddings/CADETS_E3_latent64/action_embeddings.npy" \
        "CADETS_E3 action embeddings"
    record_missing_file "${PHASE3E_CACHE_ROOT}/event_indices/CADETS_E3/event_index_train.memmap" \
        "CADETS_E3 train event index"
    record_missing_file "${PHASE3E_CACHE_ROOT}/event_indices/CADETS_E3/event_index_meta.json" \
        "CADETS_E3 event index metadata"
    preflight_memmap "ema_fixed" "required"
    preflight_memmap "s4d_complex_node" "required"
    preflight_memmap "ema_fixed_update_gate_quantile" "optional_build"

    if (( ${#missing_paths[@]} > 0 )); then
        echo "error: CADETS_E3 dual-head v2 prerequisites are missing:" >&2
        printf '  - %s\n' "${missing_paths[@]}" >&2
        echo "error: refusing to rerun Phase3E precompute or use non-CADETS artifacts" >&2
        exit 2
    fi

    mkdir -p "${RESULT_ROOT}" "${LOG_DIR}" "${ACTION_HEAD_V2_ROOT}"
    echo "[cadets dual-head v2] preflight ok"
}

run_train_one() {
    local run_only="$1"
    local family="$2"
    local checkpoint_path="${ACTION_HEAD_V2_ROOT}/CADETS_E3_${family}_PHASE3G_CONDITIONAL_DUAL_HEAD.pkl"
    echo "[cadets dual-head v2] train ${run_only} -> ${checkpoint_path}"
    env \
        DATASET="${DATASET}" \
        RUN_ONLY="${run_only}" \
        STAGE="${STAGE}" \
        RESULT_ROOT="${RESULT_ROOT}" \
        LOG_DIR="${LOG_DIR}" \
        PHASE3E_CACHE_ROOT="${PHASE3E_CACHE_ROOT}" \
        SSPM_BASE_CHECKPOINT_ROOT="${CHECKPOINT_ROOT}" \
        ACTION_HEAD_CHECKPOINT_ROOT="${ACTION_HEAD_V2_ROOT}" \
        ACTION_VALIDATION_CACHE_DIR="${ACTION_VALIDATION_CACHE_DIR}" \
        ACTION_HEAD_CHECKPOINT_PATH_${family}_PHASE3G="${checkpoint_path}" \
        PRETRAINED_RESIDUAL_EMBEDDER_PATH="${PRETRAINED_EMBEDDER}" \
        SSPM_SCORE_HEAD="conditional_action_semantic" \
        SSPM_CONDITIONAL_HEAD_ARCH="dual_lowrank_by_target_case_v2" \
        NODE_REPR_FUSION="simple_mean" \
        SSPM_TRAIN_MODE="train_conditional_and_save" \
        SSPM_CONDITIONAL_TRAIN_DATA_MODE="memmap" \
        SSPM_CONDITIONAL_MAX_EPOCHS="80" \
        SSPM_TRAIN_BACKEND="torch" \
        SSPM_INFER_BACKEND="numpy" \
        RSS_PROFILE_MODE="online_minimal" \
        MAX_TRAIN_EVENTS="0" \
        MAX_REF_EVENTS="0" \
        MAX_TEST_EVENTS="0" \
        bash "${BASE_RUNNER}"
    echo "[cadets dual-head v2] done ${run_only}"
}

preflight
run_train_one "E2_CONDITIONAL_BASE_FULL" "E2"
run_train_one "E4_CONDITIONAL_BASE_FULL" "E4"
run_train_one "E5_CONDITIONAL_BASE_FULL" "E5"

echo "[cadets dual-head v2] all requested training jobs completed"
