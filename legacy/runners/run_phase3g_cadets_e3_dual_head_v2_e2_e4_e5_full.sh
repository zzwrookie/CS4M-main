#!/usr/bin/env bash
set -euo pipefail

# Serial CADETS_E3 Phase3G dual-head v2 full inference for E2/E4/E5 NONE.
# This wrapper never launches training, precompute, RUN_ONLY=ALL, or OFSM variants.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${CS4M_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
BASE_RUNNER="${REPO_ROOT}/scripts/run/run_cadets_e3_phase3e_node_action_semantic_full.sh"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

DATASET="CADETS_E3"
STAGE="infer_ablation"
RUN_LIST=(E2_NONE E4_NONE E5_NONE)

TMP_ROOT="${TMP_ROOT:-${REPO_ROOT}/tmp}"
RESULT_ROOT="${RESULT_ROOT:-${TMP_ROOT}/tflr_light_phase3g_v2_full}"
LOG_DIR="${LOG_DIR:-${TMP_ROOT}/logs/cadets_e3_phase3g_dual_head_v2_full}"
ACTION_VALIDATION_CACHE_DIR="${ACTION_VALIDATION_CACHE_DIR:-${TMP_ROOT}/phase3g_action_validation_v2/${DATASET}}"

PHASE3E_CACHE_ROOT="${PHASE3E_CACHE_ROOT:-${REPO_ROOT}/outputs/cache/phase3e}"
CHECKPOINT_ROOT="${SSPM_BASE_CHECKPOINT_ROOT:-${REPO_ROOT}/outputs/models/sspm_phase3e}"
ACTION_HEAD_V2_ROOT="${ACTION_HEAD_V2_ROOT:-${REPO_ROOT}/outputs/models/phase3g_action_heads_v2}"
ENDPOINT_CACHE_DIR="${CONDITIONAL_ENDPOINT_SUPPRESSION_CACHE_DIR:-${REPO_ROOT}/outputs/cache/phase3g_endpoint_suppression_compact/${DATASET}}"
DEFAULT_EMBEDDER="${REPO_ROOT}/outputs/models/residual_word2vec"
DEFAULT_EMBEDDER+="/CADETS_E3_RAW_DETAIL_RULES_V1_LATENT64_word2vec_window3.pkl"
PRETRAINED_EMBEDDER="${PRETRAINED_RESIDUAL_EMBEDDER_PATH:-${DEFAULT_EMBEDDER}}"

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

require_no_forbidden_modes() {
    local run_only
    if [[ "${STAGE}" != "infer_ablation" ]]; then
        echo "error: wrapper must stay in STAGE=infer_ablation" >&2
        exit 2
    fi
    for run_only in "${RUN_LIST[@]}"; do
        if [[ "${run_only}" == "ALL" || "${run_only}" == *"OFSM"* ]]; then
            echo "error: forbidden RUN_ONLY in dual-head v2 inference list: ${run_only}" >&2
            exit 2
        fi
    done
}

validate_v2_checkpoints() {
    "${PYTHON_BIN}" - <<'PY'
import pickle
from pathlib import Path

root = Path("outputs/models/phase3g_action_heads_v2")
expected = {
    "schema": "phase3g_conditional_semantic_dual_head_v2",
    "arch": "dual_lowrank_by_target_case_v2",
    "score_head": "conditional_action_semantic",
    "context_dim": 136,
    "target_case_mode": "conditional_cold_action_else_event",
    "target_case_head_mode": "target_case_specific",
}
required_keys = (
    "event_w1",
    "event_w2",
    "event_bias",
    "action_w1",
    "action_w2",
    "action_bias",
)
paths = {
    "E2_NONE": root / "CADETS_E3_E2_PHASE3G_CONDITIONAL_DUAL_HEAD.pkl",
    "E4_NONE": root / "CADETS_E3_E4_PHASE3G_CONDITIONAL_DUAL_HEAD.pkl",
    "E5_NONE": root / "CADETS_E3_E5_PHASE3G_CONDITIONAL_DUAL_HEAD.pkl",
}
errors = []
for run_only, path in paths.items():
    if not path.exists():
        errors.append(f"{run_only}: missing v2 checkpoint: {path}")
        continue
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    config = dict(payload.get("config", {}))
    metadata = dict(payload.get("metadata", {}))
    schema = str(payload.get("schema", ""))
    arch = str(config.get("conditional_head_arch", config.get("head_arch", "")))
    checks = {
        "schema": schema,
        "arch": arch,
        "score_head": str(metadata.get("score_head", "")),
        "context_dim": int(config.get("input_dim", -1)),
        "target_case_mode": str(config.get("target_case_mode", "")),
        "target_case_head_mode": str(config.get("target_case_head_mode", "")),
    }
    for key, expected_value in expected.items():
        if checks.get(key) != expected_value:
            errors.append(
                f"{run_only}: {key} mismatch: got {checks.get(key)!r}, "
                f"expected {expected_value!r} at {path}"
            )
    for key in required_keys:
        if key not in payload:
            errors.append(f"{run_only}: missing v2 key {key}: {path}")
    print(
        f"[cadets dual-head v2 infer] checkpoint ok {run_only}: "
        f"{path} schema={schema} arch={arch}"
    )
if errors:
    print("error: CADETS_E3 v2 checkpoint preflight failed", flush=True)
    for error in errors:
        print(f"  - {error}", flush=True)
    raise SystemExit(2)
PY
}

preflight() {
    echo "[cadets dual-head v2 infer] preflight start"
    require_no_forbidden_modes
    record_missing_file "${BASE_RUNNER}" "base Phase3E runner"
    record_missing_file "${PRETRAINED_EMBEDDER}" "CADETS_E3 residual Word2Vec embedder"
    record_missing_file "${CHECKPOINT_ROOT}/CADETS_E3_E2_PHASE3E_BASE_FULL.pkl" \
        "CADETS_E3 E2 Phase3E base checkpoint"
    record_missing_file "${CHECKPOINT_ROOT}/CADETS_E3_E4_PHASE3E_BASE_FULL.pkl" \
        "CADETS_E3 E4 Phase3E base checkpoint"
    record_missing_file "${CHECKPOINT_ROOT}/CADETS_E3_E5_PHASE3E_BASE_FULL.pkl" \
        "CADETS_E3 E5 Phase3E base checkpoint"
    record_missing_file "${ACTION_HEAD_V2_ROOT}/CADETS_E3_E2_PHASE3G_CONDITIONAL_DUAL_HEAD.pkl" \
        "CADETS_E3 E2 Phase3G dual-head v2 checkpoint"
    record_missing_file "${ACTION_HEAD_V2_ROOT}/CADETS_E3_E4_PHASE3G_CONDITIONAL_DUAL_HEAD.pkl" \
        "CADETS_E3 E4 Phase3G dual-head v2 checkpoint"
    record_missing_file "${ACTION_HEAD_V2_ROOT}/CADETS_E3_E5_PHASE3G_CONDITIONAL_DUAL_HEAD.pkl" \
        "CADETS_E3 E5 Phase3G dual-head v2 checkpoint"
    record_missing_file "${PHASE3E_CACHE_ROOT}/node_embeddings/CADETS_E3_latent64/node_embeddings.npy" \
        "CADETS_E3 node embeddings"
    record_missing_file \
        "${PHASE3E_CACHE_ROOT}/action_embeddings/CADETS_E3_latent64/action_embeddings.npy" \
        "CADETS_E3 action embeddings"
    record_missing_file "${PHASE3E_CACHE_ROOT}/event_indices/CADETS_E3/event_index_validation.memmap" \
        "CADETS_E3 validation event index"
    record_missing_file "${PHASE3E_CACHE_ROOT}/event_indices/CADETS_E3/event_index_test.memmap" \
        "CADETS_E3 test event index"
    record_missing_file "${PHASE3E_CACHE_ROOT}/event_indices/CADETS_E3/event_index_meta.json" \
        "CADETS_E3 event index metadata"
    record_missing_file \
        "${PHASE3E_CACHE_ROOT}/compact_used_node_embeddings/CADETS_E3/compact_node_embeddings.npy" \
        "CADETS_E3 compact node embeddings"
    record_missing_file \
        "${PHASE3E_CACHE_ROOT}/compact_used_node_embeddings/CADETS_E3/compact_embedding_meta.json" \
        "CADETS_E3 compact embedding metadata"
    record_missing_dir "${ENDPOINT_CACHE_DIR}" "CADETS_E3 compact endpoint suppression cache root"

    if (( ${#missing_paths[@]} > 0 )); then
        echo "error: CADETS_E3 dual-head v2 inference prerequisites are missing:" >&2
        printf '  - %s\n' "${missing_paths[@]}" >&2
        echo "error: refusing to train, precompute, or fall back to v1 checkpoints" >&2
        exit 2
    fi

    validate_v2_checkpoints

    if [[ "${CONDITIONAL_ENDPOINT_SUPPRESSION_MODE:-pair_only}" != "pair_only" ]]; then
        echo "error: endpoint suppression mode must remain pair_only" >&2
        exit 2
    fi
    if [[ "${EVENT_THRESHOLD_MODE:-conditional_target_action_type_group_quantile}" \
        != "conditional_target_action_type_group_quantile" ]]; then
        echo "error: EVENT_THRESHOLD_MODE must be conditional_target_action_type_group_quantile" >&2
        exit 2
    fi
    mkdir -p "${RESULT_ROOT}" "${LOG_DIR}" "${ACTION_VALIDATION_CACHE_DIR}"
    echo "[cadets dual-head v2 infer] result_root=${RESULT_ROOT}"
    echo "[cadets dual-head v2 infer] validation_cache_root=${ACTION_VALIDATION_CACHE_DIR}"
    echo "[cadets dual-head v2 infer] preflight ok"
}

run_one() {
    local run_only="$1"
    local family="$2"
    local checkpoint_path="${ACTION_HEAD_V2_ROOT}/CADETS_E3_${family}_PHASE3G_CONDITIONAL_DUAL_HEAD.pkl"
    local gate_extra=()
    if [[ "${family}" == "E5" ]]; then
        gate_extra+=(
            SSPM_UPDATE_GATE_SCORE_SPACE="conditional_event_score"
            SSPM_UPDATE_GATE_THRESHOLD_MODE="quantile"
            SSPM_UPDATE_GATE_QUANTILE="0.999"
        )
    fi
    local head_override=()
    if [[ "${family}" == "E2" ]]; then
        head_override=(ACTION_HEAD_CHECKPOINT_PATH_E2_PHASE3G="${checkpoint_path}")
    elif [[ "${family}" == "E4" ]]; then
        head_override=(ACTION_HEAD_CHECKPOINT_PATH_E4_PHASE3G="${checkpoint_path}")
    elif [[ "${family}" == "E5" ]]; then
        head_override=(ACTION_HEAD_CHECKPOINT_PATH_E5_PHASE3G="${checkpoint_path}")
    else
        echo "error: unsupported family=${family}" >&2
        exit 2
    fi

    echo "[cadets dual-head v2 infer] start RUN_ONLY=${run_only}"
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
        CONDITIONAL_ENDPOINT_SUPPRESSION_CACHE_DIR="${ENDPOINT_CACHE_DIR}" \
        PRETRAINED_RESIDUAL_EMBEDDER_PATH="${PRETRAINED_EMBEDDER}" \
        "${head_override[@]}" \
        SSPM_SCORE_HEAD="conditional_action_semantic" \
        SSPM_CONDITIONAL_HEAD_ARCH="dual_lowrank_by_target_case_v2" \
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
        SSPM_TRAIN_BACKEND="torch" \
        SSPM_INFER_BACKEND="numpy" \
        "${gate_extra[@]}" \
        bash "${BASE_RUNNER}"
    echo "[cadets dual-head v2 infer] done RUN_ONLY=${run_only}"
}

preflight
run_one "E2_NONE" "E2"
run_one "E4_NONE" "E4"
run_one "E5_NONE" "E5"

echo "[cadets dual-head v2 infer] all requested inference jobs completed"
