#!/usr/bin/env bash
set -euo pipefail

# Active CADETS_E3 E4 best-chain runner: Phase3E online state runtime plus
# Phase3G dual conditional head, v2 checkpoint, q=0.9995 threshold.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${CS4M_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
BASE_RUNNER="${REPO_ROOT}/scripts/run/run_cadets_e3_phase3e_node_action_semantic_full.sh"
cd "${REPO_ROOT}"

export DATASET="CADETS_E3"
export STAGE="infer_ablation"
export RUN_ONLY="E4_NONE"
export EVENT_THRESHOLD_MODE="${EVENT_THRESHOLD_MODE:-target_case_quantile}"
export EVENT_THRESHOLD_QUANTILE="${EVENT_THRESHOLD_QUANTILE:-0.9995}"
export SSPM_SCORE_HEAD="${SSPM_SCORE_HEAD:-conditional_action_semantic}"
export SSPM_CONDITIONAL_HEAD_ARCH="${SSPM_CONDITIONAL_HEAD_ARCH:-dual_lowrank_by_target_case_v2}"
export SSPM_STATE_MODEL="${SSPM_STATE_MODEL:-s4d_complex_node}"
export SSPM_UPDATE_GATE_MODE="${SSPM_UPDATE_GATE_MODE:-none}"
export SSPM_UPDATE_GATE_SCORE_SPACE="${SSPM_UPDATE_GATE_SCORE_SPACE:-conditional_event_score}"
export ACTION_TYPE_ALERT_POLICY="${ACTION_TYPE_ALERT_POLICY:-cadets_e4_v2_group_v1}"
export NODE_EMBEDDING_LOOKUP_MODE="${NODE_EMBEDDING_LOOKUP_MODE:-compact_used_nodes}"
export NODE_EMBEDDING_LAZY_BACKING="${NODE_EMBEDDING_LAZY_BACKING:-compact_used_nodes}"
export CONDITIONAL_ENDPOINT_SUPPRESSION_MODE="${CONDITIONAL_ENDPOINT_SUPPRESSION_MODE:-pair_only}"

DEFAULT_ENDPOINT_CACHE="${REPO_ROOT}/outputs/cache/phase3g_endpoint_suppression_compact"
DEFAULT_ACTION_CACHE="${REPO_ROOT}/outputs/cache/phase3g_action_validation_v2_Q09995"
DEFAULT_ACTION_HEAD_ROOT="${REPO_ROOT}/outputs/models/phase3g_action_heads_v2"
DEFAULT_ACTION_HEAD="${DEFAULT_ACTION_HEAD_ROOT}/CADETS_E3_E4_PHASE3G_CONDITIONAL_DUAL_HEAD.pkl"
if [[ -z "${CONDITIONAL_ENDPOINT_SUPPRESSION_CACHE_DIR+x}" ]]; then
    export CONDITIONAL_ENDPOINT_SUPPRESSION_CACHE_DIR="${DEFAULT_ENDPOINT_CACHE}/${DATASET}"
fi
if [[ -z "${ACTION_VALIDATION_CACHE_DIR+x}" ]]; then
    export ACTION_VALIDATION_CACHE_DIR="${DEFAULT_ACTION_CACHE}/${DATASET}"
fi
if [[ -z "${ACTION_HEAD_CHECKPOINT_PATH_E4_PHASE3G+x}" ]]; then
    export ACTION_HEAD_CHECKPOINT_PATH_E4_PHASE3G="${DEFAULT_ACTION_HEAD}"
fi

exec bash "${BASE_RUNNER}"
