#!/usr/bin/env bash
set -euo pipefail

# ClearScope E5 dual-channel bounded smoke wrapper.
# This runs inference only and enables the validation-only node-support channel.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export STAGE="${STAGE:-infer_smoke}"
export OUT_TAG_OVERRIDE="${OUT_TAG_OVERRIDE:-CLEARSCOPE_E5_V33B_DUAL_CHANNEL_NODE_SUPPORT_SMOKE}"
export DUAL_CHANNEL_NODE_SUPPORT_ENABLED="${DUAL_CHANNEL_NODE_SUPPORT_ENABLED:-true}"
DUAL_SCORE_FLOOR_DEFAULT="0.12497031688690186"
if [[ -z "${DUAL_CHANNEL_NODE_SUPPORT_SCORE_FLOOR:-}" ]]; then
    DUAL_CHANNEL_NODE_SUPPORT_SCORE_FLOOR="${DUAL_SCORE_FLOOR_DEFAULT}"
fi
export DUAL_CHANNEL_NODE_SUPPORT_SCORE_FLOOR
export DUAL_CHANNEL_NODE_SUPPORT_THRESHOLD="${DUAL_CHANNEL_NODE_SUPPORT_THRESHOLD:-2}"
DUAL_CHANNEL_DEFAULT="event_semantic_node_support_ge_2"
DUAL_CHANNEL_NODE_SUPPORT_CHANNEL="${DUAL_CHANNEL_NODE_SUPPORT_CHANNEL:-${DUAL_CHANNEL_DEFAULT}}"
export DUAL_CHANNEL_NODE_SUPPORT_CHANNEL

exec bash "${SCRIPT_DIR}/run_clearscope_e5_v33b_bounded_phase3e_phase3g_smoke.sh"
