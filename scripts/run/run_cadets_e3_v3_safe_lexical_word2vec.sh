#!/usr/bin/env bash
set -euo pipefail

# CADETS_E3 FreeBSD v3 safe lexical train-only residual Word2Vec runner.
# This runner does not build Phase3E artifacts, train Phase3G, run inference, or
# read ground-truth labels.

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
STAGE="${STAGE:-train_word2vec}"
DRY_RUN="${DRY_RUN:-0}"

WORD2VEC_ROOT="${WORD2VEC_ROOT:-${REPO_ROOT}/outputs/models/residual_word2vec}"
WORD2VEC_TAG="${WORD2VEC_TAG:-FREEBSD_V3_SAFE_LEXICAL_LATENT64}"
WORD2VEC_CORPUS_ROOT="${WORD2VEC_CORPUS_ROOT:-${REPO_ROOT}/tmp/word2vec_corpus_cadets_e3_v3_safe_lexical}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/cadets_e3_v3_safe_lexical_word2vec}"

MAX_TRAIN_EVENTS="${MAX_TRAIN_EVENTS:-0}"
FETCH_SIZE="${FETCH_SIZE:-10000}"
PROGRESS_INTERVAL_EVENTS="${PROGRESS_INTERVAL_EVENTS:-100000}"
MAX_TOKENS_PER_NODE="${MAX_TOKENS_PER_NODE:-8}"
PROCESS_SEMANTICS_CONFIG="${PROCESS_SEMANTICS_CONFIG:-configs/common/process_semantics.yaml}"
WORD2VEC_WINDOW="${WORD2VEC_WINDOW:-3}"
WORD2VEC_MIN_COUNT="${WORD2VEC_MIN_COUNT:-1}"
WORD2VEC_SG="${WORD2VEC_SG:-1}"
WORD2VEC_NEGATIVE="${WORD2VEC_NEGATIVE:-5}"
WORD2VEC_EPOCHS="${WORD2VEC_EPOCHS:-10}"
WORD2VEC_WORKERS="${WORD2VEC_WORKERS:-4}"
WORD2VEC_SEED="${WORD2VEC_SEED:-0}"
WORD2VEC_OOV_POLICY="${WORD2VEC_OOV_POLICY:-unk}"

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
    if [[ "${DATASET}" != "CADETS_E3" ]]; then
        echo "error: this runner is restricted to CADETS_E3" >&2
        exit 2
    fi
    if [[ "${SEMANTIC_MODE}" != "cadets_freebsd_raw_detail_v3_safe_lexical" ]]; then
        echo "error: SEMANTIC_MODE must be cadets_freebsd_raw_detail_v3_safe_lexical" >&2
        exit 2
    fi
    if [[ "${STAGE}" != "train_word2vec" ]]; then
        echo "error: this runner supports only STAGE=train_word2vec" >&2
        exit 2
    fi
    if ! is_true "${DRY_RUN}"; then
        if [[ -z "${CLAD_DB_PASSWORD:-}" ]]; then
            echo "error: set CLAD_DB_PASSWORD before running CADETS v3 Word2Vec" >&2
            exit 2
        fi
        require_writable_dir "${WORD2VEC_ROOT}" "Word2Vec root"
        require_writable_dir "${WORD2VEC_CORPUS_ROOT}" "Word2Vec corpus root"
        require_writable_dir "${LOG_DIR}" "log directory"
    fi
}

run_train_word2vec() {
    local cmd=(
        "${PYTHON_BIN}" legacy/tools/train_residual_word2vec_models.py
        --datasets "${DATASET}"
        --out_dir "${WORD2VEC_ROOT}"
        --out_tag "${WORD2VEC_TAG}"
        --max_train_events "${MAX_TRAIN_EVENTS}"
        --fetch_size "${FETCH_SIZE}"
        --progress_interval_events "${PROGRESS_INTERVAL_EVENTS}"
        --max_tokens_per_node "${MAX_TOKENS_PER_NODE}"
        --process_semantics_config "${PROCESS_SEMANTICS_CONFIG}"
        --semantic_mode "${SEMANTIC_MODE}"
        --word2vec_window "${WORD2VEC_WINDOW}"
        --word2vec_min_count "${WORD2VEC_MIN_COUNT}"
        --word2vec_sg "${WORD2VEC_SG}"
        --word2vec_negative "${WORD2VEC_NEGATIVE}"
        --word2vec_epochs "${WORD2VEC_EPOCHS}"
        --word2vec_workers "${WORD2VEC_WORKERS}"
        --word2vec_seed "${WORD2VEC_SEED}"
        --word2vec_oov_policy "${WORD2VEC_OOV_POLICY}"
        --word2vec_corpus_file_dir "${WORD2VEC_CORPUS_ROOT}"
    )
    if is_true "${DRY_RUN}"; then
        echo "===== DRY_RUN train_word2vec CADETS_E3_FREEBSD_V3_SAFE_LEXICAL ====="
        print_command "${cmd[@]}"
        return 0
    fi
    "${cmd[@]}"
}

preflight
run_train_word2vec
