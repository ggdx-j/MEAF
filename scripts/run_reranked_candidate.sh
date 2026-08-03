#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(which python)}"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/weights/pretrained/Qwen2.5-VL-7B-Instruct}"
VIDEO_DIR="${VIDEO_DIR:-${ROOT_DIR}/data}"
QA_FILE="${QA_FILE:-${ROOT_DIR}/MAC_QA.jsonl}"
CONFIG_FILE="${CONFIG_FILE:-${ROOT_DIR}/config/reranked_candidate.json}"
OUTPUT_FILE="${OUTPUT_FILE:-${ROOT_DIR}/outputs/reranked_candidate_3108_answer_only.jsonl}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for proxy_var in ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy; do
  proxy_value="${!proxy_var:-}"
  if [[ "${proxy_value}" == socks://* ]]; then
    printf -v "${proxy_var}" '%s' "socks5://${proxy_value#socks://}"
    export "${proxy_var}"
  fi
done

cd "${ROOT_DIR}"
"${PYTHON_BIN}" src/reranked_candidate_infer.py \
  --qa-file "${QA_FILE}" \
  --video-dir "${VIDEO_DIR}" \
  --model-path "${MODEL_PATH}" \
  --config "${CONFIG_FILE}" \
  --output-file "${OUTPUT_FILE}" \
  --limit "${LIMIT:-0}" \
  "$@"
