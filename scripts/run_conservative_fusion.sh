#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(which python)}"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/weights/pretrained/Qwen2.5-VL-7B-Instruct}"
QA_FILE="${QA_FILE:-${ROOT_DIR}/MAC_QA.jsonl}"
RERANKED_FILE="${RERANKED_FILE:-${ROOT_DIR}/outputs/reranked_candidate_3108_answer_only.jsonl}"
DENSE_FILE="${DENSE_FILE:-${ROOT_DIR}/outputs/dense_keyframe_3108_answer_only.jsonl}"
ADAPTIVE_FILE="${ADAPTIVE_FILE:-${ROOT_DIR}/outputs/duration_adaptive_3108_answer_only.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-${ROOT_DIR}/outputs/conservative_fusion_3108_answer_only.jsonl}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${ROOT_DIR}"
"${PYTHON_BIN}" src/conservative_answer_fusion.py \
  --qa-file "${QA_FILE}" \
  --reranked-file "${RERANKED_FILE}" \
  --dense-file "${DENSE_FILE}" \
  --adaptive-file "${ADAPTIVE_FILE}" \
  --model-path "${MODEL_PATH}" \
  --output-file "${OUTPUT_FILE}" \
  --limit "${LIMIT:-0}" \
  "$@"
