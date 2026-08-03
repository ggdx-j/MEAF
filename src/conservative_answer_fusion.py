from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def existing_ids(path: str | Path) -> set[str]:
    output = Path(path)
    if not output.exists():
        return set()
    return {str(row["id"]) for row in load_jsonl(output) if "id" in row}


def by_id(path: str | Path) -> Dict[str, str]:
    return {
        str(row["id"]): str(row.get("model_prediction", "")).strip()
        for row in load_jsonl(path)
        if "id" in row
    }


def resolve_dtype(name: str):
    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def normalize_answer(text: str) -> str:
    text = strip_thinking(text)
    text = re.sub(r"^\s*(?:[-*]\s+|\d+\.\s+)", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clear_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(model_path: str, dtype: str, device_map: str):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=resolve_dtype(dtype),
        device_map=device_map,
        offload_folder="outputs/offload_conservative_fusion",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    return processor, model


def build_messages(question: str, reranked_answer: str, dense_answer: str, adaptive_answer: str) -> List[Dict[str, Any]]:
    prompt = (
        "You are the final answer fusion stage for an advertisement video QA task.\n"
        f"Question: {question}\n\n"
        f"Primary answer from Reranked Candidate Selection:\n{reranked_answer or '[missing]'}\n\n"
        f"Secondary answer from Dense Keyframe Evidence:\n{dense_answer or '[missing]'}\n\n"
        f"Secondary answer from Duration-Adaptive Evidence:\n{adaptive_answer or '[missing]'}\n\n"
        "Write one final answer by conservatively fusing these same-model strategy outputs.\n\n"
        "Use the reranked answer as the default primary answer. However, if Dense Keyframe Evidence and Duration-Adaptive Evidence both "
        "provide a consistent complementary detail that the reranked answer misses, include it. If the reranked answer conflicts with both secondary "
        "answers, prefer the claim that is more specific, question-relevant, and less speculative.\n\n"
        "Keep points supported by multiple answers; concrete product or service identity; visible demonstrations; "
        "readable text claims; spoken claims; numbers, offers, and guarantees; and consumer benefits when they are "
        "consistent. Keep complementary details that directly improve coverage of selling points, practical value, "
        "pain points, target audience, or emotional value.\n\n"
        "Remove hallucinated or speculative details; claims that appear in only one secondary answer and are too "
        "specific to verify; truncated fragments; unfinished sentences; bullet or list formatting artifacts; and "
        "duplicated points. Do not mention branch names, candidates, fusion, voting, models, evidence, frames, "
        "transcripts, the pipeline, or uncertainty.\n\n"
        "Return only the final answer in 2-4 concise English sentences. Prefer 60-110 words unless the question "
        "requires more detail. Output a plain paragraph only: no bullet points, numbered lists, Markdown, headings, "
        "line breaks, labels, JSON, or quotation wrappers."
    )
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]


def generate(processor, model, messages: List[Dict[str, Any]], max_new_tokens: int) -> str:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)
    try:
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        output_ids = output_ids[:, inputs.input_ids.shape[1] :]
        answer = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return normalize_answer(answer)
    finally:
        del inputs
        if "output_ids" in locals():
            del output_ids
        clear_cuda_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-file", default="MAC_QA.jsonl")
    parser.add_argument("--reranked-file", default="outputs/reranked_candidate_3108_answer_only.jsonl")
    parser.add_argument("--dense-file", default="outputs/dense_keyframe_3108_answer_only.jsonl")
    parser.add_argument("--adaptive-file", default="outputs/duration_adaptive_3108_answer_only.jsonl")
    parser.add_argument("--model-path", default="weights/pretrained/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output-file", default="outputs/conservative_fusion_3108_answer_only.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qa_rows = load_jsonl(args.qa_file)
    if args.limit > 0:
        qa_rows = qa_rows[: args.limit]

    reranked_answers = by_id(args.reranked_file)
    dense_answers = by_id(args.dense_file)
    adaptive_answers = by_id(args.adaptive_file)

    output_path = Path(args.output_file)
    if args.overwrite and output_path.exists():
        output_path.unlink()
    done = existing_ids(output_path)

    processor, model = load_model(args.model_path, args.dtype, args.device_map)

    for row in tqdm(qa_rows, desc="Conservative Answer Fusion"):
        sample_id = str(row["id"])
        if sample_id in done:
            continue
        answer = generate(
            processor,
            model,
            build_messages(
                str(row["question"]),
                reranked_answers.get(sample_id, ""),
                dense_answers.get(sample_id, ""),
                adaptive_answers.get(sample_id, ""),
            ),
            args.max_new_tokens,
        )
        append_jsonl(output_path, {"id": sample_id, "model_prediction": answer})


if __name__ == "__main__":
    main()
