"""Reranked Candidate Selection: dense evidence plus multiple answer candidates and judge selection."""

from __future__ import annotations

import argparse
import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


@dataclass
class Frame:
    timestamp: float
    image: Image.Image


@dataclass
class CandidateFrame:
    timestamp: float
    frame_index: int
    image: Image.Image
    score: float
    ahash: int
    source: str


@dataclass
class Config:
    dtype: str = "auto"
    device_map: str = "auto"
    temperature: float = 0.0
    top_p: float = 0.9
    max_new_tokens_evidence: int = 200
    max_new_tokens_final: int = 150
    max_new_tokens_judge: int = 150
    final_candidate_count: int = 3
    final_candidate_temperature: float = 0.4
    final_candidate_top_p: float = 0.9
    max_pixels: int = 589824
    max_frames: int = 16
    uniform_frames: int = 8
    keyframe_candidates_per_second: float = 1.0
    min_frame_gap_seconds: float = 0.5
    min_hash_distance: int = 8
    whisper_model_size: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    language: str = "en"
    voiceover_max_chars: int = 6000

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in allowed})


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


def load_model(model_path: str, dtype: str, device_map: str):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=resolve_dtype(dtype),
        device_map=device_map,
        offload_folder="outputs/offload_reranked_candidate",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    return processor, model


def resize_to_max_pixels(image: Image.Image, max_pixels: int) -> Image.Image:
    width, height = image.size
    if max_pixels <= 0 or width * height <= max_pixels:
        return image
    scale = (max_pixels / float(width * height)) ** 0.5
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def image_ahash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    threshold = float(small.mean())
    bits = (small > threshold).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hash_distance(left: int, right: int) -> int:
    return int((left ^ right).bit_count())


def score_frame(gray: np.ndarray, prev_gray: np.ndarray | None, timestamp: float, duration: float) -> float:
    sharpness = min(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 650.0, 1.0)
    edge_density = min(float(np.mean(cv2.Canny(gray, 80, 160) > 0)) * 5.0, 1.0)
    motion = 0.0 if prev_gray is None else min(float(np.mean(cv2.absdiff(gray, prev_gray))) / 64.0, 1.0)
    late_bonus = 0.12 if duration > 0 and timestamp >= duration * 0.75 else 0.0
    return sharpness * 0.40 + edge_density * 0.30 + motion * 0.30 + late_bonus


def make_candidate(cap, idx: int, fps: float, max_pixels: int, score: float, ahash: int, source: str) -> CandidateFrame | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    if not ok:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = resize_to_max_pixels(Image.fromarray(rgb), max_pixels)
    return CandidateFrame(
        timestamp=float(idx) / max(fps, 1e-6),
        frame_index=int(idx),
        image=image,
        score=score,
        ahash=ahash,
        source=source,
    )


def is_distinct_enough(candidate: CandidateFrame, selected: List[CandidateFrame], cfg: Config) -> bool:
    for item in selected:
        if abs(candidate.timestamp - item.timestamp) < cfg.min_frame_gap_seconds:
            return False
        if hash_distance(candidate.ahash, item.ahash) < cfg.min_hash_distance:
            return False
    return True


def sample_uniform_keyframes(video_path: Path, cfg: Config) -> List[Frame]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if frame_count <= 0:
        cap.release()
        return []

    duration = frame_count / max(fps, 1e-6)
    uniform_count = min(cfg.uniform_frames, cfg.max_frames)
    uniform_indices = np.linspace(0, max(0, frame_count - 1), uniform_count, dtype=int)

    candidates: List[CandidateFrame] = []
    prev_gray = None
    step = 1.0 / max(cfg.keyframe_candidates_per_second, 0.1)
    candidate_times = np.arange(0.0, max(duration, step), step)
    seen_indices: set[int] = set()

    for timestamp in candidate_times:
        idx = min(frame_count - 1, max(0, int(round(float(timestamp) * fps))))
        if idx in seen_indices:
            continue
        seen_indices.add(idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        score = score_frame(gray, prev_gray, float(idx) / max(fps, 1e-6), duration)
        prev_gray = gray
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        candidates.append(
            CandidateFrame(
                timestamp=float(idx) / max(fps, 1e-6),
                frame_index=idx,
                image=resize_to_max_pixels(Image.fromarray(rgb), cfg.max_pixels),
                score=score,
                ahash=image_ahash(gray),
                source="keyframe",
            )
        )

    selected: List[CandidateFrame] = []
    for idx in uniform_indices:
        nearest = min(candidates, key=lambda item: abs(item.frame_index - int(idx)), default=None)
        if nearest is not None:
            candidate = CandidateFrame(
                timestamp=nearest.timestamp,
                frame_index=nearest.frame_index,
                image=nearest.image,
                score=nearest.score + 0.20,
                ahash=nearest.ahash,
                source="uniform",
            )
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            candidate = make_candidate(cap, int(idx), fps, cfg.max_pixels, 0.5, image_ahash(gray), "uniform")
        if candidate is not None and is_distinct_enough(candidate, selected, cfg):
            selected.append(candidate)

    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if len(selected) >= cfg.max_frames:
            break
        if is_distinct_enough(candidate, selected, cfg):
            selected.append(candidate)

    if len(selected) < cfg.max_frames:
        chosen = {item.frame_index for item in selected}
        for candidate in sorted(candidates, key=lambda item: item.timestamp):
            if len(selected) >= cfg.max_frames:
                break
            if candidate.frame_index not in chosen:
                selected.append(candidate)
                chosen.add(candidate.frame_index)

    cap.release()
    selected = sorted(selected, key=lambda item: item.timestamp)[: cfg.max_frames]
    return [Frame(timestamp=item.timestamp, image=item.image) for item in selected]


def timestamps(frames: Iterable[Frame]) -> str:
    return ", ".join(f"{frame.timestamp:.1f}s" for frame in frames)


def build_evidence_prompt(question: str, frames: List[Frame], voiceover: str) -> str:
    return (
        "You are the evidence-extraction stage of an advertisement video QA pipeline.\n"
        f"Question: {question}\n"
        f"Uniform plus keyframe timestamps: {timestamps(frames)}\n\n"
        f"Whisper voiceover transcript:\n{voiceover or '[No intelligible voiceover was detected.]'}\n\n"
        "Produce a compact evidence brief for a second-stage answer writer, not a final answer. Identify the "
        "advertised product or service, visible demonstrations and use cases, concrete visual claims, readable "
        "on-screen text claims, and spoken claims in the transcript. Connect each supported feature to a stated "
        "or clearly shown consumer benefit or pain point when applicable. Preserve brand names, product names, "
        "numbers, offer terms, and mechanisms only when visible or spoken. Mark uncertain details as uncertain "
        "rather than guessing. Do not invent text, specifications, prices, guarantees, audiences, or benefits."
    )


def build_evidence_messages(question: str, frames: List[Frame], voiceover: str) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": build_evidence_prompt(question, frames, voiceover)}]
    content.extend({"type": "image", "image": frame.image} for frame in frames)
    return [{"role": "user", "content": content}]


def build_final_messages(question: str, evidence: str) -> List[Dict[str, Any]]:
    prompt = (
        "You are the final answer stage of an advertisement video QA pipeline.\n"
        f"Question: {question}\n\n"
        f"Evidence brief from the video and Whisper voiceover:\n{evidence}\n\n"
        "Answer the question directly in 2-4 concise English sentences. Use only the evidence brief; omit "
        "uncertain details. Identify the product when supported, cover the major selling points relevant to the "
        "question, and connect them to consumer value or pain points when relevant. Do not mention the pipeline, "
        "frames, transcript, evidence brief, or uncertainty labels. Do not invent details."
    )
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]


def build_judge_messages(question: str, evidence: str, candidates: List[str]) -> List[Dict[str, Any]]:
    numbered_candidates = "\n\n".join(
        f"Candidate {idx}:\n{candidate.strip()}" for idx, candidate in enumerate(candidates, start=1)
    )
    prompt = (
        "You are the judge stage of an advertisement video QA pipeline.\n"
        f"Question: {question}\n\n"
        f"Evidence brief from the video and Whisper voiceover:\n{evidence}\n\n"
        f"Answer candidates:\n{numbered_candidates}\n\n"
        "Select the single best candidate using only the evidence brief. Prefer answers that are faithful, directly "
        "answer the question, cover the main selling points and consumer value, and avoid unsupported details. "
        "Return only the chosen final answer text in 2-4 concise English sentences. Do not mention candidate "
        "numbers, judging, the pipeline, frames, transcript, evidence brief, or uncertainty labels."
    )
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def clear_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate(
    processor,
    model,
    messages: List[Dict[str, Any]],
    max_new_tokens: int,
    cfg: Config,
    temperature: float | None = None,
    top_p: float | None = None,
) -> str:
    generation_temperature = cfg.temperature if temperature is None else temperature
    generation_top_p = cfg.top_p if top_p is None else top_p
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = [item["image"] for item in messages[0]["content"] if item.get("type") == "image"]
    processor_kwargs: Dict[str, Any] = {"text": [text], "padding": True, "return_tensors": "pt"}
    if images:
        processor_kwargs["images"] = images
    inputs = processor(**processor_kwargs).to(model.device)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": generation_temperature > 0,
    }
    if generation_temperature > 0:
        generation_kwargs.update({"temperature": generation_temperature, "top_p": generation_top_p})
    try:
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_kwargs)
        output_ids = output_ids[:, inputs.input_ids.shape[1] :]
        answer = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return strip_thinking(answer)
    finally:
        del inputs
        if "output_ids" in locals():
            del output_ids
        clear_cuda_cache()


def generate_final_candidates(processor, model, question: str, evidence: str, cfg: Config) -> List[str]:
    candidates: List[str] = []
    for _ in range(max(1, cfg.final_candidate_count)):
        answer = generate(
            processor,
            model,
            build_final_messages(question, evidence),
            cfg.max_new_tokens_final,
            cfg,
            temperature=cfg.final_candidate_temperature,
            top_p=cfg.final_candidate_top_p,
        )
        if answer:
            candidates.append(answer)
    if not candidates:
        candidates.append(
            generate(
                processor,
                model,
                build_final_messages(question, evidence),
                cfg.max_new_tokens_final,
                cfg,
                temperature=0.0,
            )
        )
    return candidates


def judge_final_answer(processor, model, question: str, evidence: str, candidates: List[str], cfg: Config) -> str:
    if len(candidates) == 1:
        return candidates[0]
    answer = generate(
        processor,
        model,
        build_judge_messages(question, evidence, candidates),
        cfg.max_new_tokens_judge,
        cfg,
        temperature=0.0,
    )
    return answer or candidates[0]


def run_sample(processor, model, row: Dict[str, Any], frames: List[Frame], voiceover: str, cfg: Config) -> str:
    question = str(row["question"])
    evidence = generate(
        processor,
        model,
        build_evidence_messages(question, frames, voiceover),
        cfg.max_new_tokens_evidence,
        cfg,
    )
    candidates = generate_final_candidates(processor, model, question, evidence, cfg)
    return judge_final_answer(processor, model, question, evidence, candidates, cfg)


def load_whisper(cfg: Config):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper is required for Whisper voiceover transcription.") from exc
    return WhisperModel(cfg.whisper_model_size, device=cfg.whisper_device, compute_type=cfg.whisper_compute_type)


def transcribe_voiceover(whisper_model, video_path: Path, cfg: Config) -> str:
    try:
        segments, _ = whisper_model.transcribe(
            str(video_path),
            language=cfg.language or None,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        return text[: cfg.voiceover_max_chars].strip()
    except Exception as exc:
        print(f"Warning: Whisper failed for {video_path.name}: {exc}", flush=True)
        return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-file", default="MAC_QA.jsonl")
    parser.add_argument("--video-dir", default="data")
    parser.add_argument("--model-path", default="weights/pretrained/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--config", default="config/reranked_candidate.json")
    parser.add_argument("--output-file", default="outputs/reranked_candidate_3108_answer_only.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.load(args.config)
    rows = load_jsonl(args.qa_file)
    if args.limit > 0:
        rows = rows[: args.limit]

    output_path = Path(args.output_file)
    if args.overwrite and output_path.exists():
        output_path.unlink()
    done = existing_ids(output_path)

    whisper_model = load_whisper(cfg)
    processor, model = load_model(args.model_path, cfg.dtype, cfg.device_map)

    for row in tqdm(rows, desc="Reranked Candidate Selection"):
        sample_id = str(row["id"])
        if sample_id in done:
            continue
        video_path = Path(args.video_dir) / f"{sample_id}.mp4"
        if not video_path.exists():
            append_jsonl(output_path, {"id": sample_id, "model_prediction": "Video file missing."})
            continue
        frames = sample_uniform_keyframes(video_path, cfg)
        if not frames:
            append_jsonl(output_path, {"id": sample_id, "model_prediction": "Video could not be decoded."})
            continue
        voiceover = transcribe_voiceover(whisper_model, video_path, cfg)
        try:
            answer = run_sample(processor, model, row, frames, voiceover, cfg)
        except torch.OutOfMemoryError as exc:
            clear_cuda_cache()
            retry_frames = frames[: min(8, len(frames))]
            print(
                f"Warning: CUDA OOM for {sample_id} with {len(frames)} frames; retrying with {len(retry_frames)} frames: {exc}",
                flush=True,
            )
            try:
                answer = run_sample(processor, model, row, retry_frames, voiceover, cfg)
            except torch.OutOfMemoryError as retry_exc:
                clear_cuda_cache()
                print(f"Warning: CUDA OOM retry failed for {sample_id}: {retry_exc}", flush=True)
                answer = "Video could not be processed due to CUDA memory limits."
        append_jsonl(output_path, {"id": sample_id, "model_prediction": answer})


if __name__ == "__main__":
    main()
