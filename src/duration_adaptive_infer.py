"""Duration-Adaptive Evidence: Whisper transcript plus duration-adaptive uniform frame sampling."""

from __future__ import annotations

import argparse
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
class Config:
    dtype: str = "auto"
    device_map: str = "auto"
    temperature: float = 0.0
    top_p: float = 0.9
    max_new_tokens: int = 180
    max_new_tokens_evidence: int = 220
    max_new_tokens_final: int = 180
    max_pixels: int = 262144
    frames_short: int = 6
    frames_medium: int = 8
    frames_long: int = 10
    frames_xlong: int = 12
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
        offload_folder="outputs/offload_duration_adaptive",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    return processor, model


def choose_frame_count(duration: float, cfg: Config) -> int:
    if duration <= 30:
        return cfg.frames_short
    if duration <= 90:
        return cfg.frames_medium
    if duration <= 180:
        return cfg.frames_long
    return cfg.frames_xlong


def resize_to_max_pixels(image: Image.Image, max_pixels: int) -> Image.Image:
    width, height = image.size
    if max_pixels <= 0 or width * height <= max_pixels:
        return image
    scale = (max_pixels / float(width * height)) ** 0.5
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def sample_uniform_frames(video_path: Path, cfg: Config) -> List[Frame]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if frame_count <= 0:
        cap.release()
        return []
    duration = frame_count / max(fps, 1e-6)
    indices = np.linspace(0, max(0, frame_count - 1), choose_frame_count(duration, cfg), dtype=int)
    frames: List[Frame] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = resize_to_max_pixels(Image.fromarray(rgb), cfg.max_pixels)
        frames.append(Frame(timestamp=float(idx) / fps, image=image))
    cap.release()
    return frames


def timestamps(frames: Iterable[Frame]) -> str:
    return ", ".join(f"{frame.timestamp:.1f}s" for frame in frames)


def build_evidence_prompt(question: str, frames: List[Frame], voiceover: str) -> str:
    return (
        "You are the evidence-extraction stage of an advertisement video QA pipeline.\n"
        f"Question: {question}\n"
        f"Sampled frame timestamps: {timestamps(frames)}\n\n"
        f"Whisper voiceover transcript:\n{voiceover or '[No intelligible voiceover was detected.]'}\n\n"
        "Produce a compact evidence brief for a second-stage answer writer, not a final answer. Identify the "
        "advertised product or service, visible demonstrations and use cases, concrete visual claims, and spoken "
        "claims in the transcript. Connect each supported feature to a stated or clearly shown consumer benefit "
        "or pain point when applicable. Preserve brand names, product names, numbers, offer terms, and mechanisms "
        "only when visible or spoken. Mark uncertain details as uncertain rather than guessing. Do not use OCR or "
        "invent text, specifications, prices, guarantees, audiences, or benefits."
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


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def generate(processor, model, messages: List[Dict[str, Any]], max_new_tokens: int, cfg: Config) -> str:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = [item["image"] for item in messages[0]["content"] if item.get("type") == "image"]
    processor_kwargs: Dict[str, Any] = {"text": [text], "padding": True, "return_tensors": "pt"}
    if images:
        processor_kwargs["images"] = images
    inputs = processor(**processor_kwargs).to(model.device)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": cfg.temperature > 0,
    }
    if cfg.temperature > 0:
        generation_kwargs.update({"temperature": cfg.temperature, "top_p": cfg.top_p})
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            **generation_kwargs,
        )
    output_ids = output_ids[:, inputs.input_ids.shape[1] :]
    answer = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return strip_thinking(answer)


def load_whisper(cfg: Config):
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper is required for Whisper voiceover transcription.") from exc
    return WhisperModel(
        cfg.whisper_model_size,
        device=cfg.whisper_device,
        compute_type=cfg.whisper_compute_type,
    )


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
    parser.add_argument("--config", default="config/duration_adaptive.json")
    parser.add_argument("--output-file", default="outputs/duration_adaptive_3108_answer_only.jsonl")
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

    for row in tqdm(rows, desc="Duration-Adaptive Evidence"):
        sample_id = str(row["id"])
        if sample_id in done:
            continue
        video_path = Path(args.video_dir) / f"{sample_id}.mp4"
        if not video_path.exists():
            append_jsonl(output_path, {"id": sample_id, "model_prediction": "Video file missing."})
            continue
        frames = sample_uniform_frames(video_path, cfg)
        if not frames:
            append_jsonl(output_path, {"id": sample_id, "model_prediction": "Video could not be decoded."})
            continue
        voiceover = transcribe_voiceover(whisper_model, video_path, cfg)
        evidence = generate(
            processor,
            model,
            build_evidence_messages(str(row["question"]), frames, voiceover),
            cfg.max_new_tokens_evidence,
            cfg,
        )
        answer = generate(
            processor,
            model,
            build_final_messages(str(row["question"]), evidence),
            cfg.max_new_tokens_final,
            cfg,
        )
        append_jsonl(
            output_path,
            {
                "id": sample_id,
                "model_prediction": answer,
            },
        )


if __name__ == "__main__":
    main()
