# W-MEAF: Advertisement Video QA

W-MEAF (Whisper-Guided Multi-Path Evidence and Answer Fusion) is a test-time reasoning pipeline for advertisement video question answering. It uses Qwen2.5-VL-7B-Instruct as the multimodal backbone and faster-whisper base for voice-over transcription. The submitted score reported in the technical note is `0.6495`.

## 1. Environment

Recommended environment:

- Linux with CUDA-capable GPU
- Python 3.10+
- PyTorch with CUDA support
- `transformers`
- `accelerate`
- `opencv-python`
- `Pillow`
- `numpy`
- `tqdm`
- `faster-whisper`

Install example:

```bash
pip install torch transformers accelerate opencv-python pillow numpy tqdm faster-whisper
```

For GPU memory stability, all run scripts set:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 2. Model Weights

### Multimodal backbone

Use Qwen2.5-VL-7B-Instruct:

- Hugging Face: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- ModelScope: https://modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct

Default expected local path:

```text
weights/pretrained/Qwen2.5-VL-7B-Instruct
```

You can override it at runtime:

```bash
MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct bash scripts/run_dense_keyframe.sh
```

### ASR model

Use faster-whisper base:

- Hugging Face: https://huggingface.co/Systran/faster-whisper-base

The config files use `Systran/faster-whisper-base` by default. You may change `whisper_model_size` to a local faster-whisper checkpoint path to avoid downloading at runtime.

## 3. Data Layout

Expected project layout:

```text
W-MEAF/
  MAC_QA.jsonl
  data/
    <sample_id>.mp4
  weights/
    pretrained/
      Qwen2.5-VL-7B-Instruct/
  config/
  src/
  scripts/
```

`MAC_QA.jsonl` format:

```json
{"id": "sample-id", "question": "question text"}
```

Each video is loaded from:

```text
data/<id>.mp4
```

## 4. File Mapping

| Method component | Python file | Config file | Run script |
| --- | --- | --- | --- |
| Dense Keyframe Evidence | `src/dense_keyframe_infer.py` | `config/dense_keyframe.json` | `scripts/run_dense_keyframe.sh` |
| Reranked Candidate Selection | `src/reranked_candidate_infer.py` | `config/reranked_candidate.json` | `scripts/run_reranked_candidate.sh` |
| Duration-Adaptive Evidence | `src/duration_adaptive_infer.py` | `config/duration_adaptive.json` | `scripts/run_duration_adaptive.sh` |
| Conservative Answer Fusion | `src/conservative_answer_fusion.py` | N/A | `scripts/run_conservative_fusion.sh` |

## 5. Configuration

Main config fields:

- `dtype`: model dtype, default `auto`
- `device_map`: Hugging Face device mapping, default `auto`
- `temperature`, `top_p`: decoding parameters
- `max_new_tokens_evidence`: evidence-brief generation budget
- `max_new_tokens_final`: final-answer generation budget
- `max_pixels`: maximum pixels per input frame
- `max_frames` / `frames_short` / `frames_medium` / `frames_long` / `frames_xlong`: frame count controls
- `whisper_model_size`: local faster-whisper model path or model name
- `whisper_device`: usually `cpu`
- `whisper_compute_type`: usually `int8`
- `voiceover_max_chars`: transcript truncation length

Before running, optionally edit the three files in `config/` so `whisper_model_size` points to your local faster-whisper model.

## 6. Reproduction Steps

Run from the repository root.

### Step 1: Dense Keyframe Evidence

```bash
bash scripts/run_dense_keyframe.sh
```

Default output:

```text
outputs/dense_keyframe_3108_answer_only.jsonl
```

### Step 2: Reranked Candidate Selection

```bash
bash scripts/run_reranked_candidate.sh
```

Default output:

```text
outputs/reranked_candidate_3108_answer_only.jsonl
```

### Step 3: Duration-Adaptive Evidence

```bash
bash scripts/run_duration_adaptive.sh
```

Default output:

```text
outputs/duration_adaptive_3108_answer_only.jsonl
```

### Step 4: Conservative Answer Fusion

The fusion script expects the three branch output files. Override paths if needed:

```bash
RERANKED_FILE=outputs/reranked_candidate_3108_answer_only.jsonl \
DENSE_FILE=outputs/dense_keyframe_3108_answer_only.jsonl \
ADAPTIVE_FILE=outputs/duration_adaptive_3108_answer_only.jsonl \
bash scripts/run_conservative_fusion.sh
```

Default output:

```text
outputs/conservative_fusion_3108_answer_only.jsonl
```

## 7. Useful Overrides

```bash
MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct \
VIDEO_DIR=/path/to/videos \
QA_FILE=/path/to/MAC_QA.jsonl \
OUTPUT_FILE=/path/to/output.jsonl \
bash scripts/run_reranked_candidate.sh
```

## 8. Notes for GitHub

Do not commit large/private files:

- video data: `data/`
- model weights: `weights/`
- generated predictions: `outputs/`
- full JSONL submissions if they are private

The `.gitignore` file excludes these by default.
