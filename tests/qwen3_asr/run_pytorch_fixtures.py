#!/usr/bin/env python3
"""Run original PyTorch Qwen3-ASR on fixture audio.

This script fills the PyTorch side of the layered Qwen3-ASR parity contract. It
does not run ncnn. It writes one transcription JSON and one feature summary JSON
per fixture, and can emit a rewritten fixture file that points at those outputs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rewrite_path(path: str, rewrites: list[tuple[str, str]]) -> str:
    out = path
    for src, dst in rewrites:
        if out.startswith(src):
            return dst + out[len(src):]
    return out


def parse_rewrite(values: list[str]) -> list[tuple[str, str]]:
    out = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--path-rewrite must be FROM=TO, got {value!r}")
        src, dst = value.split("=", 1)
        out.append((src.rstrip("/"), dst.rstrip("/")))
    return out


def read_audio(path: str) -> tuple[np.ndarray, int, int]:
    data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    channels = int(data.shape[1])
    mono = data.mean(axis=1)
    return mono.astype(np.float32, copy=False), int(sample_rate), channels


def fixed_overlap_chunks(total_samples: int, frames: int, hop_length: int, overlap_frames: int) -> list[tuple[int, int]]:
    chunk_samples = frames * hop_length
    stride_samples = (frames - overlap_frames) * hop_length
    min_final_tail_samples = chunk_samples // 5
    chunks = []
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunks.append((start, end))
        if end >= total_samples:
            break
        next_start = start + stride_samples
        if next_start + chunk_samples > total_samples:
            if total_samples - end < min_final_tail_samples:
                break
            next_start = max(total_samples - chunk_samples, 0)
        if next_start <= start:
            break
        start = next_start
    return chunks


def tensor_summary(array: np.ndarray, first_value_count: int = 12) -> dict[str, Any]:
    values = np.asarray(array, dtype=np.float32)
    flat = values.reshape(-1)
    if flat.size == 0:
        return {
            "dims": values.ndim,
            "shape": list(values.shape),
            "dtype": "float32",
            "total": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "first_values": [],
        }
    return {
        "dims": values.ndim,
        "shape": list(values.shape),
        "dtype": "float32",
        "total": int(flat.size),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean(dtype=np.float64)),
        "std": float(flat.std(dtype=np.float64)),
        "first_values": [float(x) for x in flat[:first_value_count]],
    }


def feature_summary(
    feature_extractor: Any,
    audio: np.ndarray,
    sample_rate: int,
    source: str,
    input_samples: int,
    channels_after_load: int,
    chunk_index: int,
    chunk_start_sample: int,
    chunk_end_sample: int,
    frames: int,
    hop_length: int,
    n_fft: int,
    mel_bins: int,
    overlap_frames: int,
) -> dict[str, Any]:
    chunk = audio[chunk_start_sample:chunk_end_sample]
    chunk_samples = frames * hop_length
    features = feature_extractor(
        chunk,
        sampling_rate=sample_rate,
        return_tensors="np",
        padding="max_length",
        max_length=chunk_samples,
        truncation=True,
    )["input_features"]
    # WhisperFeatureExtractor returns [batch, mel, frames].
    arr = np.asarray(features[0], dtype=np.float32)
    summary = tensor_summary(arr)
    summary.update({
        "source": source,
        "sample_rate": sample_rate,
        "channels_after_load": channels_after_load,
        "input_samples": input_samples,
        "input_duration_sec": float(input_samples / sample_rate) if sample_rate else 0.0,
        "mel_bins": mel_bins,
        "frames": frames,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "padding": "PyTorch WhisperFeatureExtractor max_length chunk",
        "truncation": "fixed frame count",
        "chunk_overlap_frames": overlap_frames,
        "chunk_index": chunk_index,
        "chunk_start_sample": chunk_start_sample,
        "chunk_end_sample": chunk_end_sample,
        "chunk_samples": chunk_end_sample - chunk_start_sample,
        "h": int(arr.shape[-2]) if arr.ndim >= 2 else 1,
        "w": int(arr.shape[-1]) if arr.ndim >= 1 else int(arr.size),
    })
    return summary


def transcription_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {
        "language": getattr(value, "language", ""),
        "text": getattr(value, "text", ""),
        "time_stamps": getattr(value, "time_stamps", None),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", default="tests/qwen3_asr/fixtures.local.json")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--updated-fixtures")
    parser.add_argument("--path-rewrite", action="append", default=[], help="Rewrite fixture paths, FROM=TO")
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--frames", type=int, default=256)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--n-fft", type=int, default=400)
    parser.add_argument("--mel-bins", type=int, default=128)
    parser.add_argument("--chunk-overlap-frames", type=int, default=32)
    args = parser.parse_args()

    from qwen_asr import Qwen3ASRModel
    import torch

    fixture_doc = load_json(args.fixtures)
    rewrites = parse_rewrite(args.path_rewrite)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    load_start = time.time()
    model = Qwen3ASRModel.from_pretrained(
        args.model,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
    )
    load_time_sec = time.time() - load_start
    feature_extractor = model.processor.feature_extractor

    updated = json.loads(json.dumps(fixture_doc))
    results = []
    for fixture in updated.get("fixtures", []):
        fixture_id = fixture["id"]
        audio_path = rewrite_path(fixture["audio"], rewrites)
        audio, sample_rate, original_channels = read_audio(audio_path)
        if sample_rate != 16000:
            raise ValueError(f"{audio_path}: expected 16000 Hz, got {sample_rate}")

        start = time.time()
        transcription = model.transcribe(audio_path, language=args.language)
        infer_time_sec = time.time() - start
        first = transcription[0] if isinstance(transcription, list) and transcription else transcription
        text = transcription_to_dict(first)
        result = {
            "schema_version": 1,
            "runtime": "pytorch_qwen_asr",
            "model": args.model,
            "audio_wav": audio_path,
            "language": text.get("language") or args.language,
            "text": text.get("text") or "",
            "raw_text": text.get("text") or "",
            "time_stamps": text.get("time_stamps"),
            "total_time_ms": infer_time_sec * 1000.0,
            "audio_samples": int(audio.shape[0]),
            "audio_duration_sec": float(audio.shape[0] / sample_rate),
            "rtf": float(infer_time_sec / (audio.shape[0] / sample_rate)) if audio.shape[0] else None,
            "sample_rate": sample_rate,
            "original_channels": original_channels,
            "channels_after_load": 1,
            "load_time_sec": load_time_sec,
            "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else None,
            "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        }

        chunks = fixed_overlap_chunks(
            int(audio.shape[0]), args.frames, args.hop_length, args.chunk_overlap_frames
        )
        mel = [
            feature_summary(
                feature_extractor,
                audio,
                sample_rate,
                audio_path,
                int(audio.shape[0]),
                1,
                i,
                start_sample,
                end_sample,
                args.frames,
                args.hop_length,
                args.n_fft,
                args.mel_bins,
                args.chunk_overlap_frames,
            )
            for i, (start_sample, end_sample) in enumerate(chunks)
        ]

        result_path = out_dir / f"{fixture_id}_pytorch.json"
        mel_path = out_dir / f"{fixture_id}_pytorch_mel.json"
        write_json(result_path, result)
        write_json(mel_path, mel)
        fixture["audio"] = audio_path
        fixture["pytorch_result_json"] = str(result_path)
        fixture["pytorch_mel_summary_json"] = str(mel_path)
        results.append({
            "id": fixture_id,
            "text": result["text"],
            "rtf": result["rtf"],
            "mel_shape_first": [mel[0].get("h"), mel[0].get("w")] if mel else None,
            "chunks": len(chunks),
        })

    if args.updated_fixtures:
        write_json(args.updated_fixtures, updated)

    print(json.dumps({"fixtures": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
