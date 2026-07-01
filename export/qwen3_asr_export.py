from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor


class Qwen3ASRAudioEncoderTS(nn.Module):
    def __init__(self, audio_tower: nn.Module):
        super().__init__()
        self.audio_tower = audio_tower

    def forward(self, input_features: torch.Tensor, feature_lens: torch.Tensor) -> torch.Tensor:
        output = self.audio_tower(input_features, feature_lens=feature_lens)
        return output.last_hidden_state


class Qwen3ASRTextEmbedTS(nn.Module):
    def __init__(self, text_model: nn.Module):
        super().__init__()
        self.embed_tokens = text_model.embed_tokens

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


class Qwen3ASRTextBackboneTS(nn.Module):
    def __init__(self, text_model: nn.Module):
        super().__init__()
        self.text_model = text_model

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.text_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False)
        return output.last_hidden_state


class Qwen3ASRLmHeadTS(nn.Module):
    def __init__(self, lm_head: nn.Module):
        super().__init__()
        self.lm_head = lm_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


@dataclass
class Qwen3ASRExportBundle:
    root: Path
    processor_dir: Path
    audio_encoder_path: Path
    text_embed_path: Path
    text_backbone_path: Path
    lm_head_path: Path
    manifest_path: Path


def _resolve_dtype(name: Optional[str], default: torch.dtype) -> torch.dtype:
    if not name:
        return default
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = name.strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[key]


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return _to_jsonable(value.to_dict())
    return str(value)


def _token_to_string(token: Any) -> str:
    if isinstance(token, dict):
        return str(token.get("content", ""))
    if token is None:
        return ""
    return str(token)


def _extract_special_tokens(tokenizer_config: Dict[str, Any]) -> Dict[str, Any]:
    special: Dict[str, Any] = {}
    for key in ["bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token"]:
        special[key] = _token_to_string(tokenizer_config.get(key, ""))

    additional = tokenizer_config.get("additional_special_tokens", [])
    if isinstance(additional, list):
        special["additional_special_tokens"] = [_token_to_string(t) for t in additional if _token_to_string(t)]
    else:
        special["additional_special_tokens"] = []
    return special


def _write_bpe_tokenizer_files(processor_dir: Path, out_dir: Path) -> Dict[str, Any]:
    tokenizer_json_path = processor_dir / "tokenizer.json"
    tokenizer_config_path = processor_dir / "tokenizer_config.json"

    if not tokenizer_json_path.exists():
        return {
            "type": "huggingface",
            "processor_dir": processor_dir.name,
            "vocab_file": "",
            "merges_file": "",
            "bos": "",
            "eos": "",
            "pad": "",
            "unk": "",
            "additional_special_tokens": [],
        }

    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)

    model = tokenizer_data.get("model", {})
    vocab = model.get("vocab", {})
    merges = model.get("merges", [])

    vocab_path = out_dir / "vocab.txt"
    with open(vocab_path, "w", encoding="utf-8") as f:
        for token, _ in sorted(vocab.items(), key=lambda item: item[1]):
            f.write(f"{token}\n")

    merges_path = out_dir / "merges.txt"
    with open(merges_path, "w", encoding="utf-8") as f:
        for merge in merges:
            if isinstance(merge, list) and len(merge) == 2:
                f.write(f"{merge[0]} {merge[1]}\n")
            elif isinstance(merge, str):
                f.write(f"{merge}\n")

    tokenizer_config: Dict[str, Any] = {}
    if tokenizer_config_path.exists():
        with open(tokenizer_config_path, "r", encoding="utf-8") as f:
            tokenizer_config = json.load(f)
    special = _extract_special_tokens(tokenizer_config)

    return {
        "type": "bbpe",
        "processor_dir": processor_dir.name,
        "vocab_file": vocab_path.name,
        "merges_file": merges_path.name,
        "bos": special.get("bos_token", ""),
        "eos": special.get("eos_token", ""),
        "pad": special.get("pad_token", ""),
        "unk": special.get("unk_token", ""),
        "additional_special_tokens": special.get("additional_special_tokens", []),
    }


def _device_from_arg(device: Optional[str]) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _register_qwen3_asr_auto_classes() -> None:
    try:
        from qwen_asr.core.transformers_backend import (  # type: ignore
            Qwen3ASRConfig,
            Qwen3ASRForConditionalGeneration,
            Qwen3ASRProcessor,
        )
        from transformers import AutoConfig, AutoModel, AutoProcessor

        AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
        AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
        AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)
    except Exception:
        # Newer model repos may provide native/remote-code registrations.
        pass


def _load_model_and_processor(model_id: str, device: str, dtype: torch.dtype) -> Tuple[Any, Any]:
    _register_qwen3_asr_auto_classes()
    model = AutoModel.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, fix_mistral_regex=True, trust_remote_code=True)
    model = model.to(device).eval()
    return model, processor


def _get_thinker(model: nn.Module) -> nn.Module:
    if hasattr(model, "thinker"):
        return model.thinker
    return model


def _get_text_model(thinker: nn.Module) -> nn.Module:
    if hasattr(thinker, "model"):
        return thinker.model
    raise AttributeError("Could not find Qwen3-ASR text model at thinker.model")


def _get_audio_tower(thinker: nn.Module) -> nn.Module:
    if hasattr(thinker, "audio_tower"):
        return thinker.audio_tower
    raise AttributeError("Could not find Qwen3-ASR audio encoder at thinker.audio_tower")


def _get_lm_head(thinker: nn.Module, model: nn.Module) -> nn.Module:
    if hasattr(thinker, "lm_head"):
        return thinker.lm_head
    if hasattr(model, "lm_head"):
        return model.lm_head
    raise AttributeError("Could not find Qwen3-ASR lm_head")


def _get_thinker_config(model: nn.Module) -> Any:
    cfg = model.config
    return getattr(cfg, "thinker_config", cfg)


def _trace_audio_encoder(
    audio_tower: nn.Module,
    out_path: Path,
    device: str,
    dtype: torch.dtype,
    num_mel_bins: int,
    example_frames: int,
) -> None:
    wrapper = Qwen3ASRAudioEncoderTS(audio_tower).to(device).eval()
    example_input = torch.randn(num_mel_bins, example_frames, device=device, dtype=dtype)
    example_lens = torch.tensor([example_frames], device=device, dtype=torch.long)

    traced = torch.jit.trace(wrapper, (example_input, example_lens), strict=False)
    traced.save(str(out_path))


def _trace_text_embed(
    text_model: nn.Module,
    out_path: Path,
    device: str,
    dtype: torch.dtype,
    vocab_size: int,
) -> None:
    wrapper = Qwen3ASRTextEmbedTS(text_model).to(device).eval()
    example_ids = torch.randint(0, vocab_size, (1, 8), device=device, dtype=torch.long)
    traced = torch.jit.trace(wrapper, example_ids, strict=False)
    traced.save(str(out_path))


def _trace_text_backbone(
    text_model: nn.Module,
    out_path: Path,
    device: str,
    dtype: torch.dtype,
    hidden_size: int,
) -> None:
    wrapper = Qwen3ASRTextBackboneTS(text_model).to(device).eval()
    example_embeds = torch.randn(1, 8, hidden_size, device=device, dtype=dtype)
    example_mask = torch.ones(1, 8, device=device, dtype=torch.long)
    traced = torch.jit.trace(wrapper, (example_embeds, example_mask), strict=False)
    traced.save(str(out_path))


def _trace_lm_head(
    lm_head: nn.Module,
    out_path: Path,
    device: str,
    dtype: torch.dtype,
    hidden_size: int,
) -> None:
    wrapper = Qwen3ASRLmHeadTS(lm_head).to(device).eval()
    example_hidden = torch.randn(1, 8, hidden_size, device=device, dtype=dtype)
    traced = torch.jit.trace(wrapper, example_hidden, strict=False)
    traced.save(str(out_path))


def _run_pnnx(pnnx_bin: str, pt_path: Path, extra_args: Optional[list[str]] = None) -> None:
    cmd = [pnnx_bin, str(pt_path)]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, check=True)


def export_qwen3_asr(
    model_id: str,
    out_dir: str = "./export_qwen3_asr",
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    export_audio_encoder: bool = True,
    export_text_stack: bool = True,
    convert_ncnn: bool = False,
    pnnx_bin: str = "pnnx",
) -> Qwen3ASRExportBundle:
    device = _device_from_arg(device)
    default_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    torch_dtype = _resolve_dtype(dtype, default_dtype)

    model, processor = _load_model_and_processor(model_id, device=device, dtype=torch_dtype)

    root = Path(out_dir)
    processor_dir = root / "processor"
    root.mkdir(parents=True, exist_ok=True)
    processor_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(str(processor_dir))

    thinker = _get_thinker(model)
    text_model = _get_text_model(thinker)
    audio_tower = _get_audio_tower(thinker)
    lm_head = _get_lm_head(thinker, model)
    thinker_cfg = _get_thinker_config(model)
    audio_cfg = thinker_cfg.audio_config
    text_cfg = thinker_cfg.text_config

    audio_encoder_path = root / "audio_encoder.pt"
    text_embed_path = root / "text_embed.pt"
    text_backbone_path = root / "text_backbone.pt"
    lm_head_path = root / "lm_head.pt"

    if export_audio_encoder:
        example_frames = min(
            int(getattr(audio_cfg, "max_source_positions", 256) or 256),
            256,
        )
        _trace_audio_encoder(
            audio_tower,
            audio_encoder_path,
            device=device,
            dtype=torch_dtype,
            num_mel_bins=int(audio_cfg.num_mel_bins),
            example_frames=max(16, example_frames),
        )

    if export_text_stack:
        _trace_text_embed(
            text_model,
            text_embed_path,
            device=device,
            dtype=torch_dtype,
            vocab_size=int(text_cfg.vocab_size),
        )
        _trace_text_backbone(
            text_model,
            text_backbone_path,
            device=device,
            dtype=torch_dtype,
            hidden_size=int(text_cfg.hidden_size),
        )
        _trace_lm_head(
            lm_head,
            lm_head_path,
            device=device,
            dtype=torch_dtype,
            hidden_size=int(text_cfg.hidden_size),
        )

    tokenizer = getattr(processor, "tokenizer", None)
    tokenizer_manifest = _write_bpe_tokenizer_files(processor_dir, root)
    manifest: Dict[str, Any] = {
        "model_type": "qwen3_asr",
        "params": {
            "audio_encoder_pt": audio_encoder_path.name if export_audio_encoder else "",
            "text_embed_pt": text_embed_path.name if export_text_stack else "",
            "text_backbone_pt": text_backbone_path.name if export_text_stack else "",
            "lm_head_pt": lm_head_path.name if export_text_stack else "",
            "processor_dir": processor_dir.name,
        },
        "tokenizer": tokenizer_manifest,
        "setting": {
            "audio_token_id": int(thinker_cfg.audio_token_id),
            "audio_start_token_id": int(thinker_cfg.audio_start_token_id),
            "user_token_id": int(thinker_cfg.user_token_id),
            "audio_config": _to_jsonable(audio_cfg.to_dict() if hasattr(audio_cfg, "to_dict") else audio_cfg.__dict__),
            "text_config": _to_jsonable(text_cfg.to_dict() if hasattr(text_cfg, "to_dict") else text_cfg.__dict__),
            "support_languages": _to_jsonable(getattr(model.config, "support_languages", None)),
        },
    }

    if tokenizer is not None:
        manifest["tokenizer"].update(
            {
                "audio_token": getattr(tokenizer, "audio_token", ""),
                "audio_bos_token": getattr(tokenizer, "audio_bos_token", ""),
                "audio_eos_token": getattr(tokenizer, "audio_eos_token", ""),
            }
        )

    manifest_path = root / "model.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    if convert_ncnn:
        if export_audio_encoder:
            _run_pnnx(
                pnnx_bin,
                audio_encoder_path,
                [
                    f"inputshape=[{int(audio_cfg.num_mel_bins)},{max(16, example_frames)}]",
                    f"inputshape2=[{int(audio_cfg.num_mel_bins)},{max(32, example_frames // 2)}]",
                ],
            )
        if export_text_stack:
            _run_pnnx(pnnx_bin, text_embed_path, ["inputshape=[1,8]"])
            _run_pnnx(pnnx_bin, text_backbone_path, [f"inputshape=[1,8,{int(text_cfg.hidden_size)}],[1,8]"])
            _run_pnnx(pnnx_bin, lm_head_path, [f"inputshape=[1,8,{int(text_cfg.hidden_size)}]"])

    return Qwen3ASRExportBundle(
        root=root,
        processor_dir=processor_dir,
        audio_encoder_path=audio_encoder_path,
        text_embed_path=text_embed_path,
        text_backbone_path=text_backbone_path,
        lm_head_path=lm_head_path,
        manifest_path=manifest_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Qwen3-ASR TorchScript bundles and metadata.")
    parser.add_argument("--model-id", required=True, help="Hugging Face model id or local path.")
    parser.add_argument("--out-dir", default="./export_qwen3_asr", help="Output directory.")
    parser.add_argument("--device", default=None, help="Torch device to run export on.")
    parser.add_argument("--dtype", default=None, help="Override export dtype: fp32, fp16, bf16.")
    parser.add_argument("--no-audio-encoder", action="store_true", help="Skip exporting the audio encoder.")
    parser.add_argument("--no-text-stack", action="store_true", help="Skip exporting the text stack.")
    parser.add_argument("--convert-ncnn", action="store_true", help="Run pnnx on exported TorchScript modules.")
    parser.add_argument("--pnnx-bin", default="pnnx", help="Path to pnnx executable.")
    args = parser.parse_args()

    bundle = export_qwen3_asr(
        model_id=args.model_id,
        out_dir=args.out_dir,
        device=args.device,
        dtype=args.dtype,
        export_audio_encoder=not args.no_audio_encoder,
        export_text_stack=not args.no_text_stack,
        convert_ncnn=args.convert_ncnn,
        pnnx_bin=args.pnnx_bin,
    )

    print(f"Saved manifest: {bundle.manifest_path}")
    if bundle.audio_encoder_path.exists():
        print(f"Saved audio encoder: {bundle.audio_encoder_path}")
    if bundle.text_embed_path.exists():
        print(f"Saved text embed: {bundle.text_embed_path}")
    if bundle.text_backbone_path.exists():
        print(f"Saved text backbone: {bundle.text_backbone_path}")
    if bundle.lm_head_path.exists():
        print(f"Saved lm head: {bundle.lm_head_path}")


if __name__ == "__main__":
    main()
