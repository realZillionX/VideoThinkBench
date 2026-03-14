from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path
from typing import Optional


def ensure_diffsynth_path(diffsynth_path: Optional[str]) -> Path:
    resolved = diffsynth_path or os.environ.get("DIFFSYNTH_PATH")
    if not resolved:
        raise RuntimeError("DIFFSYNTH_PATH is required. Pass --diffsynth-path or set DIFFSYNTH_PATH.")
    path = Path(resolved).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"DiffSynth-Studio path not found: {path}")
    if path.as_posix() not in sys.path:
        sys.path.insert(0, path.as_posix())
    return path


def load_qwen_image_pipeline(model_path: str, lora_path: Optional[str] = None):
    import torch
    from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

    model_configs = [
        ModelConfig(model_id="Qwen/Qwen-Image-Edit-2511", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ]
    processor_config = ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/")

    if model_path and Path(model_path).exists():
        model_base = Path(model_path)
        model_configs = [
            ModelConfig(model_path=(model_base / "Qwen/Qwen-Image-Edit-2511").as_posix(), origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_path=(model_base / "Qwen/Qwen-Image").as_posix(), origin_file_pattern="text_encoder/model*.safetensors"),
            ModelConfig(model_path=(model_base / "Qwen/Qwen-Image").as_posix(), origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ]
        processor_config = ModelConfig(model_path=(model_base / "Qwen/Qwen-Image-Edit").as_posix(), origin_file_pattern="processor/")

    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        processor_config=processor_config,
    )
    if lora_path:
        pipe.load_lora(pipe.dit, lora_path)
    return pipe


_WAN_CKPT_PATTERN = re.compile(r"^(epoch|step)-(\d+)\.safetensors$")


def _resolve_wan_model_family(model_base: Path) -> str:
    if (model_base / "high_noise_model").is_dir() and (model_base / "low_noise_model").is_dir():
        return "wan2.2-i2v-a14b"
    if glob.glob((model_base / "diffusion_pytorch_model*.safetensors").as_posix()):
        return "wan2.2-ti2v-5b"
    raise FileNotFoundError(f"Unable to detect Wan model family under: {model_base}")


def _pick_latest_wan_checkpoint(directory: Path) -> Optional[Path]:
    if not directory.is_dir():
        return None
    candidates = []
    for path in directory.glob("*.safetensors"):
        match = _WAN_CKPT_PATTERN.match(path.name)
        if match is None:
            continue
        kind = match.group(1)
        index = int(match.group(2))
        tag = path.stem
        has_state = (directory / f"training_state_{tag}").is_dir()
        candidates.append((int(has_state), index, 1 if kind == "step" else 0, path.stat().st_mtime, path))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][-1]


def _resolve_wan_lora_refs(lora_ckpt: Optional[str], family: str) -> dict[str, Optional[Path]]:
    if not lora_ckpt:
        return {"single": None, "high_noise": None, "low_noise": None}

    ref = Path(lora_ckpt).expanduser().resolve()
    if family == "wan2.2-ti2v-5b":
        if ref.is_file():
            return {"single": ref, "high_noise": None, "low_noise": None}
        latest = _pick_latest_wan_checkpoint(ref)
        return {"single": latest, "high_noise": None, "low_noise": None}

    if ref.is_file():
        raise ValueError(
            "Wan2.2-I2V-A14B inference expects --lora to point to a directory containing "
            "high_noise/ and low_noise/ checkpoints."
        )

    high_noise = _pick_latest_wan_checkpoint(ref / "high_noise")
    low_noise = _pick_latest_wan_checkpoint(ref / "low_noise")
    if (high_noise is None) != (low_noise is None):
        raise ValueError(
            "Wan2.2-I2V-A14B LoRA directory must contain both high_noise and low_noise checkpoints."
        )
    return {"single": None, "high_noise": high_noise, "low_noise": low_noise}


def _first_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"None of the expected files exist: {', '.join(item.as_posix() for item in candidates)}")


def load_wan_video_pipeline(model_base_path: str, lora_ckpt: Optional[str] = None, device: str = "cuda"):
    import torch
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

    model_base = Path(model_base_path).expanduser().resolve()
    if not model_base.exists():
        raise FileNotFoundError(f"Wan model base path not found: {model_base}")

    family = _resolve_wan_model_family(model_base)
    tokenizer_path = model_base / "google/umt5-xxl"
    lora_refs = _resolve_wan_lora_refs(lora_ckpt, family)

    if family == "wan2.2-ti2v-5b":
        dit_files = sorted(glob.glob((model_base / "diffusion_pytorch_model*.safetensors").as_posix()))
        if not dit_files:
            raise FileNotFoundError(f"No diffusion checkpoints found under: {model_base}")
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=[
                ModelConfig(path=(model_base / "models_t5_umt5-xxl-enc-bf16.pth").as_posix()),
                ModelConfig(path=dit_files),
                ModelConfig(path=(model_base / "Wan2.2_VAE.pth").as_posix()),
            ],
            tokenizer_config=ModelConfig(path=tokenizer_path.as_posix()),
            audio_processor_config=None,
        )
        if lora_refs["single"] is not None:
            pipe.load_lora(pipe.dit, lora_refs["single"].as_posix(), alpha=1.0)
        return pipe

    high_noise_files = sorted(glob.glob((model_base / "high_noise_model" / "diffusion_pytorch_model*.safetensors").as_posix()))
    low_noise_files = sorted(glob.glob((model_base / "low_noise_model" / "diffusion_pytorch_model*.safetensors").as_posix()))
    if not high_noise_files or not low_noise_files:
        raise FileNotFoundError(f"Wan2.2-I2V-A14B checkpoints are incomplete under: {model_base}")

    vae_path = _first_existing_path(model_base / "Wan2.1_VAE.pth", model_base / "Wan2.2_VAE.pth")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(path=high_noise_files),
            ModelConfig(path=low_noise_files),
            ModelConfig(path=(model_base / "models_t5_umt5-xxl-enc-bf16.pth").as_posix()),
            ModelConfig(path=vae_path.as_posix()),
        ],
        tokenizer_config=ModelConfig(path=tokenizer_path.as_posix()),
        audio_processor_config=None,
    )
    if lora_refs["high_noise"] is not None and lora_refs["low_noise"] is not None:
        pipe.load_lora(pipe.dit, lora_refs["high_noise"].as_posix(), alpha=1.0)
        pipe.load_lora(pipe.dit2, lora_refs["low_noise"].as_posix(), alpha=1.0)
    return pipe
