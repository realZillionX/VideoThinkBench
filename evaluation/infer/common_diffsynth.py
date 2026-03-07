from __future__ import annotations

import glob
import os
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


def load_wan_video_pipeline(model_base_path: str, lora_ckpt: Optional[str] = None, device: str = "cuda"):
    import torch
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

    model_base = Path(model_base_path).expanduser().resolve()
    if not model_base.exists():
        raise FileNotFoundError(f"Wan model base path not found: {model_base}")

    dit_files = sorted(glob.glob((model_base / "diffusion_pytorch_model*.safetensors").as_posix()))
    if not dit_files:
        raise FileNotFoundError(f"No diffusion checkpoints found under: {model_base}")

    tokenizer_path = model_base / "google/umt5-xxl"
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
    if lora_ckpt:
        pipe.load_lora(pipe.dit, lora_ckpt, alpha=1.0)
    return pipe
