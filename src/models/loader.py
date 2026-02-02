"""Model loader with timing measurement."""

import time
from dataclasses import dataclass

import torch
from diffusers import LCMScheduler, StableDiffusionXLPipeline


@dataclass
class LoadedModel:
    """Container for loaded model with metadata."""

    pipe: StableDiffusionXLPipeline
    model_name: str
    model_type: str  # "teacher" or "lcm"
    base_model_key: str
    load_time: float


class ModelLoader:
    """Handles model loading with timing measurement."""

    @staticmethod
    def load_teacher_model(base_model_key: str, base_model_path: str) -> LoadedModel:
        """
        Load teacher model (original SDXL).

        Args:
            base_model_key: Key identifier for the base model (e.g., "base", "animagine")
            base_model_path: HuggingFace model path or local path

        Returns:
            LoadedModel containing the pipeline and metadata
        """
        start_time = time.perf_counter()

        pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
        ).to("cuda")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        load_time = time.perf_counter() - start_time

        return LoadedModel(
            pipe=pipe,
            model_name=f"{base_model_key}_base",
            model_type="teacher",
            base_model_key=base_model_key,
            load_time=load_time,
        )

    @staticmethod
    def load_lcm_model(
        base_model_key: str,
        lcm_checkpoint_path: str,
    ) -> LoadedModel:
        """
        Load LCM fine-tuned model with optimizations for faster inference.

        Applies several optimization techniques:
        - channels_last memory format for faster convolutions
        - VAE tiling and slicing for memory efficiency
        - QKV projection fusion for attention speedup
        - torch.compile for JIT compilation

        Args:
            base_model_key: Key identifier for the base model
            lcm_checkpoint_path: Local path to LCM checkpoint

        Returns:
            LoadedModel containing the optimized pipeline and metadata
        """
        start_time = time.perf_counter()

        pipe = StableDiffusionXLPipeline.from_pretrained(
            lcm_checkpoint_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to("cuda")

        # Set LCM scheduler
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

        # ===== LCM Optimizations =====
        # 1. Enable channels_last memory format for faster convolutions
        pipe.unet.to(memory_format=torch.channels_last)
        pipe.vae.to(memory_format=torch.channels_last)

        # 2. Enable VAE tiling for memory efficiency
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()

        # 3. Fuse QKV projections for attention speedup
        try:
            pipe.fuse_qkv_projections()
        except (AttributeError, NotImplementedError):
            # Some pipelines may not support QKV fusion
            pass

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        load_time = time.perf_counter() - start_time

        return LoadedModel(
            pipe=pipe,
            model_name=f"{base_model_key}_dobby_model",
            model_type="lcm",
            base_model_key=base_model_key,
            load_time=load_time,
        )

    @staticmethod
    def unload_model(loaded_model: LoadedModel) -> None:
        """
        Unload model and clear GPU memory.

        Args:
            loaded_model: Model to unload
        """
        del loaded_model.pipe
        torch.cuda.empty_cache()
