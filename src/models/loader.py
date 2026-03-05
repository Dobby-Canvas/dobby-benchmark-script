"""Model loader with timing measurement."""

import time
from dataclasses import dataclass

import torch
from diffusers import (DPMSolverMultistepScheduler, LCMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline,
                       UNet2DConditionModel)

from .sd15_pipe import MixDQ_SD15_Pipeline_W8A8


@dataclass
class LoadedModel:
    """Container for loaded model with metadata."""

    pipe: StableDiffusionXLPipeline
    model_name: str
    model_type: str  # "teacher" or "lcm"
    base_model_key: str
    load_time: float
    model_memory_mb: float = 0.0


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
            model_type="base_speed",
            base_model_key=base_model_key,
            load_time=load_time,
        )

    @staticmethod
    def load_lcm_model(
        base_model_key: str,
        lcm_checkpoint_path: str,
        base_model_path: str,
    ) -> LoadedModel:
        """
        Load LCM fine-tuned model with optimizations for faster inference.

        Dobby 모델은 UNet만 파인튜닝된 체크포인트이므로, 베이스 모델에서
        전체 파이프라인을 로드한 뒤 UNet만 Dobby 체크포인트로 교체합니다.

        Applies several optimization techniques:
        - channels_last memory format for faster convolutions
        - VAE tiling and slicing for memory efficiency
        - QKV projection fusion for attention speedup

        Args:
            base_model_key: Key identifier for the base model
            lcm_checkpoint_path: HuggingFace model ID or local path to Dobby UNet checkpoint
            base_model_path: HuggingFace model ID or local path to the base SDXL model

        Returns:
            LoadedModel containing the optimized pipeline and metadata
        """
        start_time = time.perf_counter()

        # Dobby 모델은 UNet 전용 체크포인트이므로 베이스 파이프라인을 먼저 로드
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

        # UNet만 Dobby 체크포인트로 교체
        pipe.unet = UNet2DConditionModel.from_pretrained(
            lcm_checkpoint_path,
            subfolder="unet",
            torch_dtype=torch.float16,
            use_safetensors=True,
        )

        pipe = pipe.to("cuda")

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
            model_type="dobby_speed",
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

    @staticmethod
    def load_base_memory_model(base_model_key: str, base_model_path: str) -> LoadedModel:

        start_time = time.perf_counter()

        pipe = StableDiffusionPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
        ).to("cuda")

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        load_time = time.perf_counter() - start_time
        model_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

        return LoadedModel(
            pipe=pipe,
            model_name=f"{base_model_key}_base",
            model_type="base_memory",
            base_model_key=base_model_key,
            load_time=load_time,
            model_memory_mb=model_memory_mb,
        )

    @staticmethod
    def load_dobby_memory_model(base_model_key: str, base_model_path: str, quant_path: str) -> LoadedModel:

        start_time = time.perf_counter()

        pipe = MixDQ_SD15_Pipeline_W8A8.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
        ).to("cuda")

        pipe.unet = UNet2DConditionModel.from_pretrained(
            quant_path,
            subfolder="quantization/unet",
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to("cuda")

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
        )

        # Quantize UNet (W8A8).
        # bos=False: BOS optimization requires a separately pre-computed tensor
        # (bos_pre_computed.pt) that is distinct from ckpt.pth and does not exist
        # for SD1.5 yet.
        pipe.quantize_unet(
            ckpt_path=quant_path,
            w_bit=8,
            a_bit=8,
            bos=False,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        load_time = time.perf_counter() - start_time
        model_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

        return LoadedModel(
            pipe=pipe,
            model_name=f"{base_model_key}_quantized",
            model_type="dobby_memory",
            base_model_key=base_model_key,
            load_time=load_time,
            model_memory_mb=model_memory_mb,
        )
