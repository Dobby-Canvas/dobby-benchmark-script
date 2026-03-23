"""Model loader with timing measurement."""

import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import gguf
import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    LCMScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from huggingface_hub import hf_hub_download
from transformers import CLIPTextModel, CLIPTokenizer

from .sd15_pipe import MixDQ_SD15_Pipeline_W8A8

UNET_PREFIX_CANDIDATES = ("model.diffusion_model.", "model.", "unet.")
TORCH_COMPATIBLE_QTYPES = {
    gguf.GGMLQuantizationType.F16,
    gguf.GGMLQuantizationType.F32,
    gguf.GGMLQuantizationType.BF16,
}


def _read_original_shape(reader: gguf.GGUFReader, tensor_name: str) -> Optional[torch.Size]:
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    return torch.Size(tuple(int(field.parts[index][0]) for index in field.data))


def _detect_unet_prefix(tensor_names: Iterable[str]) -> Optional[str]:
    tensor_name_set = set(tensor_names)
    for prefix in UNET_PREFIX_CANDIDATES:
        if any(name.startswith(prefix) for name in tensor_name_set):
            return prefix
    return None


def _dequantize_gguf_tensor(raw_tensor, target_shape: torch.Size, target_dtype: torch.dtype) -> torch.Tensor:
    tensor_type = raw_tensor.tensor_type
    if tensor_type in TORCH_COMPATIBLE_QTYPES:
        source_tensor = torch.from_numpy(raw_tensor.data.copy())
        return source_tensor.view(*target_shape).to(dtype=target_dtype)

    dequantized = gguf.quants.dequantize(raw_tensor.data, tensor_type)
    return torch.from_numpy(dequantized.copy()).view(*target_shape).to(dtype=target_dtype)


def _load_unet_state_dict_from_gguf(gguf_path: Path, target_dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    reader = gguf.GGUFReader(str(gguf_path))
    prefix = _detect_unet_prefix(tensor.name for tensor in reader.tensors)

    state_dict: Dict[str, torch.Tensor] = {}
    for tensor in reader.tensors:
        original_name = tensor.name
        if prefix and not original_name.startswith(prefix):
            continue
        key_name = original_name[len(prefix) :] if prefix else original_name

        original_shape = _read_original_shape(reader, original_name)
        if original_shape is None:
            original_shape = torch.Size(tuple(int(value) for value in reversed(tensor.shape)))

        state_dict[key_name] = _dequantize_gguf_tensor(
            raw_tensor=tensor,
            target_shape=original_shape,
            target_dtype=target_dtype,
        )
    return state_dict


def _resolve_asset_file_path(path_or_filename: str, hf_repo_id: Optional[str]) -> Path:
    local_path = Path(path_or_filename)
    if local_path.is_file():
        return local_path

    if hf_repo_id:
        downloaded_path = hf_hub_download(repo_id=hf_repo_id, filename=path_or_filename)
        return Path(downloaded_path)

    raise FileNotFoundError(
        f"Asset file not found: {path_or_filename}. " "If this path is in Hugging Face repo, set hf_asset_repo_id."
    )


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
            safety_checker=None,
            requires_safety_checker=False,
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

        pipe = MixDQ_SD15_Pipeline_W8A8.from_pretrained(base_model_path, torch_dtype=torch.float16).to("cuda")

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

    @staticmethod
    def load_dobby_ram_gguf_model(
        base_model_key: str,
        base_model_path: str,
        gguf_unet_path: str,
        unet_config_dir: str,
        hf_asset_repo_id: Optional[str] = None,
    ) -> LoadedModel:
        start_time = time.perf_counter()

        dtype = torch.float16
        local_unet_config_dir = Path(unet_config_dir)
        if local_unet_config_dir.is_dir():
            unet_config = UNet2DConditionModel.load_config(str(local_unet_config_dir))
        elif hf_asset_repo_id:
            unet_config = UNet2DConditionModel.load_config(hf_asset_repo_id, subfolder=unet_config_dir)
        else:
            raise FileNotFoundError(
                f"UNet config directory not found: {unet_config_dir}. "
                "If this directory is in Hugging Face repo, set hf_asset_repo_id."
            )

        unet = UNet2DConditionModel.from_config(unet_config).to(dtype=dtype)
        resolved_gguf_path = _resolve_asset_file_path(gguf_unet_path, hf_asset_repo_id)
        unet_state_dict = _load_unet_state_dict_from_gguf(resolved_gguf_path, target_dtype=dtype)
        unet.load_state_dict(unet_state_dict, strict=False)

        # Release temporary CPU tensors as early as possible for RAM benchmark.
        del unet_state_dict
        gc.collect()

        tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(
            base_model_path,
            subfolder="text_encoder",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        vae = AutoencoderKL.from_pretrained(
            base_model_path,
            subfolder="vae",
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        scheduler = DDIMScheduler.from_pretrained(base_model_path, subfolder="scheduler")

        pipe = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        ).to("cuda")

        pipe.enable_attention_slicing()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

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
