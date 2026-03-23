"""Benchmark runner with inference timing."""

import gc
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import psutil
import torch

from ..config import GENERAL_PROMPT, GUIDANCE_SCALE, SEED
from ..models import LoadedModel

SD15_MODEL_TYPES = {"base_memory", "dobby_memory"}

SDXL_COLUMNS = ["prompt_idx", "prompt", "base_model_key", "model_name", "model_type", "image_path", "inference_time"]
SD15_COLUMNS = [
    "prompt_idx",
    "prompt",
    "base_model_key",
    "model_name",
    "model_type",
    "image_path",
    "gpu_peak_memory_mb",
    "ram_peak_mb",
]

SD15_COLUMN_MAP = {
    "gpu_peak_memory_mb": "peak_memory_mb",
    "ram_peak_mb": "peak_ram_mb",
}


@dataclass
class InferenceResult:
    """Container for inference results."""

    prompt_idx: int
    prompt: str
    base_model_key: str
    model_name: str
    model_type: str
    image_path: str
    model_load_time: float
    inference_time: float
    peak_memory_mb: Optional[float] = None
    peak_ram_mb: Optional[float] = None


def _monitor_system_resources(
    stop_event: threading.Event,
    cpu_samples: list,
    ram_samples: list,
    gpu_samples: list,
    interval: float = 0.1,
) -> None:
    """
    Thread function to periodically sample CPU, RAM, and GPU usage during inference.

    Args:
        stop_event: Event to signal monitoring thread to stop
        cpu_samples: List to append CPU percent samples to
        ram_samples: List to append RAM usage (MB) samples to
        gpu_samples: List to append GPU memory allocated (MB) samples to
        interval: Sampling interval in seconds
    """
    process = psutil.Process()
    psutil.cpu_percent(interval=None)  # 첫 번째 읽기 버리기 (부정확)
    while not stop_event.is_set():
        cpu_samples.append(psutil.cpu_percent(interval=None))
        ram_samples.append(process.memory_info().rss / (1024 * 1024))
        if torch.cuda.is_available():
            gpu_samples.append(torch.cuda.memory_allocated() / (1024 * 1024))
        stop_event.wait(timeout=interval)


def _select_columns_for_model_type(row: dict, model_type: str) -> dict:
    """Return only the relevant columns for the given model type."""
    if model_type in SD15_MODEL_TYPES:
        result = {}
        for col in SD15_COLUMNS:
            src_key = SD15_COLUMN_MAP.get(col, col)
            if src_key in row:
                result[col] = row[src_key]
        return result
    return {k: v for k, v in row.items() if k in SDXL_COLUMNS}


class BenchmarkRunner:
    """Runs inference benchmarks and records timing."""

    def __init__(self, output_dir: str):
        """
        Initialize benchmark runner.

        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[InferenceResult] = []

    def run_inference(
        self,
        loaded_model: LoadedModel,
        prompt: str,
        num_inference_steps: int,
        prompt_idx: int,
        seed: int = SEED,
        guidance_scale: float = GUIDANCE_SCALE,
    ) -> InferenceResult:
        """
        Run single inference and measure time and memory.

        Args:
            loaded_model: Loaded model to use for inference
            prompt: Text prompt for generation
            num_inference_steps: Number of denoising steps
            prompt_idx: Index of the prompt
            seed: Random seed
            guidance_scale: Guidance scale for generation

        Returns:
            InferenceResult with timing, memory, and metadata
        """
        generator = torch.manual_seed(seed)
        full_prompt = f"{GENERAL_PROMPT}, {prompt}"

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        # 모델 로드 잔류 객체 정리 후 측정 시작
        gc.collect()

        cpu_samples: list[float] = []
        ram_samples: list[float] = []
        gpu_samples: list[float] = []
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=_monitor_system_resources,
            args=(stop_event, cpu_samples, ram_samples, gpu_samples),
            daemon=True,
        )
        monitor_thread.start()

        start_time = time.perf_counter()
        with torch.inference_mode():
            image = loaded_model.pipe(
                prompt=full_prompt,
                negative_prompt="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
                num_inference_steps=num_inference_steps,
                generator=generator,
                guidance_scale=guidance_scale,
            ).images[0]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_time = time.perf_counter() - start_time

        stop_event.set()
        monitor_thread.join()

        peak_ram_mb = max(ram_samples) if ram_samples else None
        peak_gpu_mb = max(gpu_samples) if gpu_samples else None

        if inference_time < 0:
            print(f"WARNING: Negative inference time detected: {inference_time}")
            inference_time = 0.0

        image_filename = f"{prompt_idx:02d}_{loaded_model.model_name}.png"
        image_path = self.output_dir / image_filename
        image.save(image_path)

        torch.cuda.empty_cache()

        result = InferenceResult(
            prompt_idx=prompt_idx,
            prompt=prompt,
            base_model_key=loaded_model.base_model_key,
            model_name=loaded_model.model_name,
            model_type=loaded_model.model_type,
            image_path=str(image_path),
            model_load_time=loaded_model.load_time,
            inference_time=inference_time,
            peak_memory_mb=peak_gpu_mb,
            peak_ram_mb=peak_ram_mb,
        )

        self.results.append(result)
        return result

    def save_results(self, filename: str = "benchmark_results.csv") -> pd.DataFrame:
        """
        Save all results to CSV, keeping only relevant columns per model type.

        Args:
            filename: Output CSV filename

        Returns:
            DataFrame containing all results (full, for visualization use)
        """
        full_df = pd.DataFrame([vars(r) for r in self.results])

        filtered_rows = [_select_columns_for_model_type(vars(r), r.model_type) for r in self.results]
        filtered_df = pd.DataFrame(filtered_rows)
        csv_path = self.output_dir / filename
        filtered_df.to_csv(csv_path, index=False)
        print(f"결과 저장 완료: {csv_path}")

        return full_df

    def save_result(self, result: InferenceResult, filename: Optional[str] = None) -> None:
        """
        Save a single inference result, keeping only columns relevant to its model type.

        Args:
            result: InferenceResult to save
            filename: Optional filename for the CSV file
        """
        row = _select_columns_for_model_type(vars(result), result.model_type)
        df = pd.DataFrame([row])
        if filename is None:
            filename = f"result_{result.prompt_idx:02d}_{result.model_name}.csv"
        csv_path = self.output_dir / filename
        df.to_csv(csv_path, index=False)
        print(f"단일 결과 저장 완료: {csv_path}")

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get full results as DataFrame without saving.

        Returns:
            DataFrame containing all results with all columns
        """
        return pd.DataFrame([vars(r) for r in self.results])
