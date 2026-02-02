"""Benchmark runner with inference timing."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image

from ..config import GENERAL_PROMPT, GUIDANCE_SCALE, SEED
from ..models import LoadedModel


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
        Run single inference and measure time.

        Args:
            loaded_model: Loaded model to use for inference
            prompt: Text prompt for generation
            num_inference_steps: Number of denoising steps
            prompt_idx: Index of the prompt
            seed: Random seed
            guidance_scale: Guidance scale for generation

        Returns:
            InferenceResult with timing and metadata
        """
        generator = torch.manual_seed(seed)
        full_prompt = f"{GENERAL_PROMPT}, {prompt}"

        # CUDA 동기화로 정확한 시간 측정
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        image = loaded_model.pipe(
            prompt=full_prompt,
            num_inference_steps=num_inference_steps,
            generator=generator,
            guidance_scale=guidance_scale,
        ).images[0]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_time = time.perf_counter() - start_time

        # 음수 시간 방지 (디버깅용)
        if inference_time < 0:
            print(f"WARNING: Negative inference time detected: {inference_time}")
            inference_time = 0.0

        # Save image
        image_filename = f"{prompt_idx:02d}_{loaded_model.model_name}.png"
        image_path = self.output_dir / image_filename
        image.save(image_path)

        # Clear cache
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
        )

        self.results.append(result)
        return result

    def save_results(self, filename: str = "benchmark_results.csv") -> pd.DataFrame:
        """
        Save all results to CSV.

        Args:
            filename: Output CSV filename

        Returns:
            DataFrame containing all results
        """
        df = pd.DataFrame([vars(r) for r in self.results])
        csv_path = self.output_dir / filename
        df.to_csv(csv_path, index=False)
        print(f"결과 저장 완료: {csv_path}")
        return df

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get results as DataFrame without saving.

        Returns:
            DataFrame containing all results
        """
        return pd.DataFrame([vars(r) for r in self.results])
