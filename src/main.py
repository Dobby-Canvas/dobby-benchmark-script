"""Main execution script for SDXL and SD1.5 benchmarks."""

from typing import Callable

import pandas as pd
import torch

from .benchmark import BenchmarkRunner
from .config import (BASE_MODELS, LCM_CHECKPOINT_PATHS, LCM_STEPS, PROMPTS, SD15_MODELS, SD15_QUANT_CKPT_PATHS,
                     TEACHER_STEPS)
from .models import LoadedModel, ModelLoader
from .visualization import ResultPlotter

OUTPUT_DIR = "results/"
SECTION_WIDTH = 80


def _configure_torch_inductor() -> None:
    torch._inductor.config.conv_1x1_as_mm = True
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.epilogue_fusion = False
    torch._inductor.config.coordinate_descent_check_all_directions = True


def _print_section_header(title: str) -> None:
    print(f"\n{'=' * SECTION_WIDTH}")
    print(title)
    print(f"{'=' * SECTION_WIDTH}\n")


def _run_sdxl_model_benchmark(
    runner: BenchmarkRunner,
    model_display_name: str,
    load_fn: Callable[[], LoadedModel],
    num_steps: int,
) -> None:
    """Iterate over all prompts for one SDXL model variant, reporting inference time."""
    for idx, prompt in enumerate(PROMPTS, start=1):
        print(f"  [{idx}/{len(PROMPTS)}] Prompt: {prompt[:50]}...")
        print(f"    Loading model: {model_display_name}")

        loaded_model = load_fn()
        print("    ✓ Model loaded")

        result = runner.run_inference(
            loaded_model=loaded_model,
            prompt=prompt,
            num_inference_steps=num_steps,
            prompt_idx=idx,
        )
        print(f"    ✓ Generation completed (inference time: {result.inference_time:.2f}s)")

        runner.save_result(result)
        print("    ✓ Result saved")

        ModelLoader.unload_model(loaded_model)


def _run_sd15_model_benchmark(
    runner: BenchmarkRunner,
    model_display_name: str,
    load_fn: Callable[[], LoadedModel],
    num_steps: int,
) -> None:
    """Iterate over all prompts for one SD1.5 model variant, reporting memory usage."""
    for idx, prompt in enumerate(PROMPTS, start=1):
        print(f"  [{idx}/{len(PROMPTS)}] Prompt: {prompt[:50]}...")
        print(f"    Loading model: {model_display_name}")

        loaded_model = load_fn()
        print(f"    ✓ Model loaded (model memory: {loaded_model.model_memory_mb:.2f}MB)")

        result = runner.run_inference(
            loaded_model=loaded_model,
            prompt=prompt,
            num_inference_steps=num_steps,
            prompt_idx=idx,
        )
        print(f"    ✓ Generation completed (peak memory: {result.peak_memory_mb:.2f}MB)")

        runner.save_result(result)
        print("    ✓ Result saved")

        ModelLoader.unload_model(loaded_model)


def _run_sdxl_benchmarks(runner: BenchmarkRunner) -> None:
    for base_model_key, base_model_path in BASE_MODELS.items():
        _print_section_header(f"Speed Experiment Model: {base_model_key} ({base_model_path})")

        lcm_checkpoint = LCM_CHECKPOINT_PATHS.get(base_model_key)

        print("[1/2] Base Model Speed Model Benchmark Started...")
        _run_sdxl_model_benchmark(
            runner=runner,
            model_display_name=f"{base_model_key}_base",
            load_fn=lambda: ModelLoader.load_teacher_model(
                base_model_key=base_model_key,
                base_model_path=base_model_path,
            ),
            num_steps=TEACHER_STEPS,
        )
        print("✓ Base Speed Model Benchmark Completed\n")

        print("[2/2] Dobby Speed Model Benchmark Started...")
        _run_sdxl_model_benchmark(
            runner=runner,
            model_display_name=f"{base_model_key}_dobby",
            load_fn=lambda: ModelLoader.load_lcm_model(
                base_model_key=base_model_key,
                lcm_checkpoint_path=lcm_checkpoint,
                base_model_path=base_model_path,
            ),
            num_steps=LCM_STEPS,
        )
        print("✓ Dobby Speed Model Benchmark Completed\n")


def _run_sd15_benchmarks(runner: BenchmarkRunner) -> None:

    for base_model_key, base_model_path in SD15_MODELS.items():
        _print_section_header(f"Memory Experiment Model: {base_model_key} ({base_model_path})")

        quant_path = SD15_QUANT_CKPT_PATHS.get(base_model_key)

        print("[1/2] Base Memory Model Benchmark Started...")
        _run_sd15_model_benchmark(
            runner=runner,
            model_display_name=f"{base_model_key}_base",
            load_fn=lambda: ModelLoader.load_base_memory_model(
                base_model_key=base_model_key,
                base_model_path=base_model_path,
            ),
            num_steps=TEACHER_STEPS,
        )
        print("✓ Base Memory Model Benchmark Completed\n")

        print("[2/2] Dobby Memory Model Benchmark Started...")
        _run_sd15_model_benchmark(
            runner=runner,
            model_display_name=f"{base_model_key}_quantized",
            load_fn=lambda: ModelLoader.load_dobby_memory_model(
                base_model_key=base_model_key,
                base_model_path=base_model_path,
                quant_path=quant_path,
            ),
            num_steps=TEACHER_STEPS,
        )
        print("✓ Dobby Memory Model Benchmark Completed\n")


def _print_summary(df: pd.DataFrame) -> None:
    """Print benchmark summary statistics split by model family."""
    _print_section_header("Benchmark Summary")

    sd15_mask = df["model_type"].isin(["base_memory", "dobby_memory"])
    sdxl_df = df[~sd15_mask]
    sd15_df = df[sd15_mask]

    if not sdxl_df.empty:
        sdxl_summary = sdxl_df.groupby(["base_model_key", "model_type"]).agg({
            "inference_time": ["mean", "std", "min", "max"],
        })
        print("[Speed Experiment] Model inference time statistics:")
        print(sdxl_summary.to_string())

    if not sd15_df.empty:
        sd15_summary = sd15_df.groupby(["base_model_key", "model_type"]).agg({
            "model_memory_mb": ["mean", "min", "max"],
            "peak_memory_mb": ["mean", "std", "min", "max"],
        })
        print("\n[Memory Experiment] Model memory statistics:")
        print(sd15_summary.to_string())

    counts = df.groupby(["base_model_key", "model_type"]).size()
    print("\nEach model measurement count:")
    print(counts.to_string())


def main():

    _print_section_header("Benchmark - Speed Experiment & Memory Experiment")

    runner = BenchmarkRunner(output_dir=OUTPUT_DIR)

    _run_sdxl_benchmarks(runner)
    _run_sd15_benchmarks(runner)

    _print_section_header("Result Saving & Visualization Creation")
    df = runner.save_results()

    _print_summary(df)

    print(f"\n{'=' * SECTION_WIDTH}")
    print(f"All results are saved in {OUTPUT_DIR}")
    print("=" * SECTION_WIDTH)


def _suppress_library_warnings() -> None:
    import diffusers
    from transformers import logging as transformers_logging
    transformers_logging.set_verbosity_error()
    diffusers.logging.set_verbosity_error()


if __name__ == "__main__":
    _suppress_library_warnings()
    _configure_torch_inductor()
    main()
