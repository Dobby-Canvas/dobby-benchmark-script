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
        print(f"  [{idx}/{len(PROMPTS)}] 프롬프트: {prompt[:50]}...")
        print(f"    모델 로딩 중: {model_display_name}")

        loaded_model = load_fn()
        print("    ✓ 모델 로딩 완료")

        result = runner.run_inference(
            loaded_model=loaded_model,
            prompt=prompt,
            num_inference_steps=num_steps,
            prompt_idx=idx,
        )
        print(f"    ✓ 생성 완료 (추론 시간: {result.inference_time:.2f}초)")

        runner.save_result(result)
        print("    ✓ 결과 저장 완료")

        ModelLoader.unload_model(loaded_model)


def _run_sd15_model_benchmark(
    runner: BenchmarkRunner,
    model_display_name: str,
    load_fn: Callable[[], LoadedModel],
    num_steps: int,
) -> None:
    """Iterate over all prompts for one SD1.5 model variant, reporting memory usage."""
    for idx, prompt in enumerate(PROMPTS, start=1):
        print(f"  [{idx}/{len(PROMPTS)}] 프롬프트: {prompt[:50]}...")
        print(f"    모델 로딩 중: {model_display_name}")

        loaded_model = load_fn()
        print(f"    ✓ 모델 로딩 완료 (모델 메모리: {loaded_model.model_memory_mb:.2f}MB)")

        result = runner.run_inference(
            loaded_model=loaded_model,
            prompt=prompt,
            num_inference_steps=num_steps,
            prompt_idx=idx,
        )
        print(f"    ✓ 생성 완료 (피크 메모리: {result.peak_memory_mb:.2f}MB)")

        runner.save_result(result)
        print("    ✓ 결과 저장 완료")

        ModelLoader.unload_model(loaded_model)


def _run_sdxl_benchmarks(runner: BenchmarkRunner) -> None:
    """Run base and Dobby benchmarks for all SDXL models."""
    for base_model_key, base_model_path in BASE_MODELS.items():
        _print_section_header(f"SDXL Model: {base_model_key} ({base_model_path})")

        lcm_checkpoint = LCM_CHECKPOINT_PATHS.get(base_model_key)

        print("[1/2] Base 모델 벤치마크 시작...")
        _run_sdxl_model_benchmark(
            runner=runner,
            model_display_name=f"{base_model_key}_base",
            load_fn=lambda: ModelLoader.load_teacher_model(
                base_model_key=base_model_key,
                base_model_path=base_model_path,
            ),
            num_steps=TEACHER_STEPS,
        )
        print("✓ Base 모델 벤치마크 완료\n")

        print("[2/2] Dobby 모델 벤치마크 시작...")
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
        print("✓ Dobby 모델 벤치마크 완료\n")


def _run_sd15_benchmarks(runner: BenchmarkRunner) -> None:
    """Run base and quantized benchmarks for all SD1.5 models."""
    for base_model_key, base_model_path in SD15_MODELS.items():
        _print_section_header(f"SD1.5 Model: {base_model_key} ({base_model_path})")

        quant_ckpt = SD15_QUANT_CKPT_PATHS.get(base_model_key)

        print("[1/2] SD1.5 Base 모델 벤치마크 시작...")
        _run_sd15_model_benchmark(
            runner=runner,
            model_display_name=f"{base_model_key}_base",
            load_fn=lambda: ModelLoader.load_base_memory_model(
                base_model_key=base_model_key,
                base_model_path=base_model_path,
            ),
            num_steps=TEACHER_STEPS,
        )
        print("✓ SD1.5 Base 모델 벤치마크 완료\n")

        print("[2/2] SD1.5 Quantized 모델 벤치마크 시작...")
        _run_sd15_model_benchmark(
            runner=runner,
            model_display_name=f"{base_model_key}_quantized",
            load_fn=lambda: ModelLoader.load_dobby_memory_model(
                base_model_key=base_model_key,
                base_model_path=base_model_path,
                ckpt_path=quant_ckpt,
            ),
            num_steps=TEACHER_STEPS,
        )
        print("✓ SD1.5 Quantized 모델 벤치마크 완료\n")


def _print_summary(df: pd.DataFrame) -> None:
    """Print benchmark summary statistics split by model family."""
    _print_section_header("벤치마크 요약")

    sd15_mask = df["model_type"].isin(["base_memory", "dobby_memory"])
    sdxl_df = df[~sd15_mask]
    sd15_df = df[sd15_mask]

    if not sdxl_df.empty:
        sdxl_summary = sdxl_df.groupby(["base_model_key", "model_type"]).agg({
            "inference_time": ["mean", "std", "min", "max"],
        })
        print("[SDXL] 모델별 추론 시간 통계:")
        print(sdxl_summary.to_string())

    if not sd15_df.empty:
        sd15_summary = sd15_df.groupby(["base_model_key", "model_type"]).agg({
            "model_memory_mb": ["mean", "min", "max"],
            "peak_memory_mb": ["mean", "std", "min", "max"],
        })
        print("\n[SD1.5] 모델별 메모리 통계:")
        print(sd15_summary.to_string())

    counts = df.groupby(["base_model_key", "model_type"]).size()
    print("\n각 모델별 측정 횟수:")
    print(counts.to_string())


def main():
    """Execute SDXL and SD1.5 benchmarks."""
    _configure_torch_inductor()

    _print_section_header("Benchmark - SDXL (속도) & SD1.5 (메모리)")

    runner = BenchmarkRunner(output_dir=OUTPUT_DIR)
    plotter = ResultPlotter(output_dir=OUTPUT_DIR)

    _run_sdxl_benchmarks(runner)
    _run_sd15_benchmarks(runner)

    _print_section_header("결과 저장 및 시각화 생성")
    df = runner.save_results()
    plotter.create_all_plots(df, PROMPTS)

    _print_summary(df)

    print(f"\n{'=' * SECTION_WIDTH}")
    print(f"모든 결과가 저장되었습니다: {OUTPUT_DIR}")
    print("=" * SECTION_WIDTH)


if __name__ == "__main__":
    main()
