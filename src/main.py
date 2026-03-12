"""Main execution script for SDXL and SD1.5 benchmarks."""

import argparse
from pathlib import Path
from typing import Callable

import pandas as pd
import torch

from .benchmark import BenchmarkRunner
from .config import (BASE_MODELS, LCM_CHECKPOINT_PATHS, LCM_STEPS, PROMPTS,
                     SD15_MODELS, SD15_QUANT_CKPT_PATHS, TEACHER_STEPS)
from .models import LoadedModel, ModelLoader
from .visualization import ResultPlotter

OUTPUT_DIR = "results/"
BENCHMARK_SUMMARY_SPEED_CSV = "benchmark_summary_speed.csv"
BENCHMARK_SUMMARY_GPU_CSV = "benchmark_summary_gpu.csv"
BENCHMARK_SUMMARY_RAM_CSV = "benchmark_summary_ram.csv"
BENCHMARK_IMPROVEMENT_REPORT_SPEED_TXT = "benchmark_improvement_report_speed.txt"
BENCHMARK_IMPROVEMENT_REPORT_GPU_TXT = "benchmark_improvement_report_gpu.txt"
BENCHMARK_IMPROVEMENT_REPORT_RAM_TXT = "benchmark_improvement_report_ram.txt"
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
        print(
            f"    ✓ Generation completed (inference time: {result.inference_time:.2f}s "
        )

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
        print(
            f"    ✓ Generation completed (GPU: {result.peak_memory_mb:.0f}MB "
            f"| RAM: {result.peak_ram_mb:.0f}MB "
        )

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


def _run_sd15_benchmarks(runner: BenchmarkRunner, experiment_label: str = "Memory") -> None:
    for base_model_key, base_model_path in SD15_MODELS.items():
        _print_section_header(f"{experiment_label} Experiment Model: {base_model_key} ({base_model_path})")

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


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for experiment selection."""
    parser = argparse.ArgumentParser(
        description="Dobby Canvas 이미지 생성 모델 성능 벤치마크",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
실험 선택:
  all      Speed + GPU + RAM 실험 모두 실행 (기본값)
  speed    TC1: Dobby Speed 모델 추론 시간
  gpu      TC2: Dobby Quantized 모델 GPU VRAM
  ram      TC3: Dobby Quantized 모델 시스템 RAM

예시:
  python -m src.main              # 전체 실행
  python -m src.main -e speed     # Speed 실험만
  python -m src.main -e gpu       # GPU VRAM 실험만
  python -m src.main -e ram      # RAM 실험만
        """,
    )
    parser.add_argument(
        "-e",
        "--experiment",
        choices=["all", "speed", "gpu", "ram"],
        default="all",
        help="실행할 실험 (기본: all)",
    )
    return parser.parse_args()


def _save_improvement_report(
    df: pd.DataFrame,
    output_dir: Path,
    memory_output: str = "both",
) -> None:
    """Calculate Base vs Dobby improvement rates and save to separate txt files per experiment.

    Args:
        df: Full results DataFrame
        output_dir: Output directory
        memory_output: "both" | "gpu_only" | "ram_only" - which memory reports to generate
    """
    sd15_mask = df["model_type"].isin(["base_memory", "dobby_memory"])
    sdxl_df = df[~sd15_mask]
    sd15_df = df[sd15_mask]

    if not sdxl_df.empty:
        speed_lines: list[str] = []
        speed_lines.append("=" * 60)
        speed_lines.append("Speed Experiment - Improvement Report (Base vs Dobby)")
        speed_lines.append("=" * 60)

        for base_key in sdxl_df["base_model_key"].unique():
            base_speed = sdxl_df[(sdxl_df["base_model_key"] == base_key) & (sdxl_df["model_type"] == "base_speed")]
            dobby_speed = sdxl_df[(sdxl_df["base_model_key"] == base_key) & (sdxl_df["model_type"] == "dobby_speed")]

            if base_speed.empty or dobby_speed.empty:
                continue

            t_base = base_speed["inference_time"].mean()
            t_dobby = dobby_speed["inference_time"].mean()
            speed_improvement_pct = (t_base - t_dobby) / t_base * 100 if t_base > 0 else 0.0

            speed_lines.append("")
            speed_lines.append(f"[{base_key}]")
            speed_lines.append(f"  Base 평균 추론 시간:  {t_base:.4f} 초")
            speed_lines.append(f"  Dobby 평균 추론 시간: {t_dobby:.4f} 초")
            speed_lines.append(f"  속도 개선율: {speed_improvement_pct:.2f}%")
            speed_lines.append("")

        speed_lines.append("=" * 60)
        speed_path = output_dir / BENCHMARK_IMPROVEMENT_REPORT_SPEED_TXT
        speed_path.write_text("\n".join(speed_lines), encoding="utf-8")
        print(f"\nSpeed improvement report saved to: {speed_path}")

    if not sd15_df.empty:
        output_gpu = memory_output in ("both", "gpu_only")
        output_ram = memory_output in ("both", "ram_only")

        if output_gpu:
            gpu_lines: list[str] = []
            gpu_lines.append("=" * 60)
            gpu_lines.append("GPU VRAM Experiment - Improvement Report (Base vs Dobby)")
            gpu_lines.append("=" * 60)

        if output_ram:
            ram_lines: list[str] = []
            ram_lines.append("=" * 60)
            ram_lines.append("RAM Experiment - Improvement Report (Base vs Dobby)")
            ram_lines.append("=" * 60)

        for base_key in sd15_df["base_model_key"].unique():
            base_mem = sd15_df[(sd15_df["base_model_key"] == base_key) & (sd15_df["model_type"] == "base_memory")]
            dobby_mem = sd15_df[(sd15_df["base_model_key"] == base_key) & (sd15_df["model_type"] == "dobby_memory")]

            if base_mem.empty or dobby_mem.empty:
                continue

            v_base = base_mem["peak_memory_mb"].mean()
            v_dobby = dobby_mem["peak_memory_mb"].mean()
            gpu_reduction_pct = (v_base - v_dobby) / v_base * 100 if v_base > 0 else 0.0

            m_base = base_mem["peak_ram_mb"].mean()
            m_dobby = dobby_mem["peak_ram_mb"].mean()
            ram_reduction_pct = (m_base - m_dobby) / m_base * 100 if m_base > 0 else 0.0

            if output_gpu:
                gpu_lines.append("")
                gpu_lines.append(f"[{base_key}]")
                gpu_lines.append(f"  Base 평균 피크 VRAM:  {v_base:.2f} MB")
                gpu_lines.append(f"  Dobby 평균 피크 VRAM: {v_dobby:.2f} MB")
                gpu_lines.append(f"  VRAM 절감율: {gpu_reduction_pct:.2f}%")
                gpu_lines.append("")

            if output_ram:
                ram_lines.append("")
                ram_lines.append(f"[{base_key}]")
                ram_lines.append(f"  Base 평균 피크 RAM:  {m_base:.2f} MB")
                ram_lines.append(f"  Dobby 평균 피크 RAM: {m_dobby:.2f} MB")
                ram_lines.append(f"  RAM 절감율: {ram_reduction_pct:.2f}%")
                ram_lines.append("")

        if output_gpu:
            gpu_lines.append("=" * 60)
            gpu_path = output_dir / BENCHMARK_IMPROVEMENT_REPORT_GPU_TXT
            gpu_path.write_text("\n".join(gpu_lines), encoding="utf-8")
            print(f"GPU improvement report saved to: {gpu_path}")

        if output_ram:
            ram_lines.append("=" * 60)
            ram_path = output_dir / BENCHMARK_IMPROVEMENT_REPORT_RAM_TXT
            ram_path.write_text("\n".join(ram_lines), encoding="utf-8")
            print(f"RAM improvement report saved to: {ram_path}")


def _print_summary(
    df: pd.DataFrame,
    output_dir: Path,
    memory_output: str = "both",
) -> None:
    """Print benchmark summary statistics and save to separate CSV files.

    Args:
        df: Full results DataFrame
        output_dir: Output directory
        memory_output: "both" | "gpu_only" | "ram_only" - which memory outputs to generate
    """
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

        speed_df = sdxl_summary.reset_index()
        speed_df.columns = ["base_model_key", "model_type", "inference_time_mean", "inference_time_std", "inference_time_min", "inference_time_max"]
        speed_path = output_dir / BENCHMARK_SUMMARY_SPEED_CSV
        speed_df.to_csv(speed_path, index=False)
        print(f"  → Saved: {speed_path}")

    if not sd15_df.empty:
        output_gpu = memory_output in ("both", "gpu_only")
        output_ram = memory_output in ("both", "ram_only")

        if output_gpu:
            gpu_summary = sd15_df.groupby(["base_model_key", "model_type"]).agg(
                {"peak_memory_mb": ["mean", "min", "max"]}
            )
            print("\n[GPU Memory Experiment] GPU VRAM statistics:")
            print("  - peak_memory_mb: 추론 중 최대 GPU 메모리 (MB)")
            print(gpu_summary.to_string())

            gpu_df = gpu_summary.reset_index()
            gpu_df.columns = ["base_model_key", "model_type", "peak_memory_mb_mean", "peak_memory_mb_min", "peak_memory_mb_max"]
            gpu_path = output_dir / BENCHMARK_SUMMARY_GPU_CSV
            gpu_df.to_csv(gpu_path, index=False)
            print(f"  → Saved: {gpu_path}")

        if output_ram:
            ram_summary = sd15_df.groupby(["base_model_key", "model_type"]).agg(
                {"peak_ram_mb": ["mean", "min", "max"]}
            )
            print("\n[RAM (시스템 메모리) Experiment] 프로세스 RAM 사용량 통계:")
            print(ram_summary.to_string())

            ram_df = ram_summary.reset_index()
            ram_df.columns = ["base_model_key", "model_type", "peak_ram_mb_mean", "peak_ram_mb_min", "peak_ram_mb_max"]
            ram_path = output_dir / BENCHMARK_SUMMARY_RAM_CSV
            ram_df.to_csv(ram_path, index=False)
            print(f"  → Saved: {ram_path}")

    counts = df.groupby(["base_model_key", "model_type"]).size()
    print("\nEach model measurement count:")
    print(counts.to_string())

    _save_improvement_report(df, output_dir, memory_output)


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = _parse_args()

    experiment = args.experiment

    if experiment == "all":
        _print_section_header("Benchmark - Speed Experiment & GPU & RAM Experiment")
    elif experiment == "speed":
        _print_section_header("Benchmark - Speed Experiment")
    elif experiment == "gpu":
        _print_section_header("Benchmark - GPU VRAM Experiment")
    else:
        _print_section_header("Benchmark - RAM Experiment")

    runner = BenchmarkRunner(output_dir=OUTPUT_DIR)

    if experiment in ("all", "speed"):
        _run_sdxl_benchmarks(runner)
    if experiment in ("all", "gpu", "ram"):
        label = "GPU VRAM" if experiment == "gpu" else "RAM" if experiment == "ram" else "Memory"
        _run_sd15_benchmarks(runner, experiment_label=label)

    _print_section_header("Result Saving & Visualization Creation")
    df = runner.save_results()

    if experiment == "gpu":
        memory_output = "gpu_only"
    elif experiment == "ram":
        memory_output = "ram_only"
    else:
        memory_output = "both"

    _print_summary(df, Path(OUTPUT_DIR), memory_output)

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
    main(_parse_args())
