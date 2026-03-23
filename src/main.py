"""Main execution script for SDXL and SD1.5 benchmarks."""

import argparse
import dataclasses
import multiprocessing as mp
from pathlib import Path

import pandas as pd
import torch

from .benchmark import BenchmarkRunner
from .config import (
    BASE_MODELS,
    LCM_CHECKPOINT_PATHS,
    LCM_STEPS,
    PROMPTS,
    SD15_GGUF_ASSET_REPOS,
    SD15_GGUF_UNET_CONFIG_DIRS,
    SD15_GGUF_UNET_PATHS,
    SD15_MODELS,
    SD15_QUANT_CKPT_PATHS,
    TEACHER_STEPS,
)
from .models import ModelLoader


@dataclasses.dataclass
class _SD15InferRequest:
    """Parameters passed to the subprocess worker for a single SD1.5 inference."""

    model_load_type: str  # "base_memory" | "dobby_memory_quant" | "dobby_memory_gguf"
    base_model_key: str
    base_model_path: str
    prompt: str
    num_steps: int
    prompt_idx: int
    output_dir: str
    quant_path: str | None = None
    gguf_unet_path: str | None = None
    unet_config_dir: str | None = None
    hf_asset_repo_id: str | None = None


def _subprocess_infer_sd15_worker(
    req: _SD15InferRequest,
    result_queue: "mp.Queue",
) -> None:
    """Worker that runs in a spawned subprocess for isolated RAM measurement.

    각 subprocess는 새로운 OS 주소 공간에서 시작하므로 RSS 측정값이
    이전 반복의 Python heap 잔류물이나 CUDA 런타임 초기화 비용에
    오염되지 않는다. baseline은 subprocess 시작 직후의 RSS로 고정된다.
    """
    import gc as _gc
    import threading as _threading

    import psutil as _psutil

    from .benchmark import BenchmarkRunner
    from .models import ModelLoader

    process = _psutil.Process()
    baseline_ram_mb = process.memory_info().rss / (1024 * 1024)

    ram_samples: list[float] = []
    stop_event = _threading.Event()

    def _sample_loop() -> None:
        p = _psutil.Process()
        while not stop_event.is_set():
            ram_samples.append(p.memory_info().rss / (1024 * 1024))
            stop_event.wait(timeout=0.1)

    monitor = _threading.Thread(target=_sample_loop, daemon=True)
    monitor.start()

    try:
        if req.model_load_type == "base_memory":
            loaded_model = ModelLoader.load_base_memory_model(req.base_model_key, req.base_model_path)
        elif req.model_load_type == "dobby_memory_quant":
            loaded_model = ModelLoader.load_dobby_memory_model(req.base_model_key, req.base_model_path, req.quant_path)
        else:  # dobby_memory_gguf
            loaded_model = ModelLoader.load_dobby_ram_gguf_model(
                req.base_model_key,
                req.base_model_path,
                req.gguf_unet_path,
                req.unet_config_dir,
                req.hf_asset_repo_id,
            )

        runner = BenchmarkRunner(output_dir=req.output_dir)
        result = runner.run_inference(
            loaded_model=loaded_model,
            prompt=req.prompt,
            num_inference_steps=req.num_steps,
            prompt_idx=req.prompt_idx,
        )

        stop_event.set()
        monitor.join()

        result.peak_ram_mb = max(0.0, max(ram_samples) - baseline_ram_mb)

        ModelLoader.unload_model(loaded_model)
        _gc.collect()
        result_queue.put(result)
    except Exception:
        stop_event.set()
        monitor.join()
        result_queue.put(None)
        raise


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
    load_fn,
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
    req_template: _SD15InferRequest,
) -> None:
    """Spawn one subprocess per prompt for fully isolated RAM measurement.

    subprocess(spawn)는 새로운 OS 주소 공간으로 시작하므로 Python heap 잔류물,
    CUDA 런타임 초기화 비용 등이 baseline_ram_mb에 포함되지 않는다.
    이를 통해 반복 간 측정값의 분산을 최소화할 수 있다.
    """
    ctx = mp.get_context("spawn")
    for idx, prompt in enumerate(PROMPTS, start=1):
        print(f"  [{idx}/{len(PROMPTS)}] Prompt: {prompt[:50]}...")
        print(f"    Loading model: {model_display_name}")

        req = dataclasses.replace(req_template, prompt=prompt, prompt_idx=idx)
        result_queue: mp.Queue = ctx.Queue()
        proc = ctx.Process(target=_subprocess_infer_sd15_worker, args=(req, result_queue))
        proc.start()

        try:
            result = result_queue.get(timeout=900)  # 15분 타임아웃
        except Exception:
            proc.kill()
            raise RuntimeError(f"Subprocess timed out for prompt {idx}: {prompt[:50]}")

        proc.join()

        if result is None or proc.exitcode != 0:
            raise RuntimeError(f"Subprocess inference failed (exitcode={proc.exitcode}) for prompt {idx}")

        print("    ✓ Model loaded")
        print(f"    ✓ Generation completed (GPU: {result.peak_memory_mb:.0f}MB | RAM: {result.peak_ram_mb:.0f}MB")

        runner.results.append(result)
        runner.save_result(result)
        print("    ✓ Result saved")


def _run_sdxl_benchmarks(runner: BenchmarkRunner) -> None:
    for base_model_key, base_model_path in BASE_MODELS.items():
        _print_section_header(f"Speed Experiment Model: {base_model_key} ({base_model_path})")

        lcm_checkpoint = LCM_CHECKPOINT_PATHS.get(base_model_key)

        print("[1/2] Base Model Speed Model Benchmark Started...")
        _run_sdxl_model_benchmark(
            runner=runner,
            model_display_name=f"{base_model_key}_base",
            load_fn=lambda key=base_model_key, path=base_model_path: ModelLoader.load_teacher_model(
                base_model_key=key,
                base_model_path=path,
            ),
            num_steps=TEACHER_STEPS,
        )
        print("✓ Base Speed Model Benchmark Completed\n")

        print("[2/2] Dobby Speed Model Benchmark Started...")
        _run_sdxl_model_benchmark(
            runner=runner,
            model_display_name=f"{base_model_key}_dobby",
            load_fn=lambda key=base_model_key, path=base_model_path, ckpt=lcm_checkpoint: ModelLoader.load_lcm_model(
                base_model_key=key,
                lcm_checkpoint_path=ckpt,
                base_model_path=path,
            ),
            num_steps=LCM_STEPS,
        )
        print("✓ Dobby Speed Model Benchmark Completed\n")


def _run_sd15_benchmarks(
    runner: BenchmarkRunner,
    experiment_label: str = "Memory",
    use_gguf_for_dobby: bool = False,
) -> None:
    for base_model_key, base_model_path in SD15_MODELS.items():
        _print_section_header(f"{experiment_label} Experiment Model: {base_model_key} ({base_model_path})")

        print("[1/2] Base Memory Model Benchmark Started...")
        _run_sd15_model_benchmark(
            runner=runner,
            model_display_name=f"{base_model_key}_base",
            req_template=_SD15InferRequest(
                model_load_type="base_memory",
                base_model_key=base_model_key,
                base_model_path=base_model_path,
                prompt="",
                num_steps=TEACHER_STEPS,
                prompt_idx=0,
                output_dir=str(runner.output_dir),
            ),
        )
        print("✓ Base Memory Model Benchmark Completed\n")

        print("[2/2] Dobby Memory Model Benchmark Started...")
        if use_gguf_for_dobby:
            gguf_unet_path = SD15_GGUF_UNET_PATHS.get(base_model_key)
            unet_config_dir = SD15_GGUF_UNET_CONFIG_DIRS.get(base_model_key)
            hf_asset_repo_id = SD15_GGUF_ASSET_REPOS.get(base_model_key) or None

            if not gguf_unet_path or not unet_config_dir:
                raise ValueError(
                    f"Missing GGUF RAM benchmark paths for '{base_model_key}'. "
                    "Set SD15_GGUF_UNET_PATHS and SD15_GGUF_UNET_CONFIG_DIRS in settings.py."
                )
            req_template = _SD15InferRequest(
                model_load_type="dobby_memory_gguf",
                base_model_key=base_model_key,
                base_model_path=base_model_path,
                prompt="",
                num_steps=TEACHER_STEPS,
                prompt_idx=0,
                output_dir=str(runner.output_dir),
                gguf_unet_path=gguf_unet_path,
                unet_config_dir=unet_config_dir,
                hf_asset_repo_id=hf_asset_repo_id,
            )
        else:
            quant_path = SD15_QUANT_CKPT_PATHS.get(base_model_key)
            req_template = _SD15InferRequest(
                model_load_type="dobby_memory_quant",
                base_model_key=base_model_key,
                base_model_path=base_model_path,
                prompt="",
                num_steps=TEACHER_STEPS,
                prompt_idx=0,
                output_dir=str(runner.output_dir),
                quant_path=quant_path,
            )

        _run_sd15_model_benchmark(
            runner=runner,
            model_display_name=f"{base_model_key}_quantized",
            req_template=req_template,
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
    experiment: str,
) -> None:
    """Calculate Base vs Dobby improvement rates and save experiment-specific reports.

    Args:
        df: Full results DataFrame
        output_dir: Output directory
        experiment: "all" | "speed" | "gpu" | "ram" - selected experiment scope
    """
    sd15_mask = df["model_type"].isin(["base_memory", "dobby_memory", "dobby_memory_gguf"])
    sdxl_df = df[~sd15_mask]
    sd15_df = df[sd15_mask]

    output_speed = experiment in ("all", "speed")
    output_gpu = experiment in ("all", "gpu")
    output_ram = experiment in ("all", "ram")

    if output_speed and not sdxl_df.empty:
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

    if not sd15_df.empty and (output_gpu or output_ram):
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
            # GPU 실험은 quantized 모델(dobby_memory), RAM 실험은 GGUF 모델(dobby_memory_gguf)
            dobby_mem_gpu = sd15_df[(sd15_df["base_model_key"] == base_key) & (sd15_df["model_type"] == "dobby_memory")]
            dobby_mem_ram = sd15_df[
                (sd15_df["base_model_key"] == base_key) & (sd15_df["model_type"] == "dobby_memory_gguf")
            ]

            if output_gpu and not dobby_mem_gpu.empty:
                v_base = base_mem["peak_memory_mb"].mean()
                v_dobby = dobby_mem_gpu["peak_memory_mb"].mean()
                gpu_reduction_pct = (v_base - v_dobby) / v_base * 100 if v_base > 0 else 0.0

                gpu_lines.append("")
                gpu_lines.append(f"[{base_key}]")
                gpu_lines.append(f"  Base 평균 피크 VRAM:  {v_base:.2f} MB")
                gpu_lines.append(f"  Dobby 평균 피크 VRAM: {v_dobby:.2f} MB")
                gpu_lines.append(f"  VRAM 절감율: {gpu_reduction_pct:.2f}%")
                gpu_lines.append("")

            if output_ram and not dobby_mem_ram.empty:
                m_base = base_mem["peak_ram_mb"].mean()
                m_dobby = dobby_mem_ram["peak_ram_mb"].mean()
                ram_reduction_pct = (m_base - m_dobby) / m_base * 100 if m_base > 0 else 0.0

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
    experiment: str,
) -> None:
    """Print benchmark summary statistics and save experiment-specific CSV files.

    Args:
        df: Full results DataFrame
        output_dir: Output directory
        experiment: "all" | "speed" | "gpu" | "ram" - selected experiment scope
    """
    _print_section_header("Benchmark Summary")

    sd15_mask = df["model_type"].isin(["base_memory", "dobby_memory", "dobby_memory_gguf"])
    sdxl_df = df[~sd15_mask]
    sd15_df = df[sd15_mask]

    output_speed = experiment in ("all", "speed")
    output_gpu = experiment in ("all", "gpu")
    output_ram = experiment in ("all", "ram")

    if output_speed and not sdxl_df.empty:
        sdxl_summary = sdxl_df.groupby(["base_model_key", "model_type"]).agg(
            {
                "inference_time": ["mean", "std", "min", "max"],
            }
        )
        print("[Speed Experiment] Model inference time statistics:")
        print(sdxl_summary.to_string())

        speed_df = sdxl_summary.reset_index()
        speed_df.columns = [
            "base_model_key",
            "model_type",
            "inference_time_mean",
            "inference_time_std",
            "inference_time_min",
            "inference_time_max",
        ]
        speed_path = output_dir / BENCHMARK_SUMMARY_SPEED_CSV
        speed_df.to_csv(speed_path, index=False)
        print(f"  → Saved: {speed_path}")

    if not sd15_df.empty and (output_gpu or output_ram):
        if output_gpu:
            # GPU 실험: quantized 모델(dobby_memory)만 포함
            gpu_sd15_df = sd15_df[sd15_df["model_type"].isin(["base_memory", "dobby_memory"])]
            gpu_summary = gpu_sd15_df.groupby(["base_model_key", "model_type"]).agg(
                {"peak_memory_mb": ["mean", "min", "max"]}
            )
            print("\n[GPU Memory Experiment] GPU VRAM statistics:")
            print("  - peak_memory_mb: 추론 중 최대 GPU 메모리 (MB)")
            print(gpu_summary.to_string())

            gpu_df = gpu_summary.reset_index()
            gpu_df.columns = [
                "base_model_key",
                "model_type",
                "peak_memory_mb_mean",
                "peak_memory_mb_min",
                "peak_memory_mb_max",
            ]
            gpu_path = output_dir / BENCHMARK_SUMMARY_GPU_CSV
            gpu_df.to_csv(gpu_path, index=False)
            print(f"  → Saved: {gpu_path}")

        if output_ram:
            # RAM 실험: GGUF 모델(dobby_memory_gguf) 우선, 없으면 quant 폴백
            ram_dobby_type = (
                "dobby_memory_gguf" if (sd15_df["model_type"] == "dobby_memory_gguf").any() else "dobby_memory"
            )
            ram_sd15_df = sd15_df[sd15_df["model_type"].isin(["base_memory", ram_dobby_type])]
            ram_summary = ram_sd15_df.groupby(["base_model_key", "model_type"]).agg(
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

    _save_improvement_report(df, output_dir, experiment)


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

    if experiment == "all":
        speed_runner = BenchmarkRunner(output_dir=f"{OUTPUT_DIR}speed/")
        gpu_runner = BenchmarkRunner(output_dir=f"{OUTPUT_DIR}gpu/")
        ram_runner = BenchmarkRunner(output_dir=f"{OUTPUT_DIR}ram/")  # ← 추가

        _run_sdxl_benchmarks(speed_runner)
        _run_sd15_benchmarks(gpu_runner, experiment_label="GPU Memory", use_gguf_for_dobby=False)
        _run_sd15_benchmarks(ram_runner, experiment_label="RAM", use_gguf_for_dobby=True)  # ← 추가

        combined_runner = BenchmarkRunner(output_dir=OUTPUT_DIR)
        combined_runner.results = speed_runner.results + gpu_runner.results + ram_runner.results  # ← ram 추가
        df = combined_runner.save_results()
    else:
        subdir = f"{OUTPUT_DIR}{experiment}/"
        runner = BenchmarkRunner(output_dir=subdir)

        if experiment == "speed":
            _run_sdxl_benchmarks(runner)
        else:
            label = "GPU VRAM" if experiment == "gpu" else "RAM"
            _run_sd15_benchmarks(
                runner,
                experiment_label=label,
                use_gguf_for_dobby=(experiment == "ram"),
            )

        df = runner.save_results()

    _print_section_header("Result Saving & Visualization Creation")
    output_dir_path = Path(OUTPUT_DIR)
    _print_summary(df, output_dir_path, experiment)

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
