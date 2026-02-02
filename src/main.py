"""Main execution script for SDXL benchmark."""

import os
from pathlib import Path

import torch

from .benchmark import BenchmarkRunner
from .config import (BASE_MODELS, LCM_CHECKPOINT_PATHS, LCM_STEPS, PROMPTS, TEACHER_STEPS)
from .models import ModelLoader
from .visualization import ResultPlotter


def main():
    """Execute benchmark for all base models."""
    # Configure torch.compile for optimal performance
    torch._inductor.config.conv_1x1_as_mm = True
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.epilogue_fusion = False
    torch._inductor.config.coordinate_descent_check_all_directions = True

    print("=" * 80)
    print("SDXL Benchmark - Teacher vs LCM Fine-tuned Models")
    print("=" * 80)

    # Initialize runner and plotter
    output_dir = "results/validation/all_models"
    runner = BenchmarkRunner(output_dir=output_dir)
    plotter = ResultPlotter(output_dir=output_dir)

    # Run benchmarks for each base model
    for base_model_key, base_model_path in BASE_MODELS.items():
        print(f"\n{'=' * 80}")
        print(f"Base Model: {base_model_key} ({base_model_path})")
        print(f"{'=' * 80}\n")

        # 1. Base Model Benchmark
        print(f"[1/2] Base 모델 벤치마크 시작...")

        for idx, prompt in enumerate(PROMPTS, start=1):
            print(f"  [{idx}/{len(PROMPTS)}] 프롬프트: {prompt[:50]}...")
            print(f"    모델 로딩 중: {base_model_key}_base")

            teacher_model = ModelLoader.load_teacher_model(
                base_model_key=base_model_key,
                base_model_path=base_model_path,
            )
            print(f"    ✓ 모델 로딩 완료 (소요 시간: {teacher_model.load_time:.2f}초)")

            result = runner.run_inference(
                loaded_model=teacher_model,
                prompt=prompt,
                num_inference_steps=TEACHER_STEPS,
                prompt_idx=idx,
            )
            print(f"    ✓ 생성 완료 (소요 시간: {result.inference_time:.2f}초)")

            ModelLoader.unload_model(teacher_model)

        print(f"\n✓ Base 모델 벤치마크 완료\n")

        # 2. LCM Fine-tuned Model Benchmark
        lcm_checkpoint = LCM_CHECKPOINT_PATHS.get(base_model_key)

        print(f"[2/2] Dobby  모델 벤치마크 시작...")

        for idx, prompt in enumerate(PROMPTS, start=1):
            print(f"  [{idx}/{len(PROMPTS)}] 프롬프트: {prompt[:50]}...")
            print(f"    모델 로딩 중: {base_model_key}_dobby")

            lcm_model = ModelLoader.load_lcm_model(
                base_model_key=base_model_key,
                lcm_checkpoint_path=lcm_checkpoint,
            )
            print(f"    ✓ 모델 로딩 완료 (소요 시간: {lcm_model.load_time:.2f}초)")

            result = runner.run_inference(
                loaded_model=lcm_model,
                prompt=prompt,
                num_inference_steps=LCM_STEPS,
                prompt_idx=idx,
            )
            print(f"    ✓ 생성 완료 (소요 시간: {result.inference_time:.2f}초)")

            ModelLoader.unload_model(lcm_model)

        print(f"\n✓ LCM 모델 벤치마크 완료\n")

    # Save results and create visualizations
    print("\n" + "=" * 80)
    print("결과 저장 및 시각화 생성")
    print("=" * 80 + "\n")

    df = runner.save_results()
    plotter.create_all_plots(df, PROMPTS)

    # Print summary statistics
    print("\n" + "=" * 80)
    print("벤치마크 요약")
    print("=" * 80)

    # 모델별 통계 - 프롬프트별로 측정된 model_load_time과 inference_time 모두 통계 계산
    summary = df.groupby(["base_model_key", "model_type"]).agg({
        "model_load_time": ["mean", "std", "min", "max"],
        "inference_time": ["mean", "std", "min", "max"],
    })

    print("\n모델별 통계 (프롬프트별 측정):")
    print(summary.to_string())

    # 추가: 각 모델의 프롬프트 개수 확인
    counts = df.groupby(["base_model_key", "model_type"]).size()
    print("\n\n각 모델별 측정 횟수:")
    print(counts.to_string())

    print("\n" + "=" * 80)
    print(f"모든 결과가 저장되었습니다: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
