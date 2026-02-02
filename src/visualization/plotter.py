"""Plotting utilities for benchmark results."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


class ResultPlotter:
    """Creates visualizations from benchmark results."""

    def __init__(self, output_dir: str):
        """
        Initialize plotter.

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_inference_time_comparison(self, df: pd.DataFrame, prompts: list[str]) -> None:
        """
        Plot inference time comparison across all models.

        Args:
            df: DataFrame with benchmark results
            prompts: List of prompts used
        """
        plt.figure(figsize=(14, 6))

        for model_name in df["model_name"].unique():
            model_data = df[df["model_name"] == model_name]
            times = model_data.sort_values("prompt_idx")["inference_time"].values
            plt.plot(range(1, len(prompts) + 1), times, marker="o", label=model_name)

        plt.xticks(range(1, len(prompts) + 1))
        plt.xlabel("Prompt Index")
        plt.ylabel("Inference Time (s)")
        plt.title("모델별 프롬프트별 Inference Time 비교")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True)
        plt.tight_layout()

        output_path = self.output_dir / "inference_time_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Inference time 비교 그래프 저장: {output_path}")

    def plot_model_load_time_comparison(self, df: pd.DataFrame) -> None:
        """
        Plot model loading time comparison.

        Args:
            df: DataFrame with benchmark results
        """
        # Get unique model load times
        load_times = (df.groupby("model_name")["model_load_time"].first().sort_values())

        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(load_times)), load_times.values)

        # Color bars by model type
        for i, model_name in enumerate(load_times.index):
            if "teacher" in model_name:
                bars[i].set_color("#3498db")
            else:
                bars[i].set_color("#e74c3c")

        plt.xticks(range(len(load_times)), load_times.index, rotation=45, ha="right")
        plt.ylabel("Model Load Time (s)")
        plt.title("모델별 로딩 시간 비교")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        output_path = self.output_dir / "model_load_time_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"모델 로딩 시간 비교 그래프 저장: {output_path}")

    def plot_inference_time_by_prompt(self, df: pd.DataFrame, prompts: list[str]) -> None:
        """
        Plot inference time for each prompt across models.

        Args:
            df: DataFrame with benchmark results
            prompts: List of prompts used
        """
        num_prompts = len(prompts)
        model_names = sorted(df["model_name"].unique())

        plt.figure(figsize=(18, 2.5 * num_prompts))

        for idx in range(num_prompts):
            plt.subplot(num_prompts, 1, idx + 1)
            times = [
                df[(df["prompt_idx"] == idx + 1) & (df["model_name"] == model)]["inference_time"].values[0]
                for model in model_names
            ]
            bars = plt.bar(range(len(model_names)), times)

            # Color bars by model type
            for i, model_name in enumerate(model_names):
                if "teacher" in model_name:
                    bars[i].set_color("#3498db")
                else:
                    bars[i].set_color("#e74c3c")

            plt.xticks(range(len(model_names)), model_names, rotation=45, ha="right")
            plt.ylabel("Inference Time (s)")
            plt.title(f"Prompt {idx + 1}: {prompts[idx][:60]}...")
            plt.grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()

        output_path = self.output_dir / "inference_time_by_prompt.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"프롬프트별 inference time 그래프 저장: {output_path}")

    def create_all_plots(self, df: pd.DataFrame, prompts: list[str]) -> None:
        """
        Create all visualization plots.

        Args:
            df: DataFrame with benchmark results
            prompts: List of prompts used
        """
        print("\n=== 시각화 생성 중 ===")
        self.plot_model_load_time_comparison(df)
        self.plot_inference_time_comparison(df, prompts)
        self.plot_inference_time_by_prompt(df, prompts)
        print("=== 모든 시각화 완료 ===\n")
