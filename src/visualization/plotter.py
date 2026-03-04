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

    def plot_sd15_memory_comparison(self, df: pd.DataFrame, prompts: list[str]) -> None:
        """
        Plot SD1.5 model memory usage comparison (model load memory + peak inference memory).

        Args:
            df: DataFrame with benchmark results
            prompts: List of prompts used
        """
        sd15_df = df[df["model_type"].isin(["base", "dobby"])]
        if sd15_df.empty:
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        MODEL_COLORS = {"base": "#3498db", "dobby": "#e74c3c"}
        MODEL_LABELS = {"base": "Base (FP16)", "dobby": "Quantized (W8A8)"}

        # Left: Model load memory (static VRAM after loading)
        # Use the mean of peak_memory_mb as proxy for static model memory when model_memory_mb unavailable
        if "model_memory_mb" in sd15_df.columns and sd15_df["model_memory_mb"].notna().any():
            model_mem_values = sd15_df.groupby("model_type")["model_memory_mb"].first()
        else:
            model_mem_values = sd15_df.groupby("model_type")["peak_memory_mb"].min()

        ax_bar = axes[0]
        bar_colors = [MODEL_COLORS.get(mt, "#7f8c8d") for mt in model_mem_values.index]
        bars = ax_bar.bar(
            [MODEL_LABELS.get(mt, mt) for mt in model_mem_values.index],
            model_mem_values.values,
            color=bar_colors,
            edgecolor="white",
            linewidth=1.2,
        )
        for bar, val in zip(bars, model_mem_values.values):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 10,
                f"{val:.0f} MB",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )
        ax_bar.set_ylabel("Memory (MB)")
        ax_bar.set_title("모델 로딩 후 VRAM 사용량")
        ax_bar.grid(axis="y", linestyle="--", alpha=0.5)
        ax_bar.set_ylim(0, model_mem_values.max() * 1.2)

        # Right: Peak inference memory per prompt
        ax_line = axes[1]
        for model_type in sd15_df["model_type"].unique():
            mt_data = sd15_df[sd15_df["model_type"] == model_type].sort_values("prompt_idx")
            ax_line.plot(
                mt_data["prompt_idx"].values,
                mt_data["peak_memory_mb"].values,
                marker="o",
                label=MODEL_LABELS.get(model_type, model_type),
                color=MODEL_COLORS.get(model_type, "#7f8c8d"),
                linewidth=2,
                markersize=6,
            )
        ax_line.set_xticks(range(1, len(prompts) + 1))
        ax_line.set_xlabel("Prompt Index")
        ax_line.set_ylabel("Peak VRAM (MB)")
        ax_line.set_title("추론 중 피크 VRAM 사용량 (프롬프트별)")
        ax_line.legend()
        ax_line.grid(True, linestyle="--", alpha=0.5)

        plt.suptitle("SD1.5 Base vs Quantized: 메모리 사용량 비교", fontsize=14, fontweight="bold")
        plt.tight_layout()

        output_path = self.output_dir / "memory_usage_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"메모리 사용량 비교 그래프 저장: {output_path}")

    def create_all_plots(self, df: pd.DataFrame, prompts: list[str]) -> None:
        """
        Create all visualization plots.

        SD15 models compare memory usage; SDXL models compare inference time and load time.

        Args:
            df: DataFrame with benchmark results
            prompts: List of prompts used
        """
        print("\n=== 시각화 생성 중 ===")

        sdxl_df = df[~df["model_type"].isin(["base", "dobby"])]
        if not sdxl_df.empty:
            self.plot_model_load_time_comparison(sdxl_df)
            self.plot_inference_time_comparison(sdxl_df, prompts)

        self.plot_sd15_memory_comparison(df, prompts)

        print("=== 모든 시각화 완료 ===\n")
