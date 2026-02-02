import os
import time
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, LCMScheduler
import pandas as pd
import matplotlib.pyplot as plt

GENERAL_PROMPT = "master piece, best quality, high resolution, 4k, detailed background, intricate details, vibrant colors, sharp focus, cinematic lighting, professional composition, award-winning photography"

PROMPTS = [
    "Cute animated girl, blue hair, big eyes, bright smile, sky blue dress",
    "Fantasy Knight Animated Character, Silver Armor, Sword, Blonde Hair, Dynamic Pose",
    "Cyborg boy in future city background, red eyes, mechanical arm, Any style",
    "Wizard Girl, Long Purple Hair, Magic Wands, Star Patterned Cloak, Animated",
    "Girl with animal ears, cat ears, pink hair, animation style",
    "A vibrant boy in a tracksuit, short brown hair, basketball, animation",
    "A beautiful girl in a Gothic dress, black hair, red eyes, animation",
    "A boy in a pirate costume, an eye patch, short silver hair, sea background, animation",
    "Fairy character, little wings, light green hair, forest background, animation",
    "Girl in traditional hanbok, black long hair, fan, animation style"
]

BASE_MODEL = "frankjoshua/novaAnimeXL_ilV140" # "cagliostrolab/animagine-xl-4.0" #"stabilityai/stable-diffusion-xl-base-1.0"

MODEL_CONFIGS = [
    {
        "name": "teacher",
        "desc": "SDXL (teacher)",
        "pipe_args": {
            "pretrained_model_name_or_path": BASE_MODEL,
            "torch_dtype": torch.float16,
            # "variant": "fp16",
        },
        "unet": None,
        "num_inference_steps": 20,
        "scheduler": None,
    },
    {
        "name": "lcm",
        "desc": "LCM (latent-consistency/lcm-sdxl)",
        "pipe_args": {
            "pretrained_model_name_or_path": BASE_MODEL,
            "torch_dtype": torch.float16,
            # "variant": "fp16",
        },
        "unet": {
            "pretrained_model_name_or_path": "latent-consistency/lcm-sdxl",
            "torch_dtype": torch.float16,
            # "variant": "fp16",
        },
        "num_inference_steps": 4,
        "scheduler": "lcm",
    },
    {
        "name": "lcm_finetuned",
        "desc": "LCM Fine-tuned (내가 학습한 모델)",
        "pipe_args": {
            "pretrained_model_name_or_path": BASE_MODEL,
            "torch_dtype": torch.float16,
            # "variant": "fp16",  # safetensors 사용시 variant 생략
            "use_safetensors": True,
        },
        "unet": {
            "pretrained_model_name_or_path": "results/novaAnimeXL_iV140-checkpoint3000/",
            "torch_dtype": torch.float16,
            # "variant": "fp16",
        },
        "num_inference_steps": 4,
        "scheduler": "lcm",
    },
]

RESULT_DIR = f"results/validation/{BASE_MODEL.split('/')[-1]}"
os.makedirs(RESULT_DIR, exist_ok=True)


def load_pipe(model_cfg):
    """모델 config에 따라 pipeline을 생성한다."""
    unet = None
    if model_cfg["unet"] is not None:
        unet = UNet2DConditionModel.from_pretrained(**model_cfg["unet"])
    if unet is not None:
        pipe = StableDiffusionXLPipeline.from_pretrained(unet=unet, **model_cfg["pipe_args"]).to("cuda")
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(**model_cfg["pipe_args"]).to("cuda")
    if model_cfg["scheduler"] == "lcm":
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    return pipe


def run_inference(pipe, prompt, num_inference_steps, seed=0, guidance_scale=8.0):
    generator = torch.manual_seed(seed)
    start_time = time.time()
    image = pipe(prompt=GENERAL_PROMPT + ", " + prompt,
                 num_inference_steps=num_inference_steps,
                 generator=generator,
                 guidance_scale=guidance_scale).images[0]
    end_time = time.time()
    torch.cuda.empty_cache()
    return image, end_time - start_time


def main():
    # 프롬프트별 모델별 생성 이미지 비교 Grid Plot
    from PIL import Image

    results = []
    for model_cfg in MODEL_CONFIGS:
        print(f"모델 로딩: {model_cfg['desc']}")
        pipe = load_pipe(model_cfg)
        for idx, prompt in enumerate(PROMPTS):
            print(f"[{model_cfg['name']}] 프롬프트 {idx+1}/{len(PROMPTS)}: {prompt}")
            image, elapsed = run_inference(pipe, prompt, model_cfg["num_inference_steps"])
            img_path = os.path.join(RESULT_DIR, f"{idx+1:02d}_{model_cfg['name']}.png")
            image.save(img_path)
            results.append({
                "prompt_idx": idx + 1,
                "prompt": prompt,
                "model": model_cfg["name"],
                "model_desc": model_cfg["desc"],
                "image_path": img_path,
                "inference_time": elapsed,
            })
    # 결과 DataFrame 저장
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULT_DIR, "inference_results.csv"), index=False)

    # 모델별 프롬프트별 Inference Time 비교 Plot
    plt.figure(figsize=(14, 6))
    for model_name in df["model"].unique():
        times = df[df["model"] == model_name]["inference_time"].values
        plt.plot(range(1, len(PROMPTS) + 1), times, marker='o', label=model_name)
    plt.xticks(range(1, len(PROMPTS) + 1))
    plt.xlabel("Prompt Index")
    plt.ylabel("Inference Time (s)")
    plt.title("모델별 프롬프트별 Inference Time 비교")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "inference_time_comparison.png"))

    # 프롬프트별 모델별 Inference Time bar plot
    num_prompts = len(PROMPTS)
    model_names = df["model"].unique()
    x = range(len(model_names))
    plt.figure(figsize=(18, 2.5 * num_prompts))
    for idx in range(num_prompts):
        plt.subplot(num_prompts, 1, idx + 1)
        times = [
            df[(df["prompt_idx"] == idx + 1) & (df["model"] == model)]["inference_time"].values[0]
            for model in model_names
        ]
        plt.bar(x, times, tick_label=model_names)
        plt.ylabel("Inference Time (s)")
        plt.title(f"Prompt {idx + 1}: {PROMPTS[idx]}")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "inference_time_by_prompt.png"))
    print(f"[완료] prompt별 모델별 inference time plot 저장: {RESULT_DIR}/inference_time_by_prompt.png")

    df_sorted = df.sort_values(["prompt_idx", "model"]).reset_index(drop=True)
    num_prompts = len(PROMPTS)
    model_names = df["model"].unique()
    num_models = len(model_names)
    fig, axes = plt.subplots(num_prompts, num_models, figsize=(4 * num_models, 3 * num_prompts))
    for i in range(num_prompts):
        for j, model in enumerate(model_names):
            row = df_sorted[(df_sorted["prompt_idx"] == i + 1) & (df_sorted["model"] == model)]
            img_path = row["image_path"].values[0]
            img = Image.open(img_path)
            ax = axes[i, j] if num_prompts > 1 else axes[j]
            ax.imshow(img)
            ax.axis('off')
            if i == 0:
                ax.set_title(model, fontsize=14)
            if j == 0:
                ax.set_ylabel(f"Prompt {i+1}", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "generated_images_grid.png"))
    print(f"[완료] 프롬프트별 모델별 생성 이미지 grid plot 저장: {RESULT_DIR}/generated_images_grid.png")


if __name__ == "__main__":
    main()
