"""Settings and configuration constants for the benchmark."""

from typing import Dict

# Base models to benchmark
BASE_MODELS: Dict[str, str] = {
    # "base": "stabilityai/stable-diffusion-xl-base-1.0",
    # "animagine": "cagliostrolab/animagine-xl-4.0",
    "novaAnimeXL": "frankjoshua/novaAnimeXL_ilV140",
}

# LCM checkpoint paths for each base model
LCM_CHECKPOINT_PATHS: Dict[str, str] = {
    # "base": "data_fp8/sdxl-base-fp8/",
    # "animagine": "data_fp8/animagine_fp8/",
    "novaAnimeXL": "dobby-canvas/dobby-model",
}

# General quality prompt
GENERAL_PROMPT: str = ("master piece, best quality, high resolution, 4k, detailed background, "
                       "intricate details, vibrant colors, sharp focus, cinematic lighting, "
                       "professional composition, award-winning photography")

# Test prompts
PROMPTS: list[str] = [
    "Cute animated girl, blue hair, big eyes, bright smile, sky blue dress",
    "Fantasy Knight Animated Character, Silver Armor, Sword, Blonde Hair, Dynamic Pose",
    "Cyborg boy in future city background, red eyes, mechanical arm, Any style",
    "Wizard Girl, Long Purple Hair, Magic Wands, Star Patterned Cloak, Animated",
    "Girl with animal ears, cat ears, pink hair, animation style",
    "A vibrant boy in a tracksuit, short brown hair, basketball, animation",
    "A beautiful girl in a Gothic dress, black hair, red eyes, animation",
    "A boy in a pirate costume, an eye patch, short silver hair, sea background, animation",
    "Fairy character, little wings, light green hair, forest background, animation",
    "Girl in traditional hanbok, black long hair, fan, animation style",
]

# Inference settings
TEACHER_STEPS: int = 20
LCM_STEPS: int = 4
GUIDANCE_SCALE: float = 8.0
SEED: int = 0
