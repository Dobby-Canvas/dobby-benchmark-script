"""Configuration module for benchmark script."""
from .settings import (BASE_MODELS, GENERAL_PROMPT, GUIDANCE_SCALE, LCM_CHECKPOINT_PATHS, LCM_STEPS, PROMPTS, SEED,
                       TEACHER_STEPS)

__all__ = [
    "BASE_MODELS",
    "GENERAL_PROMPT",
    "PROMPTS",
    "LCM_CHECKPOINT_PATHS",
    "TEACHER_STEPS",
    "LCM_STEPS",
    "GUIDANCE_SCALE",
    "SEED",
]
