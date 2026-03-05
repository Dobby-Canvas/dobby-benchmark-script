# Dobby Model Benchmark Script

Dobby 모델의 **속도(Speed)** 및 **메모리(Memory)** 성능을 비교하는 벤치마크 스크립트입니다.


## 🚀 주요 기능

### 1. 실험 종류

| 실험 | 모델 아키텍처 | 측정 항목 | 비교 대상 |
|------|-------------|----------|----------|
| **Speed Experiment** | SDXL | 이미지 생성 시간 (inference time) | Base vs Dobby |
| **Memory Experiment** | SD1.5 | GPU 메모리 사용량 | Base vs Dobby  |

### 2. 측정 항목

- ✅ **이미지 생성 시간 (Speed)**: 프롬프트당 이미지 생성 소요 시간
- ✅ **모델 메모리 (Memory)**: 모델 로드 시 GPU 메모리 사용량
- ✅ **피크 메모리 (Memory)**: inference 중 최대 GPU 메모리 사용량
- ✅ **CSV 자동 저장**: 실험별로 적합한 컬럼만 선택해 CSV로 저장

### 3. 자동 시각화

- 모델별 inference 시간 비교 라인 차트
- 모델별 GPU 메모리 사용량 비교 그래프
- 생성된 이미지 그리드 (모델별 × 프롬프트별)


## 📋 필수 요구사항

```
Python == 3.10
PyTorch == 2.2.1 (CUDA 12.1)
diffusers == 0.26.3
transformers == 4.38.2
torchao
mixdq-extension == 0.6
pandas, matplotlib, Pillow, numpy, accelerate, huggingface-hub
```

> 전체 의존성은 [pyproject.toml](pyproject.toml)을 참조하세요.


## 🏃 실행 방법

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.10 .venv

source .venv/bin/activate

# 의존성 설치
uv sync

# 벤치마크 실행
python -m src.main
```


## ⚙️ 설정 방법

[src/config/settings.py](src/config/settings.py)에서 모델 경로, 프롬프트, inference 스텝 수를 수정할 수 있습니다.

### 사용 모델

| 실험 | 모델 타입 | HuggingFace 경로 |
|------|---------|----------------|
| Speed | Base (SDXL) | `frankjoshua/novaAnimeXL_ilV140` |
| Speed | Dobby (LCM) | `dobby-canvas/dobby-model` |
| Memory | Base (SD1.5) | `uf/Counterfeit-V3.0` |
| Memory | Dobby (Quantized) | `dobby-canvas/dobby-model` |

### 프롬프트 커스터마이징

```python
PROMPTS: list[str] = [
    "Cute animated girl, blue hair, big eyes, bright smile, sky blue dress",
    # ... 더 많은 프롬프트 추가
]
```

### Inference 스텝 수

```python
TEACHER_STEPS: int = 20  # Base 모델 (Speed & Memory 실험)
LCM_STEPS: int = 4       # Dobby LCM 모델 (Speed 실험)
```


## 📊 출력 결과

모든 결과는 `results/` 디렉토리에 저장됩니다.

### 1. CSV 파일

**Speed Experiment** (`result_*_speed_experiment_*.csv`):

| 컬럼 | 설명 |
|------|------|
| `prompt_idx` | 프롬프트 인덱스 |
| `prompt` | 사용된 프롬프트 텍스트 |
| `base_model_key` | 실험 키 (`speed_experiment`) |
| `model_name` | 모델 이름 |
| `model_type` | 모델 타입 (`base` / `dobby`) |
| `image_path` | 생성된 이미지 경로 |
| `inference_time` | 이미지 생성 시간 (초) |

**Memory Experiment** (`result_*_memory_experiment_*.csv`):

| 컬럼 | 설명 |
|------|------|
| `prompt_idx` | 프롬프트 인덱스 |
| `prompt` | 사용된 프롬프트 텍스트 |
| `base_model_key` | 실험 키 (`memory_experiment`) |
| `model_name` | 모델 이름 |
| `model_type` | 모델 타입 (`base_memory` / `dobby_memory`) |
| `image_path` | 생성된 이미지 경로 |
| `model_memory_mb` | 모델 로드 시 GPU 메모리 (MB) |
| `peak_memory_mb` | inference 중 최대 GPU 메모리 (MB) |


### 2. 생성 이미지

- `{prompt_idx:02d}_{model_name}.png` 형식으로 저장
