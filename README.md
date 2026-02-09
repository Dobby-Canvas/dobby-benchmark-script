# Dobby Model Benchmark Script

SDXL 모델과 Dobby 모델의 성능을 비교하는 벤치마크 스크립트입니다.


## 🚀 주요 기능

### 1. 성능 측정 항목
- ✅ **모델 로딩 시간**: 각 모델이 GPU 메모리에 로드되는 시간
- ✅ **이미지 생성 시간**: 프롬프트당 이미지 생성 소요 시간
- ✅ **CSV 자동 저장**: 모든 측정 데이터를 CSV로 저장

### 2. 자동 시각화
- 모델별 로딩 시간 비교 그래프
- 모델별 inference 시간 비교 라인 차트
- 프롬프트별 inference 시간 막대 그래프
- 생성된 이미지 그리드 (모델별 × 프롬프트별)

## 📋 필수 요구사항

```bash
# Python 3.10+
# PyTorch with CUDA support
# HuggingFace diffusers
# 기타 의존성은 pyproject.toml 참조
```

## 🏃 실행 방법

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.13 .venv

source .venv/bin/activate

# 의존성 설치
uv sync

# 벤치마크 실행
python -m src.main
```

## ⚙️ 설정 방법

### 프롬프트 커스터마이징

[src/config/settings.py](src/config/settings.py)에서 테스트할 프롬프트를 수정할 수 있습니다:

```python
PROMPTS: list[str] = [
    "Cute animated girl, blue hair, big eyes, bright smile, sky blue dress",
    # ... 더 많은 프롬프트 추가
]
```

## 📊 출력 결과

모든 결과는 `results/` 디렉토리에 저장됩니다:

### 1. CSV 파일
- **benchmark_results.csv**: 모든 측정 데이터
  - `prompt_idx`: 프롬프트 인덱스
  - `prompt`: 사용된 프롬프트 텍스트
  - `base_model_key`: base 모델 키 (base/animagine/novaAnimeXL)
  - `model_name`: 모델 이름
  - `model_type`: 모델 타입 (base/dobby)
  - `image_path`: 생성된 이미지 경로
  - `model_load_time`: 모델 로딩 시간 (초)
  - `inference_time`: 이미지 생성 시간 (초)

### 2. 시각화 이미지
- **model_load_time_comparison.png**: 모델별 로딩 시간 비교
- **inference_time_comparison.png**: 모델별 inference 시간 라인 차트


### 3. 생성 이미지
- `{prompt_idx:02d}_{model_name}.png` 형식으로 저장