# SDXL Benchmark Script

Stable Diffusion XL의 Teacher 모델과 LCM Fine-tuned 모델의 성능을 비교하는 벤치마크 스크립트입니다.

## 📁 프로젝트 구조

```
dobby-benchmark-script/
├── src/
│   ├── config/              # 설정 파일
│   │   ├── __init__.py
│   │   └── settings.py      # 모델 경로, 프롬프트, 상수 정의
│   ├── models/              # 모델 로딩
│   │   ├── __init__.py
│   │   └── loader.py        # 모델 로더 (로딩 시간 측정 포함)
│   ├── benchmark/           # 벤치마크 실행
│   │   ├── __init__.py
│   │   └── runner.py        # 추론 실행 및 시간 측정
│   ├── visualization/       # 시각화
│   │   ├── __init__.py
│   │   └── plotter.py       # 그래프 및 이미지 그리드 생성
│   ├── utils/               # 유틸리티
│   │   └── __init__.py
│   ├── main.py              # 메인 실행 파일
│   └── script.py            # (레거시) 기존 스크립트
├── results/                 # 생성 결과 저장
└── pyproject.toml
```

## 🚀 주요 기능

### 1. 다중 Base Model 지원
- **stabilityai/stable-diffusion-xl-base-1.0** (base)
- **cagliostrolab/animagine-xl-4.0** (animagine)
- **frankjoshua/novaAnimeXL_ilV140** (novaAnimeXL)

### 2. 성능 측정 항목
- ✅ **모델 로딩 시간**: 각 모델이 GPU 메모리에 로드되는 시간
- ✅ **이미지 생성 시간**: 프롬프트당 이미지 생성 소요 시간
- ✅ **CSV 자동 저장**: 모든 측정 데이터를 CSV로 저장

### 3. 자동 시각화
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

## ⚙️ 설정 방법

### 1. LCM 체크포인트 경로 설정

[src/config/settings.py](src/config/settings.py) 파일에서 각 base model에 대응하는 LCM 체크포인트 경로를 설정하세요:

```python
LCM_CHECKPOINT_PATHS: Dict[str, str] = {
    "base": "results/base-checkpoint3000/",
    "animagine": "results/animagine-checkpoint3000/",
    "novaAnimeXL": "results/novaAnimeXL_iV140-checkpoint3000/",
}
```

### 2. 프롬프트 커스터마이징

[src/config/settings.py](src/config/settings.py)에서 테스트할 프롬프트를 수정할 수 있습니다:

```python
PROMPTS: list[str] = [
    "Cute animated girl, blue hair, big eyes, bright smile, sky blue dress",
    # ... 더 많은 프롬프트 추가
]
```

## 🏃 실행 방법

```bash
# 의존성 설치
uv sync

# 벤치마크 실행
python src/main.py
```

## 📊 출력 결과

모든 결과는 `results/validation/all_models/` 디렉토리에 저장됩니다:

### 1. CSV 파일
- **benchmark_results.csv**: 모든 측정 데이터
  - `prompt_idx`: 프롬프트 인덱스
  - `prompt`: 사용된 프롬프트 텍스트
  - `base_model_key`: base 모델 키 (base/animagine/novaAnimeXL)
  - `model_name`: 모델 이름 (예: base_teacher, base_lcm)
  - `model_type`: 모델 타입 (teacher/lcm)
  - `image_path`: 생성된 이미지 경로
  - `model_load_time`: 모델 로딩 시간 (초)
  - `inference_time`: 이미지 생성 시간 (초)

### 2. 시각화 이미지
- **model_load_time_comparison.png**: 모델별 로딩 시간 비교
- **inference_time_comparison.png**: 모델별 inference 시간 라인 차트
- **inference_time_by_prompt.png**: 프롬프트별 inference 시간 막대 그래프
- **generated_images_grid.png**: 생성된 이미지 그리드

### 3. 생성 이미지
- `{prompt_idx:02d}_{model_name}.png` 형식으로 저장

## 🔧 코드 구조 특징

### Clean Architecture 적용
- **관심사 분리**: 설정, 모델 로딩, 벤치마크 실행, 시각화를 별도 모듈로 분리
- **단일 책임 원칙**: 각 클래스와 함수는 하나의 명확한 역할만 수행
- **확장성**: 새로운 모델이나 벤치마크 추가가 용이한 구조

### 타입 힌팅
모든 함수와 클래스에 타입 힌팅 적용으로 코드 가독성과 안정성 향상

### Dataclass 활용
- `LoadedModel`: 로드된 모델과 메타데이터
- `InferenceResult`: 추론 결과와 측정 데이터

## 💡 사용 예시

```python
from config import BASE_MODELS, PROMPTS
from models import ModelLoader
from benchmark import BenchmarkRunner

# 벤치마크 러너 초기화
runner = BenchmarkRunner(output_dir="results/my_benchmark")

# 모델 로딩 (시간 자동 측정)
model = ModelLoader.load_teacher_model(
    base_model_key="animagine",
    base_model_path=BASE_MODELS["animagine"]
)

# 추론 실행 (시간 자동 측정)
result = runner.run_inference(
    loaded_model=model,
    prompt=PROMPTS[0],
    num_inference_steps=20,
    prompt_idx=1
)

# 결과 저장
df = runner.save_results()
```

## 🔍 주요 클래스 및 메서드

### ModelLoader
- `load_teacher_model()`: Teacher 모델 로딩
- `load_lcm_model()`: LCM Fine-tuned 모델 로딩
- `unload_model()`: 모델 언로드 및 메모리 정리

### BenchmarkRunner
- `run_inference()`: 단일 추론 실행 및 시간 측정
- `save_results()`: 결과를 CSV로 저장
- `get_results_dataframe()`: 결과를 DataFrame으로 반환

### ResultPlotter
- `plot_model_load_time_comparison()`: 모델 로딩 시간 비교
- `plot_inference_time_comparison()`: Inference 시간 비교
- `plot_inference_time_by_prompt()`: 프롬프트별 시간 비교
- `plot_generated_images_grid()`: 이미지 그리드 생성
- `create_all_plots()`: 모든 시각화 한번에 생성

## 📝 라이센스

MIT License
