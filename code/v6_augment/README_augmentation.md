# 데이터 증강 및 TTA 기능 사용 가이드

본 문서는 v6_augment 버전에서 추가된 데이터 증강 및 TTA(Test Time Augmentation) 기능의 사용법을 설명합니다.

## 🚀 주요 기능

- **훈련 데이터 증강**: 훈련 중 실시간 데이터 증강
- **검증 데이터 증강**: 검증 데이터에 대한 증강
- **검증 TTA**: 검증 시 여러 증강 변형을 사용한 앙상블 예측
- **테스트 TTA**: 최종 추론 시 TTA를 통한 성능 향상
- **다중 증강 라이브러리 지원**: Albumentations, Augraphy, 혼합 사용
- **단계별 증강 이미지 개수 조절**: 훈련과 검증에서 각각 다른 증강 개수 설정 가능
- **유연한 설정**: 각 단계(train/valid/valid_tta/test_tta)를 독립적으로 제어

## 📋 설정 방법

### 기본 설정 구조

```yaml
augmentation:
  enabled: true  # 전체 증강 활성화/비활성화
  library: "albumentations"  # "albumentations", "augraphy", "mixed", "none"
  
  # 각 단계별 증강 설정
  train:
    enabled: true    # 훈련 데이터 증강
    num_augmented_images: 2  # 훈련용 증강 이미지 생성 개수
  
  valid:
    enabled: false   # 검증 데이터 증강
    num_augmented_images: 1  # 검증용 증강 이미지 생성 개수
  
  valid_tta:
    enabled: true    # 검증 시 TTA 사용
  
  test_tta:
    enabled: true    # 테스트 시 TTA 사용
```

### 증강 라이브러리 옵션

1. **`albumentations`**: 일반적인 이미지 증강 기법
   - 회전, 뒤집기, 밝기/대비 조정
   - 노이즈 추가, 블러 효과
   - 기하학적 변형

2. **`augraphy`**: 문서 이미지 특화 증강
   - 잉크 번짐, 투과 효과
   - 문서 노이즈, 페이지 테두리
   - 스캔 품질 시뮬레이션

3. **`mixed`**: 두 라이브러리 혼합
   - 랜덤하게 albumentations 또는 augraphy 적용
   - 다양한 증강 효과 조합

4. **`none`**: 증강 비활성화

### 증강 이미지 생성 개수

**훈련/검증 단계별로 독립적으로 설정 가능:**

- `train.num_augmented_images: 1`: 훈련 시 원본 + 증강 1개 = 총 2배
- `valid.num_augmented_images: 1`: 검증 시 원본 + 증강 1개 = 총 2배

**예시:**
- 훈련: `num_augmented_images: 3` → 원본 + 증강 3개 = 총 4배
- 검증: `num_augmented_images: 1` → 원본 + 증강 1개 = 총 2배

## 🎯 사용 시나리오

### 1. 기본 훈련 증강만 사용

```yaml
# config/basic_augmentation.yaml
defaults:
  - config

augmentation:
  enabled: true
  library: "albumentations"
  
  train:
    enabled: true
    num_augmented_images: 1
  valid:
    enabled: false
    num_augmented_images: 1
  valid_tta:
    enabled: false
  test_tta:
    enabled: false
```

### 2. TTA를 활용한 고성능 설정

```yaml
# config/high_performance.yaml
defaults:
  - config

augmentation:
  enabled: true
  library: "mixed"
  
  train:
    enabled: true
    num_augmented_images: 3  # 훈련에서는 많은 증강 사용
  valid:
    enabled: false
    num_augmented_images: 1  # 검증에서는 적은 증강 사용
  valid_tta:
    enabled: true
  test_tta:
    enabled: true
```

### 3. 문서 이미지 특화 설정

```yaml
# config/document_specific.yaml
defaults:
  - config

augmentation:
  enabled: true
  library: "augraphy"
  
  train:
    enabled: true
    num_augmented_images: 2  # 문서 특화 증강 적용
  valid:
    enabled: false
    num_augmented_images: 1
  valid_tta:
    enabled: false
  test_tta:
    enabled: true
```

## 🔧 실행 방법

### 1. 기본 실행

```bash
# 기본 설정으로 실행
uv run python main.py

# 특정 설정 파일로 실행
uv run python main.py --config-name=augmentation_test

# 설정 오버라이드
uv run python main.py augmentation.library=mixed augmentation.train.num_augmented_images=4 augmentation.valid.num_augmented_images=2
```

### 2. 사전 정의된 설정 파일들

```bash
# Albumentations 테스트
uv run python main.py --config-name=augmentation_test

# Augraphy 테스트
uv run python main.py --config-name=augmentation_augraphy

# 혼합 증강 테스트
uv run python main.py --config-name=augmentation_mixed

# 증강 비활성화 (베이스라인)
uv run python main.py --config-name=no_augmentation

# 훈련과 검증에서 다른 증강 개수 사용 예시
uv run python main.py augmentation.train.num_augmented_images=5 augmentation.valid.num_augmented_images=2
```

## 🧪 테스트 실행

### 단위 테스트

```bash
# 증강 기능 테스트
uv run pytest tests/test_augmentations.py -v

# 통합 테스트
uv run pytest tests/test_integration.py -v

# 전체 테스트
uv run pytest tests/ -v
```

### 테스트 커버리지

```bash
# 커버리지 리포트 생성
uv run pytest tests/ --cov=. --cov-report=html

# 커버리지 리포트 확인
open htmlcov/index.html
```

## 📊 성능 영향

### 훈련 시간 영향

- **증강 없음**: 기준 시간
- **훈련 증강 활성화**: 설정한 `train.num_augmented_images`에 비례하여 증가
  - `num_augmented_images=1`: 약 2배 증가
  - `num_augmented_images=2`: 약 3배 증가
  - `num_augmented_images=3`: 약 4배 증가
- **검증 증강 활성화**: 설정한 `valid.num_augmented_images`에 비례하여 검증 시간 증가
- **TTA 활성화**: 검증/추론 시간 5배 증가 (5개 변형 사용)

### 메모리 사용량

- 데이터 로더의 배치 크기에 따라 메모리 사용량 증가
- TTA 사용 시 임시 메모리 사용량 증가
- `num_workers=0`으로 설정하여 메모리 안정성 확보

### 성능 향상 기대치

- **훈련 증강**: 일반적으로 3-10% 성능 향상
- **TTA**: 추론 시 1-3% 추가 성능 향상
- **문서 특화 증강**: 문서 이미지에서 5-15% 성능 향상 가능

## 🛠️ 커스터마이징

### 새로운 증강 기법 추가

1. `augmentations.py`에서 증강 함수 수정:

```python
def get_custom_augmentation(cfg: AugmentationConfig, is_train: bool = True) -> A.Compose:
    """커스텀 증강 변환 반환"""
    if is_train:
        augmentation = A.Compose([
            # 여기에 새로운 증강 기법 추가
            A.CustomTransform(p=0.5),
            # ... 기존 변환들
        ])
    return augmentation
```

2. `AugmentedDataset.setup_augmentations()`에서 새 라이브러리 옵션 추가

### TTA 변형 수정

`get_tta_transforms()` 함수에서 TTA 변형들을 수정할 수 있습니다:

```python
def get_tta_transforms(cfg: AugmentationConfig) -> List[A.Compose]:
    tta_transforms = [
        # 원본
        basic_transform,
        # 커스텀 변형 추가
        custom_transform,
        # ... 기존 변형들
    ]
    return tta_transforms
```

## 🐛 문제 해결

### 일반적인 문제들

1. **Augraphy 설치 문제**
   ```bash
   # Augraphy 설치
   uv add augraphy
   ```

2. **메모리 부족**
   - `batch_size` 감소
   - `train.num_augmented_images` 감소
   - `valid.num_augmented_images` 감소
   - `num_workers=0` 설정

3. **느린 훈련 속도**
   - 간단한 증강 기법만 사용
   - `train.num_augmented_images=1`로 설정
   - `valid.num_augmented_images=1`로 설정
   - TTA 비활성화

4. **라이브러리 호환성 문제**
   - PyTorch 버전 확인
   - Albumentations 버전 확인
   - Python 3.8+ 권장

### 로그 분석

증강 기능이 올바르게 동작하는지 로그를 통해 확인:

```
=== 데이터 준비 ===
증강된 데이터셋 생성: 원본 100개 -> 총 200개
TTA 데이터셋 생성: 원본 50개 -> 총 250개 (TTA x5)
```

## 📈 최적화 팁

1. **점진적 적용**: 처음에는 간단한 증강부터 시작
2. **실험 추적**: Wandb를 활용한 실험 결과 비교
3. **데이터 특성 고려**: 이미지 종류에 맞는 증강 기법 선택
4. **계산 자원 고려**: GPU 메모리와 훈련 시간의 균형점 찾기

## 📚 참고 자료

- [Albumentations 문서](https://albumentations.ai/)
- [Augraphy 문서](https://augraphy.readthedocs.io/)
- [TTA 기법 논문](https://arxiv.org/abs/1505.07818)
- [데이터 증강 가이드](https://www.v7labs.com/blog/data-augmentation-guide)

---

더 자세한 정보나 문제가 있으면 이슈를 등록해 주세요! 🚀 