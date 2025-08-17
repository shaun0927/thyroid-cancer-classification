# 🏥 갑상선암 분류 AI 챌린지
## Dacon 대회 - 갑상선 결절의 이진 분류

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Competition](https://img.shields.io/badge/Dacon-Competition-orange)](https://dacon.io/)

## 📋 대회 개요

이 레포지토리는 **Dacon 갑상선암 분류 해커톤**에 대한 저의 솔루션을 담고 있습니다. 이 대회의 목표는 환자 건강 데이터를 사용하여 양성과 악성 갑상선 결절을 정확하게 구별하는 AI 모델을 개발하는 것입니다.

### 🎯 챌린지 설명
- **목표**: 갑상선암의 이진 분류 (양성 vs 악성)
- **평가 지표**: Binary F1 Score
- **데이터셋 크기**: 
  - 학습 데이터: 87,160개 샘플
  - 테스트 데이터: 58,107개 샘플
- **클래스 분포**: 심한 불균형 (악성 케이스 약 12%)
- **리더보드 분할**: 30% 공개 / 70% 비공개

### 🏆 최종 성능
- **CV F1 Score**: 0.51+ (5-Fold Stratified)
- **최고 단일 모델**: 최적화된 임계값을 사용한 CatBoost
- **접근법**: 임계값 최적화를 적용한 4개 그래디언트 부스팅 모델의 앙상블

## 📊 데이터셋 설명

### 특성 (총 14개)
| 카테고리 | 특성 | 설명 |
|----------|----------|-------------|
| **인구통계** | Age, Gender, Country, Race | 환자 인구통계 정보 |
| **병력** | Family_Background, Radiation_History, Iodine_Deficiency | 관련 의료 배경 |
| **생활습관** | Smoke, Weight_Risk, Diabetes | 생활습관 및 건강 위험 요인 |
| **임상 검사** | Nodule_Size, TSH_Result, T4_Result, T3_Result | 갑상선 특이 측정값 |

### 타겟 변수
- **Cancer**: 이진 분류 (0 = 양성, 1 = 악성)

## 🚀 솔루션 접근법

### 1. 데이터 전처리 파이프라인
```python
# 특성 엔지니어링 전략
- 이진 인코딩: Gender, Family_Background, Radiation_History 등
- 레이블 인코딩: Country, Race (다중 카테고리 특성)
- StandardScaler: 수치형 특성 (Age, Nodule_Size, 호르몬 레벨)
- 결측값 처리 전략: 매핑되지 않은 범주형 값에 대해 -1 사용
```

### 2. 모델 아키텍처

#### 개별 모델 성능 (5-Fold CV)
| 모델 | F1 Score | 최적 임계값 | 주요 파라미터 |
|-------|----------|------------------|----------------|
| **CatBoost** | 0.4900 | 0.245 | depth=8, iterations=1200, GPU 가속 |
| **GBDT** | 0.4865 | 0.210 | n_estimators=600, max_depth=5 |
| **RandomForest** | 0.4830 | 0.280 | n_estimators=1000, ROS 적용 |
| **LightGBM** | 0.4802 | 0.245 | num_leaves=127, learning_rate=0.05 |

#### 앙상블 전략
- **방법**: 소프트 보팅 (확률 평균)
- **최종 임계값**: 전체 학습 세트에서 최적화
- **결과**: F1 Score > 0.51

### 3. 클래스 불균형 해결을 위한 핵심 기법

1. **테스트한 리샘플링 방법**:
   - RandomOverSampler (ROS) - RandomForest에 최적
   - SMOTE 변형
   - 클래스 가중치 균형화

2. **임계값 최적화**:
   ```python
   # 불균형 데이터에서 F1 최대화에 중요
   precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
   f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
   optimal_threshold = thresholds[f1_scores.argmax()]
   ```

3. **Scale Position Weight**: 그래디언트 부스팅 방법용

## 📁 레포지토리 구조

```
thyroid-cancer-classification/
│
├── 📂 src/                    # Source code
│   ├── preprocessing.py       # 데이터 전처리 파이프라인
│   ├── models.py              # 모델 정의 및 학습
│   ├── ensemble.py            # 앙상블 방법
│   └── utils.py               # 유틸리티 함수
│
├── 📂 notebooks/              # Jupyter notebooks
│   ├── EDA.ipynb             # 탐색적 데이터 분석
│   ├── pipeline.ipynb        # 메인 학습 파이프라인
│   └── experiments.ipynb     # 모델 실험
│
├── 📂 data/                   # 데이터 디렉토리 (포함되지 않음)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── 📂 models/                 # 저장된 모델
│   └── best_ensemble.pkl
│
├── 📂 results/                # 출력 결과
│   └── submission.csv
│
├── requirements.txt           # 의존성
├── README.md                 # 이 파일
└── LICENSE                   # MIT 라이선스
```

## 🔧 설치 및 설정

### 필수 요구사항
- Python 3.8+
- CUDA 지원 GPU (CatBoost용 권장)
- 16GB+ RAM

### 설치
```bash
# 레포지토리 클론
git clone https://github.com/shaun0927/thyroid-cancer-classification.git
cd thyroid-cancer-classification

# 가상 환경 생성
python -m venv venv
source venv/bin/activate  # Windows에서: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

## 🏃‍♂️ 사용법

### 빠른 시작
```bash
# 1. 데이터 준비
python src/preprocessing.py --input data/train.csv --output data/processed_train.csv

# 2. 모델 학습
python src/train.py --config configs/best_params.yaml

# 3. 예측 생성
python src/predict.py --model models/best_ensemble.pkl --test data/test.csv
```

### 결과 재현
```python
# 전체 파이프라인 실행
python run_pipeline.py --mode full --cv_folds 5 --optimize_threshold
```

### Jupyter 노트북
```bash
jupyter notebook notebooks/pipeline.ipynb
```

## 📈 모델 학습 세부사항

### 하이퍼파라미터 최적화
- **방법**: TPE 샘플러를 사용한 Optuna
- **전략**: 중첩 CV (5 외부 × 3 내부 폴드)
- **시행 횟수**: 모델당 100-300회
- **목표**: Binary F1 Score 최대화

### 최적 하이퍼파라미터

#### CatBoost (최고 모델)
```python
{
    'iterations': 1200,
    'depth': 8,
    'learning_rate': 0.05,
    'l2_leaf_reg': 3.0,
    'scale_pos_weight': 7.3,
    'task_type': 'GPU'
}
```

#### RandomForest (ROS 적용)
```python
{
    'n_estimators': 1000,
    'max_depth': 25,
    'min_samples_leaf': 15,
    'class_weight': 'balanced_subsample'
}
```

## 🔬 핵심 인사이트

1. **임계값 최적화가 중요**: 불균형 데이터에서 기본 0.5 임계값은 성능이 좋지 않음
2. **앙상블 다양성**: 다른 모델 유형 결합으로 견고성 향상
3. **리샘플링 영향**: RandomOverSampler가 트리 기반 모델을 크게 개선
4. **특성 중요도**: 임상 측정값(TSH, T3, T4)과 Nodule_Size가 가장 예측력이 높음

## 📊 검증 전략

```python
# 클래스 분포 유지를 위한 Stratified K-Fold
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 양성 클래스용 커스텀 F1 스코어러
from sklearn.metrics import make_scorer, f1_score
f1_scorer = make_scorer(f1_score, pos_label=1, average='binary')
```

## 🛠️ 사용 기술

- **핵심 라이브러리**: pandas, numpy, scikit-learn
- **그래디언트 부스팅**: XGBoost, LightGBM, CatBoost
- **하이퍼파라미터 튜닝**: Optuna
- **불균형 학습**: imbalanced-learn
- **시각화**: matplotlib, seaborn

## 📝 배운 점

1. **클래스 불균형 처리**: 다중 전략 필요 (리샘플링 + 임계값 + 클래스 가중치)
2. **교차 검증의 중요성**: 일반화를 위한 견고한 CV 전략 필수
3. **앙상블의 이점**: 단순 평균이 복잡한 스태킹보다 종종 더 나은 성능
4. **GPU 가속**: CatBoost GPU 모드가 5-10배 속도 향상 제공

---

**참고**: 라이선스 제한으로 인해 대회 데이터 파일은 이 레포지토리에 포함되어 있지 않습니다. 공식 Dacon 대회 페이지에서 다운로드해 주세요.
