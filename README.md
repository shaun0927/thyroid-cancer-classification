# 🏥 Thyroid Cancer Classification AI Challenge
## Dacon Competition - Binary Classification of Thyroid Nodules

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Competition](https://img.shields.io/badge/Dacon-Competition-orange)](https://dacon.io/)

## 📋 Competition Overview

This repository contains my solution for the **Dacon Thyroid Cancer Classification Hackathon**, where the goal is to develop an AI model that accurately distinguishes between benign and malignant thyroid nodules using patient health data.

### 🎯 Challenge Description
- **Objective**: Binary classification of thyroid cancer (benign vs malignant)
- **Evaluation Metric**: Binary F1 Score
- **Dataset Size**: 
  - Training: 87,160 samples
  - Test: 58,107 samples
- **Class Distribution**: Highly imbalanced (~12% malignant cases)
- **Leaderboard Split**: 30% Public / 70% Private

### 🏆 Final Performance
- **CV F1 Score**: 0.51+ (5-Fold Stratified)
- **Best Single Model**: CatBoost with optimized threshold
- **Approach**: Ensemble of 4 gradient boosting models with threshold optimization

## 📊 Dataset Description

### Features (14 total)
| Category | Features | Description |
|----------|----------|-------------|
| **Demographics** | Age, Gender, Country, Race | Patient demographic information |
| **Medical History** | Family_Background, Radiation_History, Iodine_Deficiency | Relevant medical background |
| **Lifestyle** | Smoke, Weight_Risk, Diabetes | Lifestyle and health risk factors |
| **Clinical Tests** | Nodule_Size, TSH_Result, T4_Result, T3_Result | Thyroid-specific measurements |

### Target Variable
- **Cancer**: Binary classification (0 = Benign, 1 = Malignant)

## 🚀 Solution Approach

### 1. Data Preprocessing Pipeline
```python
# Feature Engineering Strategy
- Binary Encoding: Gender, Family_Background, Radiation_History, etc.
- Label Encoding: Country, Race (multi-category features)
- StandardScaler: Numeric features (Age, Nodule_Size, hormone levels)
- Missing Value Strategy: -1 for unmapped categorical values
```

### 2. Model Architecture

#### Individual Models Performance (5-Fold CV)
| Model | F1 Score | Optimal Threshold | Key Parameters |
|-------|----------|------------------|----------------|
| **CatBoost** | 0.4900 | 0.245 | depth=8, iterations=1200, GPU-accelerated |
| **GBDT** | 0.4865 | 0.210 | n_estimators=600, max_depth=5 |
| **RandomForest** | 0.4830 | 0.280 | n_estimators=1000, with ROS |
| **LightGBM** | 0.4802 | 0.245 | num_leaves=127, learning_rate=0.05 |

#### Ensemble Strategy
- **Method**: Soft voting (average probabilities)
- **Final Threshold**: Optimized on full training set
- **Result**: F1 Score > 0.51

### 3. Key Techniques for Class Imbalance

1. **Resampling Methods Tested**:
   - RandomOverSampler (ROS) - Best for RandomForest
   - SMOTE variants
   - Class weight balancing

2. **Threshold Optimization**:
   ```python
   # Critical for maximizing F1 on imbalanced data
   precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
   f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
   optimal_threshold = thresholds[f1_scores.argmax()]
   ```

3. **Scale Position Weight**: For gradient boosting methods

## 📁 Repository Structure

```
thyroid-cancer-classification/
│
├── 📂 src/                    # Source code
│   ├── preprocessing.py       # Data preprocessing pipeline
│   ├── models.py              # Model definitions and training
│   ├── ensemble.py            # Ensemble methods
│   └── utils.py               # Utility functions
│
├── 📂 notebooks/              # Jupyter notebooks
│   ├── EDA.ipynb             # Exploratory Data Analysis
│   ├── pipeline.ipynb        # Main training pipeline
│   └── experiments.ipynb     # Model experiments
│
├── 📂 data/                   # Data directory (not included)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── 📂 models/                 # Saved models
│   └── best_ensemble.pkl
│
├── 📂 results/                # Outputs
│   └── submission.csv
│
├── requirements.txt           # Dependencies
├── README.md                 # This file
└── LICENSE                   # MIT License
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for CatBoost)
- 16GB+ RAM

### Installation
```bash
# Clone repository
git clone https://github.com/shaun0927/thyroid-cancer-classification.git
cd thyroid-cancer-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🏃‍♂️ Usage

### Quick Start
```bash
# 1. Prepare data
python src/preprocessing.py --input data/train.csv --output data/processed_train.csv

# 2. Train models
python src/train.py --config configs/best_params.yaml

# 3. Generate predictions
python src/predict.py --model models/best_ensemble.pkl --test data/test.csv
```

### Reproduce Results
```python
# Run complete pipeline
python run_pipeline.py --mode full --cv_folds 5 --optimize_threshold
```

### Jupyter Notebook
```bash
jupyter notebook notebooks/pipeline.ipynb
```

## 📈 Model Training Details

### Hyperparameter Optimization
- **Method**: Optuna with TPE sampler
- **Strategy**: Nested CV (5 outer × 3 inner folds)
- **Trials**: 100-300 per model
- **Objective**: Maximize Binary F1 Score

### Best Hyperparameters

#### CatBoost (Best Model)
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

#### RandomForest with ROS
```python
{
    'n_estimators': 1000,
    'max_depth': 25,
    'min_samples_leaf': 15,
    'class_weight': 'balanced_subsample'
}
```

## 🔬 Key Insights

1. **Threshold Optimization is Critical**: Default 0.5 threshold performs poorly on imbalanced data
2. **Ensemble Diversity**: Combining different model types improves robustness
3. **Resampling Impact**: RandomOverSampler significantly improves tree-based models
4. **Feature Importance**: Clinical measurements (TSH, T3, T4) and Nodule_Size are most predictive

## 📊 Validation Strategy

```python
# Stratified K-Fold to maintain class distribution
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Custom F1 scorer for positive class
from sklearn.metrics import make_scorer, f1_score
f1_scorer = make_scorer(f1_score, pos_label=1, average='binary')
```

## 🛠️ Technologies Used

- **Core Libraries**: pandas, numpy, scikit-learn
- **Gradient Boosting**: XGBoost, LightGBM, CatBoost
- **Hyperparameter Tuning**: Optuna
- **Imbalanced Learning**: imbalanced-learn
- **Visualization**: matplotlib, seaborn

## 📝 Lessons Learned

1. **Class Imbalance Handling**: Multiple strategies needed (resampling + threshold + class weights)
2. **Cross-Validation Importance**: Robust CV strategy essential for generalization
3. **Ensemble Benefits**: Simple averaging often outperforms complex stacking
4. **GPU Acceleration**: CatBoost GPU mode provides 5-10x speedup

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dacon for hosting the competition
- Competition participants for insightful discussions
- Open source community for amazing ML libraries

## 📧 Contact

- GitHub: [@shaun0927](https://github.com/shaun0927)
- Email: [your-email@example.com]

---

**Note**: Competition data files are not included in this repository due to licensing restrictions. Please download them from the official Dacon competition page.