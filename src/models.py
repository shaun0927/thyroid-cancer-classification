"""
Model Training Pipeline for Thyroid Cancer Classification
Author: Shaun
Date: 2025
"""

import warnings
import numpy as np
import pandas as pd
import optuna
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import RandomOverSampler
import joblib

warnings.filterwarnings('ignore')


class ModelTrainer:
    """Unified model trainer with hyperparameter optimization"""
    
    def __init__(self, model_type='catboost', use_gpu=False, random_state=42):
        """
        Initialize model trainer
        
        Args:
            model_type: One of ['catboost', 'lightgbm', 'xgboost', 'randomforest', 'gbdt']
            use_gpu: Whether to use GPU acceleration (for CatBoost)
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.best_model = None
        self.best_threshold = 0.5
        self.best_params = None
        
    def get_search_space(self, trial, model_type):
        """Define hyperparameter search space for each model"""
        
        if model_type == 'catboost':
            params = {
                'iterations': trial.suggest_int('iterations', 800, 2000, step=100),
                'depth': trial.suggest_int('depth', 4, 12),
                'learning_rate': trial.suggest_float('lr', 0.01, 0.2, log=True),
                'l2_leaf_reg': trial.suggest_float('l2', 1e-2, 30, log=True),
                'border_count': trial.suggest_int('border', 32, 255),
                'scale_pos_weight': trial.suggest_float('spw', 1.0, 10.0),
                'loss_function': 'Logloss',
                'verbose': False,
                'random_state': self.random_state
            }
            if self.use_gpu:
                params.update({'task_type': 'GPU', 'devices': '0'})
                
        elif model_type == 'lightgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 400, 1500, step=100),
                'learning_rate': trial.suggest_float('lr', 0.01, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 31, 255),
                'max_depth': trial.suggest_int('max_depth', -1, 20),
                'min_child_samples': trial.suggest_int('min_child', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': self.random_state,
                'n_jobs': -1,
                'objective': 'binary'
            }
            
        elif model_type == 'randomforest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000, step=250),
                'max_depth': trial.suggest_int('max_depth', 15, 35),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 40),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'max_features': trial.suggest_float('max_features', 0.3, 0.9),
                'class_weight': trial.suggest_categorical('class_weight', 
                    [None, 'balanced', 'balanced_subsample']),
                'n_jobs': -1,
                'random_state': self.random_state
            }
            
        elif model_type == 'gbdt':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 800, step=100),
                'learning_rate': trial.suggest_float('lr', 0.01, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'max_features': trial.suggest_float('max_features', 0.5, 1.0),
                'random_state': self.random_state
            }
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        return params
    
    def create_model(self, params, model_type):
        """Create model instance with given parameters"""
        
        if model_type == 'catboost':
            return CatBoostClassifier(**params)
        elif model_type == 'lightgbm':
            return LGBMClassifier(**params)
        elif model_type == 'randomforest':
            return RandomForestClassifier(**params)
        elif model_type == 'gbdt':
            return GradientBoostingClassifier(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def optimize_threshold(self, y_true, y_proba):
        """Find optimal threshold for F1 score"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        
        if len(f1_scores) > 0 and len(thresholds) > 0:
            best_idx = f1_scores.argmax()
            if best_idx < len(thresholds):
                return thresholds[best_idx], f1_scores[best_idx]
        
        return 0.5, 0.0
    
    def objective(self, trial, X, y, cv_folds=5):
        """Optuna objective function for hyperparameter optimization"""
        
        # Get hyperparameters
        params = self.get_search_space(trial, self.model_type)
        threshold = trial.suggest_float('threshold', 0.1, 0.45)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        f1_scores = []
        
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Apply oversampling for RandomForest
            if self.model_type == 'randomforest':
                ros = RandomOverSampler(random_state=self.random_state)
                X_train, y_train = ros.fit_resample(X_train, y_train)
            
            # Train model
            model = self.create_model(params, self.model_type)
            model.fit(X_train, y_train)
            
            # Predict
            y_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)
            
            # Calculate F1
            f1 = f1_score(y_val, y_pred)
            f1_scores.append(f1)
        
        return np.mean(f1_scores)
    
    def train(self, X, y, n_trials=100, cv_folds=5, optimize_threshold_final=True):
        """
        Train model with hyperparameter optimization
        
        Args:
            X: Features dataframe
            y: Target series
            n_trials: Number of Optuna trials
            cv_folds: Number of cross-validation folds
            optimize_threshold_final: Whether to optimize threshold on final model
            
        Returns:
            Dictionary with model, threshold, and performance metrics
        """
        print(f"\nTraining {self.model_type.upper()} model...")
        print(f"Dataset shape: {X.shape}")
        print(f"Class distribution: {y.value_counts().to_dict()}")
        
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # Optimize
        study.optimize(
            lambda trial: self.objective(trial, X, y, cv_folds),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        # Get best parameters
        best_trial = study.best_trial
        self.best_params = {k: v for k, v in best_trial.params.items() if k != 'threshold'}
        best_threshold_cv = best_trial.params['threshold']
        
        print(f"\nBest CV F1 Score: {study.best_value:.4f}")
        print(f"Best threshold (CV): {best_threshold_cv:.3f}")
        
        # Train final model on full data
        print("\nTraining final model on full dataset...")
        
        # Apply oversampling for RandomForest
        X_final, y_final = X, y
        if self.model_type == 'randomforest':
            ros = RandomOverSampler(random_state=self.random_state)
            X_final, y_final = ros.fit_resample(X, y)
        
        self.best_model = self.create_model(self.best_params, self.model_type)
        self.best_model.fit(X_final, y_final)
        
        # Optimize threshold on full training data if requested
        if optimize_threshold_final:
            y_proba_full = self.best_model.predict_proba(X)[:, 1]
            self.best_threshold, best_f1_full = self.optimize_threshold(y, y_proba_full)
            print(f"Final threshold (full data): {self.best_threshold:.3f}")
            print(f"Final F1 (full data): {best_f1_full:.4f}")
        else:
            self.best_threshold = best_threshold_cv
        
        # Calculate final metrics
        y_proba = self.best_model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= self.best_threshold).astype(int)
        
        results = {
            'model': self.best_model,
            'threshold': self.best_threshold,
            'params': self.best_params,
            'cv_f1': study.best_value,
            'final_f1': f1_score(y, y_pred),
            'auc': roc_auc_score(y, y_proba)
        }
        
        print(f"\nFinal AUC: {results['auc']:.4f}")
        
        return results
    
    def predict(self, X, threshold=None):
        """Make predictions with trained model"""
        if self.best_model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        threshold = threshold or self.best_threshold
        y_proba = self.best_model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        
        return y_pred, y_proba
    
    def save(self, filepath):
        """Save model and parameters"""
        model_data = {
            'model': self.best_model,
            'threshold': self.best_threshold,
            'params': self.best_params,
            'model_type': self.model_type
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load saved model"""
        model_data = joblib.load(filepath)
        trainer = cls(model_type=model_data['model_type'])
        trainer.best_model = model_data['model']
        trainer.best_threshold = model_data['threshold']
        trainer.best_params = model_data['params']
        return trainer


def train_all_models(X, y, n_trials=50):
    """Train all model types and return results"""
    
    model_types = ['catboost', 'lightgbm', 'randomforest', 'gbdt']
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()}")
        print(f"{'='*60}")
        
        trainer = ModelTrainer(
            model_type=model_type,
            use_gpu=(model_type == 'catboost')
        )
        
        result = trainer.train(X, y, n_trials=n_trials)
        results[model_type] = result
        
        # Save model
        model_path = Path('models') / f'{model_type}_best.pkl'
        model_path.parent.mkdir(exist_ok=True)
        trainer.save(model_path)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    
    summary_df = pd.DataFrame({
        'Model': list(results.keys()),
        'CV F1': [r['cv_f1'] for r in results.values()],
        'Final F1': [r['final_f1'] for r in results.values()],
        'AUC': [r['auc'] for r in results.values()],
        'Threshold': [r['threshold'] for r in results.values()]
    })
    
    summary_df = summary_df.sort_values('CV F1', ascending=False)
    print(summary_df.to_string(index=False))
    
    return results


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Train thyroid cancer classification models')
    parser.add_argument('--data', type=str, default='data/processed_train.csv',
                       help='Path to processed training data')
    parser.add_argument('--model', type=str, default='all',
                       choices=['catboost', 'lightgbm', 'randomforest', 'gbdt', 'all'],
                       help='Model type to train')
    parser.add_argument('--trials', type=int, default=50,
                       help='Number of Optuna trials')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(args.data)
    X = df.drop(columns=['Cancer', 'ID'])
    y = df['Cancer'].astype(int)
    
    # Train models
    if args.model == 'all':
        results = train_all_models(X, y, n_trials=args.trials)
    else:
        trainer = ModelTrainer(
            model_type=args.model,
            use_gpu=(args.model == 'catboost')
        )
        result = trainer.train(X, y, n_trials=args.trials)
        
        # Save model
        model_path = Path('models') / f'{args.model}_best.pkl'
        model_path.parent.mkdir(exist_ok=True)
        trainer.save(model_path)