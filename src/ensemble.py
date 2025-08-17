"""
Ensemble Methods for Thyroid Cancer Classification
Author: Shaun
Date: 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
import joblib
from models import ModelTrainer


class EnsemblePredictor:
    """Ensemble multiple models for improved predictions"""
    
    def __init__(self, model_paths=None, weights=None):
        """
        Initialize ensemble predictor
        
        Args:
            model_paths: List of paths to saved models
            weights: Optional weights for each model (default: equal weights)
        """
        self.models = []
        self.weights = weights
        self.ensemble_threshold = 0.5
        
        if model_paths:
            self.load_models(model_paths)
    
    def load_models(self, model_paths):
        """Load multiple saved models"""
        self.models = []
        
        for path in model_paths:
            if Path(path).exists():
                trainer = ModelTrainer.load(path)
                self.models.append(trainer)
                print(f"Loaded {trainer.model_type} from {path}")
            else:
                print(f"Warning: Model file not found: {path}")
        
        if not self.weights:
            self.weights = [1.0 / len(self.models)] * len(self.models)
        
        print(f"\nEnsemble initialized with {len(self.models)} models")
    
    def predict_proba(self, X):
        """Get ensemble probability predictions"""
        if not self.models:
            raise ValueError("No models loaded")
        
        # Get predictions from each model
        probas = []
        for model in self.models:
            _, proba = model.predict(X, threshold=0.5)  # Use default threshold for proba
            probas.append(proba)
        
        # Weighted average
        probas = np.array(probas)
        ensemble_proba = np.average(probas, axis=0, weights=self.weights)
        
        return ensemble_proba
    
    def optimize_threshold(self, X, y):
        """Find optimal ensemble threshold for F1 score"""
        y_proba = self.predict_proba(X)
        
        precision, recall, thresholds = precision_recall_curve(y, y_proba)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        
        if len(f1_scores) > 0 and len(thresholds) > 0:
            best_idx = f1_scores.argmax()
            if best_idx < len(thresholds):
                self.ensemble_threshold = thresholds[best_idx]
                best_f1 = f1_scores[best_idx]
                
                print(f"Optimal ensemble threshold: {self.ensemble_threshold:.3f}")
                print(f"Best ensemble F1 score: {best_f1:.4f}")
                
                return self.ensemble_threshold, best_f1
        
        return 0.5, 0.0
    
    def predict(self, X, threshold=None):
        """Make ensemble predictions"""
        threshold = threshold or self.ensemble_threshold
        y_proba = self.predict_proba(X)
        y_pred = (y_proba >= threshold).astype(int)
        
        return y_pred, y_proba
    
    def evaluate(self, X, y, threshold=None):
        """Evaluate ensemble performance"""
        y_pred, y_proba = self.predict(X, threshold)
        
        metrics = {
            'f1': f1_score(y, y_pred),
            'auc': roc_auc_score(y, y_proba),
            'threshold': threshold or self.ensemble_threshold
        }
        
        # Individual model performances
        print("\nIndividual Model Performances:")
        for i, model in enumerate(self.models):
            y_pred_i, y_proba_i = model.predict(X)
            f1_i = f1_score(y, y_pred_i)
            auc_i = roc_auc_score(y, y_proba_i)
            print(f"  {model.model_type}: F1={f1_i:.4f}, AUC={auc_i:.4f}, Threshold={model.best_threshold:.3f}")
        
        print(f"\nEnsemble Performance:")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  Threshold: {metrics['threshold']:.3f}")
        
        return metrics
    
    def save(self, filepath):
        """Save ensemble configuration"""
        ensemble_data = {
            'models': self.models,
            'weights': self.weights,
            'threshold': self.ensemble_threshold
        }
        joblib.dump(ensemble_data, filepath)
        print(f"Ensemble saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load saved ensemble"""
        ensemble_data = joblib.load(filepath)
        ensemble = cls()
        ensemble.models = ensemble_data['models']
        ensemble.weights = ensemble_data['weights']
        ensemble.ensemble_threshold = ensemble_data['threshold']
        return ensemble


def create_submission(ensemble, test_data_path, submission_path):
    """Create competition submission file"""
    
    # Load test data
    test_df = pd.read_csv(test_data_path)
    test_ids = test_df['ID']
    X_test = test_df.drop(columns=['ID'])
    
    # Make predictions
    y_pred, y_proba = ensemble.predict(X_test)
    
    # Create submission
    submission = pd.DataFrame({
        'ID': test_ids,
        'Cancer': y_pred
    })
    
    # Save submission
    submission.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to {submission_path}")
    print(f"Shape: {submission.shape}")
    print(f"Predictions distribution: {submission['Cancer'].value_counts().to_dict()}")
    
    return submission


def main():
    """Main ensemble pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ensemble predictions for thyroid cancer')
    parser.add_argument('--train', type=str, default='data/processed_train.csv',
                       help='Path to processed training data')
    parser.add_argument('--test', type=str, default='data/processed_test.csv',
                       help='Path to processed test data')
    parser.add_argument('--models_dir', type=str, default='models',
                       help='Directory containing saved models')
    parser.add_argument('--output', type=str, default='results/submission.csv',
                       help='Output submission file path')
    
    args = parser.parse_args()
    
    # Find all saved models
    models_dir = Path(args.models_dir)
    model_paths = list(models_dir.glob('*_best.pkl'))
    
    if not model_paths:
        print(f"No models found in {models_dir}")
        return
    
    print(f"Found {len(model_paths)} models:")
    for path in model_paths:
        print(f"  - {path.name}")
    
    # Create ensemble
    ensemble = EnsemblePredictor(model_paths)
    
    # Load training data for threshold optimization
    print("\nOptimizing ensemble threshold...")
    train_df = pd.read_csv(args.train)
    X_train = train_df.drop(columns=['Cancer', 'ID'])
    y_train = train_df['Cancer'].astype(int)
    
    # Optimize threshold
    ensemble.optimize_threshold(X_train, y_train)
    
    # Evaluate on training data
    print("\nEvaluating ensemble on training data:")
    ensemble.evaluate(X_train, y_train)
    
    # Save ensemble
    ensemble_path = Path(args.models_dir) / 'ensemble_best.pkl'
    ensemble.save(ensemble_path)
    
    # Create submission
    print("\nCreating submission...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    submission = create_submission(ensemble, args.test, output_path)
    
    print("\n✅ Ensemble pipeline completed successfully!")


if __name__ == "__main__":
    main()