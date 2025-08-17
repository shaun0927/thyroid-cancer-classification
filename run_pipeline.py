"""
Complete Pipeline for Thyroid Cancer Classification
Author: Shaun
Date: 2025

This script runs the entire pipeline from data preprocessing to submission generation.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.append('src')

from preprocessing import ThyroidDataPreprocessor
from models import train_all_models, ModelTrainer
from ensemble import EnsemblePredictor, create_submission


def main():
    parser = argparse.ArgumentParser(description='Run complete thyroid cancer classification pipeline')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['preprocess', 'train', 'ensemble', 'full'],
                       help='Pipeline mode')
    parser.add_argument('--train_path', type=str, default='data/train.csv',
                       help='Path to raw training data')
    parser.add_argument('--test_path', type=str, default='data/test.csv',
                       help='Path to raw test data')
    parser.add_argument('--trials', type=int, default=100,
                       help='Number of Optuna trials per model')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--optimize_threshold', action='store_true',
                       help='Optimize threshold for ensemble')
    
    args = parser.parse_args()
    
    # Create directories
    for dir_name in ['data', 'models', 'results', 'notebooks']:
        Path(dir_name).mkdir(exist_ok=True)
    
    print("="*70)
    print("THYROID CANCER CLASSIFICATION PIPELINE")
    print("="*70)
    
    # Step 1: Preprocessing
    if args.mode in ['preprocess', 'full']:
        print("\n" + "="*70)
        print("STEP 1: DATA PREPROCESSING")
        print("="*70)
        
        # Load raw data
        train_raw = pd.read_csv(args.train_path)
        test_raw = pd.read_csv(args.test_path)
        
        print(f"Raw train shape: {train_raw.shape}")
        print(f"Raw test shape: {test_raw.shape}")
        
        # Preprocess
        preprocessor = ThyroidDataPreprocessor()
        train_processed, test_processed = preprocessor.fit_transform(train_raw, test_raw)
        
        # Save processed data
        train_processed.to_csv('data/processed_train.csv', index=False)
        test_processed.to_csv('data/processed_test.csv', index=False)
        
        print(f"\n✅ Preprocessing completed")
        print(f"Processed train shape: {train_processed.shape}")
        print(f"Processed test shape: {test_processed.shape}")
        
        # Print class distribution
        if 'Cancer' in train_processed.columns:
            class_dist = train_processed['Cancer'].value_counts()
            print(f"\nClass distribution:")
            print(f"  Benign (0): {class_dist.get(0, 0):,} ({class_dist.get(0, 0)/len(train_processed)*100:.1f}%)")
            print(f"  Malignant (1): {class_dist.get(1, 0):,} ({class_dist.get(1, 0)/len(train_processed)*100:.1f}%)")
    
    # Step 2: Model Training
    if args.mode in ['train', 'full']:
        print("\n" + "="*70)
        print("STEP 2: MODEL TRAINING")
        print("="*70)
        
        # Load processed data
        train_df = pd.read_csv('data/processed_train.csv')
        X = train_df.drop(columns=['Cancer', 'ID'])
        y = train_df['Cancer'].astype(int)
        
        print(f"Training data shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Train all models
        results = train_all_models(X, y, n_trials=args.trials)
        
        print("\n✅ Model training completed")
        print(f"Models saved to: models/")
    
    # Step 3: Ensemble and Submission
    if args.mode in ['ensemble', 'full']:
        print("\n" + "="*70)
        print("STEP 3: ENSEMBLE AND SUBMISSION")
        print("="*70)
        
        # Find all saved models
        model_paths = list(Path('models').glob('*_best.pkl'))
        
        if not model_paths:
            print("❌ No trained models found. Please run training first.")
            return
        
        print(f"Found {len(model_paths)} models:")
        for path in model_paths:
            print(f"  - {path.name}")
        
        # Create ensemble
        ensemble = EnsemblePredictor(model_paths)
        
        # Optimize threshold if requested
        if args.optimize_threshold:
            print("\nOptimizing ensemble threshold...")
            train_df = pd.read_csv('data/processed_train.csv')
            X_train = train_df.drop(columns=['Cancer', 'ID'])
            y_train = train_df['Cancer'].astype(int)
            
            ensemble.optimize_threshold(X_train, y_train)
            
            # Evaluate
            print("\nEnsemble performance on training data:")
            ensemble.evaluate(X_train, y_train)
        
        # Save ensemble
        ensemble.save('models/ensemble_best.pkl')
        
        # Create submission
        print("\nGenerating submission...")
        submission = create_submission(
            ensemble,
            'data/processed_test.csv',
            'results/submission.csv'
        )
        
        print("\n✅ Submission created: results/submission.csv")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    # Print summary
    if args.mode == 'full':
        print("\nSummary:")
        print("1. Preprocessed data saved to: data/")
        print("2. Trained models saved to: models/")
        print("3. Submission saved to: results/submission.csv")
        print("\nNext steps:")
        print("1. Review the submission file")
        print("2. Submit to Dacon competition")
        print("3. Monitor leaderboard performance")


if __name__ == "__main__":
    main()