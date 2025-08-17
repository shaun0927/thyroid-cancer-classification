"""
Data Preprocessing Pipeline for Thyroid Cancer Classification
Author: Shaun
Date: 2025
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
import argparse

warnings.filterwarnings('ignore')


class ThyroidDataPreprocessor:
    """Preprocessor for thyroid cancer classification data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.binary_maps = {
            'Gender': {'m': 1, 'f': 0},
            'Family_Background': {'positive': 1, 'negative': 0},
            'Radiation_History': {'exposed': 1, 'unexposed': 0},
            'Iodine_Deficiency': {'deficient': 1, 'sufficient': 0},
            'Smoke': {'smoker': 1, 'non-smoker': 0},
            'Weight_Risk': {'obese': 1, 'not obese': 0},
            'Diabetes': {'yes': 1, 'no': 0}
        }
        
    def fit_transform(self, train_df, test_df):
        """
        Fit preprocessor on training data and transform both train and test
        
        Args:
            train_df: Training dataframe with target column
            test_df: Test dataframe without target column
            
        Returns:
            processed_train, processed_test: Preprocessed dataframes
        """
        # Define column types
        self.int_cols = ['Age']
        self.float_cols = ['Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result']
        self.bin_cols = list(self.binary_maps.keys())
        self.cat_cols = ['Country', 'Race']
        
        # Separate target
        target_col = 'Cancer'
        id_col = 'ID'
        
        y = train_df[target_col].astype('int8') if target_col in train_df.columns else None
        train_features = train_df.drop(columns=[target_col] if target_col in train_df.columns else [])
        
        # Combine for processing
        n_train = len(train_features)
        df_all = pd.concat([train_features, test_df], axis=0, ignore_index=True)
        
        # Process binary features
        for col, mapper in self.binary_maps.items():
            if col in df_all.columns:
                df_all[col] = (
                    df_all[col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .replace({'non obese': 'not obese'})
                    .map(mapper)
                    .fillna(-1)
                    .astype('int8')
                )
        
        # Process categorical features
        for col in self.cat_cols:
            if col in df_all.columns:
                le = LabelEncoder()
                df_all[col] = le.fit_transform(df_all[col].fillna('missing'))
                self.label_encoders[col] = le
        
        # Scale numeric features
        numeric_cols = [col for col in self.float_cols + self.int_cols if col in df_all.columns]
        if numeric_cols:
            # Fit scaler on training data only
            self.scaler.fit(df_all.iloc[:n_train][numeric_cols])
            df_all[numeric_cols] = self.scaler.transform(df_all[numeric_cols])
        
        # Split back to train/test
        processed_train = df_all.iloc[:n_train].reset_index(drop=True)
        processed_test = df_all.iloc[n_train:].reset_index(drop=True)
        
        # Add target back to train
        if y is not None:
            processed_train = pd.concat([processed_train, y.reset_index(drop=True)], axis=1)
        
        return processed_train, processed_test
    
    def transform(self, df):
        """Transform new data using fitted preprocessor"""
        df_copy = df.copy()
        
        # Process binary features
        for col, mapper in self.binary_maps.items():
            if col in df_copy.columns:
                df_copy[col] = (
                    df_copy[col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .replace({'non obese': 'not obese'})
                    .map(mapper)
                    .fillna(-1)
                    .astype('int8')
                )
        
        # Process categorical features
        for col in self.cat_cols:
            if col in df_copy.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                df_copy[col] = df_copy[col].fillna('missing')
                df_copy[col] = df_copy[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        # Scale numeric features
        numeric_cols = [col for col in self.float_cols + self.int_cols if col in df_copy.columns]
        if numeric_cols:
            df_copy[numeric_cols] = self.scaler.transform(df_copy[numeric_cols])
        
        return df_copy


def main():
    """Main preprocessing pipeline"""
    parser = argparse.ArgumentParser(description='Preprocess thyroid cancer data')
    parser.add_argument('--train', type=str, default='data/train.csv',
                       help='Path to training data')
    parser.add_argument('--test', type=str, default='data/test.csv',
                       help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Output directory for processed files')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    
    # Initialize preprocessor
    preprocessor = ThyroidDataPreprocessor()
    
    # Process data
    print("Processing data...")
    processed_train, processed_test = preprocessor.fit_transform(train, test)
    
    # Save processed data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processed_train.to_csv(output_dir / "processed_train.csv", index=False)
    processed_test.to_csv(output_dir / "processed_test.csv", index=False)
    
    print(f"✅ Processed data saved to {output_dir}")
    print(f"   - processed_train.csv: {processed_train.shape}")
    print(f"   - processed_test.csv: {processed_test.shape}")
    
    # Print class distribution
    if 'Cancer' in processed_train.columns:
        class_dist = processed_train['Cancer'].value_counts()
        print(f"\nClass distribution:")
        print(f"   - Benign (0): {class_dist.get(0, 0):,} ({class_dist.get(0, 0)/len(processed_train)*100:.1f}%)")
        print(f"   - Malignant (1): {class_dist.get(1, 0):,} ({class_dist.get(1, 0)/len(processed_train)*100:.1f}%)")


if __name__ == "__main__":
    main()