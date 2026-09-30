# scripts/data_preprocessing.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings('ignore')

def main():
    print("Heart Disease Data Preprocessing")
    print("=" * 50)
    
    # Create directories if they don't exist
    os.makedirs('../data', exist_ok=True)
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../results', exist_ok=True)
    
    # 1. Load Dataset
    print("Loading Heart Disease dataset...")
    
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv"
    ]
    
    df = None
    for url in urls:
        try:
            df = pd.read_csv(url, header=None, names=column_names)
            print(f"Dataset loaded successfully from: {url}")
            break
        except:
            continue
    
    if df is None:
        print("Could not load data from external sources. Creating sample data...")
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=303, n_features=13, n_informative=8, 
                                  n_redundant=5, n_classes=2, random_state=42)
        df = pd.DataFrame(X, columns=column_names[:-1])
        df['target'] = y
    
    print(f"Dataset shape: {df.shape}")
    
    # 2. Handle Missing Values
    print("\nHandling missing values...")
    df = df.replace('?', np.nan)
    
    for col in ['ca', 'thal']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    # 3. Data Cleaning
    print("\nCleaning data...")
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median())
    
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Removed {duplicates} duplicate rows")
    
    # 4. Feature Engineering
    print("\nPerforming feature engineering...")
    df_engineered = df.copy()
    
    df_engineered['age_group'] = pd.cut(df_engineered['age'], 
                                       bins=[20, 35, 50, 65, 100], 
                                       labels=['20-35', '36-50', '51-65', '65+'])
    
    df_engineered['chol_category'] = pd.cut(df_engineered['chol'], 
                                           bins=[0, 200, 240, 1000], 
                                           labels=['Normal', 'Borderline', 'High'])
    
    df_engineered['bp_category'] = pd.cut(df_engineered['trestbps'], 
                                         bins=[0, 120, 140, 1000], 
                                         labels=['Normal', 'Elevated', 'High'])
    
    df_engineered['hr_category'] = pd.cut(df_engineered['thalach'], 
                                         bins=[0, 60, 100, 200], 
                                         labels=['Low', 'Normal', 'High'])
    
    # 5. Data Scaling
    print("\nScaling numerical features...")
    X = df_engineered.drop('target', axis=1)
    y = df_engineered['target']
    
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Save the scaler
    import joblib
    joblib.dump(scaler, '../models/scaler.pkl')
    
    # 6. Save Processed Data
    print("\nSaving processed data...")
    df.to_csv('../data/heart_cleaned.csv', index=False)
    df_engineered.to_csv('../data/heart_engineered.csv', index=False)
    X_scaled.to_csv('../data/heart_features_scaled.csv', index=False)
    y.to_csv('../data/heart_target.csv', index=False)
    
    print("Data preprocessing completed successfully!")
    print(f"Final dataset shape: {df_engineered.shape}")
    print(f"Number of patients with heart disease: {y.sum()}")
    print(f"Number of patients without heart disease: {len(y) - y.sum()}")
    print(f"Percentage with heart disease: {y.mean()*100:.2f}%")

if __name__ == "__main__":
    main()