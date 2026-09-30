# train_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectKBest, chi2, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, confusion_matrix, 
                             classification_report, precision_recall_curve, average_precision_score)
from sklearn.pipeline import Pipeline
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_and_preprocess_data():
    """
    Load and preprocess the heart disease dataset
    """
    # Load data from UCI repository
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    try:
        df = pd.read_csv(url, header=None, names=column_names)
        print("Dataset loaded successfully from UCI repository")
    except:
        # Fallback to local file if online loading fails
        try:
            df = pd.read_csv('data/heart.csv')
            print("Dataset loaded from local file")
        except:
            # Create synthetic data as last resort
            from sklearn.datasets import make_classification
            X, y = make_classification(n_samples=303, n_features=13, n_informative=8, 
                                      n_redundant=5, n_classes=2, random_state=42)
            df = pd.DataFrame(X, columns=column_names[:-1])
            df['target'] = y
            print("Synthetic dataset created")
    
    # Clean the data
    df = df.replace('?', np.nan)
    
    # Convert columns to numeric
    for col in ['ca', 'thal']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Convert target to binary (0: no disease, 1: disease)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    return df

def perform_eda(df):
    """
    Perform exploratory data analysis
    """
    print("="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)
    
    # Dataset info
    print(f"Dataset shape: {df.shape}")
    print(f"Number of patients with heart disease: {df['target'].sum()}")
    print(f"Number of patients without heart disease: {len(df) - df['target'].sum()}")
    print(f"Percentage of patients with heart disease: {df['target'].mean()*100:.2f}%")
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Basic statistics
    print("\nDataset statistics:")
    print(df.describe())
    
    # Correlation matrix
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('results/correlation_matrix.png')
    plt.close()
    
    # Target distribution
    plt.figure(figsize=(8, 6))
    df['target'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('Distribution of Heart Disease Cases')
    plt.xlabel('Heart Disease (0 = No, 1 = Yes)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('results/target_distribution.png')
    plt.close()
    
    return df

def feature_engineering(df):
    """
    Perform feature engineering and selection
    """
    print("\n" + "="*50)
    print("FEATURE ENGINEERING AND SELECTION")
    print("="*50)
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Save the scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Feature selection using Random Forest importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    
    # Get feature importances
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Feature importances:")
    print(feature_importances)
    
    # Select top features (you can adjust this threshold)
    selected_features = feature_importances[feature_importances['importance'] > 0.02]['feature'].tolist()
    print(f"\nSelected features: {selected_features}")
    
    # Update X with selected features
    X_selected = X_scaled[selected_features]
    
    # Plot feature importances
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importances['feature'], feature_importances['importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importances from Random Forest')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('results/feature_importances.png')
    plt.close()
    
    return X_selected, y, selected_features

def train_models(X, y, selected_features):
    """
    Train and evaluate multiple machine learning models
    """
    print("\n" + "="*50)
    print("MODEL TRAINING AND EVALUATION")
    print("="*50)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define models to train
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Support Vector Machine': SVC(probability=True, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Naive Bayes': GaussianNB()
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Print results
        print(f"{name} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        print(f"  Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    return results, X_test, y_test, models

def hyperparameter_tuning(X, y):
    """
    Perform hyperparameter tuning for the best model
    """
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING")
    print("="*50)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define parameter grids for different models
    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Logistic Regression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'subsample': [0.8, 0.9, 1.0]
        }
    }
    
    # Train and tune models
    tuned_models = {}
    for model_name, param_grid in param_grids.items():
        print(f"\nTuning {model_name}...")
        
        if model_name == 'Random Forest':
            model = RandomForestClassifier(random_state=42)
        elif model_name == 'Logistic Regression':
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_name == 'Gradient Boosting':
            model = GradientBoostingClassifier(random_state=42)
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store results
        tuned_models[model_name] = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Print results
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Tuned {model_name} Accuracy: {accuracy:.4f}")
        print(f"Tuned {model_name} AUC: {auc:.4f}")
    
    return tuned_models

def evaluate_and_save_best_model(results, tuned_models, X_test, y_test, selected_features):
    """
    Evaluate all models and save the best one
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION AND SELECTION")
    print("="*50)
    
    # Combine regular and tuned models
    all_models = {}
    for name, result in results.items():
        all_models[name] = result
    
    for name, result in tuned_models.items():
        all_models[f"Tuned {name}"] = result
    
    # Find the best model based on AUC or accuracy
    best_model_name = None
    best_accuracy = 0
    
    for name, result in all_models.items():
        if 'auc' in result:
            score = result['auc']
        else:
            # For models without probability estimates, use accuracy
            score = result['accuracy']
        
        if score > best_accuracy:
            best_accuracy = score
            best_model_name = name
    
    print(f"\nBest model: {best_model_name} with score: {best_accuracy:.4f}")
    
    # Get the best model
    best_model = all_models[best_model_name]['model']
    
    # Save the best model
    joblib.dump(best_model, 'models/best_model.pkl')
    
    # Save the selected features
    with open('models/selected_features.pkl', 'wb') as f:
        pickle.dump(selected_features, f)
    
    # Generate detailed evaluation report
    generate_evaluation_report(all_models, X_test, y_test)
    
    return best_model_name, best_model, all_models

def generate_evaluation_report(all_models, X_test, y_test):
    """
    Generate a comprehensive evaluation report
    """
    print("\n" + "="*50)
    print("COMPREHENSIVE EVALUATION REPORT")
    print("="*50)
    
    # Create a comparison table
    comparison_data = []
    for name, result in all_models.items():
        if 'auc' in result:
            auc = result['auc']
        else:
            auc = "N/A"
        
        comparison_data.append({
            'Model': name,
            'Accuracy': f"{result['accuracy']:.4f}",
            'Precision': f"{result['precision']:.4f}",
            'Recall': f"{result['recall']:.4f}",
            'F1 Score': f"{result['f1']:.4f}",
            'AUC': f"{auc:.4f}" if isinstance(auc, float) else auc
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Save comparison table
    comparison_df.to_csv('results/model_comparison.csv', index=False)
    
    # Plot ROC curves for models that support probability estimates
    plt.figure(figsize=(10, 8))
    models_with_proba = [name for name, result in all_models.items() 
                         if 'y_pred_proba' in result and result['y_pred_proba'] is not None]
    
    for name in models_with_proba:
        y_pred_proba = all_models[name]['y_pred_proba']
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('results/roc_curves.png')
    plt.close()
    
    # Plot precision-recall curves
    plt.figure(figsize=(10, 8))
    for name in models_with_proba:
        y_pred_proba = all_models[name]['y_pred_proba']
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        plt.plot(recall, precision, label=f'{name} (AP = {avg_precision:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('results/precision_recall_curves.png')
    plt.close()
    
    # Plot confusion matrix for the best model
    best_model_name = max(all_models.keys(), 
                         key=lambda x: all_models[x]['auc'] if 'auc' in all_models[x] else all_models[x]['accuracy'])
    
    y_pred = all_models[best_model_name]['y_pred']
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    plt.close()
    
    # Save classification report for the best model
    report = classification_report(y_test, y_pred, target_names=['No Disease', 'Disease'])
    with open('results/classification_report.txt', 'w') as f:
        f.write(f"Classification Report for {best_model_name}\n")
        f.write("=" * 50 + "\n")
        f.write(report)

def main():
    """
    Main function to run the entire training pipeline
    """
    print("HEART DISEASE PREDICTION MODEL TRAINING")
    print("="*50)
    
    # Create directories if they don't exist
    import os
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Step 1: Load and preprocess data
    df = load_and_preprocess_data()
    
    # Step 2: Perform EDA
    df = perform_eda(df)
    
    # Step 3: Feature engineering and selection
    X, y, selected_features = feature_engineering(df)
    
    # Step 4: Train models
    results, X_test, y_test, models = train_models(X, y, selected_features)
    
    # Step 5: Hyperparameter tuning
    tuned_models = hyperparameter_tuning(X, y)
    
    # Step 6: Evaluate and save the best model
    best_model_name, best_model, all_models = evaluate_and_save_best_model(
        results, tuned_models, X_test, y_test, selected_features
    )
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print(f"Best model: {best_model_name}")
    print("Model saved as: models/best_model.pkl")
    print("Selected features saved as: models/selected_features.pkl")
    print("Results saved in the 'results' directory")
    
    return best_model, selected_features

if __name__ == "__main__":
    best_model, selected_features = main()