# app/model_evaluation.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, confusion_matrix, 
                             classification_report)

def load_model_and_features():
    """Load the trained model and selected features"""
    try:
        model = joblib.load('models/best_model.pkl')
        with open('models/selected_features.pkl', 'rb') as f:
            selected_features = pickle.load(f)
        scaler = joblib.load('models/scaler.pkl')
        return model, selected_features, scaler
    except:
        st.error("Please train a model first by running 'python train_model.py'")
        return None, None, None

def main():
    st.title("Heart Disease Prediction Model Evaluation")
    st.markdown("Evaluate your trained heart disease prediction model")
    
    # Load model and features
    model, selected_features, scaler = load_model_and_features()
    
    if model is None:
        return
    
    st.sidebar.header("Model Information")
    st.sidebar.write(f"Model type: {type(model).__name__}")
    st.sidebar.write(f"Number of features: {len(selected_features)}")
    st.sidebar.write("Selected features:")
    for feature in selected_features:
        st.sidebar.write(f"- {feature}")
    
    # Load test data
    from train_model import load_and_preprocess_data
    df = load_and_preprocess_data()
    
    # Prepare data
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Scale features
    X_scaled = scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    X_selected = X_scaled[selected_features]
    
    # Make predictions
    y_pred = model.predict(X_selected)
    y_pred_proba = model.predict_proba(X_selected)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    # Display metrics
    st.header("Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("Precision", f"{precision:.4f}")
    col3.metric("Recall", f"{recall:.4f}")
    col4.metric("F1 Score", f"{f1:.4f}")
    
    if y_pred_proba is not None:
        auc = roc_auc_score(y, y_pred_proba)
        st.metric("AUC Score", f"{auc:.4f}")
    
    # Confusion matrix
    st.header("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    st.pyplot(fig)
    
    # ROC curve (if model supports probability)
    if y_pred_proba is not None:
        st.header("ROC Curve")
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='lower right')
        st.pyplot(fig)
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        st.header("Feature Importance")
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(feature_importance['feature'], feature_importance['importance'])
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importances')
        ax.invert_yaxis()
        st.pyplot(fig)

if __name__ == "__main__":
    main()