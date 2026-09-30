# heart_disease_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, chi2, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, confusion_matrix, 
                             classification_report, silhouette_score)
from sklearn.pipeline import Pipeline
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    # Try to load from multiple possible URLs
    urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv"
    ]
    
    # Column names as per the UCI dataset description
    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    for url in urls:
        try:
            df = pd.read_csv(url, header=None, names=columns)
            # Clean the data - replace '?' with NaN and convert to numeric
            df = df.replace('?', np.nan)
            for col in ['ca', 'thal']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except:
            continue
    
    # If no URL works, load sample data
    st.warning("Could not load data from external sources. Using sample data.")
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=300, n_features=13, n_informative=8, 
                              n_redundant=5, n_classes=2, random_state=42)
    df = pd.DataFrame(X, columns=columns[:-1])
    df['target'] = y
    return df

# Data preprocessing
def preprocess_data(df):
    df_clean = df.copy()
    
    # Handle missing values
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    
    # Convert target to binary (0: no disease, 1: disease)
    df_clean['target'] = df_clean['target'].apply(lambda x: 1 if x > 0 else 0)
    
    return df_clean

# Main function
def main():
    # Title and description
    st.title("❤️ Heart Disease Prediction with Machine Learning")
    st.markdown("""
    This application implements a complete machine learning pipeline for predicting heart disease risk 
    using the UCI Heart Disease dataset. The workflow includes data preprocessing, feature selection, 
    dimensionality reduction, model training, and evaluation.
    """)
    
    # Load data
    df = load_data()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Select a section:", 
                              ["Data Overview", "Data Preprocessing", "Exploratory Data Analysis", 
                               "Dimensionality Reduction", "Feature Selection", 
                               "Model Training & Evaluation", "Unsupervised Learning", 
                               "Make a Prediction", "About"])
    
    # Data Overview section
    if options == "Data Overview":
        st.header("Data Overview")
        st.subheader("Heart Disease Dataset")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Shape:**", df.shape)
            st.write("**Preview of the dataset:**")
            st.dataframe(df.head())
        
        with col2:
            st.write("**Column Information:**")
            st.write(df.dtypes)
            
            st.write("**Missing Values:**")
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Values': df.isnull().sum(),
                'Percentage': (df.isnull().sum() / len(df)) * 100
            })
            st.dataframe(missing_df)
        
        st.subheader("Dataset Description")
        st.write("""
        The Heart Disease dataset from UCI contains 14 attributes:
        - age: age in years
        - sex: sex (1 = male; 0 = female)
        - cp: chest pain type (0-3)
        - trestbps: resting blood pressure (in mm Hg)
        - chol: serum cholesterol in mg/dl
        - fbs: fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
        - restecg: resting electrocardiographic results (0-2)
        - thalach: maximum heart rate achieved
        - exang: exercise induced angina (1 = yes; 0 = no)
        - oldpeak: ST depression induced by exercise relative to rest
        - slope: the slope of the peak exercise ST segment (0-2)
        - ca: number of major vessels (0-3) colored by fluoroscopy
        - thal: thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)
        - target: diagnosis of heart disease (0 = no disease, 1-4 = disease presence)
        """)
    
    # Data Preprocessing section
    elif options == "Data Preprocessing":
        st.header("Data Preprocessing")
        
        # Show original data
        st.subheader("Original Data")
        st.dataframe(df.head())
        
        # Preprocess data
        df_clean = preprocess_data(df)
        
        # Show cleaned data
        st.subheader("Cleaned Data")
        st.write("**Missing values handled and target variable binarized**")
        st.dataframe(df_clean.head())
        
        # Show preprocessing details
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Types after cleaning:**")
            st.write(df_clean.dtypes)
            
        with col2:
            st.write("**Missing Values after cleaning:**")
            st.write(df_clean.isnull().sum())
        
        # Data scaling options
        st.subheader("Data Scaling")
        scaling_method = st.radio("Select scaling method:", 
                                 ["None", "Standard Scaler", "MinMax Scaler"])
        
        if scaling_method != "None":
            # Separate features and target
            X = df_clean.drop('target', axis=1)
            y = df_clean['target']
            
            # Apply scaling
            if scaling_method == "Standard Scaler":
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()
                
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
            
            st.write("**Data after scaling:**")
            st.dataframe(X_scaled.head())
            
            # Update the cleaned dataframe
            df_clean = pd.concat([X_scaled, y], axis=1)
        
        # Store the cleaned data in session state
        st.session_state.df_clean = df_clean
        st.success("Data preprocessing completed! Cleaned data is ready for analysis.")
    
    # Exploratory Data Analysis section
    elif options == "Exploratory Data Analysis":
        st.header("Exploratory Data Analysis")
        
        if 'df_clean' not in st.session_state:
            st.warning("Please preprocess the data first in the 'Data Preprocessing' section.")
            return
            
        df_clean = st.session_state.df_clean
        
        # Distribution of target variable
        st.subheader("Target Variable Distribution")
        fig = px.pie(values=df_clean['target'].value_counts().values, 
                     names=['No Heart Disease', 'Heart Disease'],
                     title='Proportion of Heart Disease Cases')
        st.plotly_chart(fig)
        
        # Numerical features distribution
        st.subheader("Distribution of Numerical Features")
        numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        selected_num_col = st.selectbox("Select a numerical feature to visualize:", numerical_cols)
        
        fig = px.histogram(df_clean, x=selected_num_col, color='target', 
                          marginal="box", nbins=30,
                          title=f"Distribution of {selected_num_col} by Heart Disease Status",
                          labels={selected_num_col: selected_num_col, "target": "Heart Disease"})
        st.plotly_chart(fig)
        
        # Categorical features distribution
        st.subheader("Distribution of Categorical Features")
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        selected_cat_col = st.selectbox("Select a categorical feature to visualize:", categorical_cols)
        
        fig = px.histogram(df_clean, x=selected_cat_col, color='target', 
                          title=f"Distribution of {selected_cat_col} by Heart Disease Status",
                          labels={selected_cat_col: selected_cat_col, "target": "Heart Disease"})
        st.plotly_chart(fig)
        
        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        fig = px.imshow(df_clean.corr(), aspect="auto", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig)
    
    # Dimensionality Reduction section
    elif options == "Dimensionality Reduction":
        st.header("Dimensionality Reduction with PCA")
        
        if 'df_clean' not in st.session_state:
            st.warning("Please preprocess the data first in the 'Data Preprocessing' section.")
            return
            
        df_clean = st.session_state.df_clean
        
        # Separate features and target
        X = df_clean.drop('target', axis=1)
        y = df_clean['target']
        
        # Apply PCA
        pca = PCA()
        X_pca = pca.fit_transform(X)
        
        # Explained variance ratio
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Plot explained variance
        st.subheader("PCA Explained Variance")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(explained_variance))], 
                            y=explained_variance, name="Explained Variance"))
        fig.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(len(cumulative_variance))], 
                               y=cumulative_variance, name="Cumulative Variance"))
        fig.update_layout(title="PCA Explained Variance Ratio",
                         xaxis_title="Principal Components",
                         yaxis_title="Variance Ratio")
        st.plotly_chart(fig)
        
        # Select number of components
        st.subheader("Select Number of Principal Components")
        n_components = st.slider("Number of components to keep:", 
                                min_value=2, max_value=X.shape[1], value=5)
        
        # Apply PCA with selected components
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        # Create a DataFrame with the principal components
        pca_cols = [f"PC{i+1}" for i in range(n_components)]
        pca_df = pd.DataFrame(X_pca, columns=pca_cols)
        pca_df['target'] = y.values
        
        # Plot first two principal components
        st.subheader("First Two Principal Components")
        fig = px.scatter(pca_df, x='PC1', y='PC2', color='target',
                        title=f"First Two Principal Components (Explained Variance: {cumulative_variance[1]:.2%})",
                        labels={'target': 'Heart Disease'})
        st.plotly_chart(fig)
        
        # Store PCA results in session state
        st.session_state.X_pca = X_pca
        st.session_state.pca = pca
        st.success(f"PCA completed with {n_components} components. Explained variance: {cumulative_variance[n_components-1]:.2%}")
    
    # Feature Selection section
    elif options == "Feature Selection":
        st.header("Feature Selection")
        
        if 'df_clean' not in st.session_state:
            st.warning("Please preprocess the data first in the 'Data Preprocessing' section.")
            return
            
        df_clean = st.session_state.df_clean
        
        # Separate features and target
        X = df_clean.drop('target', axis=1)
        y = df_clean['target']
        
        # Feature selection methods
        st.subheader("Feature Selection Methods")
        method = st.radio("Select feature selection method:", 
                         ["Random Forest Feature Importance", "Recursive Feature Elimination (RFE)", 
                          "Chi-Square Test", "SelectKBest"])
        
        if method == "Random Forest Feature Importance":
            # Train a random forest classifier
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # Get feature importances
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Plot feature importances
            st.subheader("Random Forest Feature Importance")
            fig = px.bar(x=importances[indices], y=X.columns[indices], 
                        orientation='h', title="Feature Importance",
                        labels={'x': 'Importance', 'y': 'Features'})
            st.plotly_chart(fig)
            
            # Select top k features
            k = st.slider("Number of features to select:", min_value=2, max_value=X.shape[1], value=5)
            selected_features = X.columns[indices][:k]
            
        elif method == "Recursive Feature Elimination (RFE)":
            # Create a base classifier
            estimator = LogisticRegression(max_iter=1000, random_state=42)
            
            # RFE
            k = st.slider("Number of features to select:", min_value=2, max_value=X.shape[1], value=5)
            selector = RFE(estimator, n_features_to_select=k)
            selector = selector.fit(X, y)
            
            # Get selected features
            selected_features = X.columns[selector.support_]
            
            # Plot feature ranking
            st.subheader("RFE Feature Ranking")
            ranking_df = pd.DataFrame({
                'Feature': X.columns,
                'Ranking': selector.ranking_
            }).sort_values('Ranking')
            
            fig = px.bar(ranking_df, x='Ranking', y='Feature', orientation='h',
                        title="RFE Feature Ranking (1 = selected)")
            st.plotly_chart(fig)
            
        elif method == "Chi-Square Test":
            # Apply chi-square test
            chi2_scores, p_values = chi2(X, y)
            
            # Create DataFrame with results
            chi2_df = pd.DataFrame({
                'Feature': X.columns,
                'Chi2 Score': chi2_scores,
                'p-value': p_values
            }).sort_values('Chi2 Score', ascending=False)
            
            # Plot chi2 scores
            st.subheader("Chi-Square Test Results")
            fig = px.bar(chi2_df, x='Chi2 Score', y='Feature', orientation='h',
                        title="Chi-Square Test Scores")
            st.plotly_chart(fig)
            
            # Select top k features
            k = st.slider("Number of features to select:", min_value=2, max_value=X.shape[1], value=5)
            selected_features = chi2_df['Feature'].head(k).values
            
        else:  # SelectKBest
            # Use SelectKBest with f_classif
            k = st.slider("Number of features to select:", min_value=2, max_value=X.shape[1], value=5)
            selector = SelectKBest(k=k)
            selector.fit(X, y)
            
            # Get selected features
            selected_features = X.columns[selector.get_support()]
            
            # Plot scores
            scores_df = pd.DataFrame({
                'Feature': X.columns,
                'Score': selector.scores_
            }).sort_values('Score', ascending=False)
            
            st.subheader("SelectKBest Scores")
            fig = px.bar(scores_df, x='Score', y='Feature', orientation='h',
                        title="SelectKBest Scores (f_classif)")
            st.plotly_chart(fig)
        
        # Display selected features
        st.subheader("Selected Features")
        st.write(selected_features.tolist())
        
        # Store selected features in session state
        st.session_state.selected_features = selected_features
        st.session_state.X_selected = X[selected_features]
        st.success(f"Feature selection completed! {len(selected_features)} features selected.")
    
    # Model Training & Evaluation section
    elif options == "Model Training & Evaluation":
        st.header("Model Training & Evaluation")
        
        if 'df_clean' not in st.session_state:
            st.warning("Please preprocess the data first in the 'Data Preprocessing' section.")
            return
            
        df_clean = st.session_state.df_clean
        
        st.info("""
        This section allows you to train and evaluate machine learning models for heart disease prediction.
        You can either train models directly in this interface or use the dedicated training script.
        """)
        
        training_option = st.radio("Select training method:", 
                                  ["Train in-app", "Use pre-trained model"])
        
        if training_option == "Train in-app":
            st.subheader("Train Machine Learning Models")
            
            # Model selection
            st.subheader("Model Selection")
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Support Vector Machine": SVC(probability=True, random_state=42),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42)
            }
            
            selected_models = st.multiselect("Select models to train:", list(models.keys()))
            
            # Feature selection
            st.subheader("Feature Selection")
            feature_option = st.radio("Select features to use:", 
                                     ["All Features", "PCA Components", "Selected Features (from Feature Selection)"])
            
            if feature_option == "All Features":
                X = df_clean.drop('target', axis=1)
            elif feature_option == "PCA Components":
                if 'X_pca' not in st.session_state:
                    st.warning("Please perform PCA first in the 'Dimensionality Reduction' section.")
                    return
                X = st.session_state.X_pca
            else:
                if 'X_selected' not in st.session_state:
                    st.warning("Please perform feature selection first in the 'Feature Selection' section.")
                    return
                X = st.session_state.X_selected
            
            y = df_clean['target']
            
            # Train-test split
            st.subheader("Train-Test Split")
            test_size = st.slider("Test set size:", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
            random_state = st.slider("Random state:", min_value=0, max_value=100, value=42)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            if st.button("Train Models"):
                if not selected_models:
                    st.error("Please select at least one model to train.")
                    return
                    
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = {}
                for i, model_name in enumerate(selected_models):
                    status_text.text(f"Training {model_name}...")
                    model = models[model_name]
                    
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
                    results[model_name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'y_pred': y_pred,
                        'y_pred_proba': y_pred_proba
                    }
                    
                    progress_bar.progress((i + 1) / len(selected_models))
                
                status_text.text("Training completed!")
                
                # Display results
                st.subheader("Training Results")
                for model_name, result in results.items():
                    st.write(f"**{model_name}**")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{result['accuracy']:.4f}")
                    col2.metric("Precision", f"{result['precision']:.4f}")
                    col3.metric("Recall", f"{result['recall']:.4f}")
                    col4.metric("F1 Score", f"{result['f1']:.4f}")
                    
                    # Store the best model in session state
                    if 'trained_model' not in st.session_state or result['accuracy'] > st.session_state.get('trained_model_accuracy', 0):
                        st.session_state.trained_model = result['model']
                        st.session_state.trained_model_accuracy = result['accuracy']
                        st.session_state.trained_model_name = model_name
                
                st.success(f"Best model: {st.session_state.trained_model_name} with accuracy: {st.session_state.trained_model_accuracy:.4f}")
        
        else:  # Use pre-trained model
            st.subheader("Use Pre-trained Model")
            
            try:
                model = joblib.load('models/best_model.pkl')
                with open('models/selected_features.pkl', 'rb') as f:
                    selected_features = pickle.load(f)
                scaler = joblib.load('models/scaler.pkl')
                
                st.success("Pre-trained model loaded successfully!")
                st.write(f"Model type: {type(model).__name__}")
                st.write("Selected features:")
                for feature in selected_features:
                    st.write(f"- {feature}")
                    
                # Store the model in session state
                st.session_state.trained_model = model
                st.session_state.trained_model_name = "Pre-trained " + type(model).__name__
                
            except:
                st.error("No pre-trained model found. Please run 'python train_model.py' first.")
    
    # Unsupervised Learning section
    elif options == "Unsupervised Learning":
        st.header("Unsupervised Learning - Clustering")
        
        if 'df_clean' not in st.session_state:
            st.warning("Please preprocess the data first in the 'Data Preprocessing' section.")
            return
            
        df_clean = st.session_state.df_clean
        
        # Select features for clustering
        X = df_clean.drop('target', axis=1)
        
        # Clustering algorithms
        st.subheader("Clustering Algorithms")
        algorithm = st.radio("Select clustering algorithm:", ["K-Means", "Hierarchical Clustering"])
        
        if algorithm == "K-Means":
            # Determine optimal k using elbow method
            st.subheader("Elbow Method for Optimal K")
            inertia = []
            k_range = range(2, 11)
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X)
                inertia.append(kmeans.inertia_)
            
            # Plot elbow curve
            fig = px.line(x=list(k_range), y=inertia, title="Elbow Method for Optimal K",
                         labels={'x': 'Number of Clusters (K)', 'y': 'Inertia'})
            st.plotly_chart(fig)
            
            # Select k
            k = st.slider("Select number of clusters (K):", min_value=2, max_value=10, value=3)
            
            # Apply K-Means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(X, clusters)
            st.write(f"Silhouette Score: {silhouette_avg:.4f}")
            
            # Visualize clusters (using PCA for dimensionality reduction)
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            cluster_df = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1],
                'Cluster': clusters,
                'Actual': df_clean['target']
            })
            
            fig = px.scatter(cluster_df, x='PC1', y='PC2', color='Cluster',
                            title="K-Means Clustering Results (PCA Reduced)")
            st.plotly_chart(fig)
            
        else:  # Hierarchical Clustering
            st.subheader("Hierarchical Clustering Dendrogram")
            
            # Use a sample for dendrogram to avoid performance issues
            sample_size = min(50, len(X))
            X_sample = X.iloc[:sample_size]
            
            # Calculate linkage matrix
            Z = linkage(X_sample, method='ward')
            
            # Plot dendrogram
            fig = plt.figure(figsize=(10, 5))
            dendrogram(Z)
            plt.title("Dendrogram")
            plt.xlabel("Sample index")
            plt.ylabel("Distance")
            st.pyplot(fig)
            
            # Select number of clusters
            k = st.slider("Select number of clusters:", min_value=2, max_value=10, value=3)
            
            # Apply hierarchical clustering
            hc = AgglomerativeClustering(n_clusters=k)
            clusters = hc.fit_predict(X)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(X, clusters)
            st.write(f"Silhouette Score: {silhouette_avg:.4f}")
            
            # Visualize clusters (using PCA for dimensionality reduction)
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            cluster_df = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1],
                'Cluster': clusters,
                'Actual': df_clean['target']
            })
            
            fig = px.scatter(cluster_df, x='PC1', y='PC2', color='Cluster',
                            title="Hierarchical Clustering Results (PCA Reduced)")
            st.plotly_chart(fig)
        
        # Compare clusters with actual labels
        st.subheader("Cluster vs Actual Labels")
        comparison_df = pd.DataFrame({
            'Cluster': clusters,
            'Actual': df_clean['target']
        })
        
        contingency_table = pd.crosstab(comparison_df['Cluster'], comparison_df['Actual'])
        st.write("Contingency Table (Cluster vs Actual):")
        st.dataframe(contingency_table)
        
        fig = px.imshow(contingency_table, text_auto=True, 
                       title="Cluster vs Actual Labels",
                       labels=dict(x="Actual Label", y="Cluster", color="Count"))
        st.plotly_chart(fig)
    
    # Make a Prediction section
    elif options == "Make a Prediction":
        st.header("Make a Prediction")
        
        if 'trained_model' not in st.session_state:
            st.warning("Please train a model first in the 'Model Training & Evaluation' section.")
            return
            
        model = st.session_state.trained_model
        
        # Create input form for prediction
        st.subheader("Enter Patient Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", min_value=20, max_value=100, value=50)
            sex = st.selectbox("Sex", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]
            cp = st.selectbox("Chest Pain Type", 
                             options=[("Typical Angina", 0), ("Atypical Angina", 1), 
                                     ("Non-anginal Pain", 2), ("Asymptomatic", 3)], 
                             format_func=lambda x: x[0])[1]
            trestbps = st.slider("Resting Blood Pressure (mm Hg)", min_value=90, max_value=200, value=120)
            chol = st.slider("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", 
                              options=[("False", 0), ("True", 1)], format_func=lambda x: x[0])[1]
        
        with col2:
            restecg = st.selectbox("Resting Electrocardiographic Results", 
                                  options=[("Normal", 0), ("ST-T Wave Abnormality", 1), 
                                          ("Left Ventricular Hypertrophy", 2)], 
                                  format_func=lambda x: x[0])[1]
            thalach = st.slider("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
            exang = st.selectbox("Exercise Induced Angina", 
                                options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
            oldpeak = st.slider("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
            slope = st.selectbox("Slope of Peak Exercise ST Segment", 
                                options=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)], 
                                format_func=lambda x: x[0])[1]
            ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=3, value=0)
            thal = st.selectbox("Thalassemia", 
                               options=[("Normal", 3), ("Fixed Defect", 6), ("Reversible Defect", 7)], 
                               format_func=lambda x: x[0])[1]
        
        # Create feature vector
        input_features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, 
                                  thalach, exang, oldpeak, slope, ca, thal]])
        
        # Make prediction
        if st.button("Predict Heart Disease Risk"):
            prediction = model.predict(input_features)
            probability = model.predict_proba(input_features)[:, 1] if hasattr(model, "predict_proba") else None
            
            if prediction[0] == 0:
                st.success("**Prediction: No Heart Disease**")
            else:
                st.error("**Prediction: Heart Disease Detected**")
            
            if probability is not None:
                st.write(f"**Probability of Heart Disease: {probability:.2%}**")
                
                # Visualize probability
                fig = go.Figure(go.Bar(
                    x=[probability, 1-probability],
                    y=['Heart Disease', 'No Heart Disease'],
                    orientation='h',
                    marker_color=['red', 'green']
                ))
                fig.update_layout(title="Heart Disease Probability",
                                xaxis_title="Probability",
                                xaxis_range=[0, 1])
                st.plotly_chart(fig)
    
    # About section
    else:
        st.header("About This Project")
        st.markdown("""
        This application implements a complete machine learning pipeline for heart disease prediction 
        using the UCI Heart Disease dataset. The project follows these steps:
        
        1. **Data Preprocessing**: Handling missing values, encoding categorical variables, and scaling features.
        2. **Exploratory Data Analysis**: Visualizing distributions, correlations, and patterns in the data.
        3. **Dimensionality Reduction**: Using PCA to reduce feature dimensionality while preserving variance.
        4. **Feature Selection**: Applying various techniques to identify the most important features.
        5. **Model Training**: Building and evaluating multiple classification models.
        6. **Unsupervised Learning**: Applying clustering algorithms to discover patterns in the data.
        7. **Prediction**: Using the trained model to make predictions on new data.
        
        **Models Implemented:**
        - Logistic Regression
        - Decision Trees
        - Random Forest
        - Support Vector Machines
        - K-Means Clustering
        - Hierarchical Clustering
        
        **Techniques Used:**
        - Hyperparameter Tuning with GridSearchCV and RandomizedSearchCV
        - Cross-validation
        - Performance metrics (Accuracy, Precision, Recall, F1-score, ROC-AUC)
        - Data visualization with Plotly and Matplotlib
        """)
        
        st.subheader("Dataset Information")
        st.markdown("""
        The Heart Disease dataset from the UCI Machine Learning Repository contains 303 instances 
        with 14 attributes including demographic, clinical, and diagnostic information.
        
        **Attributes:**
        - age: age in years
        - sex: sex (1 = male; 0 = female)
        - cp: chest pain type (0-3)
        - trestbps: resting blood pressure (in mm Hg)
        - chol: serum cholesterol in mg/dl
        - fbs: fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
        - restecg: resting electrocardiographic results (0-2)
        - thalach: maximum heart rate achieved
        - exang: exercise induced angina (1 = yes; 0 = no)
        - oldpeak: ST depression induced by exercise relative to rest
        - slope: the slope of the peak exercise ST segment (0-2)
        - ca: number of major vessels (0-3) colored by fluoroscopy
        - thal: thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)
        - target: diagnosis of heart disease (0 = no disease, 1-4 = disease presence)
        """)
        
        st.subheader("How to Use This App")
        st.markdown("""
        1. Start with the **Data Overview** to understand the dataset.
        2. Preprocess the data in the **Data Preprocessing** section.
        3. Explore patterns and relationships in the **Exploratory Data Analysis** section.
        4. Reduce dimensionality using **PCA** in the Dimensionality Reduction section.
        5. Select important features in the **Feature Selection** section.
        6. Train and evaluate models in the **Model Training & Evaluation** section.
        7. Apply clustering algorithms in the **Unsupervised Learning** section.
        8. Make predictions on new data in the **Make a Prediction** section.
        """)

if __name__ == "__main__":
    main()