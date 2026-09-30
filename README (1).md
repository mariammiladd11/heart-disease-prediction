# Heart Disease Prediction

An end-to-end machine learning project that predicts the presence of heart disease from patient clinical data. It covers data preprocessing, exploratory analysis, feature selection, dimensionality reduction, model comparison with hyperparameter tuning, clustering, and an interactive Streamlit web app.

## Features

- **Data preprocessing:** missing-value handling, scaling (StandardScaler / MinMaxScaler) and binary target conversion
- **Exploratory data analysis:** correlation heatmap, target distribution, and interactive feature plots
- **Feature selection:** Random Forest importance, RFE, and SelectKBest
- **Dimensionality reduction:** PCA with explained-variance analysis
- **Model training and comparison:** 7 classifiers
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine
  - K-Nearest Neighbors
  - Gradient Boosting
  - Naive Bayes
- **Hyperparameter tuning:** GridSearchCV
- **Evaluation:** accuracy, precision, recall, F1-score, ROC-AUC, ROC and precision-recall curves, confusion matrix
- **Unsupervised learning:** K-Means and hierarchical clustering with PCA visualization
- **Prediction page:** enter patient values in the app and get a prediction from the trained model

## Dataset

[UCI Heart Disease dataset (Cleveland)](https://archive.ics.uci.edu/dataset/45/heart+disease): 303 patients and 13 clinical features (age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, maximum heart rate, exercise-induced angina, ST depression, slope, number of major vessels, and thalassemia).

The target is converted to binary: `0` = no heart disease, `1` = heart disease present.

## Tech Stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Plotly, SciPy, Streamlit, joblib

## Project Structure

```
.
├── data_preprocessing.py    # Loads and preprocesses the dataset
├── train_model.py           # Feature selection, model training, tuning and evaluation
├── run_training.py          # Installs requirements and runs the training pipeline
├── model_evaluation.py      # Streamlit app for evaluating the trained model
├── heart_disease_app.py     # Main Streamlit app (analysis, training, clustering, prediction)
├── requirements.txt         # Python dependencies
└── README.md
```

Running the training script also creates the `models/` (saved model, scaler and selected features) and `results/` (plots) folders.

## Installation

Python 3.9 or newer (developed and tested on Python 3.13).

```bash
git clone https://github.com/mariammiladd11/heart-disease-prediction.git
cd heart-disease-prediction

# optional: create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

## Usage

**1. Train the models** (creates the `models/` and `results/` folders):

```bash
python train_model.py
```

Or run `python run_training.py`, which installs the requirements and then trains.

**2. Launch the web app:**

```bash
streamlit run heart_disease_app.py
```

Use the sidebar to move between the sections: Data Overview, Data Preprocessing, Exploratory Data Analysis, Dimensionality Reduction, Feature Selection, Model Training & Evaluation, Unsupervised Learning, and Make a Prediction.

**3. (Optional) Evaluate the trained model:**

```bash
streamlit run model_evaluation.py
```

## Results

Trained on an 80/20 stratified train-test split (61 test patients). The best model was selected by ROC-AUC.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Tuned Logistic Regression (best)** | 0.8852 | 0.8621 | 0.8929 | 0.8772 | **0.9621** |
| Tuned Random Forest | 0.9016 | 0.8667 | 0.9286 | 0.8966 | 0.9491 |
| Tuned Gradient Boosting | 0.8689 | 0.8571 | 0.8571 | 0.8571 | 0.9351 |
| Naive Bayes (untuned) | 0.9016 | 0.8235 | 1.0000 | 0.9032 | N/A |
| Logistic Regression (untuned) | 0.8689 | 0.8125 | 0.9286 | 0.8667 | N/A |
| Random Forest (untuned) | 0.8689 | 0.8125 | 0.9286 | 0.8667 | N/A |
| Support Vector Machine | 0.8525 | 0.8065 | 0.8929 | 0.8475 | N/A |
| Gradient Boosting (untuned) | 0.8525 | 0.7879 | 0.9286 | 0.8525 | N/A |
| K-Nearest Neighbors | 0.8197 | 0.7576 | 0.8929 | 0.8197 | N/A |
| Decision Tree | 0.7049 | 0.6667 | 0.7143 | 0.6897 | N/A |

The test set is small (61 patients), so these numbers carry some uncertainty. Cross-validation accuracy for the untuned models ranged from about 0.77 to 0.85.

## Screenshots

<!-- Add screenshots of the app here, for example:
![App screenshot](results/app_screenshot.png)
-->

## Disclaimer

This project is for educational purposes only and is not a medical diagnostic tool.

## Author

**Mariam Milad**: [GitHub](https://github.com/mariammiladd11) | [LinkedIn](https://linkedin.com/in/mariammiladd11)
