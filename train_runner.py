import os
import sys
import pandas as pd
import numpy as np

# Ensure src directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import generate_synthetic_data, generate_pdf_report
from src.data_preprocessing import load_and_clean_data, get_train_test_split, create_preprocessing_pipeline
from src.feature_engineering import engineer_features
from src.model_training import train_logistic_regression, train_random_forest, train_xgboost, save_model_artifacts
from src.evaluation import get_model_comparison_metrics

def run_pipeline():
    print("=" * 60)
    print("STARTING CUSTOMER CHURN SYSTEM INITIALIZATION PIPELINE")
    print("=" * 60)
    
    # 1. Generate Synthetic Data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "telecom_churn.csv"))
    print(f"Checking for dataset at {data_path}...")
    if not os.path.exists(data_path):
        print("Dataset not found. Generating synthetic dataset (10,000 customers)...")
        df = generate_synthetic_data(file_path=data_path, num_records=10000, seed=42)
    else:
        print("Dataset found. Loading existing data...")
        df = pd.read_csv(data_path)
        
    # 2. Clean and Preprocess
    print("Cleaning dataset...")
    df_cleaned = load_and_clean_data(data_path)
    
    print("Engineering features...")
    df_engineered = engineer_features(df_cleaned)
    
    # 3. Features definition
    categorical_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'Contract', 'PaymentMethod'
    ]
    
    numerical_cols = [
        'tenure', 'MonthlyCharges', 'TotalCharges', 'CLV', 'AverageMonthlySpend', 
        'ServiceUsageScore', 'CustomerEngagementScore'
    ]
    
    # 4. Train-Test Split
    print("Splitting data into train and test sets (80/20)...")
    X_train, X_test, y_train, y_test = get_train_test_split(df_engineered, target_col='Churn', test_size=0.2, random_state=42)
    
    # Save test set evaluation copy for verify
    # 5. Fit Preprocessor Pipeline
    print("Configuring preprocessing pipeline...")
    preprocessor = create_preprocessing_pipeline(categorical_cols, numerical_cols)
    
    # 6. Train and Tune Models
    print("Starting ML Model training with hyperparameter search...")
    
    models_dict = {}
    tuning_params = {}
    training_times = {}
    
    # Logistic Regression
    lr_best, lr_params, lr_time = train_logistic_regression(X_train, y_train, preprocessor, cv=3)
    models_dict["Logistic Regression"] = lr_best
    tuning_params["Logistic Regression"] = lr_params
    training_times["Logistic Regression"] = lr_time
    print(f"Logistic Regression trained. Best Params: {lr_params}. Time: {lr_time:.2f}s")
    
    # Random Forest
    rf_best, rf_params, rf_time = train_random_forest(X_train, y_train, preprocessor, cv=3)
    models_dict["Random Forest"] = rf_best
    tuning_params["Random Forest"] = rf_params
    training_times["Random Forest"] = rf_time
    print(f"Random Forest trained. Best Params: {rf_params}. Time: {rf_time:.2f}s")
    
    # XGBoost
    xgb_best, xgb_params, xgb_time = train_xgboost(X_train, y_train, preprocessor, cv=3)
    models_dict["XGBoost"] = xgb_best
    tuning_params["XGBoost"] = xgb_params
    training_times["XGBoost"] = xgb_time
    print(f"XGBoost trained. Best Params: {xgb_params}. Time: {xgb_time:.2f}s")
    
    # 7. Evaluate and Compare Models
    print("\nEvaluating trained models on the test set...")
    df_metrics = get_model_comparison_metrics(models_dict, X_test, y_test)
    print("\nModel Evaluation Performance Metrics:")
    print("-" * 70)
    print(df_metrics.to_string(index=False))
    print("-" * 70)
    
    # Select Best Model based on F1 Score or ROC-AUC
    best_row = df_metrics.loc[df_metrics['F1 Score'].idxmax()]
    best_model_name = best_row['Model']
    best_f1 = best_row['F1 Score']
    print(f"\n---> Best Performing Model (by F1 Score): {best_model_name} (F1 Score: {best_f1:.4f})")
    
    # 8. Save artifacts (Pipelines and Preprocessor)
    print("\nSaving trained model pipelines and preprocessor...")
    save_model_artifacts(models_dict, preprocessor)
    
    # 9. Generate PDF Report
    print("\nCompiling executive analysis PDF report...")
    metrics_report_dict = {}
    for idx, row in df_metrics.iterrows():
        metrics_report_dict[row['Model']] = {
            "Accuracy": row['Accuracy'],
            "Precision": row['Precision'],
            "Recall": row['Recall'],
            "F1 Score": row['F1 Score'],
            "ROC-AUC Score": row['ROC-AUC Score']
        }
    
    report_path = generate_pdf_report(df_engineered, metrics_report_dict, best_model_name)
    print(f"PDF report successfully saved to {report_path}")
    
    print("\n" + "=" * 60)
    print("INITIALIZATION COMPLETE! READY TO RUN STREAMLIT APP.")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()
