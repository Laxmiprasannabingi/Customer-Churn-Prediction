import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

def train_logistic_regression(X_train, y_train, preprocessor, cv=3):
    """
    Trains and tunes Logistic Regression model.
    """
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    param_grid = {
        'classifier__C': [0.01, 0.1, 1.0, 10.0],
        'classifier__penalty': ['l2']
    }
    
    print("Tuning Logistic Regression...")
    start_time = time.time()
    grid_search = GridSearchCV(lr_pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    return grid_search.best_estimator_, grid_search.best_params_, training_time

def train_random_forest(X_train, y_train, preprocessor, cv=3):
    """
    Trains and tunes Random Forest Classifier.
    """
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    
    param_grid = {
        'classifier__n_estimators': [50, 100, 150],
        'classifier__max_depth': [5, 10, 15],
        'classifier__min_samples_split': [2, 5]
    }
    
    print("Tuning Random Forest...")
    start_time = time.time()
    grid_search = GridSearchCV(rf_pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    return grid_search.best_estimator_, grid_search.best_params_, training_time

def train_xgboost(X_train, y_train, preprocessor, cv=3):
    """
    Trains and tunes XGBoost Classifier.
    """
    xgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1))
    ])
    
    param_grid = {
        'classifier__n_estimators': [50, 100, 150],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.1, 0.2]
    }
    
    print("Tuning XGBoost...")
    start_time = time.time()
    grid_search = GridSearchCV(xgb_pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    return grid_search.best_estimator_, grid_search.best_params_, training_time

def save_model_artifacts(models_dict, preprocessor):
    """
    Saves trained pipeline objects and preprocessor to disk.
    """
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR, exist_ok=True)
        
    # Save each model pipeline
    for name, model_pipeline in models_dict.items():
        filename = f"{name.lower().replace(' ', '_')}.pkl"
        filepath = os.path.join(MODELS_DIR, filename)
        joblib.dump(model_pipeline, filepath)
        print(f"Saved model pipeline to {filepath}")
        
    # Save scaler/preprocessor separately
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Saved standalone preprocessor to {preprocessor_path}")

def load_model_artifacts():
    """
    Loads saved model pipelines and preprocessor from disk.
    Returns: models_dict, preprocessor
    """
    models_dict = {}
    model_names = ["logistic_regression", "random_forest", "xgboost"]
    
    for name in model_names:
        filepath = os.path.join(MODELS_DIR, f"{name}.pkl")
        if os.path.exists(filepath):
            models_dict[name.replace('_', ' ').title()] = joblib.load(filepath)
            
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
    preprocessor = joblib.load(preprocessor_path) if os.path.exists(preprocessor_path) else None
    
    return models_dict, preprocessor
