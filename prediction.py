import pandas as pd
import numpy as np
from src.feature_engineering import engineer_features

def get_retention_recommendation(probability):
    """
    Returns the retention strategy details based on churn probability.
    """
    prob_pct = probability * 100
    
    if probability < 0.20:
        return {
            "risk_level": "Low Risk",
            "strategy": "Loyalty & Relationship Nurturing",
            "color": "#22C55E", # Success Green
            "actions": [
                "Enroll in the VIP Loyalty Rewards Program.",
                "Send proactive newsletters highlighting new service features.",
                "Offer voluntary feedback surveys and product satisfaction checks.",
                "Incentivize self-service channels and digital app adoption."
            ]
        }
    elif probability <= 0.50:
        return {
            "risk_level": "Medium Risk",
            "strategy": "Value Optimization & Personalization",
            "color": "#F59E0B", # Amber
            "actions": [
                "Propose contract upgrades (e.g., month-to-month to 1-year contract) with a 10% discount.",
                "Schedule a personalized check-in call from a customer success advisor.",
                "Audit usage patterns and recommend service add-ons or plan down-sizes to fit their budget.",
                "Send target email campaigns with customized service bundle offers."
            ]
        }
    else:
        return {
            "risk_level": "High Risk",
            "strategy": "Aggressive Retention & Support",
            "color": "#EF4444", # Danger Red
            "actions": [
                "Offer an immediate 25-30% discount on monthly charges for the next 6 months.",
                "Provide a free tech support add-on or device upgrade options.",
                "Assign a dedicated account retention specialist to resolve outstanding tickets.",
                "Invite to sign a multi-year renewal contract at customized contract rates."
            ]
        }

def predict_single_customer(customer_data, model_pipeline):
    """
    Predicts churn for a single customer profile.
    customer_data: dict of customer features
    model_pipeline: loaded sklearn pipeline (includes preprocessor + classifier)
    """
    # Convert single dictionary to DataFrame
    df = pd.DataFrame([customer_data])
    
    # Run through feature engineering
    df_engineered = engineer_features(df)
    
    # Ensure correct data type formatting
    if 'TotalCharges' in df_engineered.columns:
        df_engineered['TotalCharges'] = pd.to_numeric(df_engineered['TotalCharges'], errors='coerce')
    if 'MonthlyCharges' in df_engineered.columns:
        df_engineered['MonthlyCharges'] = pd.to_numeric(df_engineered['MonthlyCharges'], errors='coerce')
    if 'tenure' in df_engineered.columns:
        df_engineered['tenure'] = pd.to_numeric(df_engineered['tenure'], errors='coerce').astype(int)
    if 'SeniorCitizen' in df_engineered.columns:
        df_engineered['SeniorCitizen'] = pd.to_numeric(df_engineered['SeniorCitizen'], errors='coerce').astype(int)
        
    # Predict Churn Probability
    if hasattr(model_pipeline, "predict_proba"):
        probability = model_pipeline.predict_proba(df_engineered)[0, 1]
    else:
        probability = model_pipeline.predict(df_engineered)[0] # Fallback
        
    prediction_label = "Yes" if probability >= 0.50 else "No"
    
    # Get retention strategy mapping
    rec = get_retention_recommendation(probability)
    
    return {
        "churn_probability": probability,
        "churn_prediction": prediction_label,
        "risk_level": rec["risk_level"],
        "strategy": rec["strategy"],
        "actions": rec["actions"],
        "color": rec["color"]
    }

def predict_batch_customers(df, model_pipeline):
    """
    Predicts churn for a batch of customers.
    df: DataFrame containing the raw customer records.
    model_pipeline: loaded sklearn pipeline.
    """
    df_temp = df.copy()
    
    # Apply feature engineering
    df_engineered = engineer_features(df_temp)
    
    # Run predictions
    if hasattr(model_pipeline, "predict_proba"):
        probabilities = model_pipeline.predict_proba(df_engineered)[:, 1]
    else:
        probabilities = model_pipeline.predict(df_engineered)
        
    df_temp['Churn_Probability'] = probabilities
    df_temp['Churn_Prediction'] = df_temp['Churn_Probability'].apply(lambda x: 'Yes' if x >= 0.50 else 'No')
    
    # Map risk levels
    risk_info = df_temp['Churn_Probability'].apply(get_retention_recommendation)
    df_temp['Risk_Level'] = [r['risk_level'] for r in risk_info]
    df_temp['Retention_Strategy'] = [r['strategy'] for r in risk_info]
    
    return df_temp
