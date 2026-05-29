import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Applies feature engineering to the input DataFrame:
    1. Customer Lifetime Value (CLV)
    2. Average Monthly Spend
    3. Service Usage Score
    4. Customer Engagement Score
    5. Heuristic Risk Category
    """
    # Create a copy to prevent SettingWithCopyWarning
    df_engineered = df.copy()
    
    # 1. Customer Lifetime Value (CLV)
    # Simple proxy: MonthlyCharges * tenure
    df_engineered['CLV'] = df_engineered['MonthlyCharges'] * df_engineered['tenure']
    
    # 2. Average Monthly Spend
    # Estimated as TotalCharges / (tenure + 1) to avoid division by zero
    df_engineered['AverageMonthlySpend'] = np.round(
        df_engineered['TotalCharges'] / (df_engineered['tenure'] + 1), 2
    )
    
    # 3. Service Usage Score
    # Count of active services: PhoneService, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport
    services = ['PhoneService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    
    service_count = np.zeros(len(df_engineered))
    for s in services:
        if s in df_engineered.columns:
            service_count += df_engineered[s].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
            
    if 'InternetService' in df_engineered.columns:
        service_count += df_engineered['InternetService'].apply(lambda x: 1 if str(x).strip().lower() in ['dsl', 'fiber optic'] else 0)
        
    df_engineered['ServiceUsageScore'] = service_count
    
    # 4. Customer Engagement Score
    # Multiply tenure by contract weight (Month-to-month = 1, One year = 2, Two year = 3)
    if 'Contract' in df_engineered.columns:
        contract_weight = df_engineered['Contract'].map({
            'Month-to-month': 1,
            'One year': 2,
            'Two year': 3
        }).fillna(1)
        df_engineered['CustomerEngagementScore'] = df_engineered['tenure'] * contract_weight
    else:
        df_engineered['CustomerEngagementScore'] = df_engineered['tenure']
        
    # 5. Risk Category (Heuristic)
    # A rule-based risk category prior to machine learning prediction
    def get_heuristic_risk(row):
        tenure_val = row.get('tenure', 0)
        contract_val = str(row.get('Contract', '')).strip().lower()
        internet_val = str(row.get('InternetService', '')).strip().lower()
        
        # High Risk: Month-to-month contract, low tenure (< 12), and has fiber optic or no tech support
        if contract_val == 'month-to-month' and tenure_val < 12:
            return 'High Risk'
        # Low Risk: Two year contract, high tenure (> 24)
        elif contract_val == 'two year' and tenure_val > 24:
            return 'Low Risk'
        # Medium Risk: Everything else
        else:
            return 'Medium Risk'
            
    df_engineered['RiskCategory'] = df_engineered.apply(get_heuristic_risk, axis=1)
    
    return df_engineered
