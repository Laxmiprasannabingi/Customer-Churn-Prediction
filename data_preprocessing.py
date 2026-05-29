import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_and_clean_data(file_path):
    """
    Loads telecom churn dataset from CSV and cleans it.
    - Handles empty space characters in 'TotalCharges'
    - Imputes null values
    - Removes duplicates
    """
    df = pd.read_csv(file_path)
    
    # Strip whitespace from string columns and column names
    df.columns = [col.strip() for col in df.columns]
    
    string_cols = df.select_dtypes(include=['object']).columns
    for col in string_cols:
        df[col] = df[col].astype(str).str.strip()
        
    # Convert 'TotalCharges' to numeric, replacing spaces with NaN
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = df['TotalCharges'].replace('', np.nan)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Impute missing TotalCharges with MonthlyCharges * tenure as a logical default
        if 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
            missing_total = df['TotalCharges'].isnull()
            df.loc[missing_total, 'TotalCharges'] = df.loc[missing_total, 'MonthlyCharges'] * df.loc[missing_total, 'tenure']
            
    # Handle missing values for other columns if any
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
                
    # Ensure SeniorCitizen is integer 0 or 1
    if 'SeniorCitizen' in df.columns:
        df['SeniorCitizen'] = df['SeniorCitizen'].fillna(0).astype(int)
        
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Convert customerID to string and ensure uppercase
    if 'customerID' in df.columns:
        df['customerID'] = df['customerID'].astype(str).str.upper()
        
    return df

def detect_outliers_iqr(df, numerical_cols):
    """
    Detects outliers using the Interquartile Range (IQR) method.
    Returns a dictionary of outlier counts and details.
    """
    outliers_info = {}
    for col in numerical_cols:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outliers_info[col] = {
                "count": len(outliers),
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "indices": outliers.index.tolist()
            }
    return outliers_info

def get_train_test_split(df, target_col='Churn', test_size=0.2, random_state=42):
    """
    Splits the dataframe into train and test sets, stratifying by target.
    """
    X = df.drop(columns=[target_col])
    
    # Convert target Churn 'Yes'/'No' to 1/0
    y = df[target_col].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def create_preprocessing_pipeline(categorical_cols, numerical_cols):
    """
    Creates a sklearn ColumnTransformer preprocessing pipeline.
    Numerical: Imputation + Scaling
    Categorical: Imputation + One-Hot Encoding
    """
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    return preprocessor
