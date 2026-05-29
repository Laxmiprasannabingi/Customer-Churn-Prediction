import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)

COLORS = {
    'Primary': '#2563EB',    # Blue
    'Secondary': '#7C3AED',  # Purple
    'Success': '#22C55E',    # Green
    'Danger': '#EF4444',     # Red
    'Background': '#F8FAFC', # Off-white
    'Muted': '#64748B',      # Slate
    'LightAccent': '#EFF6FF' # Light Blue
}

def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculates classification evaluation metrics.
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC Score": roc_auc_score(y_true, y_prob) if y_prob is not None else 0.5
    }

def get_model_comparison_metrics(models_dict, X_test, y_test):
    """
    Evaluates all models in the dictionary and returns a comparison DataFrame.
    """
    metrics_list = []
    
    for name, pipeline in models_dict.items():
        y_pred = pipeline.predict(X_test)
        
        # Check if classifier supports predict_proba
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            # Fallback for models without predict_proba
            y_prob = pipeline.decision_function(X_test) if hasattr(pipeline, "decision_function") else y_pred
            
        m = calculate_metrics(y_test, y_pred, y_prob)
        m["Model"] = name
        metrics_list.append(m)
        
    df_metrics = pd.DataFrame(metrics_list)
    # Reorder columns
    cols = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC Score"]
    return df_metrics[cols]

def plot_confusion_matrix_plotly(y_true, y_pred, model_name="Model"):
    """
    Generates an interactive Plotly Heatmap for the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    # cm layout:
    # [[TN, FP],
    #  [FN, TP]]
    
    z = cm.tolist()
    x = ['Predicted Active (No Churn)', 'Predicted Churned']
    y = ['Actual Active (No Churn)', 'Actual Churned']
    
    # Text annotations
    labels = [
        [f"True Negative<br>{cm[0][0]}", f"False Positive<br>{cm[0][1]}"],
        [f"False Negative<br>{cm[1][0]}", f"True Positive<br>{cm[1][1]}"]
    ]
    
    fig = ff_heatmap_custom(z, x, y, labels, title=f"Confusion Matrix - {model_name}")
    return fig

def ff_heatmap_custom(z, x, y, labels, title):
    """
    Helper to generate custom styled heatmap since plotly.figure_factory.create_annotated_heatmap 
    can sometimes have import issues.
    """
    fig = go.Figure(data=go.Heatmap(
        z=z, x=x, y=y,
        colorscale=[[0, '#EFF6FF'], [0.5, '#BFDBFE'], [1, '#2563EB']], # Standard shades of Blue
        showscale=True
    ))
    
    # Add annotations
    for i in range(len(y)):
        for j in range(len(x)):
            fig.add_annotation(
                x=x[j], y=y[i],
                text=labels[i][j],
                showarrow=False,
                font=dict(color='black' if z[i][j] < (np.max(z) * 0.6) else 'white', size=13, family="Inter, sans-serif")
            )
            
    fig.update_layout(
        title=title,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed")
    )
    return fig

def plot_roc_curves_plotly(models_dict, X_test, y_test):
    """
    Plots ROC curves for all models in the dictionary on a single chart.
    """
    fig = go.Figure()
    
    # Add reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(color=COLORS['Muted'], width=2, dash='dash'),
        name='Random Guess (AUC = 0.50)',
        showlegend=True
    ))
    
    color_palette = [COLORS['Primary'], COLORS['Secondary'], '#F59E0B']
    
    for idx, (name, pipeline) in enumerate(models_dict.items()):
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_prob = pipeline.decision_function(X_test) if hasattr(pipeline, "decision_function") else pipeline.predict(X_test)
            
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        
        color = color_palette[idx % len(color_palette)]
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            line=dict(color=color, width=3),
            name=f"{name} (AUC = {auc_val:.4f})"
        ))
        
    fig.update_layout(
        title="ROC-AUC Curve Comparison",
        xaxis=dict(title="False Positive Rate", gridcolor='#E2E8F0', range=[-0.01, 1.01]),
        yaxis=dict(title="True Positive Rate", gridcolor='#E2E8F0', range=[-0.01, 1.01]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig

def plot_metrics_comparison_plotly(df_metrics):
    """
    Plots a bar chart comparing performance metrics across all models.
    """
    # Melt the dataframe for plotting
    df_melted = df_metrics.melt(id_vars="Model", var_name="Metric", value_name="Score")
    
    fig = px.bar(
        df_melted,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        color_discrete_map={
            'Logistic Regression': COLORS['Primary'],
            'Random Forest': COLORS['Secondary'],
            'XGBoost': '#F59E0B'
        },
        title="Model Metrics Comparison"
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        yaxis=dict(title="Score", range=[0, 1.05], gridcolor='#E2E8F0'),
        xaxis=dict(title="Metric"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig

def get_classification_report_df(y_true, y_pred):
    """
    Generates a classification report and structures it as a readable Pandas DataFrame.
    """
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    # Format percentages and decimals
    df_report = df_report.round(4)
    # Rename index labels for presentation
    df_report.index = [
        'Active (No Churn)' if idx == '0' else 
        'Churned' if idx == '1' else idx 
        for idx in df_report.index
    ]
    return df_report
