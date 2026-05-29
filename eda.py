import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Color configuration based on requirements
COLORS = {
    'Primary': '#2563EB',    # Blue
    'Secondary': '#7C3AED',  # Purple
    'Success': '#22C55E',    # Green
    'Danger': '#EF4444',     # Red
    'Background': '#F8FAFC', # Off-white
    'Muted': '#64748B',      # Slate
    'LightAccent': '#EFF6FF' # Light Blue
}

def plot_churn_distribution(df):
    """
    Returns an interactive Plotly Pie chart showing the Churn Distribution.
    """
    churn_counts = df['Churn'].value_counts().reset_index()
    churn_counts.columns = ['Churn Status', 'Count']
    # Map Yes -> Churned, No -> Active
    churn_counts['Churn Status'] = churn_counts['Churn Status'].map({'Yes': 'Churned (Yes)', 'No': 'Active (No)'})
    
    fig = px.pie(
        churn_counts, 
        names='Churn Status', 
        values='Count',
        color='Churn Status',
        color_discrete_map={'Active (No)': COLORS['Primary'], 'Churned (Yes)': COLORS['Danger']},
        hole=0.4,
        title="Overall Churn Distribution"
    )
    fig.update_traces(textinfo='percent+value', textfont_size=14, marker=dict(line=dict(color='#FFFFFF', width=2)))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    return fig

def plot_demographics(df, feature):
    """
    Returns a grouped bar chart of Churn by Demographic feature (gender, Partner, Dependents, SeniorCitizen).
    """
    df_temp = df.copy()
    if feature == 'SeniorCitizen':
        df_temp['SeniorCitizen'] = df_temp['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
        
    counts = df_temp.groupby([feature, 'Churn']).size().reset_index(name='Count')
    
    fig = px.bar(
        counts,
        x=feature,
        y='Count',
        color='Churn',
        barmode='group',
        color_discrete_map={'No': COLORS['Primary'], 'Yes': COLORS['Danger']},
        title=f"Churn Distribution by {feature}"
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        xaxis=dict(title=feature, gridcolor='#E2E8F0'),
        yaxis=dict(title="Number of Customers", gridcolor='#E2E8F0'),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig

def plot_contract_analysis(df):
    """
    Returns an interactive bar chart analyzing Contract Types vs Churn.
    """
    counts = df.groupby(['Contract', 'Churn']).size().reset_index(name='Count')
    # Calculate churn rates for details
    totals = df.groupby('Contract').size().reset_index(name='Total')
    merged = pd.merge(counts, totals, on='Contract')
    merged['Percentage'] = (merged['Count'] / merged['Total'] * 100).round(2)
    
    fig = px.bar(
        merged,
        x='Contract',
        y='Count',
        color='Churn',
        barmode='group',
        text=merged['Percentage'].apply(lambda x: f"{x}%"),
        color_discrete_map={'No': COLORS['Primary'], 'Yes': COLORS['Danger']},
        title="Contract Type Churn Analysis"
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        xaxis=dict(gridcolor='#E2E8F0'),
        yaxis=dict(title="Customer Count", gridcolor='#E2E8F0'),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig

def plot_charges_analysis(df):
    """
    Returns an interactive box/violin plot for Monthly Charges vs Churn.
    """
    fig = px.box(
        df,
        x='Churn',
        y='MonthlyCharges',
        color='Churn',
        color_discrete_map={'No': COLORS['Primary'], 'Yes': COLORS['Danger']},
        points="all",
        title="Monthly Charges Distribution by Churn Status"
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        xaxis=dict(title="Churn Status", gridcolor='#E2E8F0'),
        yaxis=dict(title="Monthly Charges ($)", gridcolor='#E2E8F0'),
        showlegend=False
    )
    return fig

def plot_tenure_analysis(df):
    """
    Returns a histogram/KDE-like distribution plot of customer tenure vs Churn.
    """
    fig = px.histogram(
        df,
        x='tenure',
        color='Churn',
        marginal='box',
        barmode='overlay',
        color_discrete_map={'No': COLORS['Primary'], 'Yes': COLORS['Danger']},
        title="Customer Tenure Distribution (Months)"
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        xaxis=dict(title="Tenure (Months)", gridcolor='#E2E8F0'),
        yaxis=dict(title="Count", gridcolor='#E2E8F0'),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig

def plot_correlation_heatmap(df):
    """
    Calculates correlation on numeric columns (and labels encoded columns) and returns a Heatmap.
    """
    # Select numeric columns
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Add engineered features if they exist
    for col in ['CLV', 'AverageMonthlySpend', 'ServiceUsageScore', 'CustomerEngagementScore']:
        if col in df.columns:
            numeric_cols.append(col)
            
    # Include Churn numeric mapping
    df_temp = df[numeric_cols].copy()
    if 'Churn' in df.columns:
        df_temp['Churn'] = df['Churn'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)
        numeric_cols.append('Churn')
        
    corr = df_temp.corr()
    
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title="Feature Correlation Heatmap"
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
    )
    return fig

def plot_churn_by_service(df):
    """
    Plots Churn Rate for different telecom service configurations.
    """
    services = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    rates = []
    
    for service in services:
        if service in df.columns:
            # Group by service option and calculate churn rate
            grp = df.groupby(service)['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
            grp.columns = ['Option', 'ChurnRate']
            grp['Service'] = service
            rates.append(grp)
            
    df_rates = pd.concat(rates, ignore_index=True)
    
    fig = px.bar(
        df_rates,
        x='Option',
        y='ChurnRate',
        color='Service',
        facet_col='Service',
        facet_col_wrap=3,
        color_discrete_sequence=[COLORS['Primary'], COLORS['Secondary'], '#F59E0B', COLORS['Success'], COLORS['Danger'], '#14B8A6'],
        title="Churn Rate (%) by Telecom Service Configuration"
    )
    fig.update_yaxes(title="Churn Rate (%)", gridcolor='#E2E8F0')
    fig.update_xaxes(title="", tickangle=45)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        showlegend=False
    )
    return fig

def plot_retention_trends(df):
    """
    Plots Churn Rate across tenure cohorts (groups of tenure months).
    """
    df_temp = df.copy()
    
    # Categorize tenure into cohorts
    def get_cohort(tenure):
        if tenure <= 12: return '0-1 Year'
        elif tenure <= 24: return '1-2 Years'
        elif tenure <= 36: return '2-3 Years'
        elif tenure <= 48: return '3-4 Years'
        elif tenure <= 60: return '4-5 Years'
        else: return '5+ Years'
        
    df_temp['TenureCohort'] = df_temp['tenure'].apply(get_cohort)
    cohort_order = ['0-1 Year', '1-2 Years', '2-3 Years', '3-4 Years', '4-5 Years', '5+ Years']
    
    counts = df_temp.groupby(['TenureCohort', 'Churn']).size().reset_index(name='Count')
    totals = df_temp.groupby('TenureCohort').size().reset_index(name='Total')
    merged = pd.merge(counts, totals, on='TenureCohort')
    merged['ChurnRate'] = (merged['Count'] / merged['Total'] * 100).round(2)
    merged = merged[merged['Churn'] == 'Yes']
    
    # Fill in cohorts with 0% churn if missing
    for c in cohort_order:
        if c not in merged['TenureCohort'].values:
            merged = pd.concat([merged, pd.DataFrame([{'TenureCohort': c, 'Churn': 'Yes', 'Count': 0, 'Total': 1, 'ChurnRate': 0.0}])], ignore_index=True)
            
    # Sort cohorts
    merged['TenureCohort'] = pd.Categorical(merged['TenureCohort'], categories=cohort_order, ordered=True)
    merged = merged.sort_values('TenureCohort')
    
    fig = px.line(
        merged,
        x='TenureCohort',
        y='ChurnRate',
        markers=True,
        color_discrete_sequence=[COLORS['Secondary']],
        title="Customer Churn Rate by Tenure Cohort"
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        xaxis=dict(gridcolor='#E2E8F0'),
        yaxis=dict(title="Churn Rate (%)", gridcolor='#E2E8F0', range=[0, max(merged['ChurnRate'].max() + 5, 20)]),
    )
    return fig

def plot_feature_importance(importances, feature_names, top_n=15):
    """
    Plots model feature importance.
    """
    df_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True).tail(top_n)
    
    fig = px.bar(
        df_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        color_discrete_sequence=[COLORS['Primary']],
        title=f"Top {top_n} Features by Importance"
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        xaxis=dict(title="Relative Importance Score", gridcolor='#E2E8F0'),
        yaxis=dict(title="Feature Name", gridcolor='#E2E8F0')
    )
    return fig
