import os
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Database path configuration
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "churn.db"))
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "telecom_churn.csv"))
REPORTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))

def get_db_connection():
    """Establishes connection to the SQLite database."""
    db_dir = os.path.dirname(DB_PATH)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database schema if it doesn't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customerID TEXT,
            gender TEXT,
            SeniorCitizen INTEGER,
            Partner TEXT,
            Dependents TEXT,
            tenure INTEGER,
            PhoneService TEXT,
            InternetService TEXT,
            OnlineSecurity TEXT,
            OnlineBackup TEXT,
            DeviceProtection TEXT,
            TechSupport TEXT,
            Contract TEXT,
            PaymentMethod TEXT,
            MonthlyCharges REAL,
            TotalCharges REAL,
            prediction TEXT,
            probability REAL,
            risk_level TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_prediction(customer_data, prediction, probability, risk_level):
    """
    Saves a prediction record into the SQLite database.
    customer_data is a dictionary containing the customer features.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute("""
        INSERT INTO predictions (
            customerID, gender, SeniorCitizen, Partner, Dependents, tenure,
            PhoneService, InternetService, OnlineSecurity, OnlineBackup,
            DeviceProtection, TechSupport, Contract, PaymentMethod,
            MonthlyCharges, TotalCharges, prediction, probability, risk_level, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        customer_data.get("customerID", "N/A"),
        customer_data.get("gender"),
        int(customer_data.get("SeniorCitizen", 0)),
        customer_data.get("Partner"),
        customer_data.get("Dependents"),
        int(customer_data.get("tenure", 0)),
        customer_data.get("PhoneService"),
        customer_data.get("InternetService"),
        customer_data.get("OnlineSecurity"),
        customer_data.get("OnlineBackup"),
        customer_data.get("DeviceProtection"),
        customer_data.get("TechSupport"),
        customer_data.get("Contract"),
        customer_data.get("PaymentMethod"),
        float(customer_data.get("MonthlyCharges", 0.0)),
        float(customer_data.get("TotalCharges", 0.0)),
        prediction,
        float(probability),
        risk_level,
        timestamp
    ))
    conn.commit()
    conn.close()

def get_prediction_history(limit=100):
    """Retrieves prediction history from the database as a pandas DataFrame."""
    init_db()
    conn = get_db_connection()
    query = f"SELECT * FROM predictions ORDER BY id DESC LIMIT {limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def generate_synthetic_data(file_path=DATA_PATH, num_records=10000, seed=42):
    """
    Generates a realistic synthetic telecom customer churn dataset.
    Saves to the specified CSV file path.
    """
    np.random.seed(seed)
    
    # 1. Customer IDs
    customer_ids = [f"{np.random.randint(1000, 9999)}-{np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5)}" for _ in range(num_records)]
    
    # 2. Demographic Features
    gender = np.random.choice(["Male", "Female"], size=num_records)
    senior_citizen = np.random.choice([0, 1], size=num_records, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], size=num_records, p=[0.48, 0.52])
    dependents = np.random.choice(["Yes", "No"], size=num_records, p=[0.30, 0.70])
    
    # 3. Account Features
    tenure = np.random.geometric(p=0.03, size=num_records)
    # Clip tenure at 72 months (typical 6 years max contract)
    tenure = np.clip(tenure, 1, 72)
    
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], size=num_records, p=[0.55, 0.20, 0.25])
    
    phone_service = np.random.choice(["Yes", "No"], size=num_records, p=[0.90, 0.10])
    
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=num_records, p=[0.35, 0.45, 0.20])
    
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=num_records,
        p=[0.34, 0.23, 0.22, 0.21]
    )
    
    # Dependent services
    online_security = []
    online_backup = []
    device_protection = []
    tech_support = []
    
    for i in range(num_records):
        if internet_service[i] == "No":
            online_security.append("No internet service")
            online_backup.append("No internet service")
            device_protection.append("No internet service")
            tech_support.append("No internet service")
        else:
            online_security.append(np.random.choice(["Yes", "No"], p=[0.40, 0.60]))
            online_backup.append(np.random.choice(["Yes", "No"], p=[0.45, 0.55]))
            device_protection.append(np.random.choice(["Yes", "No"], p=[0.45, 0.55]))
            tech_support.append(np.random.choice(["Yes", "No"], p=[0.40, 0.60]))
            
    # 4. Charges (Monthly & Total)
    monthly_charges = np.zeros(num_records)
    for i in range(num_records):
        base = 20.0 # Base charges
        if phone_service[i] == "Yes":
            base += 15.0
        if internet_service[i] == "DSL":
            base += 30.0
        elif internet_service[i] == "Fiber optic":
            base += 50.0
            
        if online_security[i] == "Yes": base += 5.0
        if online_backup[i] == "Yes": base += 5.0
        if device_protection[i] == "Yes": base += 5.0
        if tech_support[i] == "Yes": base += 5.0
        
        # Add small random variation
        monthly_charges[i] = round(base + np.random.normal(0, 4.0), 2)
    # Clip monthly charges to realistic bounds
    monthly_charges = np.clip(monthly_charges, 18.0, 120.0)
    
    total_charges = []
    for i in range(num_records):
        # Total charges is approximately tenure * monthly charges
        mult = tenure[i]
        tc = round(monthly_charges[i] * mult + np.random.normal(0, 15.0), 2)
        if tc < monthly_charges[i]:
            tc = monthly_charges[i]
        total_charges.append(tc)
    
    # Introduce small percentage (0.1%) of missing TotalCharges for data preprocessing test
    total_charges = np.array(total_charges, dtype=object)
    missing_indices = np.random.choice(num_records, size=int(num_records * 0.001), replace=False)
    for idx in missing_indices:
        total_charges[idx] = " " # Empty space representing missing values as in standard Telco Dataset
        
    # 5. Churn Probability Logic
    churn_prob = np.zeros(num_records)
    churn = []
    for i in range(num_records):
        # Calculate Churn log-odds based on attributes
        log_odds = -1.2 # Intercept (moderately low base churn)
        
        # Tenure: strong negative correlation
        log_odds -= 0.06 * tenure[i]
        
        # Contract: Month-to-month has high churn, long contracts are loyal
        if contract[i] == "Month-to-month":
            log_odds += 1.8
        elif contract[i] == "Two year":
            log_odds -= 1.8
        elif contract[i] == "One year":
            log_odds -= 0.6
            
        # Internet Service: Fiber optic has higher churn (due to higher cost / higher expectations)
        if internet_service[i] == "Fiber optic":
            log_odds += 0.8
        elif internet_service[i] == "No":
            log_odds -= 0.8
            
        # Tech Support & Security: No internet service doesn't apply, but No support increases churn
        if tech_support[i] == "No":
            log_odds += 0.6
        if online_security[i] == "No":
            log_odds += 0.5
            
        # Senior Citizens have higher churn
        if senior_citizen[i] == 1:
            log_odds += 0.3
            
        # Dependents: Customers with dependents are less likely to churn
        if dependents[i] == "Yes":
            log_odds -= 0.3
            
        # Monthly Charges: higher costs increase churn slightly
        log_odds += 0.01 * (monthly_charges[i] - 60.0)
        
        # Payment Method: Electronic check users churn more
        if payment_method[i] == "Electronic check":
            log_odds += 0.5
            
        # Convert log-odds to probability
        prob = 1 / (1 + np.exp(-log_odds))
        churn_prob[i] = prob
        
        # Decide churn based on probability
        if np.random.rand() < prob:
            churn.append("Yes")
        else:
            churn.append("No")
            
    # Combine into DataFrame
    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "Contract": contract,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn
    })
    
    # Make sure folder exists
    data_dir = os.path.dirname(file_path)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        
    df.to_csv(file_path, index=False)
    print(f"Generated synthetic dataset with {num_records} records at {file_path}")
    return df

def generate_pdf_report(df, metrics, best_model_name, report_path=None):
    """
    Generates a professional PDF report containing the Churn analysis and model evaluation metrics.
    """
    if report_path is None:
        if not os.path.exists(REPORTS_PATH):
            os.makedirs(REPORTS_PATH, exist_ok=True)
        report_path = os.path.join(REPORTS_PATH, "churn_analysis_report.pdf")
        
    # Calculate some summary stats
    total_customers = len(df)
    
    # Ensure Churn is treated properly
    churn_col = df["Churn"].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)
    churn_rate = (churn_col.mean() * 100)
    active_customers = total_customers - churn_col.sum()
    churned_customers = churn_col.sum()
    
    # Estimate Revenue Impact (Assume total monthly charge of churned customers is lost monthly)
    monthly_loss = df[df["Churn"].apply(lambda x: str(x).strip().lower() == "yes")]["MonthlyCharges"].sum()
    annual_loss = monthly_loss * 12

    doc = SimpleDocTemplate(
        report_path,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles mapping the app's color palette
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#2563EB'), # Primary color
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#7C3AED'), # Secondary color
        spaceAfter=25
    )
    
    h1_style = ParagraphStyle(
        'Heading1Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyCustom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceAfter=10
    )
    
    table_text_style = ParagraphStyle(
        'TableText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#1E293B')
    )
    
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.white
    )

    story = []
    
    # Header Section
    story.append(Paragraph("Customer Churn Prediction System", title_style))
    story.append(Paragraph(f"Executive Churn Analysis & ML Evaluation Report &bull; Generated: {datetime.now().strftime('%Y-%m-%d')}", subtitle_style))
    story.append(Spacer(1, 10))
    
    # Executive Summary Section
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(Paragraph(
        f"This report presents the customer retention metrics and machine learning model performances based on "
        f"historical analysis of {total_customers:,} active subscriber records. Our predictive system is designed "
        f"to preemptively flag customers with high churn risk, allowing target retention operations to mitigate "
        f"avoidable revenue leakage.",
        body_style
    ))
    
    # KPI Grid Table
    kpi_data = [
        [
            Paragraph("<b>Total Customers</b>", table_text_style), 
            Paragraph(f"{total_customers:,}", table_text_style),
            Paragraph("<b>Overall Churn Rate</b>", table_text_style),
            Paragraph(f"{churn_rate:.2f}%", table_text_style)
        ],
        [
            Paragraph("<b>Active Customers</b>", table_text_style),
            Paragraph(f"{active_customers:,}", table_text_style),
            Paragraph("<b>Churned Customers</b>", table_text_style),
            Paragraph(f"{churned_customers:,}", table_text_style)
        ],
        [
            Paragraph("<b>Monthly Revenue Loss</b>", table_text_style),
            Paragraph(f"${monthly_loss:,.2f}", table_text_style),
            Paragraph("<b>Est. Annual Revenue Loss</b>", table_text_style),
            Paragraph(f"${annual_loss:,.2f}", table_text_style)
        ]
    ]
    
    kpi_table = Table(kpi_data, colWidths=[1.8*inch, 1.4*inch, 1.8*inch, 1.4*inch])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 8),
    ]))
    
    story.append(kpi_table)
    story.append(Spacer(1, 15))
    
    # Model Performance Section
    story.append(Paragraph("2. Machine Learning Model Comparison", h1_style))
    story.append(Paragraph(
        f"We trained and evaluated three machine learning classifiers (Logistic Regression, Random Forest, and XGBoost) "
        f"using stratified 5-fold cross-validation. The preprocessor handles scale-adjustments and encoding of demographics, "
        f"services, and contracts. The best model selected for real-time predictions is <b>{best_model_name}</b>.",
        body_style
    ))
    
    # Performance Table
    perf_headers = ["Model Name", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    perf_data = [[Paragraph(f"<b>{h}</b>", table_header_style) for h in perf_headers]]
    
    for model_name, m_val in metrics.items():
        row = [
            Paragraph(model_name, table_text_style),
            Paragraph(f"{m_val.get('Accuracy', 0):.4f}", table_text_style),
            Paragraph(f"{m_val.get('Precision', 0):.4f}", table_text_style),
            Paragraph(f"{m_val.get('Recall', 0):.4f}", table_text_style),
            Paragraph(f"{m_val.get('F1 Score', 0):.4f}", table_text_style),
            Paragraph(f"{m_val.get('ROC-AUC Score', 0):.4f}", table_text_style),
        ]
        perf_data.append(row)
        
    perf_table = Table(perf_data, colWidths=[1.8*inch, 0.95*inch, 0.95*inch, 0.95*inch, 0.95*inch, 0.95*inch])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2563EB')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('ALIGN', (0,1), (0,-1), 'LEFT'), # Left align model names
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
    ]))
    
    story.append(perf_table)
    story.append(Spacer(1, 15))
    
    # Retention Strategy Section
    story.append(Paragraph("3. Recommended Retention Policies", h1_style))
    story.append(Paragraph(
        "Based on predicted churn probabilities, our retention engine automatically flags at-risk subscribers "
        "and prescribes specific, budget-optimized marketing campaigns:",
        body_style
    ))
    
    retention_rules = [
        [
            Paragraph("<b>Risk Tier</b>", table_header_style),
            Paragraph("<b>Probability Bounds</b>", table_header_style),
            Paragraph("<b>Recommended Strategy & Actions</b>", table_header_style)
        ],
        [
            Paragraph("<b>Low Risk</b>", table_text_style),
            Paragraph("< 20% Churn Prob", table_text_style),
            Paragraph("<b>Loyalty Rewards</b>: Engage with proactive newsletters, feature adoption announcements, and loyalty programs to secure long-term value.", table_text_style)
        ],
        [
            Paragraph("<b>Medium Risk</b>", table_text_style),
            Paragraph("20% - 50% Churn Prob", table_text_style),
            Paragraph("<b>Personalized Offers</b>: Offer contract upgrades to longer terms (1/2 year) with a slight discount. Trigger personalized check-in calls and surveys to address pain points.", table_text_style)
        ],
        [
            Paragraph("<b>High Risk</b>", table_text_style),
            Paragraph("> 50% Churn Prob", table_text_style),
            Paragraph("<b>Aggressive Retention</b>: Direct connection to a dedicated customer support manager, special discounts up to 30% off monthly subscriptions, or free service upgrades (e.g. tech support add-on).", table_text_style)
        ]
    ]
    
    ret_table = Table(retention_rules, colWidths=[1.2*inch, 1.5*inch, 3.8*inch])
    ret_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#7C3AED')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(ret_table)
    story.append(Spacer(1, 20))
    
    # Disclaimer/Signature block
    story.append(Paragraph(
        "<i>Disclaimer: This analysis report is generated automatically by the machine learning pipeline. "
        "Predictions are probabilistic and should be combined with qualitative customer relationship insights.</i>",
        body_style
    ))
    
    doc.build(story)
    print(f"Report generated successfully at {report_path}")
    return report_path
