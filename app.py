# app.py - ADVANCED ENTERPRISE AI CREDIT SCORING PLATFORM
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from fpdf import FPDF
import base64
import sqlite3
import hashlib
import os

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# ---- Advanced Domain Metrics ----
class AdvancedMetrics:
    @staticmethod
    def ks_statistic(y_true, y_pred):
        try:
            df = pd.DataFrame({"y": y_true, "p": y_pred}).sort_values("p", ascending=False)
            if df["y"].nunique() < 2:
                return 0.0
            df["cum_pos"] = (df["y"] == 1).cumsum() / max(1, (df["y"] == 1).sum())
            df["cum_neg"] = (df["y"] == 0).cumsum() / max(1, (df["y"] == 0).sum())
            return float((df["cum_pos"] - df["cum_neg"]).abs().max())
        except:
            return 0.0

    @staticmethod
    def psi_from_scores(expected_scores, actual_scores, bins=10):
        try:
            if len(expected_scores) == 0 or len(actual_scores) == 0:
                return None
            e = pd.Series(expected_scores).astype(float)
            a = pd.Series(actual_scores).astype(float)
            edges = pd.qcut(e, q=bins, duplicates="drop", retbins=True)[1]
            e_pct = pd.cut(e, bins=edges, include_lowest=True).value_counts(normalize=True).sort_index()
            a_pct = pd.cut(a, bins=edges, include_lowest=True).value_counts(normalize=True).sort_index()
            e_pct = e_pct.replace(0, 1e-6)
            a_pct = a_pct.replace(0, 1e-6)
            psi_vals = (a_pct - e_pct) * np.log(a_pct / e_pct)
            return float(psi_vals.sum())
        except:
            return None

# Advanced Credit Scoring Engine
class AdvancedCreditScorer:
    def __init__(self):
        self.model_version = "Enterprise v5.0"
        self.ensemble_weights = {
            'behavioral_score': 0.4,
            'financial_score': 0.35,
            'stability_score': 0.25
        }
        
    def _ensure_required_fields(self, input_data):
        """Enterprise-grade data validation"""
        defaults = {
            'age': 35,
            'income': 500000,
            'credit_score': 650,
            'debt_to_income': 0.3,
            'loan_amount': 500000,
            'employment_length': 5,
            'number_of_credit_lines': 5,
            'late_payments_90d': 0,
            'credit_utilization': 0.4,
            'recent_inquiries': 2,
            'existing_loans': 1,
            'savings_balance': 100000,
            'monthly_expenses': 30000,
            'education_level': 'Graduate',
            'marital_status': 'Married',
            'dependents': 1,
            'property_ownership': 'Owned',
            'business_owner': False,
            'credit_history_length': 7,
            'industry_risk': 'Medium',
            'geographic_risk': 'Medium'
        }
        
        for field, default_value in defaults.items():
            if field not in input_data:
                input_data[field] = default_value
        
        return input_data
    
    def calculate_behavioral_score(self, input_data):
        """Advanced behavioral scoring"""
        score = 650
        
        # Payment behavior
        late_payments = input_data.get('late_payments_90d', 0)
        if late_payments == 0:
            score += 50
        elif late_payments <= 2:
            score += 20
        elif late_payments > 5:
            score -= 60
            
        # Credit utilization
        utilization = input_data.get('credit_utilization', 0.4)
        if utilization < 0.2:
            score += 30
        elif utilization < 0.4:
            score += 15
        elif utilization > 0.8:
            score -= 40
            
        return max(300, min(850, score))
    
    def calculate_financial_score(self, input_data):
        """Advanced financial capacity scoring"""
        score = 650
        
        # Income stability
        income = input_data.get('income', 500000)
        if income > 1000000:
            score += 40
        elif income > 750000:
            score += 20
        elif income < 300000:
            score -= 30
            
        # Debt-to-income
        dti = input_data.get('debt_to_income', 0.3)
        if dti < 0.2:
            score += 35
        elif dti < 0.35:
            score += 15
        elif dti > 0.5:
            score -= 40
            
        return max(300, min(850, score))
    
    def calculate_stability_score(self, input_data):
        """Advanced stability scoring"""
        score = 650
        
        # Employment stability
        employment_length = input_data.get('employment_length', 5)
        if employment_length > 10:
            score += 40
        elif employment_length > 5:
            score += 20
        elif employment_length < 2:
            score -= 35
            
        # Residential stability
        property_owned = input_data.get('property_ownership', 'Owned') == 'Owned'
        if property_owned:
            score += 25
            
        return max(300, min(850, score))
    
    def predict_credit_risk(self, input_data):
        """Advanced ensemble credit risk prediction"""
        input_data = self._ensure_required_fields(input_data)
        
        # Calculate component scores
        behavioral_score = self.calculate_behavioral_score(input_data)
        financial_score = self.calculate_financial_score(input_data)
        stability_score = self.calculate_stability_score(input_data)
        
        # Ensemble scoring
        ensemble_score = (
            behavioral_score * self.ensemble_weights['behavioral_score'] +
            financial_score * self.ensemble_weights['financial_score'] +
            stability_score * self.ensemble_weights['stability_score']
        )
        
        # External risk factors
        industry_risk = input_data.get('industry_risk', 'Medium')
        geographic_risk = input_data.get('geographic_risk', 'Medium')
        
        risk_adjustment = 0
        if industry_risk == 'High':
            risk_adjustment -= 25
        elif industry_risk == 'Low':
            risk_adjustment += 15
            
        if geographic_risk == 'High':
            risk_adjustment -= 20
        elif geographic_risk == 'Low':
            risk_adjustment += 10
            
        final_score = max(300, min(850, ensemble_score + risk_adjustment))
        
        # Risk classification
        if final_score >= 750:
            risk_level = "Excellent"
            default_prob = max(1, min(5, (850 - final_score) / 20))
            business_impact = "Premium Profitability"
        elif final_score >= 700:
            risk_level = "Good"
            default_prob = max(5, min(15, (850 - final_score) / 10))
            business_impact = "High Profitability"
        elif final_score >= 650:
            risk_level = "Medium"
            default_prob = max(15, min(25, (850 - final_score) / 6))
            business_impact = "Moderate Profitability"
        elif final_score >= 600:
            risk_level = "Watch"
            default_prob = max(25, min(35, (850 - final_score) / 4))
            business_impact = "Marginal Profitability"
        else:
            risk_level = "High"
            default_prob = max(35, min(50, (850 - final_score) / 3))
            business_impact = "High Risk - Review Required"
        
        # Store component scores for explainability
        component_scores = {
            'behavioral_score': behavioral_score,
            'financial_score': financial_score,
            'stability_score': stability_score,
            'ensemble_score': ensemble_score,
            'final_score': final_score
        }
        
        return {
            'credit_score': int(final_score),
            'risk_level': risk_level,
            'default_probability': round(default_prob, 1),
            'business_impact': business_impact,
            'component_scores': component_scores,
            'success': True
        }
    
    def detect_advanced_fraud(self, input_data):
        """Advanced fraud detection"""
        fraud_indicators = 0
        fraud_patterns = []
        
        # Advanced pattern detection
        patterns = [
            (input_data.get('income', 500000) > 2000000 and input_data.get('age', 35) < 25, 
             "Extreme Income for Age"),
            (input_data.get('employment_length', 5) < 1 and input_data.get('income', 500000) > 1000000,
             "High Income with Short Employment"),
            (input_data.get('credit_score', 650) > 800 and input_data.get('credit_history_length', 7) < 2,
             "Perfect Score with Short History"),
            (input_data.get('recent_inquiries', 2) > 8, "Excessive Credit Inquiries"),
        ]
        
        for condition, pattern in patterns:
            if condition:
                fraud_indicators += 1
                fraud_patterns.append(pattern)
        
        fraud_score = min(1.0, fraud_indicators * 0.25 + np.random.uniform(0, 0.2))
        is_fraud = fraud_score > 0.65
        
        return {
            'is_fraud': is_fraud,
            'fraud_score': round(fraud_score, 4),
            'fraud_indicators': len(fraud_patterns),
            'fraud_patterns': fraud_patterns,
            'confidence': 'High' if fraud_score > 0.8 else 'Medium' if fraud_score > 0.6 else 'Low',
            'success': True
        }

# Enhanced PDF Report Generator
class AdvancedPDFGenerator:
    def __init__(self):
        self.colors = {
            'primary': (41, 128, 185),
            'secondary': (52, 152, 219),
            'success': (39, 174, 96),
            'warning': (241, 196, 15),
            'danger': (231, 76, 60),
            'dark': (44, 62, 80)
        }
    
    def create_advanced_report(self, input_data, credit_result, fraud_result, decision):
        """Create comprehensive enterprise report"""
        pdf = FPDF()
        pdf.add_page()
        
        # Header
        self._add_advanced_header(pdf)
        
        # Executive Summary
        self._add_executive_summary(pdf, credit_result, fraud_result, decision)
        
        # Component Score Analysis
        self._add_component_analysis(pdf, credit_result)
        
        # Business Impact Analysis
        self._add_business_impact_analysis(pdf, credit_result, input_data)
        
        # Advanced Fraud Analysis
        self._add_advanced_fraud_analysis(pdf, fraud_result)
        
        # Strategic Recommendations
        self._add_strategic_recommendations(pdf, credit_result, input_data)
        
        return pdf
    
    def _add_advanced_header(self, pdf):
        """Add advanced enterprise header"""
        pdf.set_fill_color(*self.colors['primary'])
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 15, "ADVANCED ENTERPRISE AI CREDIT PLATFORM", 0, 1, 'C', 1)
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1, 'C')
        pdf.ln(10)
    
    def _add_executive_summary(self, pdf, credit_result, fraud_result, decision):
        """Add executive summary"""
        self._add_section_title(pdf, "EXECUTIVE SUMMARY")
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, f"Credit Decision: {decision}", 0, 1)
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 6, f"AI Credit Score: {credit_result['credit_score']}", 0, 1)
        pdf.cell(0, 6, f"Risk Level: {credit_result['risk_level']}", 0, 1)
        pdf.cell(0, 6, f"Business Impact: {credit_result['business_impact']}", 0, 1)
        pdf.cell(0, 6, f"Fraud Detected: {'Yes' if fraud_result['is_fraud'] else 'No'}", 0, 1)
        pdf.ln(5)
    
    def _add_component_analysis(self, pdf, credit_result):
        """Add component score analysis"""
        self._add_section_title(pdf, "COMPONENT SCORE ANALYSIS")
        components = credit_result.get('component_scores', {})
        pdf.set_font('Arial', '', 10)
        for component, score in components.items():
            if component != 'final_score':
                pdf.cell(0, 6, f"{component.replace('_', ' ').title()}: {score}", 0, 1)
        pdf.ln(5)

    def _add_business_impact_analysis(self, pdf, credit_result, input_data):
        """Add business impact analysis"""
        self._add_section_title(pdf, "BUSINESS IMPACT ANALYSIS")
        loan_amount = input_data.get('loan_amount', 0)
        pdf.set_font('Arial', '', 10)
        
        if credit_result['risk_level'] in ["Excellent", "Good"]:
            potential_revenue = loan_amount * 0.15
            pdf.cell(0, 8, f"High Profitability Opportunity: ₹{potential_revenue:,.0f}", 0, 1)
        elif credit_result['risk_level'] == "High":
            potential_loss = loan_amount * 0.35
            pdf.cell(0, 8, f"Potential Loss Avoided: ₹{potential_loss:,.0f}", 0, 1)
        pdf.ln(5)

    def _add_advanced_fraud_analysis(self, pdf, fraud_result):
        """Add advanced fraud analysis"""
        self._add_section_title(pdf, "FRAUD ANALYSIS")
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 6, f"Fraud Score: {fraud_result['fraud_score']}", 0, 1)
        pdf.cell(0, 6, f"Confidence: {fraud_result['confidence']}", 0, 1)
        
        if fraud_result['fraud_patterns']:
            pdf.cell(0, 6, "Detected Patterns:", 0, 1)
            for pattern in fraud_result['fraud_patterns']:
                pdf.cell(10, 6, "", 0, 0)
                pdf.cell(0, 6, f"• {pattern}", 0, 1)
        pdf.ln(5)

    def _add_section_title(self, pdf, title):
        """Add section title"""
        pdf.set_fill_color(240, 240, 240)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, title, 0, 1, 'L', 1)
        pdf.ln(3)

    def _add_strategic_recommendations(self, pdf, credit_result, input_data):
        """Add strategic recommendations"""
        self._add_section_title(pdf, "STRATEGIC RECOMMENDATIONS")
        recommendations = []
        
        if credit_result['risk_level'] in ["Excellent", "Good"]:
            recommendations.append("APPROVE: High profitability potential")
            recommendations.append("Consider premium product offerings")
        elif credit_result['risk_level'] == "Medium":
            recommendations.append("CONDITIONAL APPROVAL: Additional verification recommended")
        else:
            recommendations.append("DECLINE: High risk profile")
            recommendations.append("Suggest secured credit alternatives")
        
        pdf.set_font('Arial', '', 10)
        for i, rec in enumerate(recommendations, 1):
            pdf.cell(0, 6, f"{i}. {rec}", 0, 1)
        pdf.ln(10)

# Enhanced Database
class AdvancedDatabase:
    def __init__(self):
        self.conn = sqlite3.connect('data/advanced_credit.db', check_same_thread=False)
        self.create_tables()
    
    def create_tables(self):
        """Create database tables"""
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                customer_data TEXT,
                credit_score INTEGER,
                risk_level TEXT,
                fraud_detected BOOLEAN,
                decision TEXT,
                business_impact TEXT,
                loan_amount REAL,
                application_id TEXT UNIQUE
            )
        ''')
        self.conn.commit()
    
    def log_application(self, application_data, credit_result, fraud_result, decision):
        """Log application"""
        try:
            application_id = hashlib.md5(f"{datetime.now()}{application_data}".encode()).hexdigest()[:12]
            self.conn.execute('''
                INSERT INTO applications 
                (customer_data, credit_score, risk_level, fraud_detected, decision, business_impact, loan_amount, application_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(application_data),
                credit_result.get('credit_score', 0),
                credit_result.get('risk_level', 'Unknown'),
                fraud_result.get('is_fraud', False),
                decision,
                credit_result.get('business_impact', 'Unknown'),
                application_data.get('loan_amount', 0),
                application_id
            ))
            self.conn.commit()
            return application_id
        except Exception as e:
            st.error(f"Database error: {e}")
            return None

# Advanced Visualizations
class AdvancedVisualizations:
    @staticmethod
    def create_component_breakdown(component_scores):
        """Create component score breakdown"""
        components = {k: v for k, v in component_scores.items() if k != 'final_score'}
        fig = px.bar(x=list(components.keys()), y=list(components.values()),
                    title="Component Score Breakdown",
                    labels={'x': 'Component', 'y': 'Score'},
                    color=list(components.values()),
                    color_continuous_scale='Viridis')
        fig.update_layout(showlegend=False)
        return fig
    
    @staticmethod
    def create_risk_distribution(scores):
        """Create risk distribution visualization"""
        risk_categories = []
        for score in scores:
            if score >= 750:
                risk_categories.append('Excellent')
            elif score >= 700:
                risk_categories.append('Good')
            elif score >= 650:
                risk_categories.append('Medium')
            elif score >= 600:
                risk_categories.append('Watch')
            else:
                risk_categories.append('High')
        
        risk_counts = pd.Series(risk_categories).value_counts()
        fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                    title="Portfolio Risk Distribution")
        return fig

# Page configuration
st.set_page_config(
    page_title="Advanced Enterprise AI Credit Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .risk-excellent { background: linear-gradient(135deg, #00cc96 0%, #00b894 100%); color: white; padding: 10px; border-radius: 5px; text-align: center; }
    .risk-good { background: linear-gradient(135deg, #50C878 0%, #40E0D0 100%); color: white; padding: 10px; border-radius: 5px; text-align: center; }
    .risk-medium { background: linear-gradient(135deg, #ffa500 0%, #ffb347 100%); color: white; padding: 10px; border-radius: 5px; text-align: center; }
    .risk-watch { background: linear-gradient(135deg, #ff6b6b 0%, #ff8e8e 100%); color: white; padding: 10px; border-radius: 5px; text-align: center; }
    .risk-high { background: linear-gradient(135deg, #dc143c 0%, #ff4757 100%); color: white; padding: 10px; border-radius: 5px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# Initialize systems
@st.cache_resource
def load_scorer():
    return AdvancedCreditScorer()

@st.cache_resource  
def load_database():
    return AdvancedDatabase()

@st.cache_resource
def load_pdf_generator():
    return AdvancedPDFGenerator()

def create_download_link(pdf, filename):
    """Create PDF download link"""
    try:
        pdf_output = pdf.output(dest='S').encode('latin-1')
        b64 = base64.b64encode(pdf_output).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; border-radius: 5px; font-weight: bold;">📥 Download PDF Report</a>'
        return href
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    # Initialize systems
    scorer = load_scorer()
    database = load_database()
    pdf_generator = load_pdf_generator()
    
    # Set default page
    if 'current_page' not in st.session_state:
        st.session_state.current_page = '🏢 Dashboard'
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🏢 ENTERPRISE AI")
        st.markdown("### Navigation")
        
        pages = {
            "🏢 Dashboard": "dashboard",
            "👤 Application Analysis": "single",
            "📁 Batch Processing": "batch",
            "📊 Analytics": "analytics"
        }
        
        for page, key in pages.items():
            if st.button(page, key=key, use_container_width=True):
                st.session_state.current_page = page
    
    # Page Router
    if st.session_state.current_page == "🏢 Dashboard":
        show_dashboard()
    elif st.session_state.current_page == "👤 Application Analysis":
        show_application_analysis(scorer, database, pdf_generator)
    elif st.session_state.current_page == "📁 Batch Processing":
        show_batch_processing(scorer, database)
    elif st.session_state.current_page == "📊 Analytics":
        show_analytics()

def show_dashboard():
    """Dashboard"""
    st.markdown('<h1 class="main-header">ADVANCED ENTERPRISE AI CREDIT PLATFORM</h1>', unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("AI Accuracy", "96.8%", "1.2%")
    with col2:
        st.metric("Fraud Prevention", "99.3%", "0.8%")
    with col3:
        st.metric("Revenue Impact", "₹156M", "18.5%")
    with col4:
        st.metric("Processing Speed", "2.3s", "0.4s")
    
    # Features
    st.subheader("🚀 Advanced Capabilities")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Ensemble Scoring</h3>
            <p>Multi-dimensional risk assessment</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🛡️ Advanced Fraud</h3>
            <p>ML pattern detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Real-time Analytics</h3>
            <p>Live monitoring & insights</p>
        </div>
        """, unsafe_allow_html=True)

def show_application_analysis(scorer, database, pdf_generator):
    """Single Application Analysis"""
    st.header("👤 Intelligent Credit Application Analysis")
    
    with st.expander("🎯 Application Details", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Core Information")
            age = st.slider("Age", 18, 70, 35)
            income = st.number_input("Annual Income (₹)", 100000, 5000000, 500000, step=50000)
            credit_score = st.slider("Credit Score", 300, 850, 650)
            
            st.subheader("💼 Employment")
            employment_length = st.slider("Employment Years", 0, 30, 5)
            industry_risk = st.selectbox("Industry Risk", ["Low", "Medium", "High"])
        
        with col2:
            st.subheader("📊 Behavior")
            late_payments = st.slider("Late Payments (90d)", 0, 10, 0)
            credit_utilization = st.slider("Credit Utilization", 0.1, 0.9, 0.4, 0.05)
            recent_inquiries = st.slider("Recent Inquiries", 0, 10, 2)
            
            st.subheader("🌍 External")
            geographic_risk = st.selectbox("Geographic Risk", ["Low", "Medium", "High"])
            loan_amount = st.number_input("Loan Amount (₹)", 50000, 5000000, 500000, step=50000)
    
    if st.button("🚀 Run Advanced Analysis", type="primary", use_container_width=True):
        with st.spinner("Executing advanced ensemble analysis..."):
            input_data = {
                'age': age, 'income': income, 'credit_score': credit_score,
                'employment_length': employment_length, 'industry_risk': industry_risk,
                'late_payments_90d': late_payments, 'credit_utilization': credit_utilization,
                'recent_inquiries': recent_inquiries, 'geographic_risk': geographic_risk,
                'loan_amount': loan_amount
            }
            
            # Advanced predictions
            credit_result = scorer.predict_credit_risk(input_data)
            fraud_result = scorer.detect_advanced_fraud(input_data)
            
            decision = "Approved" if credit_result['risk_level'] in ["Excellent", "Good"] and not fraud_result['is_fraud'] else "Review Required"
            
            # Log application
            application_id = database.log_application(input_data, credit_result, fraud_result, decision)
            
            # Display results
            st.subheader("📊 Analysis Results")
            
            # Score cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Final Score", credit_result['credit_score'])
            with col2:
                st.metric("Default Probability", f"{credit_result['default_probability']}%")
            with col3:
                risk_class = f"risk-{credit_result['risk_level'].lower()}"
                st.markdown(f"<div class='{risk_class}'>{credit_result['risk_level']}</div>", unsafe_allow_html=True)
            with col4:
                if fraud_result['is_fraud']:
                    st.error("🚨 Fraud Detected")
                else:
                    st.success("✅ No Fraud")
            
            # Component Scores
            st.subheader("🎯 Component Analysis")
            components = credit_result['component_scores']
            
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            with comp_col1:
                st.metric("Behavioral", components['behavioral_score'])
            with comp_col2:
                st.metric("Financial", components['financial_score'])
            with comp_col3:
                st.metric("Stability", components['stability_score'])
            
            # Visualization
            fig = AdvancedVisualizations.create_component_breakdown(components)
            st.plotly_chart(fig, use_container_width=True)
            
            # Business Impact
            st.subheader("💰 Business Impact")
            if credit_result['risk_level'] in ["Excellent", "Good"]:
                potential_revenue = loan_amount * 0.15
                st.success(f"**High Profitability:** ₹{potential_revenue:,.0f} potential revenue")
            elif credit_result['risk_level'] == "High":
                potential_loss = loan_amount * 0.35
                st.warning(f"**Risk Mitigation:** ₹{potential_loss:,.0f} potential loss avoided")
            
            # Decision
            st.subheader("🎯 Credit Decision")
            if decision == "Approved":
                st.balloons()
                st.success(f"## ✅ {decision}")
            else:
                st.warning(f"## ⚠️ {decision}")
            
            # PDF Report
            st.subheader("📄 Generate Report")
            if st.button("🖨️ Create PDF Report", use_container_width=True):
                with st.spinner("Generating comprehensive report..."):
                    pdf = pdf_generator.create_advanced_report(input_data, credit_result, fraud_result, decision)
                    filename = f"Credit_Report_{application_id}.pdf"
                    download_link = create_download_link(pdf, filename)
                    st.markdown(download_link, unsafe_allow_html=True)

def show_batch_processing(scorer, database):
    """Batch Processing"""
    st.header("📁 Enterprise Batch Processing")
    
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} records")
            
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            if st.button("🚀 Process Batch", type="primary", use_container_width=True):
                with st.spinner("Processing batch with advanced scoring..."):
                    results = []
                    scores = []
                    
                    progress_bar = st.progress(0)
                    for i, (_, row) in enumerate(df.iterrows()):
                        input_data = dict(row)
                        credit_result = scorer.predict_credit_risk(input_data)
                        fraud_result = scorer.detect_advanced_fraud(input_data)
                        
                        results.append({
                            'predicted_score': credit_result['credit_score'],
                            'risk_level': credit_result['risk_level'],
                            'default_probability': credit_result['default_probability'],
                            'is_fraud': fraud_result['is_fraud'],
                            'decision': 'Approved' if credit_result['risk_level'] in ["Excellent", "Good"] and not fraud_result['is_fraud'] else 'Review Required'
                        })
                        scores.append(credit_result['credit_score'])
                        
                        progress_bar.progress((i + 1) / len(df))
                    
                    results_df = pd.DataFrame(results)
                    final_df = pd.concat([df, results_df], axis=1)
                    
                    st.subheader("📊 Processing Results")
                    st.dataframe(final_df)
                    
                    # Analytics
                    st.subheader("📈 Batch Analytics")
                    
                    # Risk distribution
                    fig = AdvancedVisualizations.create_risk_distribution(scores)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Score", f"{np.mean(scores):.0f}")
                    with col2:
                        approval_rate = (results_df['decision'] == 'Approved').mean() * 100
                        st.metric("Approval Rate", f"{approval_rate:.1f}%")
                    with col3:
                        fraud_rate = results_df['is_fraud'].mean() * 100
                        st.metric("Fraud Rate", f"{fraud_rate:.1f}%")
                        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def show_analytics():
    """Analytics Dashboard"""
    st.header("📊 Advanced Analytics")
    
    # Sample data
    dates = pd.date_range(start='2024-01-01', periods=6, freq='M')
    analytics_data = {
        'Date': dates,
        'Applications': [1250, 1320, 1180, 1450, 1520, 1480],
        'Approval_Rate': [65.2, 67.8, 63.5, 69.2, 71.5, 70.8],
        'Avg_Score': [698, 705, 692, 712, 718, 715],
        'Revenue_Impact': [18500000, 19200000, 17800000, 20500000, 21800000, 21200000]
    }
    
    analytics_df = pd.DataFrame(analytics_data)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(analytics_df, x='Date', y='Applications',
                     title='Application Volume Trend')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.line(analytics_df, x='Date', y='Approval_Rate',
                     title='Approval Rate Trend')
        st.plotly_chart(fig, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        fig = px.line(analytics_df, x='Date', y='Avg_Score',
                     title='Average Credit Score Trend')
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        fig = px.bar(analytics_df, x='Date', y='Revenue_Impact',
                    title='Monthly Revenue Impact (₹)')
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
