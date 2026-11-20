import streamlit as st
import pandas as pd
import pickle
import numpy as np
import time

# 1. Page Config
st.set_page_config(page_title="ATM Fraud Detection System", layout="wide")

# Custom CSS for UI
st.markdown("""
    <style>
    .main { background-color: #f8f9fas; }
    .metric-box { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# 2. Load Model and Scaler
@st.cache_resource
def load_assets():
    model = pickle.load(open('atm_fraud_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    return model, scaler

model, scaler = load_assets()

# --- SIDEBAR: Threshold Control ---
st.sidebar.header("Settings")
user_threshold = st.sidebar.slider(
    "Decision Threshold", 
    min_value=0.10, 
    max_value=0.99, 
    value=0.30, 
    step=0.01,
    help="Adjust this to balance between catching more frauds (Recall) or reducing false alarms (Precision)."
)

st.sidebar.markdown("---")
st.sidebar.write(f"**Target Metric:** Recall 0.90")
st.sidebar.write(f"**Optimized Threshold:** 0.30")

# --- MAINBAR: Header & Upload ---
st.title("🏦 ATM Fraud Analytics Dashboard")
st.write(f"Analyzing transactions with a decision threshold of **{user_threshold}**")

uploaded_file = st.file_uploader("Upload Transaction CSV File", type="csv")

# Model expected columns (32 features)
expected_cols = [
    'Per1', 'Per2', 'Per3', 'Per4', 'Per5', 'Per6', 'Per7', 'Per8', 'Per9', 
    'Dem1', 'Dem2', 'Dem3', 'Dem4', 'Dem5', 'Dem6', 'Dem7', 'Dem8', 'Dem9', 
    'Cred1', 'Cred2', 'Cred3', 'Cred4', 'Cred5', 'Cred6', 'Normalised_FNT', 
    'geo_score_mean', 'geo_score_max', 'instance_score_mean', 'instance_score_max', 
    'lambda_wt', 'qsets_normalized_tat_mean', 'qsets_normalized_tat_max'
]

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Input Data Preview")
    st.dataframe(df.head(5), use_container_width=True)

    if st.button("Run Fraud Analysis"):
        # Processing
        with st.spinner('Model is scanning transactions...'):
            time.sleep(1)
            
            # Ensure all 32 columns are present
            working_df = df.copy()
            for col in expected_cols:
                if col not in working_df.columns:
                    working_df[col] = 0
            
            # Align and Scale
            input_data = working_df[expected_cols]
            scaled_input = scaler.transform(input_data)
            
            # Prediction Logic
            probs = model.predict_proba(scaled_input)[:, 1]
            # Use Sidebar Threshold
            preds = (probs >= user_threshold).astype(int)

        # 3. MAINBAR RESULTS
        st.markdown("---")
        st.subheader("🔍 Analysis Report")

        # Summary Statistics
        fraud_count = int(np.sum(preds))
        total_count = len(df)
        fraud_percentage = (fraud_count / total_count) * 100

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Transactions", total_count)
        with col2:
            st.metric("Frauds Detected", fraud_count, delta=f"{fraud_percentage:.1f}% of total", delta_color="inverse")
        with col3:
            st.metric("System Confidence", "High", delta="F2-Score: 0.80")

        # Result Dataframe
        df.insert(0, 'Prediction', np.where(preds == 1, 'FRAUD', 'SAFE'))
        df.insert(1, 'Fraud_Probability', np.round(probs, 4))

        # Styling the Result Table
        def highlight_fraud(val):
            color = 'red' if val == 'FRAUD' else 'green'
            return f'color: {color}; font-weight: bold'

        st.write("### Detailed Predictions")
        fraud_df = df[df['Prediction'] == 1]
        st.dataframe(df.head(1000).style.applymap(highlight_fraud, subset=['Prediction']),use_container_width=True)


        # Download Result
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Fraud Analysis Report",
            data=csv,
            file_name='ATM_Fraud_Results.csv',
            mime='text/csv'
        )

        st.success("Analysis complete. You can download the full report above.")