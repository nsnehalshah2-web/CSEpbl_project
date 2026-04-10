import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import ssl
import os

# --- 0. INITIAL SETUP ---
ssl._create_default_https_context = ssl._create_unverified_context
st.set_page_config(page_title="CardioVanguard Pro | Nehal Shah", layout="wide")

# --- 1. THEME & STYLING ---
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #0d1117 0%, #0f172a 100%); color: white; }
    .header-box {
        background: rgba(88, 166, 255, 0.1);
        border-left: 5px solid #58a6ff;
        padding: 25px;
        border-radius: 12px;
        margin-bottom: 30px;
    }
    .details-card {
        background: rgba(255, 255, 255, 0.03);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        font-size: 0.85rem;
    }
    .metric-container {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(88, 166, 255, 0.3);
    }
    h1, h2, h3 { color: #58a6ff; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA ENGINE ---
@st.cache_resource
def build_advanced_engine():
    cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    local_file = 'heart_data.csv/processed.cleveland.data'
    try:
        if os.path.exists(local_file):
            df = pd.read_csv(local_file, names=cols, na_values='?').fillna(0)
            df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
            X = df.drop('target', axis=1); y = df['target']
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            return model, X.columns.tolist()
        return None, None
    except: return None, None

model, feature_names = build_advanced_engine()

# --- 3. HEADER ---
st.markdown("""
    <div class="header-box">
        <h1>CardioVanguard Pro</h1>
        <p>Holistic Bio-Psycho-Social Cardiac Intelligence System</p>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=80)
    st.markdown(f"""
    <div class="details-card">
        <b>Developer:</b> Nehal Shah<br>
        <b>Reg No:</b> 2427030423<br>
        <b>Guide:</b> Dr. Susheela Vishnoi<br><hr>
        <b>Analysis:</b> 32 Parameters
    </div>
    """, unsafe_allow_html=True)

# --- 4. THE 32-PARAMETER INPUTS ---
if model is not None:
    tab1, tab2, tab3, tab4 = st.tabs(["🧬 Bio-Genetic", "🏃 Lifestyle", "🧠 Psych/Env", "🏥 Clinical"])
    
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        age = c1.number_input("Age", 1, 100, 45)
        sex = c2.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        height = c3.number_input("Height (cm)", 100, 250, 175)
        weight = c4.number_input("Weight (kg)", 30, 200, 75)
        
        c5, c6 = st.columns(2)
        fam_history = c5.multiselect("Family History", ["Heart Disease", "Stroke", "Diabetes", "Hypertension"])
        waist_circ = c6.number_input("Waist Circumference (cm)", 40, 200, 85)

    with tab2:
        c1, c2, c3 = st.columns(3)
        sleep = c1.slider("Sleep Hours", 3, 12, 7)
        water = c2.slider("Water (Liters)", 1, 5, 2)
        exercise = c3.selectbox("Exercise", ["None", "Occasional", "Frequent", "Athlete"])
        
        c4, c5, c6 = st.columns(3)
        smoking = c4.radio("Smoking", ["Never", "Past", "Current"])
        alcohol = c5.selectbox("Alcohol", ["None", "Occasional", "Daily"])
        diet = c6.selectbox("Diet", ["Standard", "Keto", "Vegan", "Mediterranean"])

    with tab3:
        c1, c2 = st.columns(2)
        stress = c1.select_slider("Stress Level", ["Low", "Moderate", "High", "Extreme"])
        anger = c2.select_slider("Anger frequency", ["Low", "Mid", "High"])
        
        c3, c4, c5 = st.columns(3)
        pollution = c3.selectbox("Air Quality Exposure", ["Clean", "Moderate", "Heavy Industrial"])
        caffeine = c4.number_input("Cups of Coffee/Day", 0, 10, 2)
        anxiety = c5.checkbox("Frequent Anxiety")

    with tab4:
        c1, c2, c3 = st.columns(3)
        trestbps = c1.number_input("Resting BP", 80, 200, 120)
        chol = c2.number_input("Cholesterol", 100, 600, 210)
        fbs = c3.selectbox("FBS > 120", [0, 1])
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        thalach = st.number_input("Max HR", 60, 220, 155)
        oldpeak = st.slider("ST Depr", 0.0, 6.0, 0.5)
        ca = st.selectbox("Major Vessels", [0, 1, 2, 3])
        thal = st.selectbox("Thallium Result", [3, 6, 7])
        exang = st.checkbox("Exercise Angina")

    # --- 5. CALCULATION ---
    if st.button("Run", type="primary", use_container_width=True):
        # Base ML Prediction
        user_data = {'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs, 'restecg': 0, 'thalach': thalach, 'exang': 1 if exang else 0, 'oldpeak': oldpeak, 'slope': 1, 'ca': ca, 'thal': thal}
        df_input = pd.DataFrame([user_data])[feature_names]
        prob = model.predict_proba(df_input)[0][1]

        # Holistic Adjustments
        multiplier = 1.0
        if "Heart Disease" in fam_history: multiplier += 0.1
        if smoking == "Current": multiplier += 0.2
        if pollution == "Heavy Industrial": multiplier += 0.05
        if sleep < 6: multiplier += 0.05
        
        final_risk = min(prob * multiplier, 1.0)
        heart_age = age + (10 if final_risk > 0.6 else -2 if final_risk < 0.2 else 2)

        st.divider()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Heart Age vs Real Age", f"{heart_age} yrs", f"{heart_age - age} yrs shift")
        with col2:
            st.metric("Cardiac Risk Score", f"{final_risk*100:.1f}%")
        with col3:
            st.metric("BMI Index", f"{weight/((height/100)**2):.1f}")

        if final_risk > 0.5:
            st.error("### ⚠️ CLINICAL ALERT: HIGH RISK")
            st.info("💡 **Plan:** Cardiovascular screening and stress-test recommended. Reduce sodium and improve sleep hygiene.")
        else:
            st.success("### ✅ CLINICAL STATUS: STABLE")
            st.info("💡 **Plan:** Maintain current diet. Heart age is within optimal range.")

        # Importance Chart
        imp_df = pd.DataFrame({'Feature': feature_names, 'Weight': model.feature_importances_}).sort_values('Weight', ascending=False).head(5)
        st.plotly_chart(px.bar(imp_df, x='Weight', y='Feature', orientation='h', title="Key Risk Drivers").update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white"))

else:
    st.error("Engine Offline.")