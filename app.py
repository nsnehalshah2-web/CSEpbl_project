import streamlit as st
from predictor import get_predictor
import plotly.graph_objects as go

st.set_page_config(
    page_title="CardioVanguard Pro | Nehal Shah",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;700;900&display=swap');
    
    .stApp {
        background: radial-gradient(circle at top right, #0d1117, #010409);
        color: #e6edf3;
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(90deg, #58a6ff, #bc8cff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 3.5rem;
        margin-bottom: 0px;
    }
    
    .sidebar-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover { border-color: #58a6ff; transform: translateY(-5px); }
    
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
    }
    
    .stButton>button {
        background: #1f6feb;
        border: none;
        color: white;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-radius: 12px;
        padding: 0.8rem;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: #2ea043;
        box-shadow: 0 4px 12px rgba(46, 160, 67, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

predictor = get_predictor()

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=100)
    st.markdown("""
        <div class="sidebar-card">
            <h3 style='margin:0; color:#58a6ff;'>System Core</h3>
            <p style='font-size:0.85rem;'><b>Developer:</b> Nehal Shah</p>
            <p style='font-size:0.85rem;'><b>Reg no:</b> 2427030423</p>
            <p style='font-size:0.85rem;'><b>Guide:</b> Dr. Susheela Vishnoi</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class="sidebar-card">
            <h3 style='margin:0; color:#58a6ff;'>Model Statistics</h3>
            <p style='font-size:0.85rem;'><b>Accuracy:</b> {predictor.metadata['accuracy']:.2%}</p>
            <p style='font-size:0.85rem;'><b>Features:</b> {len(predictor.feature_names)} Parameters</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>CardioVanguard Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='opacity:0.6; font-size:1.1rem; margin-bottom:40px;'>Clinical Decision Support System with Holistic Bias Correction</p>", unsafe_allow_html=True)

with st.container():
    tab1, tab2, tab3, tab4 = st.tabs(["🧬 Bio-Genetic", "🏃 Lifestyle", "🧠 Psych/Env", "🏥 Clinical"])

    with tab1:
        c1, c2 = st.columns(2)
        age = c1.number_input("Age", 1, 120, 45)
        sex = c2.selectbox("Sex", ["Male", "Female"])
        
        c3, c4 = st.columns(2)
        fam_history = c3.multiselect("Family Medical History", ["Heart Disease", "Stroke", "Diabetes", "Hypertension"])
        ca = c4.number_input("Number of Major Vessels (0-3)", 0, 3, 0)

    with tab2:
        c1, c2 = st.columns(2)
        smoking = c1.radio("Smoking Status", ["Never", "Past", "Current"], horizontal=True)
        alcohol = c2.selectbox("Alcohol Consumption", ["None", "Occasional", "Daily"])
        
        c3, c4 = st.columns(2)
        exercise = c3.selectbox("Exercise Frequency", ["None", "Occasional", "Frequent", "Athlete"])
        diet = c4.selectbox("Dietary Consistency", ["Standard", "Mediterranean", "Vegan", "Keto"])

    with tab3:
        c1, c2 = st.columns(2)
        stress = c1.select_slider("Chronic Stress Level", options=["Low", "Moderate", "High", "Extreme"])
        anxiety = c2.checkbox("Diagnosed Anxiety Disorder")
        
        c3, c4 = st.columns(2)
        sleep = c3.slider("Average Sleep (Hours)", 4, 12, 7)
        pollution = c4.selectbox("Environmental Air Quality", ["Clean", "Moderate", "Heavy Industrial"])

    with tab4:
        c1, c2, c3 = st.columns(3)
        cp = c1.selectbox("Chest Pain Type", ["typical angina", "asymptomatic", "non-anginal", "atypical angina"])
        trestbps = c2.number_input("Resting Blood Pressure", 80, 200, 120)
        chol = c3.number_input("Serum Cholestrol (mg/dl)", 100, 600, 210)
        
        c4, c5, c6 = st.columns(3)
        thalch = c4.number_input("Maximum Heart Rate", 60, 220, 150)
        oldpeak = c5.slider("ST Depression", 0.0, 6.0, 0.5)
        slope = c6.selectbox("ST Slope", ["flat", "upsloping", "downsloping"])
        
        c7, c8 = st.columns(2)
        fbs = c7.selectbox("Fasting Blood Sugar > 120", ["False", "True"])
        restecg = c8.selectbox("Resting ECG Result", ["normal", "lv hypertrophy", "st-t abnormality"])
        
        thal = st.selectbox("Thallium Stress Result", ["normal", "fixed defect", "reversable defect"])
        exang = st.checkbox("Exercise Induced Angina (Check if Yes)")

if st.button("Run diagnostic"):
    with st.spinner("Analyzing Bio-Markers..."):
        clinical_data = {
            "age": float(age), "sex": sex, "cp": cp, "trestbps": float(trestbps),
            "chol": float(chol), "fbs": fbs, "restecg": restecg, "thalch": float(thalch),
            "exang": "True" if exang else "False", "oldpeak": float(oldpeak),
            "slope": slope, "ca": float(ca), "thal": thal
        }
        
        holistic_data = {
            "family_history": fam_history, "smoking": smoking, "stress": stress,
            "anxiety": anxiety, "pollution": pollution, "sleep": float(sleep),
            "exercise": exercise, "diet": diet
        }
        
        res = predictor.predict_risk(clinical_data, holistic_data)
        
        st.divider()
        m1, m2, m3 = st.columns(3)
        
        with m1:
            st.markdown(f"""<div class='metric-card'>
                <p style='color:#8b5cf6; font-weight:bold;'>BLOOD AGE</p>
                <h2 style='margin:0; font-size:3rem;'>{res['heart_age']} <span style='font-size:1rem; opacity:0.6;'>YRS</span></h2>
                <p style='color:{"#ef4444" if res["age_gap"] > 0 else "#10b981"};'>{res['age_gap']:+d} Years Flux</p>
            </div>""", unsafe_allow_html=True)
            
        with m2:
            st.markdown(f"""<div class='metric-card'>
                <p style='color:#3b82f6; font-weight:bold;'>CONFIDENCE RATE</p>
                <h2 style='margin:0; font-size:3rem;'>{res['final_risk_score']*100:.1f}<span style='font-size:1.5rem;'>%</span></h2>
                <p style='font-weight:black; color:{"#ef4444" if res["final_risk_score"] > 0.5 else "#10b981"};'>{res['status'].upper()}</p>
            </div>""", unsafe_allow_html=True)
            
        with m3:
            st.markdown(f"""<div class='metric-card'>
                <p style='color:#10b981; font-weight:bold;'>STATISTICAL PRECISION</p>
                <h2 style='margin:0; font-size:3rem;'>{predictor.metadata['accuracy']*100:.1f}<span style='font-size:1.5rem;'>%</span></h2>
                <p style='opacity:0.6;'>Model Accuracy</p>
            </div>""", unsafe_allow_html=True)

        st.subheader("💡 Holistic Analysis")
        cols = st.columns(2)
        with cols[0]:
            if res['adjustments']:
                for adj in res['adjustments']:
                    st.info(adj)
            else:
                st.write("No major lifestyle modifiers detected.")
        
        with cols[1]:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = res['final_risk_score']*100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Cardiac Vulnerability Index", 'font': {'size': 20, 'color': "white"}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickcolor': "white"},
                    'bar': {'color': "#58a6ff"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.2)'},
                        {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.2)'},
                        {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.2)'}],
                }
            ))
            fig.update_layout(paper_bgcolor = 'rgba(0,0,0,0)', plot_bgcolor = 'rgba(0,0,0,0)', font = {'color': "white", 'family': "Inter"})
            st.plotly_chart(fig, use_container_width=True)

        if res['final_risk_score'] > 0.6:
            st.error("🚨 **CLINICAL ALERT**: Profile indicates high vulnerability. Consider immediate cardiology consultation.")
        elif res['final_risk_score'] > 0.3:
            st.warning("⚠️ **PREVENTIVE NOTICE**: Moderate risk factors detected. Focus on lifestyle optimization.")
        else:
            st.success("✅ **STABLE STATUS**: Cardiovascular markers are within optimal ranges.")

else:
    st.info("👋 Welcome to CardioVanguard Pro. Fill in the parameters and initialize analysis.")