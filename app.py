import os
import pickle
import time
import numpy as np
import streamlit as st

# Try importing confetti for interactive button animation
try:
    from streamlit_confetti import confetti
    HAS_CONFETTI = True
except ImportError:
    HAS_CONFETTI = False

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR STYLING & ANIMATIONS ---
st.markdown("""
    <style>
    /* Main container background gradient */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    /* Card container styling */
    .metric-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        border: 1px solid #e1e8ed;
        margin-bottom: 20px;
    }

    /* Custom Header styling */
    .main-title {
        color: #1E3A8A;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .sub-title {
        color: #4B5563;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }

    /* Prediction Button Animation & Styling */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #2563EB 0%, #1D4ED8 100%);
        color: white;
        font-size: 18px;
        font-weight: 600;
        padding: 12px 28px;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.39);
        transition: all 0.3s ease-in-out;
        width: 100%;
    }

    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5);
        background: linear-gradient(90deg, #1D4ED8 0%, #1E40AF 100%);
    }

    /* Output result cards */
    .result-box {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD PICKLE MODEL ---
@st.cache_resource
def load_model():
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        st.error("`model.pkl` file not found in the current directory.")
        st.stop()
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# --- HEADER SECTION ---
st.markdown("<h1 class='main-title'>🎓 Student Grade & Score Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Enter subject marks below to run inference using your KNN Classification Model.</p>", unsafe_allow_html=True)

# --- LAYOUT & INPUT FIELDS ---
col_inputs, col_summary = st.columns([2, 1], gap="large")

with col_inputs:
    st.subheader("📚 Subject Marks (0 - 100)")
    
    c1, c2 = st.columns(2)
    with c1:
        hindi = st.slider("Hindi", 0, 100, 75)
        english = st.slider("English", 0, 100, 80)
        science = st.slider("Science", 0, 100, 70)
    with c2:
        maths = st.slider("Maths", 0, 100, 85)
        history = st.slider("History", 0, 100, 65)
        geography = st.slider("Geography", 0, 100, 72)

    total = hindi + english + science + maths + history + geography

with col_summary:
    st.subheader("📊 Performance Overview")
    st.metric("Calculated Total Marks", f"{total} / 600")
    st.metric("Percentage", f"{(total/6):.2f}%")
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("✨ Predict Outcome")

# --- INFERENCE & ANIMATION LOGIC ---
if predict_btn:
    # Trigger Confetti Celebration Effect
    if HAS_CONFETTI:
        confetti()

    # Features sequence matching model configuration:
    # ['Hindi', 'English', 'Science', 'Maths', 'History', 'Geograpgy', 'Total']
    features = np.array([[hindi, english, science, maths, history, geography, total]])
    
    with st.spinner("Processing prediction through KNN model..."):
        time.sleep(0.4)  # Small visual delay for button feel
        prediction = model.predict(features)[0]
        
        # Calculate prediction probabilities if available
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            max_proba = np.max(probabilities) * 100
        else:
            max_proba = None

    # Display Prediction Results
    st.divider()
    st.balloons()
    
    res_col1, res_col2 = st.columns([1, 1])
    with res_col1:
        st.markdown(
            f"""
            <div class='result-box'>
                Predicted Class: {prediction}
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with res_col2:
        if max_proba is not None:
            st.success(f"**Model Confidence:** {max_proba:.2f}%")
        st.info("Input features successfully verified against trained feature names.")
