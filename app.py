import time
import numpy as np
import pickle
import streamlit as st

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Academic Score Predictor",
    page_icon="🎓",
    layout="centered"
)

# -----------------------------------------------------------------------------
# Custom Styling (CSS & Animations)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
        color: #f8fafc;
    }

    /* Card Containers */
    .css-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* Custom Header Text */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a855f7 0%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 8px;
    }
    
    .sub-title {
        font-size: 1rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 32px;
    }

    /* Streamlit Button Styling Override */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #8b5cf6 0%, #ec4899 100%);
        color: white;
        border: none;
        padding: 12px 28px;
        font-size: 1.1rem;
        font-weight: 700;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(236, 72, 153, 0.4);
        transition: all 0.3s ease;
        width: 100%;
        cursor: pointer;
    }

    div.stButton > button:first-child:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 6px 20px rgba(236, 72, 153, 0.6);
        color: white;
    }

    /* Metric Display Box */
    .result-box {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.15) 0%, rgba(236, 72, 153, 0.15) 100%);
        border: 1px solid rgba(236, 72, 153, 0.3);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-top: 20px;
    }

    .result-score {
        font-size: 3.5rem;
        font-weight: 900;
        color: #f43f5e;
        text-shadow: 0 0 10px rgba(244, 63, 94, 0.5);
    }
</style>
""", unsafe_allow_allowed_html=True)

# -----------------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading `model.pkl`: {e}")
    st.stop()

# Feature names derived directly from your pickled KNN model
feature_names = ['Hindi', 'English', 'Science', 'Maths', 'History', 'Geograpgy', 'Total']

# -----------------------------------------------------------------------------
# UI Header
# -----------------------------------------------------------------------------
st.markdown("<h1 class='main-title'>Academic Score Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Enter subject marks to predict the final target outcome using K-Nearest Neighbors</p>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Input Form
# -----------------------------------------------------------------------------
st.markdown("<div class='css-card'>", unsafe_allow_html=True)
st.subheader("📝 Enter Subject Marks")

col1, col2 = st.columns(2)

with col1:
    hindi = st.number_input("Hindi", min_value=0, max_value=100, value=75, step=1)
    science = st.number_input("Science", min_value=0, max_value=100, value=80, step=1)
    history = st.number_input("History", min_value=0, max_value=100, value=70, step=1)

with col2:
    english = st.number_input("English", min_value=0, max_value=100, value=82, step=1)
    maths = st.number_input("Maths", min_value=0, max_value=100, value=85, step=1)
    geography = st.number_input("Geography", min_value=0, max_value=100, value=78, step=1)

# Auto-calculate total score
calculated_total = hindi + english + science + maths + history + geography

st.markdown("---")
st.markdown(f"**Calculated Total Score:** `{calculated_total} / 600`")

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Prediction & Animation Logic
# -----------------------------------------------------------------------------
if st.button("🚀 Predict Outcome"):
    # Visual Effects: Spinner & Progress Simulation
    with st.spinner("Analyzing performance patterns..."):
        progress_bar = st.progress(0)
        for percent in range(100):
            time.sleep(0.008)
            progress_bar.progress(percent + 1)
        progress_bar.empty()

    # Model Input Array
    input_data = np.array([[hindi, english, science, maths, history, geography, calculated_total]])
    
    # Make Prediction
    prediction = model.predict(input_data)[0]
    
    # Confetti Effect
    st.balloons()
    
    # Display Result Card
    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    st.markdown("### Output Prediction")
    st.markdown(f"<div class='result-score'>{prediction}</div>", unsafe_allow_html=True)
    
    # Probabilities if available
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        confidence = np.max(probabilities) * 100
        st.markdown(f"**Prediction Confidence:** `{confidence:.2f}%`")
        
    st.markdown("</div>", unsafe_allow_html=True)
