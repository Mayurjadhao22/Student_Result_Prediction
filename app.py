import pickle
import numpy as np
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Student Performance & Career Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling and button click effects
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header Styling */
    .main-title {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Card Container */
    .css-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }

    /* Prediction Button Effect */
    div.stButton > button:first-child {
        background-color: #2563EB;
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        width: 100%;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
    
    div.stButton > button:first-child:hover {
        background-color: #1D4ED8;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4);
    }

    div.stButton > button:first-child:active {
        transform: scale(0.97);
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("Error: `model.pkl` file not found. Please place it in the same directory as `app.py`.")
    st.stop()

# Header Section
st.markdown("<h1 class='main-title'>🎓 Student Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Enter academic marks below to predict student performance/category.</p>", unsafe_allow_html=True)

# Sidebar Info
st.sidebar.header("📊 Model Specifications")
st.sidebar.info("""
- **Algorithm:** K-Neighbors Classifier
- **Input Features:** 7 Academic Scores
- **Metric:** Minkowski Distance
""")

# Input Form
st.markdown("### 📝 Enter Subject Marks")

col1, col2 = st.columns(2)

with col1:
    hindi = st.slider("Hindi Score", min_value=0, max_value=100, value=75, step=1)
    english = st.slider("English Score", min_value=0, max_value=100, value=80, step=1)
    science = st.slider("Science Score", min_value=0, max_value=100, value=85, step=1)
    maths = st.slider("Maths Score", min_value=0, max_value=100, value=90, step=1)

with col2:
    history = st.slider("History Score", min_value=0, max_value=100, value=70, step=1)
    geography = st.slider("Geography Score", min_value=0, max_value=100, value=72, step=1)
    
    # Auto-calculate total or allow manual input
    auto_total = st.checkbox("Calculate Total Automatically", value=True)
    if auto_total:
        total = hindi + english + science + maths + history + geography
        st.number_input("Total Marks (Calculated)", value=total, disabled=True)
    else:
        total = st.number_input("Total Marks", min_value=0, max_value=600, value=472, step=1)

st.divider()

# Prediction Action
predict_btn = st.button("🚀 Run Prediction")

if predict_btn:
    # Button animation effect (Balloons & Spinner)
    with st.spinner("Analyzing marks and computing nearest neighbors..."):
        features = np.array([[hindi, english, science, maths, history, geography, total]])
        prediction = model.predict(features)[0]
        
        # Check if model supports probability estimation
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            confidence = np.max(probabilities) * 100
        else:
            confidence = None

    st.balloons()
    
    st.markdown("---")
    st.markdown("### 🎯 Result")
    
    res_col1, res_col2 = st.columns([2, 1])
    
    with res_col1:
        st.success(f"**Predicted Category:** Class {prediction}")
        
    with res_col2:
        if confidence is not None:
            st.metric(label="Model Confidence", value=f"{confidence:.2f}%")
        else:
            st.metric(label="Status", value="Complete")
