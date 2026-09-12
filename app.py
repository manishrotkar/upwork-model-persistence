import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(
    page_title="Production Model Inference",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Interactive Machine Learning Predictor")
st.write("Adjust feature inputs below to get real-time predictions from the serialized `joblib` model artifact.")

MODEL_PATH = "rf_pipeline.joblib"

@st.cache_resource
def load_model():
    """Load and cache the trained end-to-end pipeline model artifact."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

pipeline = load_model()

if pipeline is None:
    st.error(f"Model file `{MODEL_PATH}` not found! Please run `python3 pipeline_persistence.py` first to generate the model artifact.")
else:
    st.subheader("📊 Feature Inputs")
    
    # Generate 10 input sliders matching our trained synthetic dataset's 10 features
    col1, col2 = st.columns(2)
    inputs = []
    
    for i in range(10):
        target_col = col1 if i % 2 == 0 else col2
        val = target_col.slider(
            label=f"Feature {i+1}",
            min_value=-3.0,
            max_value=3.0,
            value=0.0,
            step=0.1
        )
        inputs.append(val)
    
    st.divider()
    
    # Real-Time Inference Action
    if st.button("🚀 Predict Target Class", type="primary", use_container_width=True):
        input_data = np.array(inputs).reshape(1, -1)
        
        prediction = pipeline.predict(input_data)[0]
        probabilities = pipeline.predict_proba(input_data)[0]
        
        st.subheader("🎯 Prediction Results")
        
        if prediction == 1:
            st.success(f"**Predicted Class:** Class 1 (Positive) — Confidence: {probabilities[1]*100:.1f}%")
        else:
            st.info(f"**Predicted Class:** Class 0 (Negative) — Confidence: {probabilities[0]*100:.1f}%")
            
        st.write("### Prediction Probabilities")
        st.progress(float(probabilities[1]), text=f"Probability of Class 1: {probabilities[1]*100:.1f}%")