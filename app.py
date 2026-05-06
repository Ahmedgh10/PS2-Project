import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn

# Define the precise PyTorch architecture that was exported
class ANN_Baseline(nn.Module):
    def __init__(self, input_dim):
        super(ANN_Baseline, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

# Streamlit App Configurations
st.set_page_config(page_title="Academic Risk Predictor", page_icon="🎓", layout="centered")

st.title("🎓 Prediction of Students at Risk of Academic Failure")
st.write("""
This system uses a Deep Learning (Artificial Neural Network) model trained on behavioral and academic indicators 
to proactively identify students who are at risk of failing an academic course.
""")

# Load the saved model and scaler
@st.cache_resource
def load_assets():
    scaler = joblib.load('scaler.pkl')
    model = ANN_Baseline(10)
    model.load_state_dict(torch.load('ann_model.pth', weights_only=True))
    model.eval()
    return scaler, model

try:
    scaler, model = load_assets()
except Exception as e:
    st.error("Error loading model artifacts. Make sure `scaler.pkl` and `ann_model.pth` exist in the directory.")
    st.stop()

# --- Sidebar Inputs ---
st.sidebar.header("Student Information")
st.sidebar.write("Adjust the parameters to see the risk probability.")

age = st.sidebar.slider("Age", 16, 60, 21)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
ses = st.sidebar.selectbox("Socio-Economic Status", ["Low", "Medium", "High"])
gpa = st.sidebar.slider("GPA", 0.0, 4.0, 2.5, step=0.01)
attendance = st.sidebar.slider("Attendance Rate (%)", 0.0, 100.0, 75.0, step=0.5)
assignments = st.sidebar.slider("Assignments Submitted", 0, 30, 8)
online_hours = st.sidebar.slider("Online Learning Hours / Week", 0.0, 40.0, 5.0, step=0.5)
forum_part = st.sidebar.slider("Forum Participation (Posts/Replies)", 0, 50, 3)
lab_perf = st.sidebar.slider("Lab/Practical Performance (%)", 0.0, 100.0, 60.0, step=1.0)
extra_curr = st.sidebar.selectbox("Participates in Extracurriculars?", ["No", "Yes"])

st.sidebar.divider()
predict_btn = st.sidebar.button("Predict Risk Status", type="primary", use_container_width=True)

# --- Main App Logic ---
if predict_btn:
    # 1. Map Categorical UI inputs to ordinal/binary matching the training process
    gender_map = {"Female": 0, "Male": 1}
    ses_map = {"Low": 0, "Medium": 1, "High": 2}
    extra_map = {"No": 0, "Yes": 1}
    
    # Needs exact column order: Age, Gender, Socio_Economic_Status, GPA, Attendance_Rate, 
    # Assignments_Submitted, Online_Learning_Hours, Forum_Participation, Lab_Performance, Extracurricular
    features = np.array([[
        age, 
        gender_map[gender], 
        ses_map[ses], 
        gpa, 
        attendance, 
        assignments, 
        online_hours, 
        forum_part, 
        lab_perf, 
        extra_map[extra_curr]
    ]])
    
    # 2. Scale features
    features_scaled = scaler.transform(features)
    
    # 3. Create tensor and Predict
    features_tensor = torch.FloatTensor(features_scaled)
    
    with torch.no_grad():
        logits = model(features_tensor)
        prob = torch.sigmoid(logits).item()
        
    # 4. Display Results
    st.divider()
    
    if prob >= 0.5:
        st.error(f"### ⚠️ At Risk of Academic Failure")
        st.write(f"The model predicts a **{prob:.1%}** probability that this student is at risk.")
        
        st.markdown("#### Triage Recommendations:")
        st.warning("""
        - 📅 **Schedule Meeting**: Initiate a 1-on-1 advisor check-in immediately.
        - 📚 **Tutoring**: Recommend complementary learning resources.
        - 📊 **Monitoring**: Keep a close eye on their subsequent assignments and class attendance.
        """)
    else:
        st.success(f"### ✅ Standard Progression (Not At Risk)")
        st.write(f"The model predicts only a **{prob:.1%}** probability of academic failure.")
        st.info("No immediate intervention is required, but encourage continued consistent performance.")