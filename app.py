import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Employee Attrition Prediction")
st.caption("Random Forest | K-Fold Cross-Validated Model")

# -------------------------------------------------
# LOAD ARTIFACTS
# -------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("employee_attrition_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("model_features.pkl")
    return model, scaler, features

if not (
    os.path.exists("employee_attrition_model.pkl")
    and os.path.exists("scaler.pkl")
    and os.path.exists("model_features.pkl")
):
    st.error("❌ Model files not found. Train the model first.")
    st.stop()

model, scaler, feature_names = load_artifacts()

# -------------------------------------------------
# INPUT UI (CLEAN & GROUPED)
# -------------------------------------------------
st.subheader("🧑‍💼 Employee Details")

col1, col2 = st.columns(2)
user_input = {}

with col1:
    for feature in feature_names[: len(feature_names)//2]:
        user_input[feature] = st.slider(
            label=feature,
            min_value=0.0,
            max_value=100.0,
            value=50.0
        )

with col2:
    for feature in feature_names[len(feature_names)//2 :]:
        user_input[feature] = st.slider(
            label=feature,
            min_value=0.0,
            max_value=100.0,
            value=50.0
        )

input_df = pd.DataFrame([user_input])

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
st.markdown("---")
if st.button("🔮 Predict Attrition"):
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("📌 Prediction Result")

    if pred == 1:
        st.error(f"⚠️ Employee likely to LEAVE\n\nProbability: {prob:.2f}")
    else:
        st.success(f"✅ Employee likely to STAY\n\nProbability: {1 - prob:.2f}")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("ℹ️ About Model")
st.sidebar.markdown("""
- **Model:** Random Forest Classifier  
- **Learning:** Supervised  
- **Validation:** K-Fold Cross Validation  
- **Training Accuracy:** 1.0  
- **Deployment:** Joblib-based  
""")

