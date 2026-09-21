import os
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# --- CONFIGURATION ---
REPO_ID = "BenBlay/visit-with-us-model"
MODEL_FILENAME = "best_tourism_model.joblib"

# Loading the model from the Hugging Face Model Hub
@st.cache_resource  # Caches the model so it does not reload on every button click
def load_model():
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
    return joblib.load(model_path)

model = load_model()

# Streamlit UI for Tourism Package Prediction
st.set_page_config(page_title="Wellness Package Predictor", layout="centered")
st.title("Visit with Us: Travel Package Predictor")
st.write("Determine the likelihood of a customer purchasing the new **Wellness Tourism Package**.")

st.divider()

# Create two columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    Age = st.slider("Age", 18, 70, 30)
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    DurationOfPitch = st.slider("Duration of Pitch (mins)", 0, 100, 15)
    Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    NumberOfPersonVisiting = st.slider("Total Persons Visiting", 1, 5, 2)
    NumberOfFollowups = st.slider("Follow-ups by Salesperson", 1, 10, 3)

with col2:
    ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    PreferredPropertyStar = st.selectbox("Preferred Hotel Rating", [1, 2, 3, 4, 5])
    MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unmarried"])
    NumberOfTrips = st.slider("Trips per Year", 1, 20, 3)
    Passport = st.selectbox("Holds a Passport?", ["Yes", "No"])
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
    NumberOfChildrenVisiting = st.slider("Children Visiting (< age 5)", 0, 5, 1)

Designation = st.selectbox("Customer Designation", ["Executive", "Manager", "AVP", "VP", "Sr. Manager"])
MonthlyIncome = st.number_input("Monthly Income ($)", min_value=1000.0, value=25000.0)

# ----------------------------
# Prepare input data
# ----------------------------
# Note: The keys here MUST match the exact column names used during training (Xtrain)
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])

# Set the classification threshold (0.45 as defined in train.py)
classification_threshold = 0.45

st.divider()

# Predict button
if st.button("Run Prediction"):
    # Since we used a Pipeline in train.py, scaling/encoding is handled automatically
    prob = model.predict_proba(input_data)[0, 1]
    pred = int(prob >= classification_threshold)

    if pred == 1:
        st.success(f"**PREDICTION: YES!** This customer is highly likely to purchase the package (Probability: {prob:.2f})")
        st.balloons()
    else:
        st.warning(f"**PREDICTION: NO.** This customer is unlikely to purchase the package (Probability: {prob:.2f})")
