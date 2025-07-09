import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import folium_static

# Load trained model and encoders
try:
    model = joblib.load("blood_availability_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")  # Load dictionary of encoders
    feature_names = joblib.load("feature_names.pkl")  # Load feature names from training
except FileNotFoundError:
    st.error("Model or encoders not found. Please train the model first.")
    st.stop()

# Streamlit UI
st.title("Blood Availability")

# User Inputs
blood_group = st.selectbox("Select Blood Group", label_encoders["Blood Group"].classes_.tolist())
hospital_name = st.selectbox("Select Hospital", ["JNM", "AIIMS Kalyani"])
location = st.selectbox("Enter Location", ["Kalyani","Kolkata", "Howrah"])
units_available = st.number_input("Units Available", min_value=0, max_value=20, value=5)

# Hospital locations for mapping
hospital_locations = {
    "JNM": [22.9833, 88.4220],  # Kalyani
    "AIIMS Kalyani": [22.9750, 88.4345]  # AIIMS Kalyani
}

# Predict Button
if st.button("Check Availability"):
    # Prepare input data with the correct feature names and order
    input_data = pd.DataFrame([[blood_group, hospital_name, location, units_available]], 
                              columns=feature_names)
    
    # Encode categorical features using the saved encoders
    for col in ["Blood Group", "Hospital Name", "Location"]:
        input_data[col] = label_encoders[col].transform([input_data[col][0]])
    
    # Ensure column order matches training data
    input_data = pd.DataFrame([input_data.values[0]], columns=feature_names)
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    result = "Available" if prediction == 1 else "Not Available"

    st.subheader(f"Result: {result}")

    # Display hospital on the map with a green (available) or red (not available) marker
    if hospital_name in hospital_locations:
        map_location = hospital_locations[hospital_name]
        m = folium.Map(location=map_location, zoom_start=15)
        
        # Choose marker color based on availability
        marker_color = "green" if prediction == 1 else "red"
        
        # Add marker
        folium.Marker(
            location=map_location, 
            popup=f"{hospital_name} - {result}", 
            icon=folium.Icon(color=marker_color)
        ).add_to(m)
        
        # Show the map
        folium_static(m)
