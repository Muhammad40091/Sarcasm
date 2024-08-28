import pandas as pd
import numpy as np
import joblib
import streamlit as st

st.title("Sarcasm Text Detection")

# Load the model
model = joblib.load('sarcasm_detect.joblib')

# Ensure the model contains the vectorizer and classifier
try:
    vectorizer = model.named_steps['vect']
    clf = model.named_steps['clf']
except AttributeError:
    st.error("Model does not contain expected components.")
    st.stop()

text = st.text_input("Enter Text:", placeholder='Prima Donna Surgeon Storms Out Of Half-Full Operating Theater')

# Create two columns: one for the button and one for the color palette dropdown
col1, col2 = st.columns([2, 1])

with col1:
    if st.button("Detect 🔍"):
        if text:
            try:
                # Transform the input text to the format used by the model
                input_vectorized = vectorizer.transform([text])
                # Make prediction
                prob = clf.predict_proba(input_vectorized)[0, 1]
                prediction = clf.predict(input_vectorized)[0]
                
                if prediction == 'Sarcastic':
                    st.write("Sarcastic")
                else:
                    st.write("Not Sarcastic")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.write("Not Sarcastic")

with col2:
    # Add custom CSS for styling the selectbox
    
    color = st.selectbox(
        "Select a Color Palette",
        options=[
            "Red", "Green", "Blue", "Yellow", "Orange", 
            "Purple", "Pink", "Brown", "Black", "White"
        ],  
    )
    
    # Dictionary to map color names to their hex codes
    color_map = {
        "Red": "#FF0000",
        "Green": "#00FF00",
        "Blue": "#0000FF",
        "Yellow": "#FFFF00",
        "Orange": "#FFA500",
        "Purple": "#800080",
        "Pink": "#FFC0CB",
        "Brown": "#A52A2A",
        "Black": "#000000",
        "White": "#FFFFFF"
    }

    # Show the selected color as a colored box
    st.markdown(
        f"<div style='background-color:{color_map[color]}; "
        f"width:30%; height:30px; border-radius:5px;'></div>",
        unsafe_allow_html=True
    )