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