import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer

st.title("Sarcasm Text Detection")

# Load the model
model = joblib.load('sarcasm_detect.joblib')

# Define the vectorizer used in the model
vectorizer = model.named_steps['vect']

text = st.text_input("Enter Text: ", placeholder='Write a text for detect sarcasm or not sarcasm')

col2, col3 = st.columns(2)

def handle_input_text():
    if len(text) != 0:
        # Transform the input text to the format used by the model
        input_vectorized = vectorizer.transform([text])
        # Make prediction
        prob = model.named_steps['clf'].predict_proba(input_vectorized)[0, 1]
        # Convert probability to prediction
        prediction = model.named_steps['clf'].predict(input_vectorized)[0]
        if prediction == 'Sarcastic':
            col3.write("Sarcastic")
        else:
            col3.write("Not Sarcastic")
    else:
        col3.write("")

col2.button("Detect🔍", on_click=handle_input_text)