import pandas as pd
import numpy as np
import joblib
import streamlit as st
import random

st.title("Sarcasm Text Detection")

# Load the model
model = joblib.load('sarcasm_detect.joblib')

sarcastic_sentences = [
   
    "Deli Worker Searches For Palest, Mealiest Tomato To Put On Customer’s Sandwich",
    "Cancer Researchers Develop Highly Promising New Pink Consumer Item",
    "Pope Francis Renounces Papacy After Falling In Love With Beautiful American Divorcee",
    "Insufferable Man Utters Words ‘Craft Beer Movement’",
    "Anaheim Police Chief John Welter: 'Look, Our Job Is To Shoot People'",
    "Single, Unemployed Mother Leeching Off Government",
    "Mayor Of Phoenix Apologizes For Naming Berlin Germany Of 1941 As Sister City",
    "Report: Average American Feels Comfortable In Own Skin For Only 6% Of Day",
    "Zoologists: Ape Neurology Much Like That Of Banana-Obsessed Humans",
    "Breaking: You Have Reached Your Free Article Limit",
    "Nation Could Really Use A Few Days Where It Isn’t Gripped By Something"
]

placeholder_text = random.choice(sarcastic_sentences)

# Ensure the model contains the vectorizer and classifier
try:
    vectorizer = model.named_steps['vect']
    clf = model.named_steps['clf']
except AttributeError:
    st.error("Model does not contain expected components.")
    st.stop()

text = st.text_input("Enter Text:", placeholder=placeholder_text)

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
        st.write("Please enter some text.")