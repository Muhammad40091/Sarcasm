import pandas as pd
import numpy as np
import joblib
import streamlit as st
import random
from sklearn.feature_extraction.text import CountVectorizer

st.title("Sarcasm Text Detection")

# Define a list of sarcastic sentences
sarcastic_sentences = [
    "Overwhelmed Archaeologists Struggling To Keep Pace With Glut Of Early Humans Thawed Out By Climate Change",
    "Study Finds Majority Of Accidental Heroin Overdoses Could Be Prevented With Less Heroin",
    "Deli Worker Searches For Palest, Mealiest Tomato To Put On Customer’s Sandwich",
    "Cancer Researchers Develop Highly Promising New Pink Consumer Item",
    "Pope Francis Renounces Papacy After Falling In Love With Beautiful American Divorcee",
    "Insufferable Man Utters Words ‘Craft Beer Movement’",
    "Anaheim Police Chief John Welter: 'Look, Our Job Is To Shoot People'",
    "Single, Unemployed Mother Leeching Off Government",
    "Mayor Of Phoenix Apologizes For Naming Berlin Germany Of 1941 As Sister City",
    "Study: Coffee Drinkers At Far Higher Risk Of Having Mug Crash To Floor In Slow Motion After Hearing Their Father Is Dead",
    "Report: Average American Feels Comfortable In Own Skin For Only 6% Of Day",
    "Report: You Were Lonely Before The Pandemic Started, And You’ll Be Lonely After It Ends",
    "Zoologists: Ape Neurology Much Like That Of Banana-Obsessed Humans",
    "Breaking: You Have Reached Your Free Article Limit",
    "Nation Could Really Use A Few Days Where It Isn’t Gripped By Something"
]

# Load the model
model = joblib.load('sarcasm_detect.joblib')

# Define the vectorizer used in the model
vectorizer = model.named_steps['vect']

# Select a random sarcastic sentence for the placeholder
placeholder_text = random.choice(sarcastic_sentences)

# Create the text input with the random sarcastic sentence as placeholder
text = st.text_input("Enter Text: ", placeholder=placeholder_text)

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