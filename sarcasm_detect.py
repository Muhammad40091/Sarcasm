import pandas as pd
import re
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

st.title("Sarcasm Text Detection")

# Load the model and tokenizer
try:
    model = tf.keras.models.load_model('sarcasm_detection_model.h5')
    
    # Load the tokenizer
    with open('tokenizer.json') as f:
        tokenizer_json = f.read()
        tokenizer = tokenizer_from_json(tokenizer_json)
        
except Exception as e:
    st.error(f"Error loading the model or tokenizer: {e}")
    st.stop()

# Provide some example text suggestions
examples = [
    "Insufferable Man Utters Words ‘Craft Beer Movement’",
    "Anaheim Police Chief John Welter: 'Look, Our Job Is To Shoot People'",
    "Single, Unemployed Mother Leeching Off Government",
    "Mayor Of Phoenix Apologizes For Naming Berlin Germany Of 1941 As Sister City",
    "Report: Average American Feels Comfortable In Own Skin For Only 6% Of Day",
    "Zoologists: Ape Neurology Much Like That Of Banana-Obsessed Humans",
    "Breaking: You Have Reached Your Free Article Limit",
    "Nation Could Really Use A Few Days Where It Isn’t Gripped By Something"
]

# Text input from user with a placeholder
text = st.text_input("Enter Text:", placeholder=examples[3])

# Create two columns: one for the button and one for the color palette dropdown
col1, col2 = st.columns([2, 1])

with col1:
    if st.button("Detect 🔍"):
        if text.strip():  # Check if text is not empty after stripping whitespace
            try:
                # Preprocess the input text
                def preprocess_text(text):
                    text = re.sub(r'http\S+', '', text)  # Remove URLs
                    text = re.sub(r'[^\w\s,¡!¿?()\'\"@#€£$%^&*+=\[\]{};:\'\"“”‘’—•✓✗💬💔❤️👍👎😂🤣😃😄😁😆😅🙃😉😊😎😍😘😗😙😚😋😛😜😝😎😏😒😓😔😕😲😳😧😨😰😥😢😭😪😓😵😡😠😤😣😖😞😓😩😫😢😥😬]', '', text)
                    text = ' '.join(text.split())
                    return text

                text = preprocess_text(text)
                
                # Tokenize and pad the input text
                maxlen = 100
                input_seq = tokenizer.texts_to_sequences([text])
                input_pad = pad_sequences(input_seq, maxlen=maxlen)
                
                # Make prediction
                prob = model.predict(input_pad)[0, 1]
                prediction = np.argmax(model.predict(input_pad), axis=1)[0]

                # Display the prediction
                if prediction == 1:  # Assuming '1' indicates sarcasm in your target variable
                    st.write("Sarcastic")
                else:
                    st.write("Not Sarcastic")

                # Display probability for additional insight
                st.write(f"Probability of Sarcasm: {prob:.2f}")
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.write("Please enter some text.")

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
