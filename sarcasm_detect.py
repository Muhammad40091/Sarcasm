# Import some of the most important libraries for using this notebook
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st

# Streamlit app title
st.title("Sarcasm Text Detection")

# Configuration
vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 20000
epochs = 5

# Load the data
sarcasm_df = pd.read_csv("Data.csv")

# Split the columns
input_seq = sarcasm_df['headlines']
target_seq = sarcasm_df['target']

# Initialize the tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(input_seq)

# Load the model
try:
    model = tf.keras.models.load_model('sarcasm_detect.h5')
except Exception as e:
    st.error(f"Error loading the model: {e}")
    model = None

# Streamlit input
text = st.text_input("Enter Text:", placeholder='Write a text to detect sarcasm or not sarcasm')

# Define the function for handling input and prediction
def handle_input_text():
    if model is None:
        st.error("Model not loaded correctly.")
        return
    
    if len(text) == 0:
        st.warning("Please enter some text.")
        return

    try:
        input_sentences = tokenizer.texts_to_sequences([text])
        input_padded_sentences = pad_sequences(input_sentences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
        probs = model.predict(input_padded_sentences)
        preds = np.round(probs).astype(int)[0][0]
        
        if preds == 1:
            st.write("Sarcastic")
        else:
            st.write("Not Sarcastic")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Streamlit button to trigger detection
st.button("Detect🔍", on_click=handle_input_text)