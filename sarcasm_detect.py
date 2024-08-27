# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import streamlit as st

st.title("Sarcasm Text Detection")

# Define constants
vocab_size = 10000
embedding_dim = 16
max_sequence_length = 100  # Updated variable name
padding_type = 'post'
trunc_type = 'post'
oov_tok = "<OOV>"
epochs = 5

# Read csv file using pandas
sarcasm_df = pd.read_csv("Data.csv")

# Tokenization and padding setup
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(sarcasm_df['headlines'])
word_index = tokenizer.word_index

# Load the pre-trained model
model = tf.keras.models.load_model('sarcasm_detect.h5')

# Streamlit input text
text = st.text_input("Enter Text:", placeholder='Write a text to detect sarcasm or not sarcasm')

col2, col3 = st.columns(2)

def handle_input_text():
    if len(text) > 0:
        # Tokenize and pad input text
        input_sentences = tokenizer.texts_to_sequences([text])
        input_padded_sentences = pad_sequences(input_sentences, maxlen=max_sequence_length, padding=padding_type, truncating=trunc_type)
        
        # Predict sarcasm
        probs = model.predict(input_padded_sentences)
        preds = int(np.round(probs[0][0]))  # Accessing the first prediction
        
        # Display result
        if preds == 1:
            col3.write("Sarcastic")
        else:
            col3.write("Not Sarcastic")
    else:
        col3.write("")

col2.button("Detect🔍", on_click=handle_input_text)