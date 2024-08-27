import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import streamlit as st

# Title for the Streamlit app
st.title("Sarcasm Text Detection")

# Define parameters for the tokenizer and model
vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 20000
epochs = 5

# Read CSV file using pandas
sarcasm_df = pd.read_csv("Data.csv")

# Split the data into input (headlines) and target (labels)
input_seq = sarcasm_df['headlines']
target_seq = sarcasm_df['target']

# Initialize and fit the tokenizer on the input sequences
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(input_seq)
word_index = tokenizer.word_index

# Load the pre-trained sarcasm detection model
model = tf.keras.models.load_model('sarcasm_detect.h5')

# Create a text input field in the Streamlit app
text = st.text_input("Enter Text: ", placeholder='Write a text to detect sarcasm or not sarcasm')

# Create columns for the button and the result display
col2, col3 = st.columns(2)

# Function to handle the detection process
def handle_input_text():
    if text:  # Check if the input text is not empty
        input_sentences = tokenizer.texts_to_sequences([text])
        input_padded_sentences = pad_sequences(input_sentences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
        probs = model.predict(input_padded_sentences)
        preds = int(np.round(probs[0][0]))  # Extract prediction and round it
        if preds == 1:
            col3.write("Sarcastic")
        else:
            col3.write("Not Sarcastic")
    else:
        col3.write("Please enter some text to analyze.")

# Add a button to trigger the detection process
col2.button("Detect🔍", on_click=handle_input_text)