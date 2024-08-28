import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
import tensorflow as tf
import json

# Load the data
df = pd.read_csv('Emoji dataset.csv')

# Preprocess text data
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters except emojis and spaces
    text = re.sub(r'[^\w\s,¡!¿?()\'\"@#€£$%^&*+=\[\]{};:\'\"“”‘’—•✓✗💬💔❤️👍👎😂🤣😃😄😁😆😅🙃😉😊😎😍😘😗😙😚😋😛😜😝😎😏😒😓😔😕😲😳😧😨😰😥😢😭😪😓😵😡😠😤😣😖😞😓😩😫😢😥😬]', '', text)
    text = ' '.join(text.split())
    return text

# Apply preprocessing
df['Comments'] = df['Comments'].apply(preprocess_text)

# Define features and target
X = df['Comments']
y = df['Target']

# Encode the target labels
y = LabelEncoder().fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Tokenize the text
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure uniform input size
maxlen = 100  # Choose a maximum sequence length
X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)

# Build the model
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=maxlen),
    LSTM(64, return_sequences=True),
    LSTM(64),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # Assuming binary classification (sarcastic, not sarcastic)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Save the model
model.save('sarcasm_detection_model.h5')

# Save the tokenizer
tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w') as f:
    f.write(tokenizer_json)
