import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample data
sentences = [
    ("hello world", "hi world"),
    ("apple orange", "apple banana"),
    ("machine learning", "deep learning"),
    ("openai", "openai gpt"),
    # Add more data here
]

# Combine sentences to form corrupted input
corrupted_sentences = [s1 + ' ' + s2 + ' ' + s1 for s1, s2 in sentences]

# Tokenization and padding
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(corrupted_sentences)
vocab_size = len(tokenizer.word_index) + 1

max_length = max([len(s) for s in corrupted_sentences])
X = tokenizer.texts_to_sequences(corrupted_sentences)
X = pad_sequences(X, maxlen=max_length, padding='post')

# Build the autoencoder model (same as before)
input_layer = Input(shape=(max_length,))
embedding = Embedding(vocab_size, 64, input_length=max_length)(input_layer)
bi_lstm = Bidirectional(LSTM(32))(embedding)
decoded = Dense(vocab_size, activation='softmax')(bi_lstm)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
autoencoder.fit(X, X, epochs=10, batch_size=4)

# Encode the sentences
encoder = Model(inputs=input_layer, outputs=bi_lstm)
encoded_data = encoder.predict(X)

# Now you can use the encoded_data for prediction
new_samples = [
    ("hello world", "hi there"),
    ("apple banana", "banana apple"),
    ("machine learning", "learning machine"),
    ("openai", "gpt openai"),
    # Add more new samples here
]

new_encoded_samples = []

for s1, s2 in new_samples:
    new_input = tokenizer.texts_to_sequences([s1 + ' ' + s2 + ' ' + s1])
    new_input = pad_sequences(new_input, maxlen=max_length, padding='post')
    new_encoded = encoder.predict(new_input)
    new_encoded_samples.append(new_encoded)

# Now you can use new_encoded_samples for further analysis or predictions
print("Encoded data for new samples:")
print(new_encoded_samples)

