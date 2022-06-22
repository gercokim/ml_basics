import tensorflow as tf
from keras.datasets import imdb
from keras_preprocessing.sequence import pad_sequences
import os
import numpy as np


# Number of unique words in the dataset
VOCAB_SIZE = 88584

MAX_LEN = 250
BATCH_SIZE = 64

# Every review is already encoded as a vector of integers 
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=VOCAB_SIZE)

# An example of a review
print(train_data[0])

# Preprocessing - Making every review same length (250 words) in order for fixed input layer size
train_data = pad_sequences(train_data, MAX_LEN)
test_data = pad_sequences(test_data, MAX_LEN)

# Building the model - the last sigmoid activation function helps us set a threshold for pos vs neg reviews
# The embedding layer creates the relational vectors to group similar words
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compiling the model
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['accuracy'])

# Training the model
history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)

# Evaluating results on test data
results = model.evaluate(test_data, test_labels)
print(results)