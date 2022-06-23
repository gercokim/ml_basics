import tensorflow as tf
from keras_preprocessing import sequence
import os
import numpy as np

# Importing file path of shakespeare play
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# Reading and decoding text
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# Length of text / number of chars in text
print('Length of text: {} chars'.format(len(text)))

# First 250 chars
print(text[:250])

# Preprocess text / encoding as vector of ints
vocab = sorted(set(text)) # Sorts unique chars in text
# Creating mapping from unique chars to indices
char2idx = {u : i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# Returns vector of ints from text
def text_to_int(text):
    return np.array([char2idx[c] for c in text])
text_as_int = text_to_int(text)

# Comparing og text to encoded text
print("Text: ", text[:10])
print("Encoded: ", text_as_int[:10])

# Return vector of ints to text
def int_to_text(ints):
    try:
        ints = ints.numpy()
    except:
        pass
    return ''.join(idx2char[ints])

print(int_to_text(text_as_int[5:7]))