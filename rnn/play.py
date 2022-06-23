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

print(int_to_text(text_as_int[:10]))

# Creating training examples
# Each training example will use a sequence as input and outputs the same sequence shifted to the right by one letter
# EX: input: Hell | output: ello
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

# Creating training dataset / stream of chars
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

# Creating batches of length 101 from dataset
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

# Splitting sequences of 101 into input and output
def split_input_target(chunk):
    input_text = chunk[:-1] # hell
    target_text = chunk[1:] # ello
    return input_text, target_text

# Transforms each batch within the dataset into input:output
dataset = sequences.map(split_input_target)

# Hyperparameters and making training batches
BATCH_SIZE = 64
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 256
RNN_UNITS = 1024

# Buffer size to shuffle dataset
BUFFER_SIZE = 10000

data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
