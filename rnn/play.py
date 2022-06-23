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

# Building the model - we use a function to create different models with different parameters, specifically batch_size
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)

# Creating a loss function because TF does not have a built in loss function that analyzes 3D nested array of probabilities
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# Compiling the model
model.compile(optimizer='adam', loss=loss)

# Saving checkpoints as the model trains
# Directory where the checkpoints will be saved
checkpoint_dir = 'rnn\_training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

history = model.fit(data, epochs=50, callbacks=[checkpoint_callback])

# Saving the model
#model.save('rnn\models\play_model.h5')

# Rebuilding the model with batch_size = 1, so we can input one piece of text
model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)

# Find the latest checkpoint that stores the models weights from previous training
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

# Generating text from any string
def generate_text(model, start_string):
    # Number of characters to generate
    num_generate = 800

    # Preprocessing input text
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store results
    text_generated = []

    # Lower temp => more predictable text
    # Higher temp => more surprising text
    temperature = 1.0

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        
        # Flattens batch list
        predictions = tf.squeeze(predictions, 0)

        # Uses categorical distribution to predict character from model
        predictions = predictions/temperature
        pred_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # Add predicted char to next input to the model
        input_eval = tf.expand_dims([pred_id], 0)
        text_generated.append(idx2char[pred_id])
    return (start_string + ''.join(text_generated))

inp = input("Type a starting string: ")
print(generate_text(model, inp))

