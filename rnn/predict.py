import tensorflow as tf
from keras.datasets import imdb
from keras_preprocessing.sequence import pad_sequences
import numpy as np

MAX_LEN = 250

# Loading in model
model = tf.keras.models.load_model("rnn\models\mov_reviews.h5")

# Encoding input text review in the same format as data fed to the model

# Dict of indices of vocab words in imdb
word_index = imdb.get_word_index()

# Encodes text into a vector of integers
def encode_text(text):
    # Converts words in input text sentence to tokens
    tokens = tf.keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    return pad_sequences([tokens], MAX_LEN)[0]

# Input text review
text = "I loved that movie, it was so amazing"
encoded = encode_text(text)
print(encoded)

# A dictionary that reverses key, value pairs from og word index dictionary
reverse_word_index = {value: key for (key, value) in word_index.items()}

# Decodes input text (not a necessary function)
def decode_text(integers):
    text = ""
    for num in integers:
        if num != 0:
            text += reverse_word_index[num] + " "
    # Returns result text minus the last whitespace
    return text[:-1]

print(decode_text(encoded))


# Making binary prediction on input reviews
def predict(text):
    encoded_text = encode_text(text)
    pred = np.zeros((1, 250))
    pred[0] = encoded_text 
    result = model.predict(pred)
    print(result[0])

pos_review = "That movie was so awesome. I really loved and would watch it again"
predict(pos_review)

neg_review = "That movie really sucked. I hated it and wouldn't watch it again. Was one of the worst things I've ever watched "
predict(neg_review)