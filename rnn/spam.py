import tensorflow as tf
import pandas as pd
import tensorflow_datasets as tfds
import numpy as np
import matplotlib as plt

# Retrieving file paths of data
train_file_path = 'rnn\data\itrain-data.tsv'
test_file_path = 'rnn\data\ivalid-data.tsv'

# For Mac OS
# train_file_path = 'data/itrain-data.tsv'
# test_file_path = 'data/ivalid-data.tsv'

# Loading data into dataframes 
# Col 0 = label, Col 1 = feature data
train_df = pd.read_table(train_file_path, header=None)
test_df = pd.read_table(test_file_path, header=None)
#print(train_df.head())
#print(test_df.head())
print(train_df.to_numpy().size, test_df.to_numpy().size)

# Converting categorical labels into binary 
train_df[0] = train_df[0].map({'ham': 1, 'spam': 0})
test_df[0] = test_df[0].map({'ham': 1, 'spam' : 0})

# Some hyperparameters
BUFFER_SIZE = 500
BATCH_SIZE = 64
SEED = 42
VOCAB_SIZE = 4000
EMBEDDING_DIM = 32


# Converting train examples/labels into numpy arrays
train_example = train_df.pop(1).to_numpy()
train_label = train_df.pop(0).to_numpy()

test_example = test_df.pop(1).to_numpy()
test_label = test_df.pop(0).to_numpy()

# Converting test+train example/label numpy arrays into tfds
train_dataset = tf.data.Dataset.from_tensor_slices((train_example, train_label))
test_dataset = tf.data.Dataset.from_tensor_slices((train_example, train_label))

# Shuffling and batching datasets
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Seeing the training data
for example, label in train_dataset.take(1):
  print('texts: ', example.numpy()[:3])
  print()
  print('labels: ', label.numpy()[:3])

# Preprocessing the data into a vector
text_encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)

# Setting the dataset vocabulary
text_encoder.adapt(train_dataset.map(lambda text, label : text))
vocab = np.array(text_encoder.get_vocabulary())

# Creating the model
model = tf.keras.Sequential([
  # encoder layer converts text into sequence indices
  text_encoder,
  tf.keras.layers.Embedding(len(text_encoder.get_vocabulary()), EMBEDDING_DIM, mask_zero=True),
  # RNN layer wrapped by bidirectional layer
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(1)
])

# Compiling the model
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer='adam', metrics='accuracy')

# Training the model
history = model.fit(train_dataset, epochs=20, validation_data = test_dataset, validation_steps=30)

# Saving the model
model.save("rnn\models\spam_model", save_format='tf')

sample_text = 'Congratulations! You won a $100 gift card. Click this link to claim it'
predictions = model.predict(np.array([sample_text]))
print(predictions)

# Predicting ham or spam
def predict_message(pred_text):
    return 0

def test_predictions():
  test_messages = ["how are you doing today",
                   "sale today! to stop texts call 98912460324",
                   "i dont want to go. can we try it a different day? available sat",
                   "our new mobile video service is live. just install on your phone to start watching.",
                   "you have won £1000 cash! call to claim your prize.",
                   "i'll bring it tomorrow. don't forget the milk.",
                   "wow, is your arm alright. that happened to me one time too"
                  ]

  test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
  passed = True

  for msg, ans in zip(test_messages, test_answers):
    prediction = predict_message(msg)
    if prediction[1] != ans:
      passed = False

  if passed:
    print("Tests have passed")
  else:
    print("Tests have failed")

#test_predictions()

