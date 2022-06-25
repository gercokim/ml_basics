import tensorflow as tf
import pandas as pd
import tensorflow_datasets as tfds
import numpy as np
import matplotlib as plt

# Retrieving file paths of data
train_file_path = 'rnn\data\itrain-data.tsv'
test_file_path = 'rnn\data\ivalid-data.tsv'

# Loading data into dataframes 
# Col 0 = label, Col 1 = feature data
train_df = pd.read_table(train_file_path, header=None)
test_df = pd.read_table(test_file_path, header=None)
#print(train_df.head())
#print(test_df.head())
print(train_df.count(), test_df.count())

BUFFER_SIZE = 500
BATCH_SIZE = 64

train_example = train_df.pop(1).to_numpy()
train_label = train_df.pop(0).to_numpy()

print(train_example)
print(train_label)





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

test_predictions()

