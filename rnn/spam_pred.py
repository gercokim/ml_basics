import tensorflow as tf
import numpy as np

# Loading model
model = tf.keras.models.load_model('rnn\models\spam_model')

# Predicting whether message is spam or ham
def predict_message(pred_text):
    pred = np.array([pred_text])
    predictions = model.predict(pred)
    if predictions[0] >= 0:
        return 'ham'
    else:
        return 'spam'

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
    if prediction != ans:
      passed = False

  if passed:
    print("Tests have passed")
  else:
    print("Tests have failed")

test_predictions()

inp = input("Type in a text: ")
print("This message is: ", predict_message(inp))