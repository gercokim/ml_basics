
import tensorflow as tf
import tensorflow_datasets as tfds

import math
import numpy as np
import matplotlib.pyplot as pyplot

# Load in tensorflow fashion mnist dataset
data = tf.keras.datasets.fashion_mnist

# obtain training and testing data
(image_train, label_train), (image_test, label_test) = data.load_data()

# defining class name list
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(label_train)

# Initial image data inspection before preprocessing
pyplot.figure()
pyplot.imshow(image_train[0])
pyplot.colorbar()
pyplot.grid(False)
#pyplot.show()

# preprocessing data
image_train = image_train / 255.0
image_test = image_test / 255.0

# number of training and testing data
print(len(image_train), len(image_test))

# displays first 25 images and class names from training set
pyplot.figure(figsize=(10, 10))
for i in range(25):
    pyplot.subplot(5, 5, i+1)
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.grid(False)
    pyplot.imshow(image_train[i], cmap=pyplot.cm.binary)
    pyplot.xlabel(class_names[label_train[i]])
#pyplot.show()

# Building the model (3 layer network)
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)), # transforms images from 2d array of 28 by 28 to 1d array of 784 (input)
    tf.keras.layers.Dense(128, activation=tf.nn.relu), # densely connected layer of 128 nodes (hidden)
    tf.keras.layers.Dense(10, activation=tf.nn.softmax) # Each node of the 10 nodes represents clothing, holding a probability value (output)
])

# Compiling the model
model.compile(optimizer='adam', # adjusting parameters
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # loss function
              metrics=['accuracy']) # used to monitor training and testing steps

BATCH_SIZE = 32
# Training the model
model.fit(image_train, label_train, epochs=10) #, steps_per_epoch=math.ceil(len(image_train)/BATCH_SIZE)) #32 is batch size

# Assessing performance
test_loss, test_acc = model.evaluate(image_test, label_test, verbose=2) #, steps=math.ceil(len(image_test))/BATCH_SIZE)
print('Acc on test data: ', test_acc)

# Making predictions
predictions = model.predict(image_test)
print(predictions[0])

# The index with the greatest probability
print(np.argmax(predictions[0]))

print("Predicted clothe for first example: ", class_names[np.argmax(predictions[0])])
print("Actual clothe for first example: ", class_names[label_test[0]])

# Shows image of the specified test example, prediction, and probability of said prediction, as well as the actual result
def im_plot(i, preds, label, img):
    label, img = label[i], img[i]
    pyplot.grid(False)
    pyplot.xticks([])
    pyplot.yticks([])

    pyplot.imshow(img, cmap=pyplot.cm.binary)

    pred_label = np.argmax(preds)
    if pred_label == label:
        color = 'blue'
    else:
        color = 'red'
    
    pyplot.xlabel("{} {:2.0f}% ({})".format(class_names[pred_label], 100*np.max(preds), class_names[label]), color=color)

# plots the probability distribution
def plot_value_array(i, preds, label):
    label = label[i]
    pyplot.grid(False)
    pyplot.xticks(range(10))
    pyplot.yticks([])
    thisplot = pyplot.bar(range(10), preds, color="#777777")
    pyplot.ylim([0, 1])
    pred_label = np.argmax(preds)
    thisplot[pred_label].set_color('red')
    thisplot[label].set_color('blue')

# First item in the testing data
i = 0
pyplot.figure(figsize=(6, 3))
pyplot.subplot(1, 2, 1)
im_plot(i, predictions[i], label_test, image_test)
pyplot.subplot(1, 2, 2)
plot_value_array(i, predictions[i], label_test)
pyplot.show()

# displaying first 15 testing examples and their results
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
pyplot.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    pyplot.subplot(num_rows, 2*num_cols, 2*i+1)
    im_plot(i, predictions[i], label_test, image_test)
    pyplot.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], label_test)
pyplot.tight_layout()
pyplot.show()
