import tensorflow as tf

import os
import math
import numpy as np
import matplotlib.pyplot as pyplot
import plot

PATH = 'cats_and_dogs'

test_dir = os.path.join(PATH, 'test')

BATCH_SIZE = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

test_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_data_gen = test_image_generator.flow_from_directory(test_dir, target_size=(IMG_HEIGHT,IMG_WIDTH),batch_size=BATCH_SIZE, shuffle=False, class_mode='binary')


model = tf.keras.models.load_model('models')

probabilities = model.predict(test_data_gen)
#probabilities = probabilities.flatten()
print(probabilities, len(probabilities[0]))

sample_testing_images, _ = next(test_data_gen)
print(len(sample_testing_images))

# 0, 4 - 5, 9 - 10, 14
for i in range(10):
    plot.plotImages(sample_testing_images[5*i:5*i+4], probabilities=probabilities)