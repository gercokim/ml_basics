import tensorflow as tf

import os
import math
import numpy as np
import matplotlib.pyplot as pyplot

PATH = 'cats_and_dogs'

test_dir = os.path.join(PATH, 'test')

BATCH_SIZE = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

test_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_data_gen = test_image_generator.flow_from_directory(test_dir, target_size=(IMG_HEIGHT,IMG_WIDTH),batch_size=BATCH_SIZE, shuffle=False, class_mode='binary')

def plotImages(images_arr, probabilities = False):
    fig, axes = pyplot.subplots(len(images_arr), 1, figsize=(5,len(images_arr) * 3))
    if probabilities is False:
      for img, ax in zip( images_arr, axes):
          ax.imshow(img)
          ax.axis('off')
    else:
      for img, probability, ax in zip( images_arr, probabilities, axes):
        ax.imshow(img)
        ax.axis('off')
        if probability[0] > 0.5:
            ax.set_title("%.2f" % (probability[0]*100) + "% cat")
        else:
            ax.set_title("%.2f" % ((1-probability[0])*100) + "% dog")
    pyplot.show()

model = tf.keras.models.load_model('models')

probabilities = model.predict(test_data_gen)
#probabilities = probabilities.flatten()
print(probabilities, len(probabilities[0]))

sample_testing_images, _ = next(test_data_gen)
#print(sample_testing_images)

plotImages(sample_testing_images[17:23], probabilities=probabilities)