from sklearn.utils import shuffle
import tensorflow as tf

import os
import numpy as np
import matplotlib.pyplot as pyplot

PATH = 'cats_and_dogs'

train_dir = os.path.join(PATH, 'train')
valid_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')

# Number of files in each directory
num_train = sum([len(files) for r, d, files in os.walk(train_dir)])
num_val = sum([len(files) for r, d, files in os.walk(valid_dir)])
num_test = len(os.listdir(os.path.join(test_dir, 'data')))

print("Total training images: ", num_train)
print("Total validation images: ", num_val)
print("Total testing images: ", num_test)


# Hyperparameters for preprocessing and training
BATCH_SIZE = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

# Create image generators / augmenting data
train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
valid_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(train_dir, target_size=(IMG_HEIGHT,IMG_WIDTH),batch_size=BATCH_SIZE, class_mode='binary')
valid_data_gen = valid_image_generator.flow_from_directory(valid_dir, target_size=(IMG_HEIGHT,IMG_WIDTH),batch_size=BATCH_SIZE, class_mode='binary')
test_image_gen = test_image_generator.flow_from_directory(test_dir, target_size=(IMG_HEIGHT,IMG_WIDTH),batch_size=BATCH_SIZE, shuffle=False, class_mode='binary')

# plots images of data with array of images and probabilities list
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
          if probability > 0.5:
              ax.set_title("%.2f" % (probability*100) + "% dog")
          else:
              ax.set_title("%.2f" % ((1-probability)*100) + "% cat")
    pyplot.show()

sample_training_images, _ = next(train_data_gen)
plotImages(sample_training_images[:5])

# Reducing the chances of overfitting by using random transformations
train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=40, horizontal_flip=True, zoom_range=0.2, shear_range=0.2)

# New data generator from new image generator
train_data_gen = train_image_generator.flow_from_directory(train_dir, target_size=(IMG_HEIGHT,IMG_WIDTH),batch_size=BATCH_SIZE, class_mode='binary')

augmented_images = [train_data_gen[0][0][0] for i in range(5)]

# Plots the same image 5 times with different variations
plotImages(augmented_images)




