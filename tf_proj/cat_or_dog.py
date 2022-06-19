from sklearn.utils import shuffle
import tensorflow as tf

import os
import math
import numpy as np
import matplotlib.pyplot as pyplot
import plot

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

train_data_gen = train_image_generator.flow_from_directory(train_dir, target_size=(IMG_HEIGHT,IMG_WIDTH),batch_size=BATCH_SIZE, shuffle=True, class_mode='binary')
valid_data_gen = valid_image_generator.flow_from_directory(valid_dir, target_size=(IMG_HEIGHT,IMG_WIDTH),batch_size=BATCH_SIZE, class_mode='binary')
test_data_gen = test_image_generator.flow_from_directory(test_dir, target_size=(IMG_HEIGHT,IMG_WIDTH),batch_size=BATCH_SIZE, shuffle=False, class_mode='binary')

sample_training_images, _ = next(train_data_gen)
plot.plotImages(sample_training_images[:5])

# Reducing the chances of overfitting by using random transformations
train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# New data generator from new image generator
train_data_gen = train_image_generator.flow_from_directory(train_dir, target_size=(IMG_HEIGHT,IMG_WIDTH),batch_size=BATCH_SIZE, class_mode='binary')

augmented_images = [train_data_gen[0][0][0] for i in range(5)]

# Plots the same image 5 times with different variations
plot.plotImages(augmented_images)

# Building the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu, input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])

# Compiling the model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Training the model
history = model.fit(train_data_gen, epochs=epochs, steps_per_epoch=int(np.ceil(num_train / float(BATCH_SIZE))), validation_data=valid_data_gen, validation_steps=int(np.ceil(num_val / float(BATCH_SIZE))))

# Saving model in models folder
model.save('models\cats_v_dogs.h5')

# Visualizing accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

pyplot.figure(figsize=(8, 8))
pyplot.subplot(1, 2, 1)
pyplot.plot(epochs_range, acc, label='Training Accuracy')
pyplot.plot(epochs_range, val_acc, label='Validation Accuracy')
pyplot.legend(loc='lower right')
pyplot.title('Training and Validation Accuracy')

pyplot.subplot(1, 2, 2)
pyplot.plot(epochs_range, loss, label='Training Loss')
pyplot.plot(epochs_range, val_loss, label='Validation Loss')
pyplot.legend(loc='upper right')
pyplot.title('Training and Validation Loss')
pyplot.show()
