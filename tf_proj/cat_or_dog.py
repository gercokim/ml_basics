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
num_test = total_test = len(os.listdir(test_dir))

print("Total training images: ", num_train)
print("Total validation images: ", num_val)
print("Total testing images: ", num_test)


# Hyperparameters for preprocessing and training
BATCH_SIZE = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150
