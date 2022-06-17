import tensorflow as tf
import tensorflow_datasets as tfds
import math
import numpy as np
import matplotlib.pyplot as pyplot
import fashion_mnist as f

# Load in data
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

# Initialize training and testing data
train, test = dataset['train'], dataset['test']
print(type(train))

# Defining class names list
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Format of the dataset
num_train = metadata.splits['train'].num_examples
num_test = metadata.splits['test'].num_examples

# Pre process data 
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

# Applying normalizer to data
train = train.map(normalize)
test = test.map(normalize)

# Caching the data in memory
train = train.cache()
test = test.cache()

# Visualizing an example in data
for image, label in test.take(1):
    break
image = image.numpy().reshape((28, 28))

# Plotting image
pyplot.figure()
pyplot.imshow(image, cmap=pyplot.cm.binary)
pyplot.colorbar()
pyplot.grid(False)
pyplot.show()

#Verifying first 25 examples in data
pyplot.figure(figsize=(10, 10))
i = 0
for (image, label) in test.take(25):
    image = image.numpy().reshape((28,28))
    pyplot.subplot(5, 5, i+1)
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.grid(False)
    pyplot.imshow(image, cmap=pyplot.cm.binary)
    pyplot.xlabel(class_names[label])
    i+=1
pyplot.show()

# Building the model
model = tf.keras.Sequential([
    # 2 pairs of Conv/MaxPool. First layer produces 32 convoluted images, then reduced by MaxPool
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu, input_shape=(28, 28, 1)), 
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compiling the model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Training the model

# Defining iteration behavior for training
    # Repeat forever by specifiying repeat()
    # Randomize order of training data with shuffle(size)
    # Specify the batch size for fit() with batch()
BATCH_SIZE = 32
train = train.cache().repeat().shuffle(num_train).batch(BATCH_SIZE)
test = test.cache().batch(BATCH_SIZE)

model.fit(train, epochs=10, steps_per_epoch=math.ceil(num_train/BATCH_SIZE))

# Evaluate model
test_loss, test_acc = model.evaluate(test, steps=math.ceil(num_test/BATCH_SIZE))
print("Acc on test dataset: ", test_acc)

# Make predictions
for image_test, label_test in test.take(1):
    image_test = image_test.numpy()
    label_test = label_test.numpy()
    predictions = model.predict(image_test)
print("Prediction shape: ", predictions.shape)

print("Predicted clothe for first example: ", class_names[np.argmax(predictions[0])])
print("Actual clothe for first example: ", class_names[label_test[0]])

# First item in the testing data
i = 0
pyplot.figure(figsize=(6, 3))
pyplot.subplot(1, 2, 1)
f.im_plot(i, predictions[i], label_test, image_test)
pyplot.subplot(1, 2, 2)
f.plot_value_array(i, predictions[i], label_test)
pyplot.show()

# displaying first 15 testing examples and their results
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
pyplot.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    pyplot.subplot(num_rows, 2*num_cols, 2*i+1)
    f.im_plot(i, predictions[i], label_test, image_test)
    pyplot.subplot(num_rows, 2*num_cols, 2*i+2)
    f.plot_value_array(i, predictions[i], label_test)
pyplot.tight_layout()
pyplot.show()