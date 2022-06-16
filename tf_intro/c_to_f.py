from pickletools import optimize
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pyplot

# rudimentary training data 
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float) # feature
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float) # label

for i, c in enumerate(celsius):
    print("{} degrees Celsius = {} degrees fahrenheit".format(c, fahrenheit[i]))

# Building the model (Dense network)

# The first step is creating our layers, in this case just one
# input_shape[1] means input is one value, i.e. shape is one dimensional array with one value
# units=1 means one neuron in the layer (how many internal variables a layer has)
layer_0 = tf.keras.layers.Dense(units=1, input_shape=[1])

# The model
# Sequential model takes in a list of layers as argument, specifying calculation order from input to oupt
model = tf.keras.Sequential([layer_0])

# The common way to define models with Sequential
'''
model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1, input_shape=1)
        ]) 
'''

# Compiling the model
# model is given: 
    # loss function - the type of function to calculate the loss
    # optimizer function - a way of adjusting internal values to reduce loss
# 0.1 parameter is the learning rate
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

# Training the model
history = model.fit(celsius, fahrenheit, epochs=500, verbose=False)
print("Finished training the model")

# Displaying loss statistics
pyplot.xlabel('Epoch Number')
pyplot.ylabel('Loss Magnitude')
pyplot.plot(history.history['loss'])
pyplot.show()

# Using the model to predict values
print(model.predict([100.0]))

# Layer internal variables
# The weight and bias is close to the formula for celsius to fahrenheit
print("Layer variables: ", layer_0.get_weights())