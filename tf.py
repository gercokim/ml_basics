import tensorflow as tf
import numpy as np

rank0 = tf.constant(4)
print(rank0)
rank2 = tf.constant([[1, 2, 3], [4, 5, 6]])
print(rank2[:, 1].numpy())