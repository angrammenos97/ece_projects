# Computational Intelligence Task 3.1
# Anastasios Gramemnos    9212
# avgramme@ece.auth.gr    March 2022

from numpy.core.defchararray import array
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import time

## MNIST dataset parameters.
num_classes = 10    # total classes (0-9 digits).
num_features = 784  # data features (img shape: 28*28).

## Training parameters.
epochs = 100
batch_size = 1
verbose = 2
validation_split = 0.2
shuffle = True
use_multiprocessing = True

## Prepare MNIST data.
#Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#Input data edit
#Append default train and test data
x_data = np.empty((x_train.shape[0]+x_test.shape[0],x_train.shape[1], x_train.shape[2]), dtype=np.uint8)
x_data[0:x_train.shape[0],:,:] = x_train
x_data[x_train.shape[0]:x_train.shape[0]+x_test.shape[0],:,:] = x_test
#Convert to float32.
# x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_data = np.array(x_data, np.float32)
#Flatten images to 1-D vector of 784 features (28*28).
x_data = x_data.reshape([-1, num_features])
#Normalize images value from [0, 255] to [0, 1].
x_data = x_data / 255.
#Output data edit
#Append default train and test data
y_data = np.append(y_train, y_test)

## Define Sequential model with 4 layers
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(num_features, )),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

## Train model
tic = time.time()
history = model.fit(
    x=x_data, y=y_data, batch_size=batch_size, epochs=epochs, verbose=verbose,
    validation_split=validation_split, shuffle=shuffle,
    use_multiprocessing=use_multiprocessing
)
toc = time.time()
print("Elapsed time = ", toc-tic)

## Save resutls
mdict = {
    'default_b1_loss':history.history['loss'],
    'default_b1_val_loss':history.history['val_loss'],
    'default_b1_accuracy':history.history['accuracy'],
    'default_b1_val_accuracy':history.history['val_accuracy']
}
savemat("Task3_1_default_b1.mat", appendmat=True, mdict=mdict)