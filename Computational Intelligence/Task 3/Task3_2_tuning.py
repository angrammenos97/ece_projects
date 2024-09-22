# Computational Intelligence Task 3.2
# Anastasios Gramemnos    9212
# avgramme@ece.auth.gr    March 2022

from numpy.core.defchararray import array
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import keras_tuner as kt
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import time

## MNIST dataset parameters.
num_classes = 10    # total classes (0-9 digits).
num_features = 784  # data features (img shape: 28*28).

## Training parameters.
epochs = 10000
patience = 200
batch_size = 512
verbose = 2
validation_split = 0.2
shuffle = True
use_multiprocessing = True

## Prepare MNIST data.
#Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize pixel values between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

## Define custom metrics
def recall_m(y_true, y_pred):
    y_true = K.ones_like(y_true) 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    
    recall = true_positives / (all_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    y_true = K.ones_like(y_true) 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

## Define Sequential model with 4 layers
def model_builder(hp):
    #Hyperparameters for test
    hp_l1_units = hp.Choice('l1_units', values=[64,128])
    hp_l2_units = hp.Choice('l2_units', values=[256,521])
    hp_a = hp.Choice('l2', values=[0.1,0.001,0.000001])
    hp_lr = hp.Choice('learning_rate', values=[0.1,0.01,0.001])

    #Layers definition
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(
            units=hp_l1_units, 
            activation='relu',
            kernel_initializer=tf.keras.initializers.HeNormal(),
            kernel_regularizer=tf.keras.regularizers.L2(l2=hp_a)
        ),
        tf.keras.layers.Dense(
            units=hp_l2_units, 
            activation='relu',
            kernel_initializer=tf.keras.initializers.HeNormal(),
            kernel_regularizer=tf.keras.regularizers.L2(l2=hp_a)
        ),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    #Model compilation
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=hp_lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy', f1_score, precision_m, recall_m]
    )
    return model

## Instantiate tuner and perform search
tuner = kt.RandomSearch(
    model_builder,
    objective='val_accuracy',
    max_trials=36
)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
tic = time.time()
tuner.search(x_train, y_train, epochs=epochs, validation_split=validation_split, callbacks=[stop_early])
toc = time.time()

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
print("Elapsed time = ", toc-tic)
