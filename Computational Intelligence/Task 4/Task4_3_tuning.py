# Computational Intelligence Task 4.3
# Anastasios Gramemnos    9212
# avgramme@ece.auth.gr    March 2022

import time
import numpy as np

import tensorflow as tf
from tensorflow.keras.datasets import boston_housing

import keras_tuner as kt
from keras.layers import Layer
from keras import backend as K
from keras.layers import Dropout
from keras.layers.core import Dense
from keras.models import Sequential
from keras.initializers import Initializer

from scipy.io import savemat

from sklearn.cluster import KMeans


## Boston Housing dataset parameters.
num_features = 13

## Training parameters.
test_split = 0
learning_rate = 0.001
batch_size = 4
epochs = 100
verbose = 1
validation_split = 0.2
use_multiprocessing = True

## Create initilazer using KMean to set centers for RBF layer.
class FindCenters(Initializer):
    def __init__(self, X, n_clusters):
        self.X = X
        self.n_clusters = n_clusters
        super().__init__()

    def __call__(self, shape, dtype=None):
        km = KMeans(n_clusters=self.n_clusters)
        km.fit(self.X)
        centers = km.cluster_centers_
        return np.transpose(centers)

## Create custom layer to implement RBF hidden layer.
class RBFLayer(Layer):
    def __init__(self, centers,  num_units, sigma=None, **kwargs):
        self.centers = centers
        self.sigma = sigma
        self.num_units = num_units
        super(RBFLayer, self).__init__(**kwargs)     

    def build(self, input_shape):
        #define centers
        self.c_i = self.add_weight(
                        name='c_i',
                        shape=(input_shape[1], self.num_units),
                        initializer=self.centers,
                        trainable=False)
        #define sigma
        if self.sigma == None:
            dmax = 0
            for i in range(self.num_units):
                for j in range(self.num_units):
                    d = np.linalg.norm(self.c_i[:,i]-self.c_i[:,j])
                    if d > dmax:
                        dmax = d
            self.sigma = dmax/np.sqrt((2*self.num_units), dtype=np.float32)
        super(RBFLayer, self).build(input_shape)

    def call(self, x):
        C_i = tf.expand_dims(tf.transpose(self.c_i), -1)
        res = K.exp(tf.math.reduce_sum(-K.pow(C_i-tf.transpose(x), 2), axis=1)/(2*(self.sigma**2)))
        res = tf.transpose(res)
        return res

## Prepare data.
# Load dataset
(x_train, y_train), (x_test, y_test) = boston_housing.load_data(test_split=test_split)
x_train = np.array(x_train, np.float32)
y_train = np.array(y_train, np.float32)
x_test = np.array(x_test, np.float32)
y_test = np.array(y_test, np.float32)
# Normalize data
mins = np.min(x_train, axis=0)
maxs = np.max(x_train, axis=0)
x_train = (x_train - mins)/(maxs-mins)
x_test = (x_test - mins)/(maxs-mins)
# Standarize data
# x_train = (x_train - np.mean(x_train, axis=0))/np.std(x_train, axis=0)
# x_test = (x_test - np.mean(x_test, axis=0))/np.std(x_test, axis=0)

## Define RBF model.
def model_builder(hp):
    #Hyperparameters for test
    samples_percent = np.array([0.05, 0.15, 0.3, 0.5])
    num_units = (len(x_train[:,0]*(1-validation_split))*samples_percent).astype(np.int)
    hp_rbf_units = hp.Choice('rbf_units', values=num_units.tolist())
    hp_hidden_units = hp.Choice('hidden_units', values=[32, 64, 128, 256])
    hp_dropout_rate = hp.Choice('dropout_rate', values=[0.2, 0.35, 0.5])

    #Layers definition
    model = Sequential()
    model.add(RBFLayer(centers=FindCenters(x_train, hp_rbf_units), num_units=hp_rbf_units, input_shape=(num_features,)))
    model.add(Dense(hp_hidden_units))
    model.add(Dropout(rate=hp_dropout_rate))
    model.add(Dense(1))
    #Model compilation
    model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=tf.keras.losses.MeanSquaredError())
    return model

## Instantiate tuner and perform search
tuner = kt.RandomSearch(
    hypermodel=model_builder,
    objective='val_mean_squared_error',
    max_trials=48
)
tic = time.time()
tuner.search(x=x_train, y=y_train, epochs=epochs, validation_split=validation_split, 
            batch_size=batch_size, use_multiprocessing=use_multiprocessing, verbose=verbose)
toc = time.time()

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
print("Elapsed time = ", toc-tic)
print("Best parameters: ", best_hps.values)
