# Computational Intelligence Task 4.3
# Anastasios Gramemnos    9212
# avgramme@ece.auth.gr    March 2022

import time
import numpy as np

import tensorflow as tf
from tensorflow.keras.datasets import boston_housing

from keras.layers import Layer
from keras import backend as K
from keras.layers import Dropout
from keras.layers.core import Dense
from keras.models import Sequential
from keras.initializers import Initializer
from keras.callbacks import ModelCheckpoint

from scipy.io import savemat

from sklearn.cluster import KMeans


## Boston Housing dataset parameters.
num_features = 13

## Training parameters.
batch_size = 4
epochs = 100
validation_split = 0.2
test_split = 0.25
#best from tuning
rbf_units = 151
hidden_units = 128
dropout_rate = 0.5

## Define custom metrics.
def rmse_score(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true)))

def r2_score(y_true, y_pred):
    SS_res = np.sum(np.square(y_true - y_pred))
    SS_tot = np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - SS_res/SS_tot

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

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'centers': self.centers,
            'num_units': self.num_units,
            'sigma': self.sigma
        })
        return config

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
model = Sequential()
model.add(RBFLayer(centers=FindCenters(x_train, rbf_units), num_units=rbf_units, input_shape=(num_features,)))
model.add(Dense(hidden_units))
model.add(Dropout(rate=dropout_rate))
model.add(Dense(1))
model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
            loss=tf.keras.losses.MeanSquaredError())

## Save best model
best_weights_filepath = '.mdl_wts.hdf5'
mcp_save = ModelCheckpoint(best_weights_filepath, save_best_only=True, monitor='val_loss', mode='min')

## Train model.
tic = time.time()
history = model.fit(x=x_train, y=y_train, epochs=epochs, validation_split=validation_split, 
                    batch_size=batch_size, callbacks=[mcp_save])
toc = time.time()
print("Elapsed time = ", toc-tic)

## Evaluater model.
model.load_weights(best_weights_filepath)   #reload best weights
y_pred = model.predict(x_test)
rmse = rmse_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("RMSE= ", rmse)
print("R2= ", r2)

## Save results.
mdict = {
    'loss':history.history['loss'],
    'val_loss':history.history['val_loss'],
    "rmse":rmse,
    "r2":r2
}
savemat("Task4_3_output.mat", appendmat=True, mdict=mdict)