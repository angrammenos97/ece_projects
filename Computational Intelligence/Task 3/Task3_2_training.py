# Computational Intelligence Task 3.2
# Anastasios Gramemnos    9212
# avgramme@ece.auth.gr    March 2022

from numpy.core.defchararray import array
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import time
from sklearn.metrics import confusion_matrix

## MNIST dataset parameters.
num_classes = 10    # total classes (0-9 digits).
num_features = 784  # data features (img shape: 28*28).

## Training parameters.
epochs = 1000
verbose = 1
validation_split = 0.2
shuffle = True
use_multiprocessing = True

## Best model hyperparameters
hp_l1_units = 128   #layer 1 number of units
hp_l2_units = 256   #layer 2 number of units
hp_a = 1e-06        #l2 regularization
hp_lr = 0.001       #learning rate

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
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=hp_lr),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy', f1_score, precision_m, recall_m]
)

model.summary()

## Save best model
best_weights_filepath = '.mdl_wts.hdf5'
mcp_save = ModelCheckpoint(best_weights_filepath, save_best_only=True, monitor='val_loss', mode='min')

## Train model
tic = time.time()
history = model.fit(
    x=x_train, y=y_train, epochs=epochs, verbose=verbose,
    validation_split=validation_split, shuffle=shuffle,
    use_multiprocessing=use_multiprocessing,
    callbacks=[mcp_save]
)
toc = time.time()
print("Elapsed time = ", toc-tic)

## Evaluate model
model.load_weights(best_weights_filepath)   #reload best weights
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
results = confusion_matrix(y_test, y_pred)
print(results)

## Save resutls
mdict = {
    'loss':history.history['loss'],
    'val_loss':history.history['val_loss'],
    'accuracy':history.history['accuracy'],
    'val_accuracy':history.history['val_accuracy'],
    'precision':history.history['precision_m'],
    'val_precision':history.history['val_precision_m'],
    'recall':history.history['recall_m'],
    'val_recall':history.history['val_recall_m'],
    'f1_score':history.history['f1_score'],
    'val_f1_score':history.history['val_f1_score'],
    'confusion_matrix':results
}
savemat("Task3_2_output.mat", appendmat=True, mdict=mdict)