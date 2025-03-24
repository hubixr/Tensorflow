import tensorflow as tf
print(tf.__version__)

import numpy as np
import matplotlib.pyplot as plt

#Tensorflow does not support my GPU so I have to use the CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#Create data
X = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

#Visualise data
plt.scatter(X, y)

#1 Create model   
model = tf.keras.Sequential([
    tf.keras.Input(shape=(3,)), #input layer 3 neurons
    tf.keras.layers.Dense(100, activation='relu'), #hidden layer 100 neurons
    tf.keras.layers.Dense(100, activation='relu'), #hidden layer 100 neurons
    tf.keras.layers.Dense(100, activation='relu'), #hidden layer 100 neurons
    tf.keras.layers.Dense(1, activation='none') # output layer 1 neuron
])

#2 Compile model
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.Adam(lr=0.0001),
              metrics=['mae'])

#3 Fit model
# mode.fit(X_train, y_train, epochs=100)