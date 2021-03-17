import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import cv2

import pickle
import numpy as np

x = pickle.load(open("x.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

#normalisation
x=x/255.0


dense_layers = [1]
layer_sizes = [32]
conv_layers = [4]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)
            tensorboard = TensorBoard (log_dir="logs/{}".format(NAME))
            model = Sequential() #a sequential cnn model to create

            #added a neuron to network
            model.add(Conv2D(layer_size, (3,3), input_shape=x.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))
            #tring hit and trial to find best network to work 

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size,(3,3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2,2)))

            #flattening image data
            model.add(Flatten())
            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))#activation function

            model.compile(loss="binary_crossentropy",
                        optimizer="adam",
                        metrics=['accuracy'])

            model.fit(x, y, batch_size=4, epochs=1, validation_split=0.1, callbacks=[tensorboard]) # change parameters to increase accuracy of data
            model.save('model_name.model')#finally saving the model
