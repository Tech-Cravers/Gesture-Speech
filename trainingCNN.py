import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np

x = pickle.load(open("x.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

x=x/255.0
for i in range(0,1):
    model = Sequential() #a sequential cnn model to create

    #added a neuron to network
    model.add(Conv2D(64, (3,3), input_shape=x.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    print("size of kernel : "+str(i))
    #tring hit and trial to find best network to work 

    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    #flattening image data
    model.add(Flatten())
    model.add(Dense(64))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))#activation function

    model.compile(loss="binary_crossentropy",
                 optimizer="adam",
                 metrics=['accuracy'])

    model.fit(x, y, batch_size=4, epochs=1, validation_split=0.1) # change parameters to increase accuracy of data
