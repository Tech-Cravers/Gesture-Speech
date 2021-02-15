from keras.models import Sequential 
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle

X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

X=X/255.0

model = Sequential()
model.add(    Conv2D(64, (3,3), input_shape=X.shape[1:])    )
model.add(Activation("relu"))
model.add(MaxPooling2D(poo_size=(2,2)))

model.add(Conv2D(64,(3,3))
model.add(Activation("relu"))
model.add(MaxPooling2D(poo_size=(2,2)))

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",
             optimizer="adam",
             metrics=['accuracy'])
model.fit(X, y, batch_size=32, epochs=3, validation_spit=0.1)