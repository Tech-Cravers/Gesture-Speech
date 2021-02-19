import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np

x = pickle.load(open("x.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

x=x/255.0

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=x.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",
             optimizer="adam",
             metrics=['accuracy'])

model.fit(x, y, batch_size=32, epochs=10, validation_split=0.1)
"""
Traceback (most recent call last):
  File "d:/Project/gesture-Speech/Gesture-Speech/trainingCNN.py", line 31, in <module>
    model.fit(x, y, batch_size=32, epochs=10, validation_split=0.1)
  File "C:\Users\rushi\AppData\Local\Programs\Python\Python38\lib\site-packages\tensorflow\python\keras\engine\training.py", line 1040, in fit
    data_adapter.train_validation_split(
  File "C:\Users\rushi\AppData\Local\Programs\Python\Python38\lib\site-packages\tensorflow\python\keras\engine\data_adapter.py", line 1357, in train_validation_split
    raise ValueError(
ValueError: `validation_split` is only supported for Tensors or NumPy arrays, found following types in the input: [<class 'int'>, <class 'int'>, <class 'int'>, <class 'int'>, <class 'int'>,...<class 'int'>]
"""
