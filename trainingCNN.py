import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, InputLayer
#from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

import pickle
import numpy as np

x = pickle.load(open("x.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

x = x/255.0
#normalisation

model = Sequential() #a sequential cnn model to create

#added a neuron to network
model.add(Conv2D(32,(2,2), input_shape=x.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(64,(2,2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(128 ,(2,2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3)))

#flattening image data
model.add(Flatten())

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(27))#extra 1 output to find if no alphabet is detected 26 alphabets
model.add(Activation('softmax'))#activation function

opt=tf.keras.optimizers.Adam(learning_rate=1e-7)#prev value: 1e-7
model.compile(loss="sparse_categorical_crossentropy",
                        optimizer='adam',
                        metrics=['accuracy']) 
model.summary()
'''
NUM_TRAIN_IMAGES = len(y)*0.009
NUM_TEST_IMAGES = len(y)*0.001
NUM_EPOCHS = 20 
BS = 10

trainlabel = y
trainimages = x
traingen=ImageDataGenerator(rotation_range=40,
                            zoom_range=0.2,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.2,
                            fill_mode='nearest',
                            horizontal_flip=True,
                            rescale=1/255.0,
                            validation_split=0.1)

traindata_generator = traingen.flow(trainimages,trainlabel,subset='training')
validationdata_generator = traingen.flow(trainimages,trainlabel,subset='validation')
'''
model.fit(x,y, batch_size=5, epochs=2, validation_split=0.1) # change parameters to increase accuracy of data
'''
H = model.fit(
	x=traindata_generator,
	steps_per_epoch=NUM_TRAIN_IMAGES // BS,
	validation_data=validationdata_generator,
	validation_steps=NUM_TEST_IMAGES // BS,
	epochs=NUM_EPOCHS)
'''
model.save('model_name.model')#finally saving the model