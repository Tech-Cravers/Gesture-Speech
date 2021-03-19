import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, InputLayer
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

import pickle
import numpy as np

x = pickle.load(open("xblack_raw.pickle","rb"))
y = pickle.load(open("yblack_raw.pickle","rb"))

#normalisation
trainlabel = y
trainimages = x
traingen=ImageDataGenerator(rotation_range=40,
                            zoom_range=0.2,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.2,
                            horizontal_flip=True,
                            rescale=1/255.0,
                            validation_split=0.2)

traindata_generator = traingen.flow(trainimages,trainlabel,subset='training')
validationdata_generator = traingen.flow(trainimages,trainlabel,subset='validation')

print(traindata_generator)

model = Sequential() #a sequential cnn model to create

#added a neuron to network
model.add(Conv2D(64,(3,3), input_shape=x.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


#flattening image data
model.add(Flatten())

model.add(Dense(256))
model.add(Activation('relu'))

model.add(Dense(26))
model.add(Activation('softmax'))#activation function

'''
#for hit and trial method
dense_layers = [1, 2]
layer_sizes = [32,64,128]
conv_layers = [1,2,3,4]

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

            #dense layer
            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(26))
            model.add(Activation('softmax'))#activation function
'''
opt=tf.keras.optimizers.Adam(learning_rate=1e-7)
model.compile(loss="sparse_categorical_crossentropy",
                        optimizer='adam',
                        metrics=['accuracy'])
model.summary()
model.fit(traindata_generator, batch_size=4, epochs=2, validation_data=validationdata_generator ) # change parameters to increase accuracy of data
model.save('model_name.model')#finally saving the model