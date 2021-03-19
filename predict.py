import cv2
import tensorflow as tf
import numpy as np

ALPHABET = [] #array containing letters to categorize 
alpha = 'a'
for i in range(0, 26): 
    ALPHABET.append(alpha) 
    alpha = chr(ord(alpha) + 1)
CATEGORIES=ALPHABET

def prepare(filepath):
    IMG_SIZE = 128  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("model_name.model")

prediction = model.predict([prepare('D:\Project\gesture-Speech\\testing data 1\\z.jpg')])
print(prediction)  # will be a list in a list.
print( ALPHABET[int(np.argmax(prediction[0]))])