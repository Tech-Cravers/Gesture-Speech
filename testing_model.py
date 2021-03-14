import cv2
import tensorflow as tf

ALPHABET = [] #array containing letters to categorize 
alpha = 'a'
for i in range(0, 26): 
    ALPHABET.append(alpha) 
    alpha = chr(ord(alpha) + 1)

def prepare(filepath):
    IMG_SIZE = 70  # change in accordance to input of model
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("model_name.model")

//test image is given as jpg format here
prediction = model.predict([prepare('test0.jpg')])
print(prediction)  # will be a list in a list.
print(ALPHABET[int(prediction[0][0])])
