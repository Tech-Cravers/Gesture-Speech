import cv2
import tensorflow as tf

CATEGORIES = ["Dog", "Cat"]
//still need to change to alphabets

def prepare(filepath):
    IMG_SIZE = 70  # change in accordance to input of model
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("64x3-CNN.model")

prediction = model.predict([prepare('doggo.jpg')])
print(prediction)  # will be a list in a list.
print(CATEGORIES[int(prediction[0][0])])
