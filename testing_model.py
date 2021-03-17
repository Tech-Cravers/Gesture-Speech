import cv2
import tensorflow as tf
import numpy as np
import scipy.ndimage as sci
from gtts import gTTS

# This module is imported so that we can  
# play the converted audio 
import os 

#custom function to crop the picture to symmetric size of 700x700 for 720x1280 input resolution
def cropIt(gray,top=10,left=290,right=290,down=10):
    w, h = gray.shape
    croped_image = gray[top:(w-down), right:(h-left)]
    #print(gray.shape)
    return croped_image

#to normalize the images to same no. of pixels
def resizeIt(img,size=100,median=8):
    img=np.float32(img)
    r,c=img.shape
    #filtering then resizing image
    resized_img=cv2.resize(img,(size,size))
    filtered_img=sci.median_filter(resized_img,median)
    return np.uint8(filtered_img)


def prepare(img0):
    IMG_SIZE = 200  # change in accordance to input of model
    img0=np.float32(img0)

    img_cropped = cropIt(img0,10,145,145,10)
    #cv2.imshow(" ",np.uint8(img_cropped))
    #cv2.waitKey(0)
    img_resized=resizeIt(img_cropped,IMG_SIZE,5) # resize to normalize data size
    #ret,imgTh0=cv2.threshold(img_resized, 20, 255,cv2.THRESH_BINARY)
    imgTh=cv2.adaptiveThreshold(img_resized,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,5)
    #print(imgTh.shape)
    return imgTh

ALPHABET = [] #array containing letters to categorize 
alpha = 'a'
for i in range(0, 26): 
    ALPHABET.append(alpha) 
    alpha = chr(ord(alpha) + 1)

model = tf.keras.models.load_model("model_name.model")
cap = cv2.VideoCapture(0) #to load video file 
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret==False:
        break
    # Our operations on the frame come here
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_test = prepare(img_gray)

    # Display the resulting frame
    cv2.imshow('testing this',np.uint8(img_test))
    IMG_SIZE = 200
    prediction = model.predict([img_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)])
    text = ALPHABET[int(prediction[0][0])]
    #print(prediction)  # will be a list in a list.
    print(text)


  
    # The text that you want to convert to audio 
    mytext = str(text)
  
    # Language in which you want to convert 
    language = 'en'
  
    # Passing the text and language to the engine,  
    # here we have marked slow=False. Which tells  
    # the module that the converted audio should  
    # have a high speed 
    myobj = gTTS(text=mytext, lang=language, slow=False) 
  
    # Saving the converted audio in a mp3 file named 
    # welcome  
    myobj.save("welcome.mp3") 
  
    # Playing the converted file 
    os.system("mpg321 welcome.mp3") 
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

