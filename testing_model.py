import cv2
import tensorflow as tf
import numpy as np
import scipy.ndimage as sci
from gtts import gTTS

# This module is imported so that we can  
# play the converted audio 
import os

#to normalize the images to same no. of pixels
def resizeIt(img,size=100,median=2):
    img=np.float32(img)
    r,c=img.shape
    #filtering then resizing image
    resized_img=cv2.resize(img,(size,size))
    filtered_img=sci.median_filter(resized_img,median)
    return np.uint8(filtered_img)

def preprocessing(img0,IMG_SIZE=100):
    img_resized=resizeIt(img0,IMG_SIZE,1) # resize to normalize data size
    #cv2.imshow("intermidieate",img_resized)
    img_blur = cv2.GaussianBlur(img_resized,(5,5),0)
    ret,img_th = cv2.threshold(img_blur,30,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)  
    imgTh=cv2.adaptiveThreshold(img_th ,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,40)
    #edges = cv2.Canny(img_resized,170, 300)
    return img_th

ALPHABET = [] #array containing letters to categorize 
alpha = 'a'
for i in range(0, 27): 
    ALPHABET.append(alpha) 
    alpha = chr(ord(alpha) + 1)

prev=""
model = tf.keras.models.load_model("model_name.model")
cap = cv2.VideoCapture(0) #to load video file 

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret==False:
        break
    # Our operations on the frame come here
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    IMG_SIZE = 200
    #print(img_gray.shape)
    w, h = img_gray.shape
    top=180
    left=h-195
    down=w-50
    right=195
    img_rect = cv2.rectangle(img_gray,(right,top),(left,down),(255,0,0),2)
    img_gray = img_gray[top:down, right:left]
    #print(img_gray.shape)
    img_test = preprocessing(img_gray,IMG_SIZE)

    cv2.imshow('whole input frame', np.uint8(img_rect))
    cv2.imshow('qwerty',img_test)
    # Display the resulting frame
    #cv2.imshow('testing this',np.uint8(img_test))
    
    prediction = model.predict([img_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)])
    #print(img_test)
    
    text = ALPHABET[int(np.argmax(prediction[0]))]

    now=text
    if now!=prev:
        print(text)
    prev=text

    '''
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
    myobj.save("audio.mp3") 

    # Playing the converted file 
    os.system("mpg123 welcome.mp3")
    
    from playsound import playsound
    playsound('audio.mp3')
    '''
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()