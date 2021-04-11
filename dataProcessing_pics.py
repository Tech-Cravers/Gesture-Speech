#stop ongoing process by 'Q' while selecting image viewer window

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import scipy.ndimage as sci


#custom function to crop the picture to symmetric size of 700x700 for 720x1280 input resolution
def cropIt(gray,top=10,left=290,right=290,down=10):
    w, h = gray.shape
    croped_image = gray[top:(w-down), right:(h-left)]
    return croped_image

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
    
    imgTh=cv2.adaptiveThreshold(img_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,3)
    ret,img_th = cv2.threshold(imgTh,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)

    #edges = cv2.Canny(img_resized,170, 300)
    return img_th

#choose the directory u want to process in which video data is present 
# videos must be named after the small case letter, it represents in gesture of hand
DATADIR = "D:\Project\gesture-Speech\dataset_20"

ALPHABET = [] #array containing letters to categorize and create path to video
alpha = 'a'
for i in range(0, 27): # { indicate there is no alpabet there additional feature to add
    ALPHABET.append(alpha) 
    alpha = chr(ord(alpha) + 1)

#contains the data set to be extracted
training_data=[] # [ feature , label ]format 
#print(ALPHABET)

IMG_SIZE=200
#to iterate over every alphabet
for category in ALPHABET:
    path = os.path.join(DATADIR,category)  # create path to directory
    print(path)
    for img_path in os.listdir(path):  # iterate over each image 
        #print(img_path)
        img0 = cv2.imread(os.path.join(path,img_path) ,cv2.IMREAD_GRAYSCALE)  # convert to array
        img_processed=preprocessing(img0,IMG_SIZE)
        cv2.imshow("input",img_processed)
        cv2.waitKey(1)

        class_num =ALPHABET.index(category)
        training_data.append([img_processed, class_num])  # add image and classification to our training_data
        
        if cv2.waitKey(1) & 0xFF == ord('q'):#break ongoing process by Q
            break




cv2.destroyAllWindows()
print("-------------Ultimated processing-----------------")

import random

random.shuffle(training_data)

x = []
y = []

for features,label in training_data:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

#Let's save this data, so that we don't need to keep calculating it every time we want to play with the neural network model:
import pickle

pickle_out = open("x.pickle","wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()
print('Pickle file created successfully if named <X.pickle> and <Y.pickle> !!!')
print('We can always load it in to our current script, or a totally new one by doing:')
print('pickle_in = open("X.pickle","rb")')
print('X = pickle.load(pickle_in)')
print('pickle_in = open("y.pickle","rb")')
print('y = pickle.load(pickle_in)')
print('Thank you :)')