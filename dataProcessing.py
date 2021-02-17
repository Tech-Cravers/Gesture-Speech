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
def resizeIt(img,size=100):
    img=np.float32(img)
    r,c=img.shape
    #filtering then resizing image
    filtered_img=sci.gaussian_filter(img,10)
    resized_img=cv2.resize(filtered_img,(size,size))
    return np.uint8(resized_img)


#choose the directory u want to process in which video data is present 
# videos must be named after the small case letter, it represents in gesture of hand
DATADIR = "D:\Project\gesture-Speech\\americanData"

ALPHABET = [] #array containing letters to categorize and create path to video
alpha = 'a'
for i in range(0, 26): 
    ALPHABET.append(alpha) 
    alpha = chr(ord(alpha) + 1)
#print(ALPHABET) # initialized it

#contains the data set to be extracted
training_data=[] # [ feature , label ]format 

#to iterate over every alphabet
for category in ALPHABET:
    path = os.path.join(DATADIR,category)+'.mp4'  # create path to video file
#    print(path)
    cap = cv2.VideoCapture(path) #to load video file 

    #stores index of every alphabet to categorize
    class_num =ALPHABET.index(category) # get the classification  (0 or 1 or 2 and soo on). 0=a 1=b 2=c ...

    #iterates over every frame of video
    while(cap.isOpened() ):
        ret, frame = cap.read() #If u want to modify it to 320x240. Just use ret = cap.set(3,320) and ret = cap.set(4,240)
        
        #to exit when frames are over or video is fully iterated
        if ret==False:
            break
        
        #conversion of image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #print(gray.shape)
        #cv2.imshow('input gray',gray)

        #cropping using custom function
        croped_image=cropIt(gray) #cropping image to get symmetric dimensions

        #normalising using custom function
        IMG_SIZE=100
        img=resizeIt(croped_image,IMG_SIZE) # resize to normalize data size
        #print(croped_image.shape)
        cv2.imshow('Result',img)# to give visual of each frame being processed
          
        training_data.append([img, class_num])  # add image and classification to our training_data
        #print(len(training_data))

        #use to save images
        #cv2.imwrite("result.bmp",croped_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):#break ongoing process by Q
            break
        
    #destroying every window after each video process to make way for next video
    cap.release()
    cv2.destroyAllWindows()
    
    print('---------------Completed: '+category+' !!!------------------')    

    #if input("Enter 'stop' to inturrupt otherwise press anything:")=='stop':
    #    break   

print("-------------Ultimated processing-----------------")

import random

random.shuffle(training_data)

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#Let's save this data, so that we don't need to keep calculating it every time we want to play with the neural network model:
import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()
print('Pickle file created successfully named <X.pickle> and <Y.pickle> !!!')
print('We can always load it in to our current script, or a totally new one by doing:')
print('pickle_in = open("X.pickle","rb")')
print('X = pickle.load(pickle_in)')
print('pickle_in = open("y.pickle","rb")')
print('y = pickle.load(pickle_in)')
print('Thank you :)')