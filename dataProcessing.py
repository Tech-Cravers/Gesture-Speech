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
    resized_img=cv2.resize(img,(size,size))
    filtered_img=sci.median_filter(resized_img,8)
    return np.uint8(filtered_img)


#choose the directory u want to process in which video data is present 
# videos must be named after the small case letter, it represents in gesture of hand
DATADIR = "D:\Project\gesture-Speech\\americanData"
PROC_DIR= "D:\Project\gesture-Speech\\processed_image"

ALPHABET = [] #array containing letters to categorize and create path to video
alpha = 'a'
for i in range(0, 26): 
    ALPHABET.append(alpha) 
    alpha = chr(ord(alpha) + 1)
#print(ALPHABET) # initialized it

#contains the data set to be extracted
training_data=[] # [ feature , label ]format 
print(ALPHABET)

#to iterate over every alphabet
for category in ALPHABET:
    path = os.path.join(DATADIR,category)+'.mp4'  # create path to video file
#    print(path)
    cap = cv2.VideoCapture(path) #to load video file 

    #counting frames processed
    count = 0
    #stores index of every alphabet to categorize
    class_num =ALPHABET.index(category) # get the classification  (0 or 1 or 2 and soo on). 0=a 1=b 2=c ...
#    print(class_num)

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
        #cv2.waitKey(0)

        #cropping using custom function
        croped_img=cropIt(gray) #cropping image to get symmetric dimensions
        #cv2.imshow('cropped',croped_image)
        #cv2.waitKey(0)

        #normalising using custom function
        IMG_SIZE=500
        resized_img=resizeIt(croped_img,IMG_SIZE) # resize to normalize data size
        #cv2.imshow('resized',resized_img)
        #cv2.waitKey(0)

        #canny not using 
        '''
        edge_map = cv2.Canny(resized_img,50,150)
        img = edge_map
        cv2.imshow('Result',img)# to give visual of each frame being processed
        #cv2.waitKey(0)
        '''
        training_data.append([img, class_num])  # add image and classification to our training_data

        #use to save images
        newpath = r'D:\\Project\\gesture-Speech\\processed_image\\' 
        newpath = newpath+category
        if not os.path.exists(newpath):
            os.makedirs(newpath)# create folder if not present
        
        full_path = os.path.join(PROC_DIR,category,str(count))+'.bmp'
        cv2.imwrite(full_path, img)
        count=count+1 #updating to next value
        print("saved image no. "+str(count)+" to location : "+full_path)

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
print('Pickle file created successfully named <X.pickle> and <Y.pickle> !!!')
print('We can always load it in to our current script, or a totally new one by doing:')
print('pickle_in = open("X.pickle","rb")')
print('X = pickle.load(pickle_in)')
print('pickle_in = open("y.pickle","rb")')
print('y = pickle.load(pickle_in)')
print('Thank you :)')
