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
def resizeIt(img,size=100,median_val=1):
    img=np.float32(img)
    r,c=img.shape
    #filtering then resizing image
    resized_img=cv2.resize(img,(size,size))
    filtered_img=sci.median_filter(resized_img,median_val)
    return np.uint8(filtered_img)


#choose the directory u want to process in which video data is present 
# videos must be named after the small case letter, it represents in gesture of hand
DATADIR = "D:\Project\gesture-Speech\\videoData"
PROC_DIR= "D:\Project\gesture-Speech\\pic_data mix"

ALPHABET = [] #array containing letters to categorize and create path to video
alpha = 'a'
for i in range(0, 26): 
    ALPHABET.append(alpha) 
    alpha = chr(ord(alpha) + 1)
#print(ALPHABET) # initialized it
print(ALPHABET)

#to iterate over every alphabet
for category in ALPHABET:
    path = os.path.join(DATADIR,category)+'.mp4'  # create path to video file
    print(path)
    cap = cv2.VideoCapture(path) #to load video file 

    #counting frames processed
    count = 0
    #stores index of every alphabet to categorize
    class_num = ALPHABET.index(category) # get the classification  (0 or 1 or 2 and soo on). 0=a 1=b 2=c ...
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

        #normalising using custom function
        IMG_SIZE=100
        resized_img=resizeIt(gray,IMG_SIZE) # resize to normalize data size
        #cv2.imshow('resized',resized_img)
        #cv2.waitKey(0)

        #use to save images
        newpath = r'D:\\Project\\gesture-Speech\\processed_image\\' 
        newpath = newpath+category
        if not os.path.exists(newpath):
            os.makedirs(newpath)# create folder if not present
        
        full_path = os.path.join(PROC_DIR,category,str(count))+'.jpg'
        cv2.imwrite(full_path,resized_img)
        count=count+1 #updating to next value
        print("saved image no. "+str(count)+" to location : "+full_path)

        if cv2.waitKey(1) & 0xFF == ord('q'):#break ongoing process by Q
            break
        
    #destroying every window after each video process to make way for next video
    cap.release()
    cv2.destroyAllWindows()
    
    print('---------------Completed: '+category+' !!!------------------')     

print("-------------Ultimated processing-----------------")
