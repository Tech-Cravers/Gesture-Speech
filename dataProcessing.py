import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import scipy.ndimage as sci
#from tqdm import tqdm

def cropIt(gray,top=10,left=290,right=290,down=10):
    w, h = gray.shape
    croped_image = gray[top:(w-down), right:(h-left)]
    return croped_image

def resizeIt(img,size=100):
    img=np.float32(img)
    r,c=img.shape

    filtered_img=sci.gaussian_filter(img,10)
    resized_img=cv2.resize(filtered_img,(size,size))
    return np.uint8(resized_img)


#choose the directory u want to process
DATADIR = "D:\Project\gesture-Speech\\americanData"
ALPHABET = []

alpha = 'a'
for i in range(0, 26): 
    ALPHABET.append(alpha) 
    alpha = chr(ord(alpha) + 1)
#print(ALPHABET)

training_data=[]
for category in ALPHABET:
    path = os.path.join(DATADIR,category)+'.mp4'  # create path to video file
    print(path)
    cap = cv2.VideoCapture(path)
    count=0
    while(cap.isOpened() ):
        ret, frame = cap.read() #I want to modify it to 320x240. Just use ret = cap.set(3,320) and ret = cap.set(4,240)
        
        if ret==False:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #print(gray.shape)
        #cv2.imshow('input gray',gray)

        croped_image=cropIt(gray) #cropping image to get symmetric dimensions

        img=resizeIt(croped_image)
        #print(croped_image.shape)
        cv2.imshow('Result',img)
        



        #count+=1
        #print(count) #counts frames
        
        #use to save images
        #cv2.imwrite("result.bmp",croped_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
    print('---------------Completed: '+category+' !!!------------------')    

    if input("Enter 'stop' to inturrupt otherwise press anything:")=='stop':
        break   

print("-------------Ultimated processing-----------------")
    
