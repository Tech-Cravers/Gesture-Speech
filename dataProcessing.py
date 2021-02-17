import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
#from tqdm import tqdm

#choose the directory u want to process
DATADIR = "D:\Project\gesture-Speech\\americanData"
ALPHABET = []

alpha = 'a'
for i in range(0, 26): 
    ALPHABET.append(alpha) 
    alpha = chr(ord(alpha) + 1)
print(ALPHABET)

for category in ALPHABET:
    path = os.path.join(DATADIR,category)+'.mp4'  # create path to video file
    print(path)
    cap = cv2.VideoCapture(path)
    while(cap.isOpened()):
        ret, frame = cap.read() #I want to modify it to 320x240. Just use ret = cap.set(3,320) and ret = cap.set(4,240)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(gray.shape)
        cv2.imshow('input gray',gray)

        w, h = gray.shape
        top=10
        left=290
        right=290
        down=10

        croped_image = gray[top:(w-down), right:(h-left)]

        print(croped_image.shape)
        
        cv2.imshow('cropped',croped_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if input("Enter 'stop' to inturrupt otherwise press anything:")=='stop':
            break
    cap.release()
    cv2.destroyAllWindows()    
    
