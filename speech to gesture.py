#import library
import cv2
import speech_recognition as sr
import os

# Initialize recognizer class (for recognizing the speech)

r = sr.Recognizer()

# Reading Microphone as source
# listening the speech and store in audio_text variable

with sr.Microphone() as source:
    print('Turn on data traffic for accessing speech recogniser :)')
    print("Say Something")

    audio_text = r.listen(source)
    print("Time over, thanks")
# recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
    
    try:
        # using google speech recognition
        text = r.recognize_google(audio_text)
        print("Text: "+text)
    except:
         print("Sorry, I did not get that, please try again")

for elem in text:
    path = r'D:\Project\gesture-Speech\Gesture-Speech\images'
    if elem ==' ':
        full_path = os.path.join(path,'blank')+'.jpg'
        print(full_path)
        img = cv2.imread(full_path ,cv2.IMREAD_GRAYSCALE)
        cv2.imshow('blank',img)
        cv2.waitKey(1000)
        cv2.destroyWindow('blank')
    else:
        full_path = os.path.join(path,elem)+'.jpg'
        print(full_path)
        img = cv2.imread(full_path ,cv2.IMREAD_GRAYSCALE)
        cv2.imshow(elem,img)
        cv2.waitKey(1000)
        cv2.destroyWindow(elem)

cv2.destroyAllWindows()