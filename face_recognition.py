import cv2 as cv
import numpy as np
import os

haar_cascade = cv.CascadeClassifier('haarcascade_face.xml')

people = []
base_dir = r'F:\MS\SRH study\IV sem\Computer vision\image and video reading\image\face_rec_train'
for person in os.listdir(base_dir):
    if os.path.isdir(os.path.join(base_dir, person)):
        people.append(person)

#features=np.lead('features.npy')
#labels=np.lead('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

#validation
img=cv.imread(r"F:\MS\SRH study\IV sem\Computer vision\image and video reading\image\face_rec_train\Gina Stiebitz\1545805.entity.jpg")
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#cv.imshow('person',gray)

#detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

for (x,y,w,h) in faces_rect:
    faces_roi=gray[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(faces_roi)
    print(f'label = {label} with a confidence of {confidence}')
    cv.putText(img,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0),thickness=2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
cv.imshow('detected face',img)
cv.waitKey(0)