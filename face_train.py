import os
import cv2 as cv
import numpy as np

# Folder names
people = []
base_dir = r'F:\MS\SRH study\IV sem\Computer vision\image and video reading\image\face_rec_train'

# Loop to get all the people's names (assuming each folder represents a person)
for person in os.listdir(base_dir):
    if os.path.isdir(os.path.join(base_dir, person)):
        people.append(person)
print(people)

# Set the directory where the images are present
DIR = base_dir

# Reading the Haar cascade file
haar_cascade = cv.CascadeClassifier('haarcascade_face.xml')

# Lists to hold features and labels
features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for image in os.listdir(path):  # Images inside each folder
            img_path = os.path.join(path, image)
            img_array = cv.imread(img_path)
            
            if img_array is None:
                print(f"Failed to load image {img_path}")
                continue

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('training done -------')
print(f'Length of the features = {len(features)}')
print(f'Length of the labels = {len(labels)}')

# Convert features and labels into numpy arrays
features = np.array(features, dtype='object')
labels = np.array(labels)

# Initialize face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the recognizer on the features list and the labels list
face_recognizer.train(features, labels)

#giving a path so that we can use this file and save it to use it in another directory
face_recognizer.save('face_trained.yml')

# Save the trained model and label data
face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)

