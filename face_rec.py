import face_recognition as fr
from face_recognition.api import face_encodings
import numpy as np
import pickle
import os
import matplotlib.image as plt


def load_data():
    file = open("data.pkl", "rb")
    encoded = pickle.load(file)
    file.close()
    return encoded

def unknown_image_encoded(img):
    """
    encode a face given the file name
    """
    face = fr.load_image_file("faces/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding


def main(im):
    print("Start")
    faces = load_data()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    print("Reading image")
    img = plt.imread(im)
    #img = cv2.imread(im, 1)
    #img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    #img = img[:,:,::-1]
    face_locations = fr.face_locations(img)
    unknown_face_encodings = fr.face_encodings(img, face_locations)
    width = int(img.shape[1] * 50 / 100)
    height = int(img.shape[0] * 50 / 100)
    dim = (width, height)

    print("Recognition")
    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)

        matches = fr.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        # use the known face with the smallest distance to the new face
        face_distances = fr.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    return name

