import os
import face_recognition as fr
import pickle
import matplotlib.image as plt
import numpy as np

list_encode = []
# list_com = []
match_face = False
percentage_com = 0
# webcam = cv2.VideoCapture(0)
# width = int(webcam.get(3))
# height = int(webcam.get(4))

def load_data():
    file = open("data.pkl", "rb")
    encoded = pickle.load(file)
    file.close()
    return encoded

def main(im) :
    try :
        print("Reading image")
        img = plt.imread(im)
        face_locations = fr.face_locations(img)
        unknown_face_encodings = fr.face_encodings(img, face_locations)

        for face_encoding in unknown_face_encodings:
            list_encode.append(face_encoding)

        try :
            # result = fr.compare_faces([list_encode[0][0]], list_encode[0][1])
            percentage = fr.face_distance([list_encode[0]], list_encode[1])
            percentage = (1-percentage)*100
            result = fr.compare_faces([list_encode[0]], list_encode[1])
            return result, percentage
        except Exception as e:
            return ("Encode error:"+e)
    except :
        pass
