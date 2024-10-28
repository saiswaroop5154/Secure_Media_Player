# register_user.py

import cv2
import dlib
import numpy as np
import os

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_rec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

USER_FACE_DATA_DIR = "user_faces/"

if not os.path.exists(USER_FACE_DATA_DIR):
    os.makedirs(USER_FACE_DATA_DIR)

def register_user(username):
    cap = cv2.VideoCapture(0)
    print("Capturing face for user registration...")

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) > 0:
            face = faces[0]
            shape = predictor(gray, face)
            face_descriptor = face_rec.compute_face_descriptor(frame, shape)
            np.save(f"{USER_FACE_DATA_DIR}{username}_face_data.npy", face_descriptor)
            print(f"Face data saved for {username}.")
            break

        cv2.imshow('Register Face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    username = input("Enter your username: ")
    register_user(username)
