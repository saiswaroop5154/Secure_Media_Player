# face_login.py

import cv2
import dlib
import numpy as np
import os

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_rec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

USER_FACE_DATA_DIR = "user_faces/"

def load_registered_faces():
    user_faces = {}
    for filename in os.listdir(USER_FACE_DATA_DIR):
        if filename.endswith(".npy"):
            username = filename.split("_")[0]
            user_faces[username] = np.load(f"{USER_FACE_DATA_DIR}{filename}")
    return user_faces

def recognize_face(face_descriptor, registered_faces):
    for username, registered_face in registered_faces.items():
        distance = np.linalg.norm(np.array(face_descriptor) - np.array(registered_face))
        if distance < 0.6:
            return username
    return None

def face_login():
    registered_faces = load_registered_faces()
    cap = cv2.VideoCapture(0)

    print("Looking for your face to log in...")

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) > 0:
            face = faces[0]
            shape = predictor(gray, face)
            face_descriptor = face_rec.compute_face_descriptor(frame, shape)
            recognized_user = recognize_face(face_descriptor, registered_faces)

            if recognized_user:
                print(f"Login successful! Welcome, {recognized_user}")
                break
            else:
                print("Face not recognized. Try again.")

        cv2.imshow('Face Login', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    face_login()
